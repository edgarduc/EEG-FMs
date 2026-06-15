from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from .constants import DATA_DIR, EEGMAT_DIR, OUTPUTS_DIR
from .data import ensure_eegmat, load_eegmat_trials, read_subject_info, split_subjects
from .models import build_feature_extractor
from .training import run_probe_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run calibration and uncertainty experiments on EEGMAT with one frozen EEG foundation model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("reve", "cbramod", "labram"),
        help="Model to run for this invocation.",
    )
    parser.add_argument("--dataset", default="mat", choices=("mat",), help="Dataset to use. Currently only EEGMAT.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Project data directory.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR, help="Directory for one-run JSON outputs.")
    parser.add_argument("--force-data-download", action="store_true", help="Re-download EEGMAT files even if present.")

    parser.add_argument("--seed", type=int, default=0, help="Seed for subject split and probe initialization.")
    parser.add_argument("--window-seconds", type=float, default=10.0, help="EEG trial/window duration.")
    parser.add_argument("--stride-seconds", type=float, default=None, help="Window stride. Defaults to window duration.")
    parser.add_argument("--normalize", choices=("per_window", "none"), default="per_window", help="EEG normalization.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional small-data debug limit.")

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for frozen feature extraction.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--epochs", type=int, default=300, help="Linear probe training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Linear probe learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Linear probe AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout before the linear classifier.")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Number of independently initialized probes.")

    parser.add_argument("--embedding-pooling", choices=("flatten", "mean"), default="flatten")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal error rate; 0.1 targets 90%% coverage.")
    parser.add_argument("--ece-bins", type=int, default=15, help="Number of ECE bins.")
    parser.add_argument("--montage-drop-prob", type=float, default=0.2, help="Channel zeroing probability for montage shift.")
    parser.add_argument(
        "--snr-levels",
        type=float,
        nargs="+",
        default=[10.0, 5.0, 0.0, -5.0, -15.0],
        help="Gaussian-noise SNR levels in dB.",
    )

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token for gated model access. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument("--cbramod-source-dir", type=Path, default=None, help="Optional local CBraMod source checkout.")
    parser.add_argument("--cbramod-weights-path", type=Path, default=None, help="Optional local CBraMod weights path.")
    parser.add_argument("--labram-source-dir", type=Path, default=None, help="Optional local LaBraM source checkout.")
    parser.add_argument("--labram-weights-path", type=Path, default=None, help="Optional local LaBraM weights path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.ensemble_size < 1:
        raise SystemExit("--ensemble-size must be at least 1.")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be between 0 and 1.")
    if not 0.0 <= args.montage_drop_prob < 1.0:
        raise SystemExit("--montage-drop-prob must be in [0, 1).")

    device = resolve_device(args.device)
    data_dir = args.data_dir / "eegmat" / EEGMAT_DIR.name
    ensure_eegmat(data_dir, force_download=args.force_data_download)

    trials = load_eegmat_trials(
        data_dir,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        target_sfreq=200.0,
        normalize=args.normalize,
        max_subjects=args.max_subjects,
    )
    splits = split_subjects(trials, seed=args.seed, train_fraction=0.6, calibration_fraction=0.2)
    extractor = build_feature_extractor(
        args.model,
        device=device,
        pooling=args.embedding_pooling,
        hf_token=args.hf_token,
        cbramod_source_dir=args.cbramod_source_dir,
        cbramod_weights_path=args.cbramod_weights_path,
        labram_source_dir=args.labram_source_dir,
        labram_weights_path=args.labram_weights_path,
    )

    metrics = run_probe_experiment(
        extractor=extractor,
        trials=trials,
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        ensemble_size=args.ensemble_size,
        alpha=args.alpha,
        ece_bins=args.ece_bins,
        montage_drop_prob=args.montage_drop_prob,
        snr_levels=args.snr_levels,
    )

    run = {
        "config": _safe_config(args, device),
        "dataset": {
            "name": "eegmat",
            "data_dir": str(data_dir),
            "num_trials": len(trials),
            "num_subjects": len({trial.subject for trial in trials}),
            "subject_info_rows": len(read_subject_info(data_dir)),
            "label_mapping": {"0": "background_before_arithmetic", "1": "during_arithmetic"},
        },
        "subject_splits": split_subject_lists(trials, splits),
        "metrics": metrics,
    }
    output_path = write_run_json(run, args.output_dir, args.model, args.seed)
    print(f"Wrote {output_path}")


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_subject_lists(trials, splits: dict[str, list[int]]) -> dict[str, list[str]]:
    return {
        name: sorted({trials[idx].subject for idx in indices})
        for name, indices in splits.items()
    }


def write_run_json(run: dict[str, object], output_dir: Path, model: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir / model / f"{timestamp}_seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        json.dump(_jsonable(run), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def _safe_config(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    config = vars(args).copy()
    config["hf_token"] = "<provided>" if config.get("hf_token") else None
    config["device_resolved"] = str(device)
    return config


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
