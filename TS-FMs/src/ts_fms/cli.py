from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_fms.config import load_config, merge_cli_overrides
from ts_fms.data import load_study
from ts_fms.data.splits import subject_disjoint_split
from ts_fms.models import build_backbone
from ts_fms.training import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train frozen TS-FM EEG attention probes and evaluate on a subject-disjoint test split."
    )
    parser.add_argument("--config", type=Path, help="YAML config file.")
    parser.add_argument("--study", choices=["bcic-iv-2a", "mat"], help="Study to download/load.")
    parser.add_argument("--data-dir", type=Path, help="Dataset download/cache root.")
    parser.add_argument("--cache-dir", type=Path, help="Auxiliary cache root.")
    parser.add_argument("--subjects", type=int, nargs="+", help="Optional subject IDs to include.")
    parser.add_argument("--max-subjects", type=int, help="Limit subjects for smoke tests.")
    parser.add_argument("--seed", type=int, help="Subject split/training seed.")
    parser.add_argument("--sfreq", type=float, help="Resampling frequency.")
    parser.add_argument("--window-seconds", type=float, help="MAT window length.")
    parser.add_argument("--stride-seconds", type=float, help="MAT window stride.")

    parser.add_argument("--moment-checkpoint", help="MOMENT Hugging Face checkpoint.")
    parser.add_argument("--tspulse-checkpoint", help="TSPulse Hugging Face model ID or local path.")
    parser.add_argument("--tspulse-revision", help="TSPulse Hugging Face revision/variant.")
    parser.add_argument(
        "--tspulse-module",
        help="Optional package.module:factory that returns a torch.nn.Module for TSPulse.",
    )
    parser.add_argument("--sequence-length", type=int, help="Crop/pad trials to this sample count.")
    parser.add_argument(
        "--backbones",
        default="moment,tspulse",
        help="Comma-separated protocol order. Default: moment,tspulse.",
    )
    parser.add_argument(
        "--allow-random-backbone",
        action="store_true",
        default=None,
        help="Use only for pipeline smoke tests when a real checkpoint is unavailable.",
    )

    parser.add_argument("--epochs", type=int, help="Training epochs per backbone.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--device", help="Device: auto, cpu, cuda, cuda:0, mps.")
    parser.add_argument("--output-dir", type=Path, help="Directory for JSON result files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = merge_cli_overrides(load_config(args.config), args)
    study = load_study(config.study, config.model.sequence_length)
    splits = subject_disjoint_split(
        study.subjects,
        test_fraction=config.study.test_subject_fraction,
        val_fraction=config.study.val_subject_fraction,
        seed=config.study.seed,
    )

    requested_backbones = [item.strip().lower() for item in args.backbones.split(",") if item.strip()]
    results = []
    for name in requested_backbones:
        if name == "moment":
            checkpoint = config.model.moment_checkpoint
            module_factory = None
        elif name == "tspulse":
            checkpoint = config.model.tspulse_checkpoint
            module_factory = config.model.tspulse_module
        elif name == "random" and config.model.allow_random_backbone:
            checkpoint = None
            module_factory = None
        else:
            raise ValueError(f"Unsupported backbone '{name}'.")

        backbone_name = name
        if name == "random":
            backbone_kind = "random"
        else:
            backbone_kind = name
        print(f"Running {backbone_name} on {config.study.name}...")
        backbone = build_backbone(
            backbone_kind,
            checkpoint,
            module_factory,
            revision=config.model.tspulse_revision if name == "tspulse" else None,
        )
        result = train_and_evaluate(
            backbone_name=backbone_name,
            backbone=backbone,
            study=study,
            splits=splits,
            config=config,
        )
        results.append(result)
        print(json.dumps({"backbone": backbone_name, "test": result["test"]}, indent=2))

    summary = {
        "study": config.study.name,
        "split_subjects": {
            "train": splits.train_subjects,
            "val": splits.val_subjects,
            "test": splits.test_subjects,
        },
        "results": [{"backbone": item["backbone"], "test": item["test"]} for item in results],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
