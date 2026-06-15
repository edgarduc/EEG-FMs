from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ts_fms.config import ExperimentConfig
from ts_fms.data.splits import SplitIndices
from ts_fms.data.types import EEGStudy, EEGTensorDataset
from ts_fms.models import ConcatenatedLinearProbe, EEGAttentionProbe, FrozenBackbone
from ts_fms.training.metrics import classification_metrics


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_and_evaluate(
    *,
    backbone_name: str,
    backbone: FrozenBackbone,
    study: EEGStudy,
    splits: SplitIndices,
    config: ExperimentConfig,
) -> dict[str, Any]:
    device = resolve_device(config.train.device)
    backbone = backbone.to(device)
    probe = _build_probe(backbone, study, config, device)
    probe = probe.to(device)

    loaders = {
        "train": _loader(study, splits.train, config, shuffle=True),
        "val": _loader(study, splits.val, config, shuffle=False),
        "test": _loader(study, splits.test, config, shuffle=False),
    }

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start = time()

    for epoch in range(1, config.train.epochs + 1):
        train_loss = _train_one_epoch(backbone, probe, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(backbone, probe, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        score = val_metrics["balanced_accuracy"]
        if score > best_val:
            best_val = score
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.train.patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    test_metrics = evaluate(backbone, probe, loaders["test"], device)
    result = {
        "backbone": backbone_name,
        "pooling": config.head.pooling,
        "study": config.study.name,
        "class_names": study.class_names,
        "num_trials": int(len(study.y)),
        "num_channels": int(study.num_channels),
        "splits": {
            "train_subjects": splits.train_subjects,
            "val_subjects": splits.val_subjects,
            "test_subjects": splits.test_subjects,
            "train_trials": int(len(splits.train)),
            "val_trials": int(len(splits.val)),
            "test_trials": int(len(splits.test)),
        },
        "best_val_balanced_accuracy": float(best_val),
        "test": test_metrics,
        "history": history,
        "elapsed_seconds": time() - start,
    }

    _save_result(result, config)
    return result


def evaluate(
    backbone: FrozenBackbone,
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    backbone.eval()
    probe.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = probe(backbone.encode_channels(x))
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            targets.append(y.numpy())
    return classification_metrics(np.concatenate(targets), np.concatenate(predictions))


def _train_one_epoch(
    backbone: FrozenBackbone,
    probe: EEGAttentionProbe,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    backbone.eval()
    probe.train()
    total_loss = 0.0
    total_examples = 0

    for x, y in tqdm(loader, leave=False, desc="train"):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            embeddings = backbone.encode_channels(x)
        logits = probe(embeddings)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(y)
        total_examples += len(y)

    return total_loss / max(total_examples, 1)


def _build_probe(
    backbone: FrozenBackbone,
    study: EEGStudy,
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    sample = torch.from_numpy(study.x[:1]).float().to(device)
    with torch.no_grad():
        sample_embeddings = backbone.encode_channels(sample)
    embedding_dim = int(sample_embeddings.shape[-1])
    if config.head.pooling == "concat_linear":
        return ConcatenatedLinearProbe(
            embedding_dim=embedding_dim,
            num_channels=study.num_channels,
            num_classes=study.num_classes,
        )
    if config.head.pooling != "attention":
        raise ValueError("head.pooling must be either 'attention' or 'concat_linear'.")
    return EEGAttentionProbe(
        embedding_dim=embedding_dim,
        num_channels=study.num_channels,
        electrode_positions=torch.from_numpy(study.electrode_positions),
        num_classes=study.num_classes,
        hidden_dim=config.head.hidden_dim,
        num_heads=config.head.num_heads,
        dropout=config.head.dropout,
        channel_id_dim=config.head.channel_id_dim,
        position_dim=config.head.position_dim,
    )


def _loader(study: EEGStudy, indices: np.ndarray, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        EEGTensorDataset(study, indices),
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _save_result(result: dict[str, Any], config: ExperimentConfig) -> None:
    output_dir = config.train.output_dir / config.study.name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result['backbone']}_{config.head.pooling}_seed{config.study.seed}.json"
    payload = {
        "config": _jsonable(asdict(config)),
        "result": _jsonable(result),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
