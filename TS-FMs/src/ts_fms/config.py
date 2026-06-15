from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StudyConfig:
    name: str = "bcic-iv-2a"
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    subjects: list[int] | None = None
    test_subject_fraction: float = 0.2
    val_subject_fraction: float = 0.2
    seed: int = 13
    sfreq: float = 128.0
    fmin: float = 1.0
    fmax: float = 40.0
    tmin: float = 0.0
    tmax: float = 4.0
    window_seconds: float = 4.0
    stride_seconds: float = 4.0
    max_subjects: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    moment_checkpoint: str = "AutonLab/MOMENT-1-large"
    tspulse_checkpoint: str | None = "ibm-granite/granite-timeseries-tspulse-r1"
    tspulse_revision: str | None = "tspulse-block-dualhead-512-p16-r1"
    tspulse_module: str | None = None
    sequence_length: int = 512
    embedding_dim: int | None = None
    allow_random_backbone: bool = False


@dataclass(frozen=True)
class HeadConfig:
    pooling: str = "attention"
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    channel_id_dim: int = 32
    position_dim: int = 32


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    patience: int = 5
    device: str = "auto"
    output_dir: Path = Path("runs")


@dataclass(frozen=True)
class ExperimentConfig:
    study: StudyConfig = field(default_factory=StudyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _coerce_paths(section: dict[str, Any], path_keys: set[str]) -> dict[str, Any]:
    return {key: Path(value) if key in path_keys and value is not None else value for key, value in section.items()}


def _dataclass_from_dict(cls: type, values: dict[str, Any]):
    field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    return cls(**values)


def load_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    study_raw = _coerce_paths(raw.get("study", {}), {"data_dir", "cache_dir"})
    train_raw = _coerce_paths(raw.get("train", {}), {"output_dir"})

    return ExperimentConfig(
        study=_dataclass_from_dict(StudyConfig, study_raw),
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
        head=_dataclass_from_dict(HeadConfig, raw.get("head", {})),
        train=_dataclass_from_dict(TrainConfig, train_raw),
    )


def merge_cli_overrides(config: ExperimentConfig, args: Any) -> ExperimentConfig:
    study_updates: dict[str, Any] = {}
    model_updates: dict[str, Any] = {}
    head_updates: dict[str, Any] = {}
    train_updates: dict[str, Any] = {}

    for key in (
        "study",
        "data_dir",
        "cache_dir",
        "subjects",
        "max_subjects",
        "seed",
        "sfreq",
        "window_seconds",
        "stride_seconds",
    ):
        value = getattr(args, key, None)
        if value is not None:
            target_key = "name" if key == "study" else key
            study_updates[target_key] = value

    for key in (
        "moment_checkpoint",
        "tspulse_checkpoint",
        "tspulse_revision",
        "tspulse_module",
        "sequence_length",
        "allow_random_backbone",
    ):
        value = getattr(args, key, None)
        if value is not None:
            model_updates[key] = value

    if getattr(args, "no_attention_pooling", False):
        head_updates["pooling"] = "concat_linear"
    if getattr(args, "pooling", None) is not None:
        head_updates["pooling"] = args.pooling

    for key in ("epochs", "batch_size", "lr", "device", "output_dir"):
        value = getattr(args, key, None)
        if value is not None:
            train_updates[key] = value

    return ExperimentConfig(
        study=StudyConfig(**{**config.study.__dict__, **study_updates}),
        model=ModelConfig(**{**config.model.__dict__, **model_updates}),
        head=HeadConfig(**{**config.head.__dict__, **head_updates}),
        train=TrainConfig(**{**config.train.__dict__, **train_updates}),
    )
