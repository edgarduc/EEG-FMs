from __future__ import annotations

import csv
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import EEGMAT_BASE_URL, EEGMAT_DIR


@dataclass(frozen=True)
class Trial:
    eeg: np.ndarray
    label: int
    subject: str
    channel_names: tuple[str, ...]
    sfreq: float


class EEGTrialDataset(Dataset):
    def __init__(self, trials: list[Trial], indices: Iterable[int] | None = None):
        self.trials = trials
        self.indices = list(range(len(trials))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        trial = self.trials[self.indices[item]]
        return {
            "eeg": torch.from_numpy(trial.eeg).float(),
            "label": torch.tensor(trial.label, dtype=torch.long),
            "subject": trial.subject,
            "channel_names": list(trial.channel_names),
            "sfreq": trial.sfreq,
        }


def collate_trials(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "eeg": torch.stack([item["eeg"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "subject": [item["subject"] for item in batch],
        "channel_names": batch[0]["channel_names"],
        "sfreq": batch[0]["sfreq"],
    }


def ensure_eegmat(data_dir: Path = EEGMAT_DIR, force_download: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = [data_dir / f"Subject{i:02d}_{suffix}.edf" for i in range(36) for suffix in (1, 2)]
    expected.extend([data_dir / "subject-info.csv", data_dir / "RECORDS"])
    if not force_download and all(path.exists() for path in expected):
        return data_dir

    records_path = data_dir / "RECORDS"
    _download_file(f"{EEGMAT_BASE_URL}/RECORDS", records_path)
    _download_file(f"{EEGMAT_BASE_URL}/subject-info.csv", data_dir / "subject-info.csv")

    records = [line.strip() for line in records_path.read_text().splitlines() if line.strip()]
    for record in records:
        src = f"{EEGMAT_BASE_URL}/{record}"
        dst = data_dir / record
        if force_download or not dst.exists():
            _download_file(src, dst)
    return data_dir


def load_eegmat_trials(
    data_dir: Path = EEGMAT_DIR,
    *,
    window_seconds: float = 10.0,
    stride_seconds: float | None = None,
    target_sfreq: float = 200.0,
    normalize: str = "per_window",
    max_subjects: int | None = None,
) -> list[Trial]:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("mne is required to read EEGMAT EDF files. Install requirements.txt.") from exc

    stride_seconds = window_seconds if stride_seconds is None else stride_seconds
    paths = sorted(data_dir.glob("Subject*_*.edf"))
    if max_subjects is not None:
        allowed = {f"Subject{i:02d}" for i in range(max_subjects)}
        paths = [path for path in paths if _subject_id(path) in allowed]

    trials: list[Trial] = []
    for path in paths:
        label = _label_from_path(path)
        subject = _subject_id(path)
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.pick_types(eeg=True, verbose=False)
        if target_sfreq and not math.isclose(float(raw.info["sfreq"]), float(target_sfreq), rel_tol=0.0, abs_tol=1e-3):
            raw.resample(float(target_sfreq), verbose=False)
        data = raw.get_data()
        data = data.astype(np.float32, copy=False) * 1e6
        channel_names = tuple(_clean_channel_name(str(ch)) for ch in raw.ch_names)
        sfreq = float(raw.info["sfreq"])
        trials.extend(
            _segment_recording(
                data,
                label=label,
                subject=subject,
                channel_names=channel_names,
                sfreq=sfreq,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
                normalize=normalize,
            )
        )

    if not trials:
        raise RuntimeError(f"No EEGMAT EDF files found under {data_dir}.")
    return trials


def split_subjects(
    trials: list[Trial],
    *,
    seed: int,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, list[int]]:
    subjects = sorted({trial.subject for trial in trials})
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_subjects = len(shuffled)
    n_train = max(1, int(round(train_fraction * n_subjects)))
    n_calibration = max(1, int(round(calibration_fraction * n_subjects)))
    if n_train + n_calibration >= n_subjects:
        n_calibration = max(1, n_subjects - n_train - 1)

    train_subjects = set(shuffled[:n_train])
    calibration_subjects = set(shuffled[n_train : n_train + n_calibration])
    test_subjects = set(shuffled[n_train + n_calibration :])
    if not test_subjects:
        raise RuntimeError("Subject split produced an empty test split.")

    splits = {"train": [], "calibration": [], "test": []}
    for idx, trial in enumerate(trials):
        if trial.subject in train_subjects:
            splits["train"].append(idx)
        elif trial.subject in calibration_subjects:
            splits["calibration"].append(idx)
        elif trial.subject in test_subjects:
            splits["test"].append(idx)
    return splits


def read_subject_info(data_dir: Path = EEGMAT_DIR) -> list[dict[str, str]]:
    path = data_dir / "subject-info.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(destination)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _segment_recording(
    data: np.ndarray,
    *,
    label: int,
    subject: str,
    channel_names: tuple[str, ...],
    sfreq: float,
    window_seconds: float,
    stride_seconds: float,
    normalize: str,
) -> list[Trial]:
    window = int(round(window_seconds * sfreq))
    stride = int(round(stride_seconds * sfreq))
    if window <= 0 or stride <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive.")

    trials: list[Trial] = []
    for start in range(0, max(1, data.shape[1] - window + 1), stride):
        segment = data[:, start : start + window]
        if segment.shape[1] != window:
            continue
        segment = _normalize(segment, normalize).astype(np.float32, copy=False)
        trials.append(
            Trial(
                eeg=segment,
                label=label,
                subject=subject,
                channel_names=channel_names,
                sfreq=sfreq,
            )
        )
    return trials


def _normalize(segment: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return segment
    if mode == "per_window":
        mean = segment.mean(axis=1, keepdims=True)
        std = segment.std(axis=1, keepdims=True)
        return (segment - mean) / np.maximum(std, 1e-6)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _label_from_path(path: Path) -> int:
    if path.name.endswith("_1.edf"):
        return 0
    if path.name.endswith("_2.edf"):
        return 1
    raise ValueError(f"Cannot infer EEGMAT label from {path.name}")


def _subject_id(path: Path) -> str:
    match = re.match(r"(Subject\d+)_\d+\.edf", path.name)
    if not match:
        raise ValueError(f"Cannot infer EEGMAT subject from {path.name}")
    return match.group(1)


def _clean_channel_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r"^EEG\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*-\s*(Ref|REF|LE|A1|A2)$", "", cleaned)
    cleaned = cleaned.replace(".", "").strip()
    return cleaned
