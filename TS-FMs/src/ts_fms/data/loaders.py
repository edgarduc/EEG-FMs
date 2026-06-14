from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import mne
import numpy as np
import pandas as pd

from ts_fms.config import StudyConfig
from ts_fms.data.electrodes import electrode_positions
from ts_fms.data.preprocessing import crop_or_pad, standardize_per_trial
from ts_fms.data.types import EEGStudy


BCIC_CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

MAT_BASE_URL = "https://physionet.org/files/eegmat/1.0.0"


def load_study(config: StudyConfig, sequence_length: int) -> EEGStudy:
    study_name = config.name.lower()
    if study_name in {"bcic-iv-2a", "bcic_iv_2a", "bnci2014-001", "bnci2014_001"}:
        return load_bcic_iv_2a(config, sequence_length)
    if study_name in {"mat", "eegmat", "mental-arithmetic"}:
        return load_mat(config, sequence_length)
    raise ValueError(f"Unknown study '{config.name}'. Valid choices: bcic-iv-2a, mat.")


def load_bcic_iv_2a(config: StudyConfig, sequence_length: int) -> EEGStudy:
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise ImportError("BCIC-IV-2a loading requires the 'moabb' package.") from exc

    mne.set_log_level("WARNING")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    subjects = config.subjects
    if subjects is None:
        subjects = list(range(1, 10))
    if config.max_subjects is not None:
        subjects = subjects[: config.max_subjects]

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4,
        channels=BCIC_CHANNELS,
        resample=config.sfreq,
        fmin=config.fmin,
        fmax=config.fmax,
        tmin=config.tmin,
        tmax=config.tmax,
    )
    x, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    class_names = sorted({str(label) for label in labels})
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    y = np.asarray([label_to_id[str(label)] for label in labels], dtype=np.int64)
    subject_ids = metadata["subject"].to_numpy(dtype=np.int64)

    x = crop_or_pad(np.asarray(x, dtype=np.float32), sequence_length)
    x = standardize_per_trial(x)

    return EEGStudy(
        x=x,
        y=y,
        subjects=subject_ids,
        channel_names=BCIC_CHANNELS,
        electrode_positions=electrode_positions(BCIC_CHANNELS),
        class_names=class_names,
        sfreq=float(config.sfreq),
    )


def load_mat(config: StudyConfig, sequence_length: int) -> EEGStudy:
    mne.set_log_level("WARNING")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    mat_dir = config.data_dir / "eegmat"
    mat_dir.mkdir(parents=True, exist_ok=True)

    subject_info_path = _download_mat_file(mat_dir, "subject-info.csv")
    subject_info = pd.read_csv(subject_info_path)
    all_subjects = _mat_subjects(subject_info)
    if config.subjects is not None:
        all_subjects = [subject for subject in all_subjects if subject in set(config.subjects)]
    if config.max_subjects is not None:
        all_subjects = all_subjects[: config.max_subjects]

    trials: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    channel_names: list[str] | None = None
    samples_per_window = int(round(config.window_seconds * config.sfreq))
    stride_samples = int(round(config.stride_seconds * config.sfreq))

    for subject in all_subjects:
        for suffix, label in (("_1", 0), ("_2", 1)):
            file_name = f"Subject{subject:02d}{suffix}.edf"
            raw_path = _download_mat_file(mat_dir, file_name)
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose="ERROR")
            raw.pick_types(eeg=True)
            raw.filter(config.fmin, config.fmax, verbose="ERROR")
            if raw.info["sfreq"] != config.sfreq:
                raw.resample(config.sfreq, verbose="ERROR")
            data = raw.get_data().astype(np.float32)
            if channel_names is None:
                channel_names = list(raw.ch_names)
            data = data[: len(channel_names)]
            for start in range(0, max(1, data.shape[-1] - samples_per_window + 1), stride_samples):
                window = data[:, start : start + samples_per_window]
                if window.shape[-1] < samples_per_window:
                    continue
                trials.append(crop_or_pad(window, sequence_length))
                labels.append(label)
                subjects.append(subject)

    if not trials or channel_names is None:
        raise RuntimeError("MAT loader produced no trials. Check dataset download and preprocessing settings.")

    return EEGStudy(
        x=standardize_per_trial(np.stack(trials, axis=0)),
        y=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=np.int64),
        channel_names=channel_names,
        electrode_positions=electrode_positions(channel_names),
        class_names=["rest", "mental_arithmetic"],
        sfreq=float(config.sfreq),
    )


def _download_mat_file(destination_dir: Path, file_name: str) -> Path:
    path = destination_dir / file_name
    if path.exists():
        return path
    url = f"{MAT_BASE_URL}/{file_name}"
    urlretrieve(url, path)
    return path


def _mat_subjects(subject_info: pd.DataFrame) -> list[int]:
    for column in ("Subject", "subject", "Subject ID", "SubjectID"):
        if column in subject_info.columns:
            values = subject_info[column].tolist()
            return sorted(_parse_subject_id(value) for value in values)
    return list(range(36))


def _parse_subject_id(value: object) -> int:
    text = str(value)
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        raise ValueError(f"Could not parse MAT subject id from {value!r}")
    return int(digits)
