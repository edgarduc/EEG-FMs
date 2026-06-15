from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import EEGTrialDataset, collate_trials
from .metrics import (
    classification_metrics,
    conformal_coverage,
    conformal_threshold,
    ensemble_uncertainty,
    fit_temperature,
)
from .models import FrozenFeatureExtractor


@dataclass(frozen=True)
class ShiftSpec:
    name: str
    kind: str
    snr_db: float | None = None


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))


def run_probe_experiment(
    *,
    extractor: FrozenFeatureExtractor,
    trials,
    splits: dict[str, list[int]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    ensemble_size: int,
    alpha: float,
    ece_bins: int,
    montage_drop_prob: float,
    snr_levels: Iterable[float],
) -> dict[str, object]:
    train_features, train_labels = extract_features(
        extractor, EEGTrialDataset(trials, splits["train"]), batch_size, num_workers, device
    )
    calibration_features, calibration_labels = extract_features(
        extractor, EEGTrialDataset(trials, splits["calibration"]), batch_size, num_workers, device
    )

    num_classes = int(torch.cat([train_labels, calibration_labels]).max().item()) + 1
    probes = []
    temperatures = []
    for member in range(ensemble_size):
        member_seed = seed + 10_000 * member
        probe = train_linear_probe(
            train_features,
            train_labels,
            num_classes=num_classes,
            seed=member_seed,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            device=device,
        )
        with torch.no_grad():
            calibration_logits = probe(calibration_features.to(device)).cpu()
        temperature = fit_temperature(calibration_logits, calibration_labels)
        probes.append(probe.cpu())
        temperatures.append(temperature)

    with torch.no_grad():
        primary_calibration_logits = probes[0](calibration_features).cpu() / temperatures[0]
        calibration_probs = F.softmax(primary_calibration_logits, dim=1)
    qhat = conformal_threshold(calibration_probs, calibration_labels, alpha)

    shifts = [ShiftSpec("subject_disjoint", "none"), ShiftSpec("montage_drop", "montage")]
    shifts.extend(ShiftSpec(f"noise_{snr:g}db", "noise", snr_db=float(snr)) for snr in snr_levels)

    results: dict[str, object] = {
        "temperature": temperatures[0],
        "ensemble_temperatures": temperatures,
        "conformal_alpha": alpha,
        "conformal_threshold": qhat,
        "splits": {key: len(value) for key, value in splits.items()},
        "shifts": {},
    }

    for shift in shifts:
        test_features, test_labels = extract_features(
            extractor,
            EEGTrialDataset(trials, splits["test"]),
            batch_size,
            num_workers,
            device,
            shift=shift,
            seed=seed,
            montage_drop_prob=montage_drop_prob,
        )
        shift_result = evaluate_shift(
            probes=probes,
            temperatures=temperatures,
            features=test_features,
            labels=test_labels,
            alpha=alpha,
            qhat=qhat,
            ece_bins=ece_bins,
        )
        results["shifts"][shift.name] = shift_result
    return results


def extract_features(
    extractor: FrozenFeatureExtractor,
    dataset: EEGTrialDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    shift: ShiftSpec | None = None,
    seed: int = 0,
    montage_drop_prob: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_trials,
    )
    features = []
    labels = []
    mask_cache: torch.Tensor | None = None
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(seed + _shift_seed_offset(shift))

    for batch in loader:
        eeg = batch["eeg"].float()
        if shift is not None and shift.kind != "none":
            eeg, mask_cache = apply_shift(
                eeg,
                shift=shift,
                seed=seed,
                montage_drop_prob=montage_drop_prob,
                mask_cache=mask_cache,
                noise_generator=noise_generator,
            )
        batch_features = extractor(eeg.to(device), batch["channel_names"]).detach().cpu()
        features.append(batch_features)
        labels.append(batch["label"].detach().cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    device: torch.device,
) -> LinearProbe:
    torch.manual_seed(seed)
    probe = LinearProbe(features.shape[1], num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    features = features.to(device)
    labels = labels.to(device)

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


def evaluate_shift(
    *,
    probes: list[LinearProbe],
    temperatures: list[float],
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    qhat: float,
    ece_bins: int,
) -> dict[str, object]:
    del alpha
    with torch.no_grad():
        primary_logits = probes[0](features).cpu() / temperatures[0]
        metrics = classification_metrics(primary_logits, labels, ece_bins=ece_bins)
        probs = F.softmax(primary_logits, dim=1)
        conformal = conformal_coverage(probs, labels, qhat)

        member_probs = []
        for probe, temperature in zip(probes, temperatures):
            logits = probe(features).cpu() / temperature
            member_probs.append(F.softmax(logits, dim=1))
        ensemble_probs = torch.stack(member_probs, dim=0)

    return {
        "classification": metrics,
        "conformal": conformal,
        "uncertainty": ensemble_uncertainty(ensemble_probs),
    }


def apply_shift(
    eeg: torch.Tensor,
    *,
    shift: ShiftSpec,
    seed: int,
    montage_drop_prob: float,
    mask_cache: torch.Tensor | None,
    noise_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if shift.kind == "montage":
        if mask_cache is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 991)
            keep = torch.rand(eeg.shape[1], generator=generator) > montage_drop_prob
            if not keep.any():
                keep[torch.randint(0, eeg.shape[1], (1,), generator=generator)] = True
            mask_cache = keep.float().view(1, -1, 1)
        return eeg * mask_cache, mask_cache
    if shift.kind == "noise":
        if shift.snr_db is None:
            raise ValueError("Noise shift requires snr_db.")
        signal_power = eeg.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        noise_power = signal_power / (10.0 ** (shift.snr_db / 10.0))
        noise = torch.randn(eeg.shape, generator=noise_generator, dtype=eeg.dtype) * noise_power.sqrt()
        return eeg + noise, mask_cache
    raise ValueError(f"Unsupported shift kind: {shift.kind}")


def _shift_seed_offset(shift: ShiftSpec | None) -> int:
    if shift is None:
        return 0
    text = f"{shift.name}:{shift.kind}:{shift.snr_db}"
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 10_000
