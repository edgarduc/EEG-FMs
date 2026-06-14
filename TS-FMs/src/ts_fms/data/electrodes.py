from __future__ import annotations

import math

import numpy as np


_TEN_TWENTY_2D = {
    "FP1": (-0.35, 0.95),
    "FP2": (0.35, 0.95),
    "F7": (-0.9, 0.55),
    "F3": (-0.45, 0.55),
    "FZ": (0.0, 0.6),
    "F4": (0.45, 0.55),
    "F8": (0.9, 0.55),
    "FC5": (-0.7, 0.3),
    "FC3": (-0.4, 0.32),
    "FC1": (-0.18, 0.35),
    "FCZ": (0.0, 0.36),
    "FC2": (0.18, 0.35),
    "FC4": (0.4, 0.32),
    "FC6": (0.7, 0.3),
    "T7": (-1.0, 0.0),
    "C5": (-0.72, 0.0),
    "C3": (-0.48, 0.0),
    "C1": (-0.22, 0.0),
    "CZ": (0.0, 0.0),
    "C2": (0.22, 0.0),
    "C4": (0.48, 0.0),
    "C6": (0.72, 0.0),
    "T8": (1.0, 0.0),
    "TP7": (-0.9, -0.3),
    "CP5": (-0.7, -0.32),
    "CP3": (-0.4, -0.35),
    "CP1": (-0.18, -0.38),
    "CPZ": (0.0, -0.4),
    "CP2": (0.18, -0.38),
    "CP4": (0.4, -0.35),
    "CP6": (0.7, -0.32),
    "TP8": (0.9, -0.3),
    "P7": (-0.85, -0.55),
    "P3": (-0.45, -0.58),
    "PZ": (0.0, -0.62),
    "P4": (0.45, -0.58),
    "P8": (0.85, -0.55),
    "PO7": (-0.55, -0.82),
    "PO3": (-0.3, -0.82),
    "POZ": (0.0, -0.85),
    "PO4": (0.3, -0.82),
    "PO8": (0.55, -0.82),
    "O1": (-0.35, -0.95),
    "OZ": (0.0, -1.0),
    "O2": (0.35, -0.95),
}


def normalize_channel_name(name: str) -> str:
    return name.upper().replace(".", "").replace(" ", "").replace("EEG", "")


def electrode_positions(channel_names: list[str]) -> np.ndarray:
    positions: list[tuple[float, float, float]] = []
    fallback_count = 0
    total = max(len(channel_names), 1)

    for idx, name in enumerate(channel_names):
        key = normalize_channel_name(name)
        if key in _TEN_TWENTY_2D:
            x, y = _TEN_TWENTY_2D[key]
        else:
            angle = 2.0 * math.pi * fallback_count / total
            x, y = 0.15 * math.cos(angle), 0.15 * math.sin(angle)
            fallback_count += 1
        radius_sq = min(x * x + y * y, 1.0)
        z = math.sqrt(max(0.0, 1.0 - radius_sq))
        positions.append((x, y, z))

    return np.asarray(positions, dtype=np.float32)
