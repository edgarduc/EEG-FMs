from __future__ import annotations

import numpy as np


def crop_or_pad(x: np.ndarray, target_length: int) -> np.ndarray:
    if x.shape[-1] == target_length:
        return x
    if x.shape[-1] > target_length:
        return x[..., :target_length]

    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_length - x.shape[-1])
    return np.pad(x, pad_width, mode="constant")


def standardize_per_trial(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / np.maximum(std, eps)).astype(np.float32)
