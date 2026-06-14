from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EEGStudy:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    channel_names: list[str]
    electrode_positions: np.ndarray
    class_names: list[str]
    sfreq: float

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)


class EEGTensorDataset(Dataset):
    def __init__(self, study: EEGStudy, indices: np.ndarray):
        self.x = torch.from_numpy(study.x[indices]).float()
        self.y = torch.from_numpy(study.y[indices]).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
