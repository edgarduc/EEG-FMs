from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_subjects: list[int]
    val_subjects: list[int]
    test_subjects: list[int]


def subject_disjoint_split(
    subjects: np.ndarray,
    *,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> SplitIndices:
    unique_subjects = np.array(sorted({int(subject) for subject in subjects}))
    if len(unique_subjects) < 3:
        raise ValueError("Subject-disjoint train/val/test split needs at least 3 subjects.")

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_fraction))
    remaining_after_test = len(shuffled) - n_test
    n_val = max(1, round(remaining_after_test * val_fraction))
    if remaining_after_test - n_val < 1:
        raise ValueError("Split fractions leave no training subjects.")

    test_subjects = shuffled[:n_test]
    val_subjects = shuffled[n_test : n_test + n_val]
    train_subjects = shuffled[n_test + n_val :]

    def indices_for(selected: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(subjects, selected))

    return SplitIndices(
        train=indices_for(train_subjects),
        val=indices_for(val_subjects),
        test=indices_for(test_subjects),
        train_subjects=[int(x) for x in sorted(train_subjects)],
        val_subjects=[int(x) for x in sorted(val_subjects)],
        test_subjects=[int(x) for x in sorted(test_subjects)],
    )
