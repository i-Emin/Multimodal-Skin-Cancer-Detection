"""Group-aware internal splits with explicit leakage checks.

These splits are for internal experiments. They do not recreate the published
ISIC-DICM-17K train/validation or Kaggle test protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SplitSpec:
    patient_col: str
    image_col: str
    label_col: str
    lesion_col: str | None = None
    test_fraction: float = 0.15
    validation_fraction: float = 0.15
    seed: int = 42


def _require_clean_ids(frame: pd.DataFrame, spec: SplitSpec) -> None:
    required = [spec.patient_col, spec.image_col, spec.label_col]
    if spec.lesion_col:
        required.append(spec.lesion_col)
    absent = [name for name in required if name not in frame.columns]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    if frame.empty:
        raise ValueError("Cannot split an empty dataset")
    if frame[spec.image_col].isna().any() or frame[spec.image_col].duplicated().any():
        raise ValueError("Image IDs must be present and unique")
    if frame[spec.patient_col].isna().any():
        raise ValueError("Patient IDs are incomplete; patient-safe splitting is not established")
    if frame[spec.label_col].isna().any():
        raise ValueError("Labels must be present")
    if not (0 < spec.test_fraction < 1 and 0 < spec.validation_fraction < 1):
        raise ValueError("Split fractions must lie between zero and one")
    if spec.test_fraction + spec.validation_fraction >= 1:
        raise ValueError("Train fraction must be positive")


def _assert_disjoint(frame: pd.DataFrame, assigned: pd.Series, column: str) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a = set(frame.loc[assigned == left, column].dropna().astype(str))
        b = set(frame.loc[assigned == right, column].dropna().astype(str))
        overlap = a & b
        if overlap:
            raise ValueError(f"{column} leaks between {left} and {right}: {len(overlap)} IDs")


def make_internal_split(frame: pd.DataFrame, spec: SplitSpec) -> pd.Series:
    """Return train/val/test labels for rows, grouped by patient.

    This method does not guarantee exact class balance. Inspect the printed
    class counts and reject a split if any required class is missing.
    """
    _require_clean_ids(frame, spec)
    groups = frame[spec.patient_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=spec.test_fraction, random_state=spec.seed)
    train_val_pos, test_pos = next(splitter.split(frame, frame[spec.label_col], groups))
    relative_val = spec.validation_fraction / (1 - spec.test_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=spec.seed + 1)
    train_rel, val_rel = next(second.split(frame.iloc[train_val_pos],
                                            frame.iloc[train_val_pos][spec.label_col],
                                            groups.iloc[train_val_pos]))
    assigned = pd.Series(index=frame.index, dtype="string", name="split")
    assigned.iloc[test_pos] = "test"
    assigned.iloc[train_val_pos[val_rel]] = "val"
    assigned.iloc[train_val_pos[train_rel]] = "train"
    if assigned.isna().any():
        raise AssertionError("Some rows have no split")
    _assert_disjoint(frame, assigned, spec.patient_col)
    if spec.lesion_col:
        _assert_disjoint(frame, assigned, spec.lesion_col)
    _assert_disjoint(frame, assigned, spec.image_col)
    classes = set(frame[spec.label_col].astype(str))
    for name in ("train", "val", "test"):
        present = set(frame.loc[assigned == name, spec.label_col].astype(str))
        if present != classes:
            raise ValueError(f"{name} is missing classes: {sorted(classes - present)}")
    return assigned


def split_summary(frame: pd.DataFrame, assigned: pd.Series, spec: SplitSpec) -> pd.DataFrame:
    """Counts by split and label, suitable for a protocol record."""
    if not assigned.index.equals(frame.index):
        raise ValueError("Split labels must have the same index as the source frame")
    return (frame.assign(split=assigned)
            .groupby(["split", spec.label_col], dropna=False)
            .agg(images=(spec.image_col, "size"), patients=(spec.patient_col, "nunique"))
            .reset_index())

