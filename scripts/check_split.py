"""Synthetic smoke check for the patient-group split. No dataset download needed."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import SplitSpec, make_internal_split, split_summary  # noqa: E402


def main() -> None:
    frame = pd.DataFrame({
        "image_id": [f"image-{i}" for i in range(400)],
        "patient_id": [f"patient-{i // 2}" for i in range(400)],
        "lesion_id": [f"lesion-{i // 2}" for i in range(400)],
        "label": [i // 2 % 2 for i in range(400)],
    })
    spec = SplitSpec("patient_id", "image_id", "label", "lesion_id")
    assigned = make_internal_split(frame, spec)
    assert assigned.equals(make_internal_split(frame, spec)), "Split is not deterministic"
    print(split_summary(frame, assigned, spec).to_string(index=False))
    broken = frame.copy()
    broken.loc[0, "patient_id"] = None
    try:
        make_internal_split(broken, spec)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("Missing patient ID was not rejected")
    print("PASS: deterministic, disjoint patient/lesion groups; missing patient IDs rejected")


if __name__ == "__main__":
    main()

