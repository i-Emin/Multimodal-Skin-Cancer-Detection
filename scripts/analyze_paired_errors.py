"""Summarize paired image-only versus fusion errors without publishing image IDs.

Example:
    python scripts/analyze_paired_errors.py --image /content/drive/MyDrive/YZM5110/results/baselines/image/validation_predictions.csv --fusion /content/drive/MyDrive/YZM5110/results/baselines/fusion/validation_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> dict[str, tuple[int, int]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["isic_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate image ID in {path}")
    return {row["isic_id"]: (int(row["y_true"]), int(row["y_pred"])) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image, fusion = load(args.image), load(args.fusion)
    if image.keys() != fusion.keys():
        raise ValueError("Models have different validation image IDs")
    counts = {label: {k: 0 for k in ("both_correct", "both_wrong", "fusion_fixes", "fusion_breaks")}
              for label in (0, 1)}
    for image_id, (actual, pred_image) in image.items():
        actual_fusion, pred_fusion = fusion[image_id]
        if actual != actual_fusion:
            raise ValueError(f"Label mismatch for {image_id}")
        correct_image, correct_fusion = pred_image == actual, pred_fusion == actual
        key = ("both_correct" if correct_image and correct_fusion else
               "both_wrong" if not correct_image and not correct_fusion else
               "fusion_fixes" if correct_fusion else "fusion_breaks")
        counts[actual][key] += 1
    lines = ["# Paired baseline error analysis", "",
             "The same validation images and fixed 0.5 threshold are compared; only aggregate counts are published.", "",
             "| True class | Both correct | Both wrong | Fusion fixes image error | Fusion breaks image success |",
             "|---|---:|---:|---:|---:|"]
    for label in (0, 1):
        d = counts[label]
        lines.append(f"| {label} | {d['both_correct']} | {d['both_wrong']} | {d['fusion_fixes']} | {d['fusion_breaks']} |")
    d = counts[1]
    lines.extend(["", f"For class 1, fusion fixes {d['fusion_fixes']} image-model errors but breaks {d['fusion_breaks']} correct image predictions, a net {d['fusion_breaks']-d['fusion_fixes']} additional false negatives.",
                  "These are diagnostic error counts, not a patient-independent performance estimate or proof of a causal mechanism.", ""])
    output = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()

