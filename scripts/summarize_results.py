"""Create a report-ready Markdown table and SVG from saved Colab JSON results.

Example:
    python scripts/summarize_results.py --drive-root /content/drive/MyDrive/YZM5110/results

This reads small measured artifacts only. It does not copy checkpoints or raw data.
"""
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


FIELDS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing measured output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def minutes(data: dict) -> str:
    seconds = data.get("total_train_seconds")
    return fmt(None if seconds is None else seconds / 60)


def svg(rows: list[tuple[str, dict]], target: Path) -> None:
    width, height = 820, 90 + len(rows) * 57
    x0, usable = 250, 500
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Measured validation macro F1 and ROC AUC">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="Arial,sans-serif" font-size="19" font-weight="bold" fill="#172435">Measured validation metrics</text>',
        '<text x="24" y="55" font-family="Arial,sans-serif" font-size="12" fill="#536275">Published split has patient/lesion overlap; scores are not independent-patient estimates.</text>',
    ]
    for i, (label, metrics) in enumerate(rows):
        y = 91 + i * 57
        f1, auc = metrics["macro_f1"], metrics["roc_auc"]
        parts.extend([
            f'<text x="24" y="{y+13}" font-family="Arial,sans-serif" font-size="13" fill="#172435">{escape(label)}</text>',
            f'<rect x="{x0}" y="{y}" width="{usable*f1:.1f}" height="15" fill="#237c87"/>',
            f'<text x="{x0+usable*f1+7:.1f}" y="{y+12}" font-family="Arial,sans-serif" font-size="12" fill="#172435">F1 {f1:.3f}</text>',
            f'<rect x="{x0}" y="{y+21}" width="{usable*auc:.1f}" height="15" fill="#e69948"/>',
            f'<text x="{x0+usable*auc+7:.1f}" y="{y+33}" font-family="Arial,sans-serif" font-size="12" fill="#172435">AUC {auc:.3f}</text>',
        ])
    parts.append('</svg>')
    target.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True,
                        help="Directory containing baselines/ and improvement/")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()
    baseline = {mode: read(args.drive_root / "baselines" / mode / "metrics.json")
                for mode in ("metadata", "image", "fusion")}
    compare = read(args.drive_root / "improvement" / "comparison.json")
    improved = read(args.drive_root / "improvement" / "fusion" / "metrics.json")
    conditions = compare["conditions"]
    rows = [("Metadata-only", baseline["metadata"]),
            ("Image-only", baseline["image"]),
            ("Fusion control, clean", conditions["control_clean"]),
            ("Fusion control, missing", conditions["control_missing"]),
            ("Dropout 0.3, clean", conditions["dropout_0_3_clean"]),
            ("Dropout 0.3, missing", conditions["dropout_0_3_missing"])]

    lines = ["# Measured results", "",
             "These values come from the student's saved Colab JSON artifacts. The authors' ISIC 2020 test scores are a different protocol and are not included in this table.", "",
             "| Model / condition | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC |",
             "|---|---:|---:|---:|---:|"]
    for label, data in rows:
        lines.append("| " + label + " | " + " | ".join(fmt(data[k]) for k in FIELDS) + " |")
    lines.extend(["", "## Pre-specified comparisons", "",
        f"- Fusion minus image-only, clean macro-F1: {fmt(conditions['control_clean']['macro_f1'] - baseline['image']['macro_f1'])}.",
        f"- Dropout minus control, fully missing metadata macro-F1 (primary): {fmt(compare['delta_macro_f1_missing'])}.",
        f"- Dropout minus control, clean metadata macro-F1 (cost check): {fmt(compare['delta_macro_f1_clean'])}.",
        "", "## Resources", "",
        "| Training run | Time (minutes) | Peak allocated GPU memory (MB) |",
        "|---|---:|---:|"])
    for mode in ("metadata", "image", "fusion"):
        d = baseline[mode]
        lines.append(f"| {mode} | {minutes(d)} | {fmt(d.get('peak_gpu_memory_mb'))} |")
    lines.append(f"| dropout 0.3 | {minutes(improved)} | {fmt(improved.get('peak_gpu_memory_mb'))} |")
    lines.extend(["", "## Confusion matrices", "",
                  "Rows are true class 0/1 and columns are predicted class 0/1.", ""])
    for label, data in rows:
        lines.append(f"- {label}: `{data['confusion_matrix']}`")
    lines.extend(["", "**Interpretation limit:** This is a single-seed comparison on the authors' validation split, which shares patient and lesion identifiers with training. These values do not estimate new-patient clinical performance.", ""])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "measured_results.md"
    figure = args.output_dir / "model_metrics.svg"
    report.write_text("\n".join(lines), encoding="utf-8")
    svg(rows, figure)
    print(report)
    print(figure)


if __name__ == "__main__":
    main()

