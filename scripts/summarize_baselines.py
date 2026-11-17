"""Generate the baseline report and figure from saved, measured Colab JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_results import svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/baselines"))
    parser.add_argument("--report", type=Path, default=Path("reports/baseline_results.md"))
    parser.add_argument("--figure", type=Path, default=Path("figures/baseline_metrics.svg"))
    args = parser.parse_args()
    data = {}
    for mode in ("metadata", "image", "fusion"):
        folder = args.root / mode
        data[mode] = {kind: json.loads((folder / f"{kind}.json").read_text(encoding="utf-8"))
                      for kind in ("metrics", "run")}
    signatures = {(d["run"]["seed"], d["run"]["epochs"], d["run"]["batch_size"],
                   d["run"]["lr"], d["run"]["gpu"], d["run"]["source_sha256"])
                  for d in data.values()}
    if len(signatures) != 1:
        raise ValueError("Baseline settings or code digests differ; inspect before summarizing")
    rows = [("Metadata-only", data["metadata"]["metrics"]),
            ("Image-only", data["image"]["metrics"]),
            ("Fusion", data["fusion"]["metrics"])]
    image, fusion = data["image"]["metrics"], data["fusion"]["metrics"]
    lines = [
        "# Measured baseline results",
        "",
        "Source: the student's saved Colab `metrics.json` and `run.json` in `results/baselines/`. This is **our published-manifest validation** experiment, not Ahammed et al.'s separate ISIC 2020 Kaggle test.",
        "",
        "| Model | Macro-F1 | Balanced accuracy | ROC-AUC | Melanoma recall | Training minutes | Peak allocated GPU MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, d in rows:
        lines.append(f"| {label} | {d['macro_f1']:.4f} | {d['balanced_accuracy']:.4f} | {d['roc_auc']:.4f} | {d['per_class']['1']['recall']:.4f} | {d['total_train_seconds']/60:.2f} | {d['peak_gpu_memory_mb']:.1f} |")
    lines.extend([
        "", "![Measured baseline validation metrics](../figures/baseline_metrics.svg)", "",
        "**Figure 2.** Macro-F1 and ROC-AUC of the three completed baseline runs on the authors' validation split. Values were generated from saved JSON files. Group overlap prevents a patient-independent interpretation.",
        "", "## Matched image versus fusion comparison", "",
        f"- Fusion minus image-only macro-F1: {fusion['macro_f1']-image['macro_f1']:+.4f}; balanced accuracy: {fusion['balanced_accuracy']-image['balanced_accuracy']:+.4f}; ROC-AUC: {fusion['roc_auc']-image['roc_auc']:+.4f}.",
        f"- Image-only confusion matrix: {image['confusion_matrix']}; fusion: {fusion['confusion_matrix']}. Rows are true class 0/1, columns predicted 0/1. Fusion has {fusion['confusion_matrix'][1][0]-image['confusion_matrix'][1][0]:+d} more false negatives and {fusion['confusion_matrix'][0][1]-image['confusion_matrix'][0][1]:+d} fewer false positives at the fixed 0.5 threshold.",
        "- The opposite directions for thresholded macro-F1 and ranking ROC-AUC suggest that the current decision threshold may affect the comparison. We do not retune the threshold on this validation set and then report the tuned score as an unbiased test result.",
        "", "## Protocol and limits", "",
        "All three runs used seed 42, 8 epochs, batch size 32, learning rate 1e-4, and NVIDIA L4. Their embedded training-source SHA-256 was `" + data["fusion"]["run"]["source_sha256"] + "`. The fixed final epoch was reported. The authors' manifests contain shared patient and lesion IDs across train/validation; single-run scores do not establish independent-patient generalization. GPU memory is PyTorch peak allocated memory, not total device occupancy.",
        "",
        "Paper scores must be shown separately: Ahammed et al.'s Table 5 reports transfer EfficientNet-B3 image AUC 0.8754 and image+metadata AUC 0.8851 on the distinct ISIC 2020 test protocol. Our backbone, training budget, split, and reporting rule differ.", "",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    svg(rows, args.figure)
    print(args.report)
    print(args.figure)


if __name__ == "__main__":
    main()

