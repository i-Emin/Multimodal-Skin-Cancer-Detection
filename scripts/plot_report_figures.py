"""Rebuild publication-style baseline plots from archived experiment outputs.

Run from the repository root: python scripts/plot_report_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path("results/baselines")
OUT = Path("figures")
COLORS = {"metadata": "#64748b", "image": "#167d90", "fusion": "#df873f"}


def load_history(mode: str) -> dict[str, list[float]]:
    with (ROOT / mode / "history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(1, 9)):
        raise ValueError(f"Expected eight ordered epochs for {mode}")
    return {key: [float(row[key]) for row in rows]
            for key in ("epoch", "train_loss", "macro_f1", "roc_auc")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    histories = {mode: load_history(mode) for mode in COLORS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), constrained_layout=True)
    panels = (("train_loss", "Training cross-entropy"),
              ("macro_f1", "Validation macro-F1"),
              ("roc_auc", "Validation ROC-AUC"))
    for ax, (key, title) in zip(axes, panels):
        for mode, hist in histories.items():
            ax.plot(hist["epoch"], hist[key], marker="o", markersize=3.5,
                    linewidth=1.8, color=COLORS[mode], label=mode.title())
        ax.set(title=title, xlabel="Epoch", xlim=(1, 8))
        ax.set_xticks(range(1, 9))
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Value")
    axes[2].legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(OUT / "training_curves.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    matrices = {mode: np.asarray(json.loads((ROOT / mode / "metrics.json").read_text())
                                 ["confusion_matrix"]) for mode in ("image", "fusion")}
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.6), constrained_layout=True)
    for ax, (mode, matrix) in zip(axes, matrices.items()):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=981)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=15,
                    color="white" if value > 550 else "#142435", fontweight="bold")
        ax.set(title=mode.title(), xlabel="Predicted class", ylabel="True class",
               xticks=[0, 1], yticks=[0, 1])
        ax.set_xticklabels(["0 non-MEL", "1 MEL"])
        ax.set_yticklabels(["0 non-MEL", "1 MEL"])
    fig.savefig(OUT / "baseline_confusion.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 3.15))
    ax.set(xlim=(0, 10.8), ylim=(0, 3.15))
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.11",
                                    linewidth=1.3, edgecolor=color, facecolor="white"))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10,
                color="#142435", linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#66758b"})

    box(0.25, 1.88, 1.5, 0.65, "Dermoscopic\nimage", "#167d90")
    box(2.25, 1.88, 1.65, 0.65, "ImageNet\nResNet18", "#167d90")
    box(0.25, 0.36, 1.5, 0.65, "Age, sex,\nsite", "#df873f")
    box(2.25, 0.36, 1.65, 0.65, "Train-fitted\nencoding", "#df873f")
    box(4.42, 0.36, 1.52, 0.65, "64-unit\nmetadata MLP", "#df873f")
    box(6.4, 1.12, 1.38, 0.65, "Concatenate\n512 + 64", "#56647a")
    box(8.25, 1.12, 1.35, 0.65, "Two-layer\nclassifier", "#56647a")
    ax.text(10.3, 1.44, "MEL /\nnon-MEL", ha="center", va="center", fontsize=10,
            color="#142435")
    for a in [(1.87, 2.2, 2.12, 2.2), (1.87, 0.68, 2.12, 0.68),
              (4.02, 0.68, 4.28, 0.68), (5.99, 0.68, 6.28, 1.3),
              (4.02, 2.2, 6.28, 1.62), (7.9, 1.44, 8.13, 1.44),
              (9.72, 1.44, 9.95, 1.44)]:
        arrow(*a)
    ax.text(4.55, 0.03, "Improvement: mask full encoded vector for 30% of training examples",
            fontsize=8.4, color="#9d5b1e")
    fig.savefig(OUT / "fusion_architecture.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / "training_curves.svg")
    print(OUT / "baseline_confusion.svg")
    print(OUT / "fusion_architecture.svg")


if __name__ == "__main__":
    main()

