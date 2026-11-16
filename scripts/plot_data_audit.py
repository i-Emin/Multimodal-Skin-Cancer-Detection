"""Render the data-audit figure from observed summary counts, not raw records."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "results" / "data_audit_counts.json"
OUTPUT = ROOT / "figures" / "data_audit.svg"


def main() -> None:
    data = json.loads(COUNTS.read_text(encoding="utf-8"))
    train, valid = data["train"], data["validation"]
    overlap = data["cross_split_distinct_id_overlap"]

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    fig.suptitle("ISIC-DICM-17K published split audit", fontsize=14, fontweight="bold")

    ax = axes[0]
    x = range(2)
    ax.bar(x, [train["class_0"], valid["class_0"]], label="Class 0", color="#2563eb")
    ax.bar(x, [train["class_1"], valid["class_1"]],
           bottom=[train["class_0"], valid["class_0"]], label="Class 1", color="#f59e0b")
    ax.set_xticks([0, 1], ["Train", "Validation"])
    ax.set_ylabel("Images")
    ax.set_title("Balanced class counts")
    ax.legend(frameon=False, fontsize=8)
    for i, part in enumerate([train, valid]):
        ax.text(i, part["total"] + 300, f'{part["total"]:,}', ha="center", fontsize=9)
    ax.set_ylim(0, 17000)

    ax = axes[1]
    vals = [train["missing_patient_id"] / train["total"] * 100,
            valid["missing_patient_id"] / valid["total"] * 100]
    ax.bar(["Train", "Validation"], vals, color="#64748b")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rows missing patient ID (%)")
    ax.set_title("Incomplete patient IDs")
    for i, (v, part) in enumerate(zip(vals, [train, valid])):
        ax.text(i, v + 2, f'{v:.1f}%\n({part["missing_patient_id"]:,})', ha="center", fontsize=9)

    ax = axes[2]
    labels = ["Image ID", "Patient ID", "Lesion ID"]
    vals = [overlap["isic_id"], overlap["patient_id"], overlap["lesion_id"]]
    ax.bar(labels, vals, color=["#16a34a", "#dc2626", "#dc2626"])
    ax.set_ylim(0, 600)
    ax.set_ylabel("Distinct IDs in both splits")
    ax.set_title("Train/validation overlap")
    for i, v in enumerate(vals):
        ax.text(i, v + 12, str(v), ha="center", fontsize=9)
    ax.tick_params(axis="x", labelrotation=20)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()

