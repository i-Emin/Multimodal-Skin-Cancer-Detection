"""Quick, dependency-free integrity check for the archived measured outputs.

Run before reviewing the report: python scripts/smoke_test_saved_results.py
This does not retrain the model or assert patient-independent validity.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-10:
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify_metrics(name: str, data: dict) -> None:
    tn, fp = data["confusion_matrix"][0]
    fn, tp = data["confusion_matrix"][1]
    n = tn + fp + fn + tp
    if n != 1962 or data["n"] != n:
        raise AssertionError(f"{name}: expected 1,962 validation images")
    close(data["accuracy"], (tn + tp) / n, f"{name} accuracy")
    close(data["balanced_accuracy"], ((tn / (tn + fp)) + (tp / (tp + fn))) / 2,
          f"{name} balanced accuracy")
    f1_0 = 2 * tn / (2 * tn + fn + fp)
    f1_1 = 2 * tp / (2 * tp + fn + fp)
    close(data["macro_f1"], (f1_0 + f1_1) / 2, f"{name} macro-F1")
    if data["reported_epoch"] != 8:
        raise AssertionError(f"{name}: fixed final epoch should be 8")


def main() -> None:
    baselines = {}
    for mode in ("metadata", "image", "fusion"):
        path = f"results/baselines/{mode}/"
        baselines[mode] = read(path + "metrics.json")
        run = read(path + "run.json")
        verify_metrics(mode, baselines[mode])
        if run["seed"] != 42 or run["epochs"] != 8:
            raise AssertionError(f"{mode}: unexpected training configuration")

    improved = read("results/improvement/fusion/metrics.json")
    improved_run = read("results/improvement/fusion/run.json")
    compare = read("results/improvement/comparison.json")
    verify_metrics("dropout", improved)
    if improved_run["metadata_dropout_p"] != 0.3:
        raise AssertionError("Improvement run used a different dropout probability")
    if compare["source_sha256"] != improved_run["source_sha256"]:
        raise AssertionError("Improvement source digest differs across files")
    conditions = compare["conditions"]
    for name, data in conditions.items():
        verify_metrics(name, {**data, "reported_epoch": 8})
    close(conditions["control_clean"]["macro_f1"], baselines["fusion"]["macro_f1"],
          "control checkpoint versus archived fusion baseline")
    close(conditions["dropout_0_3_clean"]["macro_f1"], improved["macro_f1"],
          "improvement checkpoint versus archived result")
    for suffix, key in (("clean", "delta_macro_f1_clean"),
                        ("missing", "delta_macro_f1_missing")):
        delta = (conditions[f"dropout_0_3_{suffix}"]["macro_f1"]
                 - conditions[f"control_{suffix}"]["macro_f1"])
        close(compare[key], delta, key)
    print("PASS: 3 baselines, 1 improvement, 4 paired evaluation conditions, fixed epoch 8")


if __name__ == "__main__":
    main()

