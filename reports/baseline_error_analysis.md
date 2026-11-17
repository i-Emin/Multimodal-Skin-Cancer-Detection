# Paired baseline error analysis

The same 1,962 validation images and fixed 0.5 threshold are compared; only aggregate counts are published. Derived from the student's saved `validation_predictions.csv` files for image-only and fusion using [`scripts/analyze_paired_errors.py`](../scripts/analyze_paired_errors.py).

| True class | Both correct | Both wrong | Fusion fixes image error | Fusion breaks image success |
|---|---:|---:|---:|---:|
| 0 (non-melanoma) | 798 | 101 | 60 | 22 |
| 1 (melanoma) | 693 | 148 | 46 | 94 |

For melanoma, fusion fixes 46 image-model errors but changes 94 correct image predictions into errors: a net 48 additional false negatives. For non-melanoma, it fixes 60 image-model errors and breaks 22: a net 38 fewer false positives. The two models thus trade sensitivity for specificity at this fixed threshold. This is a failure-pattern observation, not proof of which feature caused an individual prediction or a patient-independent performance estimate.

The per-image predictions remain in the student's Drive. Raw image IDs and patient data are not included in this public summary. Recompute with:

```bash
python scripts/analyze_paired_errors.py --image /content/drive/MyDrive/YZM5110/results/baselines/image/validation_predictions.csv --fusion /content/drive/MyDrive/YZM5110/results/baselines/fusion/validation_predictions.csv --output reports/baseline_error_analysis.md
```

