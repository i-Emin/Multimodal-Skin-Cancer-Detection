# Measured results

These values come from the student's saved Colab JSON artifacts. The authors' ISIC 2020 test scores are a different protocol and are not included in this table.

| Model / condition | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Metadata-only | 0.6580 | 0.6580 | 0.6572 | 0.7145 |
| Image-only | 0.8191 | 0.8191 | 0.8190 | 0.9085 |
| Fusion control, clean | 0.8140 | 0.8140 | 0.8133 | 0.9135 |
| Fusion control, missing | 0.8084 | 0.8084 | 0.8076 | 0.9096 |
| Dropout 0.3, clean | 0.8267 | 0.8267 | 0.8266 | 0.9120 |
| Dropout 0.3, missing | 0.8211 | 0.8211 | 0.8210 | 0.9094 |

## Pre-specified comparisons

- Fusion minus image-only, clean macro-F1: -0.0057.
- Dropout minus control, fully missing metadata macro-F1 (primary): 0.0133.
- Dropout minus control, clean metadata macro-F1 (cost check): 0.0133.

## Resources

| Training run | Time (minutes) | Peak allocated GPU memory (MB) |
|---|---:|---:|
| metadata | 0.2416 | 17.2416 |
| image | 96.3143 | 602.0152 |
| fusion | 95.5242 | 808.6031 |
| dropout 0.3 | 93.6848 | 600.6554 |

## Confusion matrices

Rows are true class 0/1 and columns are predicted class 0/1.

- Metadata-only: `[[692, 289], [382, 599]]`
- Image-only: `[[820, 161], [194, 787]]`
- Fusion control, clean: `[[858, 123], [242, 739]]`
- Fusion control, missing: `[[853, 128], [248, 733]]`
- Dropout 0.3, clean: `[[834, 147], [193, 788]]`
- Dropout 0.3, missing: `[[831, 150], [201, 780]]`

**Interpretation limit:** This is a single-seed comparison on the authors' validation split, which shares patient and lesion identifiers with training. These values do not estimate new-patient clinical performance.

