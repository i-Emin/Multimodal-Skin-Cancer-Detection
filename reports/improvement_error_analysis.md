# Paired error analysis of metadata dropout

Source: four saved per-image validation prediction CSV files in the student's Drive, evaluated on the same 1,962 image IDs. This repository retains aggregate counts only. Melanoma is class 1. All values are descriptive for the published validation split, which has patient and lesion overlap with training.

| Paired comparison | Both correct | Both wrong | Second model fixes first | Second model breaks first | Net correct |
|---|---:|---:|---:|---:|---:|
| Control clean → dropout clean | 1,509 | 252 | 113 | 88 | +25 |
| Control missing → dropout missing | 1,491 | 256 | 120 | 95 | +25 |
| Control clean → control missing | 1,550 | 329 | 36 | 47 | -11 |
| Dropout clean → dropout missing | 1,591 | 320 | 20 | 31 | -11 |

| Comparison | Non-melanoma fixed / broken | Melanoma fixed / broken |
|---|---:|---:|
| Control clean → dropout clean | 31 / 55 | 82 / 33 |
| Control missing → dropout missing | 29 / 51 | 91 / 44 |
| Control clean → control missing | 8 / 13 | 28 / 34 |
| Dropout clean → dropout missing | 10 / 13 | 10 / 18 |

The dropout model corrects more melanoma errors than it introduces, while increasing non-melanoma errors. Its clean-to-missing correctness changes affect fewer images (51 versus 83), but both models lose 11 net correct predictions. The pre-specified macro-F1 degradation is essentially equal (0.00564 versus 0.00563), so the evidence does not support a distinct gain in missing-input macro-F1 resilience. The 0.5-threshold operating point and score ranking differ, and this single-seed split cannot quantify uncertainty or new-patient effects.

