# Reproduction record

## Current state

The starter notebook has been generated and JSON-validated locally. It has **not** been executed on Colab or against the dataset. Hardware timings, metrics, and trained checkpoints do not yet exist.

## Protocol gate before training

1. Run the notebook on the official ISIC-DICM-17K bundle and record CSV columns, file paths, missingness, label distribution, duplicate IDs, and patient/lesion ID coverage.
2. Compare the authors' published `train-set-metadata.csv` and `valid-set-metadata.csv` manifests to the official bundle by `isic_id`. These files provide the authors' binary `class` field; the bundle's `diagnosis_1` is not a substitute. Document whether its ISIC 2020 test evaluation can be repeated.
3. Verify that no patient, lesion, or duplicate image crosses train/validation/test boundaries. Where identifiers are missing, state what cannot be established.
4. Fit metadata preprocessing on the training split only. Exclude diagnosis, diagnosis confirmation, label, and post-diagnosis variables from inputs.
5. Freeze the split, preprocessing, seeds, and metric definitions before comparing methods.

If the published split or test protocol cannot be reproduced, label the run a **new internal experiment**, not a direct numerical reproduction.

`src/splits.py` provides a deterministic patient-group split for an **internal experiment only**. It refuses incomplete patient IDs, duplicate image IDs, split overlap, and splits missing a class. It does not silently fall back to image-level splitting. Verify it without data using `python scripts/check_split.py`. Actual class counts and split identifiers must be recorded after inspecting the downloaded CSV.

Initial published-manifest counts and group-overlap findings are recorded in `reports/data_audit.md`. Do not treat its validation set as patient or lesion independent.

## Planned comparison

- Image-only model.
- Metadata-only model.
- Image plus metadata late fusion.
- One justified improvement and an isolating control, trained under matched conditions.

Report macro-F1, balanced accuracy, accuracy, ROC-AUC where valid, confusion matrix, per-class scores, training time, and uncertainty. Keep authors' results in a separate table from locally measured results.

## Data and licensing

Official collection: https://api.isic-archive.com/doi/isic-dicm-17k/

Do not commit the downloaded bundle, images, raw patient metadata, credentials, or large checkpoints. The ISIC page lists CC-0, CC-BY, and CC-BY-NC licenses across content; preserve required attribution and noncommercial limits.

