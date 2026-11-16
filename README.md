# Multimodal Skin Cancer Detection — YZM5110

Research project on binary melanoma classification using dermoscopic images and clinical metadata. This repository is **in preparation**: no experiment has been run or reproduced yet.

## Research question

Under a leakage-safe evaluation, does adding age, sex, and anatomical site to a dermoscopic image model improve melanoma classification? After establishing an image-only and late-fusion baseline, we will test one targeted change under matched conditions.

## Source and data

- Selected study: [Ahammed et al., CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html).
- Dataset: [official ISIC-DICM-17K collection](https://api.isic-archive.com/doi/isic-dicm-17k/), 17,060 images (8,530 per class).
- The paper uses 15,098 training and 1,962 validation images, plus a separate ISIC 2020 test evaluation. A newly generated split of the bundle is a different experiment.
- The collection includes images under CC-0, CC-BY, and CC-BY-NC terms. Check individual attribution and use restrictions.

## Repository map

| Path | Purpose |
|---|---|
| `notebooks/` | Colab entry points, starting with the data audit |
| `src/` | Reusable data, model, training, and evaluation code (planned) |
| `configs/` | Exact experiment settings and seeds (planned) |
| `results/reproduction/` | Measured baseline outputs (currently empty) |
| `results/improvement/` | Matched comparison and ablation outputs (currently empty) |
| `figures/` | Figures generated from saved results (currently empty) |
| `reports/` | Literature matrix, report, and presentation (planned) |
| `REPRODUCE.md` | Protocol, commands, environment, and known deviations |

## Start in Colab

1. Open [`notebooks/01_dataset_audit_colab.ipynb`](notebooks/01_dataset_audit_colab.ipynb) in Google Colab.
2. Run the environment and official-download cells. The complete bundle is about 6.9 GB compressed.
3. Inspect the actual CSV columns and image paths. Fill the column variables in the audit cell only after inspection.
4. Save the schema, missingness, label, duplicate, and image-match outputs before defining a split or model.

Training and measured result commands will be added after the data audit. See [REPRODUCE.md](REPRODUCE.md) for the current protocol gate.

The patient-group split smoke check runs without the dataset:

```bash
python scripts/check_split.py
```

## Literature review

The six verified starting sources and the instructor's required comparison fields are in [`reports/literature_matrix.md`](reports/literature_matrix.md). This is a reading matrix; detailed critical analysis remains to be completed from the full papers.

## Result provenance

Published paper numbers and our own measurements will be stored separately. There are no measured results in this repository yet. No clinical use is intended.

## AI assistance

Codex assisted with repository structure, source verification, and the initial audit notebook. The student must inspect the data, run experiments, verify citations, interpret findings, and defend the work.

