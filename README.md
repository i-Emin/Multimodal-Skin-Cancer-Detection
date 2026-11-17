# Multimodal Skin Cancer Detection - YZM5110

Binary melanoma classification using dermoscopic images and clinical metadata. This is a **partial, reduced-compute reproduction** of [Ahammed et al. (CVPR Workshops 2025)](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html), followed by one pre-specified missing-metadata experiment.

**Status (17 November 2026):** The data audit, metadata-only training, and image-only training have been performed in the student's Colab. Fusion was restarted after a Colab session interruption. Measured outputs currently remain in the student's Google Drive and are **not yet archived here**. No completed fusion or improvement result is claimed. See [delivery status](reports/delivery_status.md).

## Research design

- **Question:** Does adding age, sex, and anatomical site improve binary classification relative to image-only, and does metadata dropout help when clinical inputs are unavailable?
- **Data:** [ISIC-DICM-17K official collection](https://api.isic-archive.com/doi/isic-dicm-17k/), 17,060 images. The authors' train and validation manifests supply the binary `class` target. The bundle's `diagnosis_1` field is not used as target or input.
- **Baselines:** metadata-only MLP, image-only ImageNet-pretrained ResNet18, and concatenation fusion using the same ResNet18 branch. Same published 15,098/1,962 split, seed 42, 8 epochs, batch size 32, and learning rate 1e-4.
- **Improvement:** mask the encoded metadata vector for 30% of fusion training examples. Evaluate the existing fusion control and improvement on clean and fully missing metadata. Missing-condition macro-F1 is the primary outcome; clean-condition macro-F1 is the cost check. This plan was written before fusion results were observed.
- **Limit:** the published validation split shares 433 nonmissing patient IDs and 488 lesion IDs across train/validation. These are distinct identifier counts. Scores are not patient-independent estimates. The paper's separate ISIC 2020 test is not evaluated.

## Repository map

| Location | Purpose |
|---|---|
| [`notebooks/01_dataset_audit_colab.ipynb`](notebooks/01_dataset_audit_colab.ipynb) | Download and inspect the official bundle |
| [`notebooks/02_baselines_colab.ipynb`](notebooks/02_baselines_colab.ipynb) | Train and save three baselines |
| [`notebooks/03_metadata_dropout_colab.ipynb`](notebooks/03_metadata_dropout_colab.ipynb) | Train the improvement and compare clean/missing conditions |
| [`src/baseline.py`](src/baseline.py) | Versioned dataset, model, training, and evaluation implementation |
| [`reports/literature_matrix.md`](reports/literature_matrix.md) | Six-paper comparison matrix |
| [`reports/data_audit.md`](reports/data_audit.md) | Data checks, split overlap, and Figure 1 |
| [`reports/improvement_hypothesis.md`](reports/improvement_hypothesis.md) | Pre-specified intervention and outcomes |
| [`reports/report_draft.md`](reports/report_draft.md) | Report text awaiting final measured outputs |
| [`REPRODUCE.md`](REPRODUCE.md) | Commands, artifacts, deviations, and limits |
| [`scripts/summarize_results.py`](scripts/summarize_results.py) | Build a measured-results table and SVG from saved Colab JSON files |
| [`data/README.md`](data/README.md) | Data access and licenses; raw data is not uploaded |

## Run

Select a GPU Colab runtime. Run notebook 02 top to bottom for baselines. Once the fusion `final_model.pt` is saved, run notebook 03 for the matched improvement. Notebook 03 can restore the official dataset in a new runtime, but it requires the fusion checkpoint in Google Drive. Full instructions and result paths are in [REPRODUCE.md](REPRODUCE.md).

Do not use "Run all" in notebook 02 after metadata and image training have completed: it repeats them. A Colab runtime reset clears in-memory model weights; `history.csv` alone cannot resume a model. `final_model.pt` is saved at the fixed final epoch.

## Provenance and responsible use

Paper scores, preliminary Colab displays, and archived measured results are kept separate. Report claims only from saved artifacts. Dataset image licenses include CC-0, CC-BY, and CC-BY-NC; consult the collection for attribution and noncommercial limits. Do not upload raw patient metadata, image bundles, credentials, or large checkpoints. This is a research exercise, not a clinical decision tool.

Codex assisted with source discovery, repository organization, code, audit visualization, and editing. The student remains responsible for running and checking experiments, reading sources, interpreting outcomes, and defending the submission.

