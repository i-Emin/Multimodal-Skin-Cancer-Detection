# Multimodal Skin Cancer Detection - YZM5110

Binary melanoma classification using dermoscopic images and clinical metadata. This is a **partial, reduced-compute reproduction** of [Ahammed et al. (CVPR Workshops 2025)](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html), followed by one pre-specified missing-metadata experiment.

**Status (17 November 2026):** The audit, three baselines, and metadata-dropout experiment are complete. Small measured result artifacts and report figures are archived here. The final [research report](reports/report_draft.md) and [measured comparison](reports/generated/measured_results.md) include the results and limits; the submission PDF is generated locally by `scripts/build_report_pdf.py --final`.

## Research design

- **Question:** Does adding age, sex, and anatomical site improve binary classification relative to image-only, and does metadata dropout help when clinical inputs are unavailable?
- **Data:** [ISIC-DICM-17K official collection](https://api.isic-archive.com/doi/isic-dicm-17k/), 17,060 images. The authors' train and validation manifests supply the binary `class` target. The bundle's `diagnosis_1` field is not used as target or input.
- **Baselines:** metadata-only MLP, image-only ImageNet-pretrained ResNet18, and concatenation fusion using the same ResNet18 branch. Same published 15,098/1,962 split, seed 42, 8 epochs, batch size 32, and learning rate 1e-4.
- **Improvement:** mask the encoded metadata vector for 30% of fusion training examples. Evaluate the existing fusion control and improvement on clean and fully missing metadata. Missing-condition macro-F1 is the primary outcome; clean-condition macro-F1 is the cost check. This plan was written after interim fusion epochs were visible, but before the completed fusion result and before any improved-model outcome.
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
| [`reports/baseline_results.md`](reports/baseline_results.md) | Three measured baseline runs and Figure 2 |
| [`reports/baseline_error_analysis.md`](reports/baseline_error_analysis.md) | Aggregate paired error analysis without raw IDs |
| [`reports/improvement_hypothesis.md`](reports/improvement_hypothesis.md) | Pre-specified intervention and outcomes |
| [`reports/report_draft.md`](reports/report_draft.md) | Completed report text and numbered references |
| [`reports/generated/measured_results.md`](reports/generated/measured_results.md) | Six-condition measured table and matrices |
| [`reports/improvement_error_analysis.md`](reports/improvement_error_analysis.md) | Aggregate paired error analysis for the intervention |
| [`REPRODUCE.md`](REPRODUCE.md) | Commands, artifacts, deviations, and limits |
| [`scripts/summarize_baselines.py`](scripts/summarize_baselines.py) | Rebuild baseline table and Figure 2 from archived JSON files |
| [`scripts/summarize_results.py`](scripts/summarize_results.py) | Rebuild the complete comparison from saved JSON |
| [`scripts/plot_report_figures.py`](scripts/plot_report_figures.py) | Rebuild training curves, confusion matrices, and model diagram |
| [`scripts/build_report_pdf.py`](scripts/build_report_pdf.py) | Render the paginated A4 submission PDF |
| [`data/README.md`](data/README.md) | Data access and licenses; raw data is not uploaded |
| [`data/provenance.md`](data/provenance.md) | Pinned upstream manifest reference and retrospective version limit |
| [`scripts/smoke_test_saved_results.py`](scripts/smoke_test_saved_results.py) | Fast check of archived outputs, with no download or GPU |

## Run

Select a GPU Colab runtime. Run notebook 02 top to bottom for baselines. Once the fusion `final_model.pt` is saved, run notebook 03 for the matched improvement. Notebook 03 can restore the official dataset in a new runtime and resume its own training from an epoch checkpoint in Drive, but it requires the fusion baseline checkpoint. Full instructions and result paths are in [REPRODUCE.md](REPRODUCE.md).

Do not use "Run all" in notebook 02 after metadata and image training have completed: it repeats them. A Colab runtime reset clears in-memory model weights; `history.csv` alone cannot resume a model. `final_model.pt` is saved at the fixed final epoch. Notebook 03 also saves an epoch checkpoint for resumption.

## Main measured finding

For a quick check of the saved results, run `python scripts/smoke_test_saved_results.py` from the repository root. It verifies the archived confusion matrices, metrics, fixed epoch, and intervention deltas without downloading the dataset. Full model training requires the licensed data bundle and a GPU.

On the published validation split, image-only macro-F1 is 0.8190; standard fusion is 0.8133 but has slightly higher ROC-AUC (0.9135 versus 0.9085). The p=0.3 metadata-dropout fusion has macro-F1 0.8266 with clean inputs and 0.8210 when all metadata are masked. It exceeds standard fusion by about 0.0133 in both conditions, so the run does not establish an additional missing-input robustness benefit. The split shares patient and lesion identifiers between training and validation; none of these scores estimates independent-patient performance.

## Provenance and responsible use

Paper scores, preliminary Colab displays, and archived measured results are kept separate. Report claims only from saved artifacts. Dataset image licenses include CC-0, CC-BY, and CC-BY-NC; consult the collection for attribution and noncommercial limits. Do not upload raw patient metadata, image bundles, credentials, or large checkpoints. This is a research exercise, not a clinical decision tool.

Codex assisted with source discovery, repository organization, code, audit visualization, and editing. The student remains responsible for running and checking experiments, reading sources, interpreting outcomes, and defending the submission.

