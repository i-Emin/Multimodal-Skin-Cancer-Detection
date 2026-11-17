# Reproduction record

## Scope and data

This project tests the **concept** of combining dermoscopic images and clinical metadata on the [Ahammed et al. (2025)](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html) data and published manifests. It does not repeat the original external ISIC 2020 test. The paper's reported score and our validation measurements answer different questions and must remain in separate tables.

The official bundle has 17,060 images and metadata rows. The authors' manifests have 15,098 train and 1,962 validation samples with balanced binary `class` labels. The bundle's `diagnosis_1` is not the target. The published split has 433 shared nonmissing patient IDs and 488 shared lesion IDs across train/validation; image IDs have zero overlap. Missing patient IDs are common. See [data audit](reports/data_audit.md).

## Colab sequence

1. Select a GPU runtime. Run [notebook 01](notebooks/01_dataset_audit_colab.ipynb) for the data audit if starting from scratch.
2. Run [notebook 02](notebooks/02_baselines_colab.ipynb) for three baselines. It reuses `/content/isic-dicm-17k` or downloads the official ~6.9 GB bundle and saves under `MyDrive/YZM5110/results/baselines/<mode>/`.
3. Confirm `MyDrive/YZM5110/results/baselines/fusion/final_model.pt` exists. Then run [notebook 03](notebooks/03_metadata_dropout_colab.ipynb). It can restore the dataset after a runtime reset and saves under `MyDrive/YZM5110/results/improvement/`.
4. Archive small, non-sensitive JSON/CSV outputs and generated figures in this public repository after review. Keep weights, raw images, and raw patient metadata private.

If notebook 02 is interrupted after metadata/image complete, run its setup cells and invoke only the unfinished mode. The current code writes `history.csv` each epoch but writes model weights only at the final epoch. A history file cannot resume training.

Equivalent command for one mode in a prepared GPU environment:

```bash
python -m src.baseline --mode fusion --data-dir /content/isic-dicm-17k --output-dir /content/drive/MyDrive/YZM5110/results/baselines --epochs 8 --batch-size 32 --lr 1e-4 --seed 42 --save-checkpoint
```

Repeat with `--mode metadata` and `--mode image`. Each mode writes `run.json` (configuration, dependencies, GPU, source digest), `history.csv`, `metrics.json` (fixed final epoch, confusion matrix, per-class scores, time, peak GPU memory), `validation_predictions.csv`, and optional `final_model.pt`. The notebooks embed the versioned source and record its SHA-256. The original manifests are read from the authors' GitHub at runtime; freeze their revision for a final archival rerun.

After all four training runs and notebook 03's matched evaluation complete, generate the report table and SVG from the saved JSON outputs:

```bash
python scripts/summarize_results.py --drive-root /content/drive/MyDrive/YZM5110/results --output-dir reports/generated
```

The output is `reports/generated/measured_results.md` and `model_metrics.svg`. Review and commit these only after checking the source JSONs. The script fails on missing measured files rather than inserting placeholder scores.

## Improvement

The [hypothesis](reports/improvement_hypothesis.md) was recorded before the baseline fusion outcome. The only deliberate training change is `metadata_dropout_p=0.3`. The control is the existing p=0 fusion checkpoint with matched seed, split, epochs, batch size, learning rate, and backbone. Notebook 03 evaluates both on clean and fully missing metadata and saves `comparison.json` and per-image prediction CSVs.

## Deviations and limits

- ResNet18 replaces the selected EfficientNet-B3 comparison; our budget is eight epochs rather than 100.
- AdamW with a fixed learning rate and training crop/flip differ from the paper. Validation uses resize and center crop at 224 pixels.
- The paper's separate ISIC 2020 test is not evaluated. Our published validation split has patient/lesion overlap.
- A single seed and split cannot establish stable between-run uncertainty. Image-level intervals cannot repair group leakage.

The matched controls support a within-implementation comparison. They do not establish numerical reproduction of the paper's external-test score or generalization to new patients.

## Data and licenses

Collection: https://api.isic-archive.com/doi/isic-dicm-17k/ . Images carry CC-0, CC-BY, or CC-BY-NC terms. Follow individual attribution and noncommercial limits. Do not commit the raw bundle, credentials, or large checkpoints.

