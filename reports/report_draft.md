# YZM5110 research report — working draft

Repository: https://github.com/i-Emin/Multimodal-Skin-Cancer-Detection

**Status (17 November 2026):** Baseline fusion is being rerun after a Colab interruption; the pre-specified improvement remains to be run. This draft contains no invented scores. Replace the result fields from saved Colab JSON files before PDF submission.

## Abstract

This study examines whether clinical metadata improves binary melanoma classification from dermoscopic images and whether training with missing metadata improves robustness. We audit ISIC-DICM-17K and train metadata-only, image-only, and late-fusion baselines on the authors' published split. We then compare the fusion control against a single pre-specified intervention that masks all clinical features for 30% of training examples. The published split contains shared patient and lesion identifiers across training and validation, limiting claims of independent generalization. Final measured results and conclusions will be inserted from Colab artifacts.

## 1. Research question and source study

Ahammed et al. (CVPR Workshops 2025) curate ISIC-DICM-17K and investigate dermoscopic images combined with age, sex, and anatomical location. The specific published comparison is the transfer-learning EfficientNet-B3 row in their Table 5: image-only AUC **0.8754** and image+metadata AUC **0.8851** on the **10,982-image ISIC 2020 Kaggle test**. The predicted direction is a positive image+metadata difference; a zero or negative within-implementation difference would question whether this benefit transfers to our reduced setup. These published scores are **not directly comparable** to our 1,962-image validation scores. The paper's Table 5 also shows that adding metadata did not help every transfer-learning backbone, which tempers a universal-benefit claim. Our ResNet18 implementation is a reduced-backbone, partial reproduction of the comparison, not an exact reproduction of the paper's external-test score.

## 2. Literature synthesis

**Search and selection.** We began with the selected CVPRW paper and its official dataset repository, then searched PubMed/PMC, publisher pages, and citation/DOI links for skin-lesion image+metadata models, fusion comparisons, dataset provenance, and missing-feature ablations. We retained six papers spanning the source study, prior aggregation, competing fusion mechanisms, feature ablation, data construction, and historical multi-image fusion. We checked titles, years, and claims against primary full texts where available. This is a focused narrative selection, not a systematic review.

The six-paper comparison matrix is in [literature_matrix.md](literature_matrix.md). Ahammed et al. provide the direct dataset and image/metadata comparison. Pacheco and Krohling motivate clinical-feature aggregation, but their smartphone images differ from our dermoscopy setting. Das et al. compare simple and attention-based fusion on HAM10000, motivating a concatenation baseline before adding complexity; the prose and preprocessing figure describe different split proportions, so its protocol needs caution. Sonuç et al. emphasize clinical-feature ablations in a seven-class task; its scores cannot be transferred to our binary task. Pacheco et al.'s PAD-UFES-20 data paper documents a different capture domain. Yap et al. (2018) provide a useful counterexample: adding metadata to their two-image model lowered melanoma AUC from 0.866 to 0.861, although multiclass mAP rose slightly. Different labels, devices, and splits prevent direct comparison of numerical scores across these studies.

## 3. Data and split audit

The official bundle contains 17,060 JPEG images and a 17,060-row, 28-column metadata table. The authors' manifests assign 15,098 training and 1,962 validation samples, balanced within each split. Every manifest ID matches the bundle table, and no image ID occurs in both splits. However, 433 distinct nonmissing patient IDs and 488 lesion IDs occur in both splits; 7,447 training and 1,221 validation rows lack patient IDs. These are overlapping *identifier* counts, not affected-row counts. The archive `diagnosis_1` field is not used as the binary target; we use the authors' `class` field. The full audit and Figure 1 are in [data_audit.md](data_audit.md).

## 4. Methods and reproducibility

**Technical mechanism.** The image branch maps a dermoscopic image to a 512-dimensional ResNet18 feature vector. The metadata branch median-imputes and standardizes age, one-hot encodes sex and general anatomical site using training-only statistics, then maps that vector through a 64-unit ReLU layer. Fusion concatenates the two representations and applies a two-layer classification head to produce two logits. Cross-entropy trains all parameters; AdamW updates them. Metadata-only and image-only models remove the other modality. This is late feature fusion, unlike attention methods that model pairwise interactions before classification.

**Protocol.** The baselines use the same published manifests, seed 42, eight epochs, batch size 32, and learning rate 1e-4. Image and fusion use ImageNet-pretrained ResNet18. Training images are resized/cropped/flipped; validation uses resize and center crop at 224 pixels with ImageNet normalization. The original paper used 100 epochs, a cosine schedule with warm-up, Adam for EfficientNet/ResNet, early stopping, and best-epoch test reporting; our fixed-final-epoch, eight-epoch AdamW run is a deliberate compute reduction. The training code, source digest, history, predictions, settings, and metrics are saved as described in [REPRODUCE.md](../REPRODUCE.md). The student's L4 GPU is the training device. Measured wall time and GPU memory must come from run artifacts.

## 5. Improvement hypothesis and controlled comparison

Before observing the fusion result, we fixed one intervention: mask the entire encoded metadata vector to zero for 30% of training examples. The matched p=0 fusion baseline is the control. Both models are evaluated on clean validation data and with all validation metadata masked. The primary outcome is the difference in macro-F1 under missing metadata; clean-input macro-F1 is the cost check. Balanced accuracy, ROC-AUC, confusion matrices, training time, and peak GPU memory are secondary observations. The fixed plan is in [improvement_hypothesis.md](improvement_hypothesis.md).

## 6. Results — pending measured artifacts

| Model / condition | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | Time | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Metadata-only, clean | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON |
| Image-only, clean | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON |
| Fusion control, clean | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON |
| Fusion control, missing metadata | pending JSON | pending JSON | pending JSON | pending JSON | same training run | same training run |
| Fusion dropout 0.3, clean | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON | pending JSON |
| Fusion dropout 0.3, missing metadata | pending JSON | pending JSON | pending JSON | pending JSON | same training run | same training run |

Required result artifacts: `results/baselines/{metadata,image,fusion}/metrics.json`, `run.json`, and `results/improvement/comparison.json` from Google Drive. Report the final fixed epoch for all baselines. Calculate fusion minus image-only on the same validation set, and intervention minus control separately for clean and missing-metadata conditions. Add confusion matrices and at least one failure analysis from prediction CSVs.

## 7. Limitations and interpretation

The authors' validation split has patient and lesion overlap. Missing group IDs prevent a complete patient-independent check on all examples. A higher validation score therefore cannot establish performance on new patients. The reduced backbone and training recipe differ from the selected paper, while its separate test set is not evaluated. A single seed and a single split do not quantify variability. The missing-metadata experiment measures resilience to a defined artificial absence condition, not the full range of real clinical missingness. No clinical deployment claim follows from these results.

## References

1. Ahammed et al. *Skin Lesion Classification Using Dermoscopic Images and Clinical Metadata: Insights from Multimodal Models*. CVPR Workshops, 2025. https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html
2. Pacheco and Krohling. *The impact of patient clinical information on automated skin cancer detection*. Computers in Biology and Medicine, 2020. https://doi.org/10.1016/j.compbiomed.2019.103545
3. Das, Agarwal, and Shetty. *Comparative analysis of multimodal architectures for effective skin lesion detection using clinical and image data*. Frontiers in AI, 2025. https://doi.org/10.3389/frai.2025.1608837
4. Sonuç et al. *Interpretable multimodal fusion for skin lesion classification using dermoscopic images and patient metadata*. Frontiers in Medicine, 2026. https://doi.org/10.3389/fmed.2026.1871689
5. Pacheco et al. *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones*. Data in Brief, 2020. https://doi.org/10.1016/j.dib.2020.106221
6. Yap, Yolland, and Tschandl. *Multimodal skin lesion classification using deep learning*. Experimental Dermatology, 2018. https://doi.org/10.1111/exd.13777

