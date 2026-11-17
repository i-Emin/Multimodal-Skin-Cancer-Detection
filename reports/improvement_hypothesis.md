# Improvement experiment: metadata dropout

Written on 2026-11-17 while the initial fusion baseline was still running. Interim fusion epoch metrics were visible in Colab, but the completed fixed-final-epoch fusion result and all improved-model outcomes had not been observed when this plan was written. The intervention and primary endpoint were fixed at that point.

## Question and intervention

Does training the fusion model with the **entire clinical-metadata vector masked to zero for 30% of training examples** make it more robust when metadata is unavailable at evaluation time? A zero vector represents missing metadata after train-fitted scaling/one-hot encoding. The mask is applied only during training; evaluation uses clean inputs or the explicitly masked condition.

## Matched control and outcomes

- Control: existing fusion model (`p=0`), trained with seed 42, 8 epochs, batch size 32, learning rate 1e-4, ImageNet-pretrained ResNet18, same published train/validation manifests.
- Intervention: same setup with `p=0.3` metadata dropout; no other deliberate changes.
- Primary comparison: validation **macro-F1** under fully missing metadata (intervention minus control).
- Cost check: validation macro-F1 under clean metadata. Report whether robustness comes with clean-input loss.
- Also report balanced accuracy, ROC-AUC, confusion matrices, training time, and peak GPU memory for both conditions where available.

Do not tune `p` or change the condition after inspecting these results. The published validation split has patient and lesion overlap, so this experiment is a **within-split robustness comparison**, not an independent clinical generalization claim. A later group-controlled replication would strengthen the conclusion.

