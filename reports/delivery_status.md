# Delivery status and final-run checklist

Updated: 17 November 2026. Due: **18 November 2026 before the presentation session**, according to the course project brief. The student plans to finish today.

## Completed or prepared

| Requirement | Status | Evidence / caveat |
|---|---|---|
| Public project repository and licensed data instructions | Prepared | `README.md`, `REPRODUCE.md`, `data/README.md` |
| Initial data audit and visual | Prepared | `reports/data_audit.md`, `figures/data_audit.svg`, `results/data_audit_counts.json` |
| Six-paper source list and comparison structure | Draft | `reports/literature_matrix.md`; close reading and critical synthesis still need strengthening |
| Reproduction implementation and three baselines | Code prepared | `src/baseline.py`, notebook 02; metadata/image completed in student's Drive, fusion being rerun |
| Pre-specified one-change improvement and control | Code prepared | `reports/improvement_hypothesis.md`, notebook 03; no improved result yet |
| Report structure and references | Draft | `reports/report_draft.md`; real measured values, critical synthesis, and PDF still needed |
| Saved-result table and figure generator | Prepared | `scripts/summarize_results.py`; must run on real Drive JSONs after improvement |

## After fusion finishes

1. Confirm `results/baselines/fusion/final_model.pt` and `metrics.json` exist in Drive. Do not infer completion from `history.csv` alone.
2. Run notebook 03 on the L4 GPU. A fresh Colab runtime may need to re-download the official data. Keep the fusion checkpoint in Drive.
3. Provide `metrics.json`, `run.json`, `history.csv`, and `validation_predictions.csv` for metadata, image, and fusion, plus `results/improvement/comparison.json`, its run artifacts, and four condition prediction CSVs. Share small files only; keep weights and raw data private.
4. Generate final tables/figures from saved outputs. Inspect confusion matrices, class-specific recall/precision, at least a few favorable and unfavorable examples, time and peak GPU memory, and clean/missing tradeoff. Preserve the fixed final epoch and report a negative improvement honestly if observed.
5. Complete the report's eight required sections, put the repository URL on page 1, export `<LastName>_YZM5110_Research.pdf`, inspect the rendered pages, and add the PDF to the repository or submit it through the course channel as required.
6. Pin the final public repository version with a commit or tag and record its identifier in the report. Verify a clean Colab launch, the exact data access steps, source revision, dependencies, configs, and figure/table scripts.

## Highest-priority gaps

- **Measured artifacts are not yet in GitHub.** The student's Drive contains metadata/image runs; fusion is ongoing. Results in screenshots are preliminary until the saved JSON/CSV is reviewed.
- **Improvement training and matched evaluation remain.** The planned p=0.3 run may take a similar time to fusion on the L4.
- **The literature matrix is a starting matrix**, not yet the required in-depth critique of at least five papers. Check objectives, preprocessing/split units, hardware/code availability, baseline fairness, and limitations against full texts.
- **No final PDF is complete.** The report draft contains conspicuous pending-result fields and must not be submitted as final.
- **No patient-independent evaluation has been run.** The published split's group overlap must be disclosed prominently; do not claim clinical generalization.

Course brief: `YZM5110_Research_Project.pdf`, supplied by the student. This checklist summarizes its deliverables; the PDF itself remains the authority for grading and submission details.

