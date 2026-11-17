# Delivery status

Updated 17 November 2026. The course brief requests the research PDF before the 18 November presentation session.

| Requirement | Current evidence |
|---|---|
| Public reproducible repository | `README.md`, `REPRODUCE.md`, three Colab notebooks, `src/baseline.py`, scripts and settings in saved `run.json` files |
| Official data audit | `reports/data_audit.md`, `results/data_audit_counts.json`, `figures/data_audit.svg` |
| At least five papers with critical comparison | Six numbered sources in `reports/report_draft.md` and `reports/literature_matrix.md` |
| Partial reproduction | Three fixed-epoch baseline outputs in `results/baselines/` |
| One controlled improvement | Metadata dropout 0.3, with four matched clean/missing conditions in `results/improvement/` |
| Quantitative results and figures | `reports/generated/measured_results.md`, four baseline/report figures and six-condition comparison figure |
| Failure and limitations analysis | `reports/baseline_error_analysis.md`, `reports/improvement_error_analysis.md`, and report Sections 7–8 |
| PDF | `output/pdf/Ipek_YZM5110_Research.pdf`, rendered and visually checked locally; repository URL appears on page 1 |

The validation split has overlapping patient and lesion identifiers, so all measured scores are within that published split and are not independent-patient estimates. The paper's separate ISIC 2020 test score is not numerically compared with our validation scores. Large checkpoints, raw images, metadata, and per-image predictions remain in the student's Drive.

The final PDF must be submitted through the course's requested channel. A public repository revision or tag should be retained for the submitted version. The project brief, supplied by the student, is authoritative for exact submission mechanics.

