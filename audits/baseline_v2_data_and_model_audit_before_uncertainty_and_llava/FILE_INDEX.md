# File and directory index

| Path | Purpose |
| --- | --- |
| `README.md` | Overall objective and baseline protection rules. |
| `00_scope_and_rules/` | Audit scope, allowed actions, and prohibited actions. |
| `01_input_inventory/` | Inventory of datasets, checkpoints, code, and hashes. |
| `02_checkpoint_verification/` | Checkpoint SHA-256, architecture, classes, and preprocessing verification. |
| `03_dataset_integrity/` | Image decoding, counts, dimensions, and manifest reconciliation. |
| `04_duplicates_and_leakage/` | Exact duplicates, near-duplicates, and split leakage checks. |
| `05_label_and_image_quality_review/` | Review of suspicious labels and low-quality images. |
| `06_baseline_reproduction/` | Baseline evaluation reproduction without training. |
| `07_error_analysis_and_improvement/` | Error diagnosis, remedies, and measurable acceptance tests. |
| `08_uncertainty_and_ood_readiness/` | Requirements for calibration, OOD detection, and Unknown decisions. |
| `09_other_class_decision/` | Comparison plan for 121+OOD versus a 122-class Other experiment. |
| `10_llava_readiness_gate/` | Conditions and boundaries for LLaVA integration. |
| `11_final_decision/` | Final findings and recommended next action. |
| `configs/` | Audit-only configuration that does not change the baseline. |
| `scripts/` | Tested read-only programs for static analysis, external asset verification, image integrity, prediction export, and uncertainty analysis. |
| `tests/` | Unit tests for audit calculations. |
| `working_copies/` | Isolated copies of baseline code when experimental edits are required. |
| `audit_results/` | Generated audit results that do not replace baseline reports. |

The `train`, `val`, and `test` images and `.pt` weights remain outside GitHub. Their locations and hashes will be recorded in the input inventory.
