# Baseline V2 Data and Model Audit Before Uncertainty and LLaVA

This directory is an isolated workspace for auditing Dataset V2 and the 121-class checkpoint, then identifying evidence-based improvements before uncertainty handling or LLaVA integration.

The audit is not limited to confirming reported numbers. Every discovered issue must be connected to evidence, impact, a proposed remedy, and a measurable acceptance test.

## Protection boundaries

- Do not edit the baseline files in `scripts/`, `configs/`, or `reports/baseline_121/`.
- Do not modify `final_food_dataset_v2` or `best_checkpoint.pt`.
- Copy code that needs experimental changes into `working_copies/` under a new name.
- Give every cleaned dataset a new version, such as `final_food_dataset_v3_clean_121`.
- Give every new checkpoint a new name, SHA-256, and independent reports.
- Keep dataset images and large weights outside ordinary Git and supply their locations through arguments or local configuration.
- Never use the test split to tune hyperparameters or decision thresholds.

## Workflow

1. Inventory the available assets.
2. Verify checkpoint identity, preprocessing, and class order.
3. Validate image integrity and reconcile the dataset with its manifest.
4. Check duplicates and split leakage.
5. Review suspicious labels and low-quality images.
6. Reproduce the baseline evaluation when the test images are available.
7. Convert errors into measurable improvement proposals.
8. Assess readiness for calibration, uncertainty, and OOD detection.
9. Decide whether a separate 122-class `Other` experiment is justified.
10. Apply a readiness gate before LLaVA integration.

See `FILE_INDEX.md` for the purpose of every file and directory.
