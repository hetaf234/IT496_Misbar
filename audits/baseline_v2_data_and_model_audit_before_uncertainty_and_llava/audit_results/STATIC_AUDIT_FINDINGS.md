# Interim Static Audit Findings

Generated from committed metadata and reports. This does not replace image-level or model-level verification.

## Verified now

- Manifest: 105,681 rows and 121 classes.
- Splits: 76,198 train, 3,630 validation, 25,853 test.
- Exact duplicate groups: 0; cross-split: 0.
- Same-pHash cross-split groups: 0.
- All registry/report/config consistency checks pass: True.

## Improvement-relevant findings

- **HIGH** — 9 classes have fewer than 30 test images; perfect accuracy is not reliable evidence.
  Action: Build a separately sourced external test set with at least 50-100 diverse images per affected class.
- **MEDIUM** — 11 more classes have only 30-99 test images.
  Action: Expand independent evaluation coverage and retain confidence intervals in reports.
- **HIGH** — Committed results do not contain per-image logits or probabilities.
  Action: Re-run evaluation and save logits for calibration, risk-coverage, selective accuracy, and threshold selection. This is evaluation, not retraining.
- **HIGH** — No unknown/OOD evaluation set is present.
  Action: Create licensed non-food, unsupported-food, ambiguous/multi-dish, packaging, and poor-quality OOD groups before choosing abstention or an `other` class.
- **MEDIUM** — Largest error pair: steak -> filet_mignon (34 images).
  Action: Visually review high-confusion pairs for label-policy overlap, improve data first, and fine-tune only if the error is learnable.
- **HIGH** — The manifest records 5 images with a side shorter than 32 pixels.
  Action: Inspect these files and replace or exclude unusable images only in a new dataset version.

## Current gate decision

Do not add LLaVA or train an `other` class yet. Obtain the complete images, export baseline logits, calibrate confidence, and test abstention on a designed OOD set. The current checkpoint remains protected.

See `static_audit.json` for machine-readable class confidence intervals and details.
