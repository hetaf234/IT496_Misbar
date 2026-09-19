# Audit Status

## Completed

- Protected work is isolated on `audit/baseline-v2-pre-llava-uncertainty`.
- Registry, metric, manifest, split-count, and preprocessing consistency checks pass.
- The available checkpoint SHA-256, architecture, 121-class mapping, preprocessing metadata, classifier-head shape, and strict state-dict load are verified.
- Static analysis identifies class-level evidence limits, weakest classes, primary confusion pairs, and manifest quality flags.
- Read-only tools for full image decoding/hash verification and per-image prediction export are implemented.
- Validation-only temperature scaling and confidence-threshold selection are implemented and unit-tested.
- The user-facing Accept / Clarify / Abstain protocol and the decision gate for an `other` experiment are documented.

## Interim findings requiring action

- Nine classes have fewer than 30 test images; eleven more have fewer than 100.
- Five Kunafa images have a recorded side shorter than 32 pixels, including one validation image.
- The largest reported directional confusion is `steak` to `filet_mignon` with 34 errors.
- Existing reports lack per-image logits, so calibration and abstention performance are not yet measurable.
- No representative OOD set is available, so neither an unknown threshold nor a 122nd `other` class is approved.
- The inference script has a Windows CP1252 console-print portability issue; UTF-8 mode works and the model itself is valid.

## Waiting for external assets

- Complete `final_food_dataset_v2/train`, `val`, and `test` image directories
- A documented external/OOD evaluation set grouped by failure type

## Next execution after images arrive

1. Verify every dataset path, image decode, dimension, and SHA-256.
2. Visually review all tiny files and prioritized confusion samples.
3. Export validation logits and fit calibration/thresholds without test access.
4. Export test and OOD logits and evaluate the frozen policy.
5. Decide among no model change, data cleanup plus fine-tuning, or a separate 122-class experiment.
6. Apply the LLaVA readiness gate only after the classifier decision layer passes.

No retraining has been run and no protected baseline file has been changed.
