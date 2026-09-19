# Uncertainty and OOD Decision Protocol

## Purpose

The application must not present an unsupported food label as a fact. The baseline classifier always returns one of its 121 classes, so a separate decision layer must decide whether that prediction is safe to show.

## Evaluation order

1. Export logits from the unchanged baseline for validation and test.
2. Fit probability temperature on validation only.
3. Select decision thresholds on validation only.
4. Freeze the policy.
5. Evaluate it once on test and on a separately designed OOD set.
6. Approve LLaVA integration only after the classifier-only policy is documented.

The test set must never select temperature, thresholds, prompts, or hyperparameters.

## Proposed user-facing decisions

The final numeric boundaries will come from validation results; they are not hard-coded in advance.

- **Accept:** show the Top-1 food when calibrated confidence is above the validated acceptance threshold.
- **Clarify:** when evidence is intermediate, show up to three materially plausible candidates and ask the user to choose or provide another image. Do not cycle through all five predictions as if each is equally credible.
- **Abstain:** for low confidence, likely OOD, severe image-quality failure, or user rejection of all candidates, state that the food could not be identified reliably. Request a clearer image or a text description.
- **LLaVA review:** later, LLaVA may describe visible evidence or help ask a clarification question. It must not silently override the classifier or turn a rejected unknown into a confident 121-class label.

## Required OOD groups

- Non-food objects and scenes
- Foods outside the 121 supported classes
- Multiple dishes with no single dominant target
- Packaged food, menus, logos, and text-heavy images
- Cropped, blurred, dark, occluded, or extremely small images
- Visually similar unsupported regional dishes

Results must be reported separately for each group so a large easy non-food group cannot hide failures on unsupported foods.

## `other` class gate

Do not add a 122nd `other` class merely because softmax is overconfident. Train it only if all conditions hold:

1. Confidence calibration and abstention remain insufficient on a representative OOD set.
2. The proposed `other` data has a stable, documented sampling policy.
3. It does not contain mislabeled examples of the 121 known classes.
4. A separate 122-class experiment improves OOD behavior without materially reducing known-class performance.

Otherwise, keep 121 outputs and implement `unknown` as a decision-layer state rather than a visual class.

## Acceptance evidence

At minimum, report known-class coverage, selective accuracy, error-detection rate, calibration error, and OOD acceptance/rejection rates. Include confidence intervals and per-group OOD results before declaring the layer production-ready.
