# Team workflow

## Protect the baseline

`main` represents the documented project state. Never overwrite the verified 121-class checkpoint or edit its test reports. New work happens in a branch and is merged through a Pull Request.

## Adding an `other` class

1. Create a branch such as `friend/add-other-class`.
2. Create a new dataset version; do not alter `final_food_dataset_v2` in place.
3. Document source, licence, train/validation/test counts, and duplicate checks.
4. Change the classifier output count from 121 to 122 only in the new experiment configuration.
5. Train/evaluate a separate checkpoint and give it a new hash.
6. Compare it against the preserved baseline before merging.

## Adding LLaVA

LLaVA should initially be an optional downstream assistant, not a replacement for the food classifier:

```text
image -> 121-class classifier -> top-k/confidence -> optional LLaVA review -> response
```

Add LLaVA code under a new, clearly named folder and commit its configuration, prompt template, model identifier, dependencies, tests, and decision policy. Do not commit LLaVA weights or access tokens. Keep classifier-only evaluation separate from any LLaVA-assisted evaluation.

## Before merging

- Explain what changed and why.
- Keep model/dataset hashes current.
- Run the relevant lightweight checks.
- Preserve the existing reports; add new reports beside them, never over them.
