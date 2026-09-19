# Checkpoint Verification Result

## Status

The available external checkpoint is the registered protected baseline.

## Verified properties

- Filename: `best_checkpoint.pt`
- Size: 1,040,301,749 bytes
- SHA-256: `662b398c7e377cca59b984ede3c2d6460c70eef5d7c74beff3da57cda8c6c469`
- Architecture metadata: `DINOv2_ViTB14`
- Class mapping: 121 unique contiguous indices, 0 through 120
- Classifier head: 121 outputs by 768 input features
- Preprocessing: bicubic resize to 378, center crop to 336, ImageNet normalization
- State dictionary: strict model load succeeded

No model file was modified.

## Operational finding

On a Windows console using CP1252, the baseline inference script can raise `UnicodeEncodeError` while printing the Unicode arrow in the preprocessing description. Checkpoint loading and validation complete before that print failure. Running Python in UTF-8 mode (`python -X utf8 ...`) succeeds.

This is a command-line portability issue, not a model-weight failure. Any future fix must be a focused change reviewed separately; this audit does not alter the protected baseline script.
