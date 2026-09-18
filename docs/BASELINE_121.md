# Verified 121-class baseline

This repository records, but does not retrain or replace, the existing baseline.

- Architecture: DINOv2 ViT-B/14.
- Input: resize 378 then center crop 336 for evaluation/inference.
- Classes: 121 (Food-101 plus 20 Arabic/MENA dishes).
- Best checkpoint epoch: 3.
- Validation Top-1: 96.997%.
- Final independent-test Top-1: 94.987%.
- Final independent-test Top-5: 99.149%.
- Test images: 25,853.

The test report is retained in `reports/baseline_121/`. The exact training configuration and scripts are retained in `configs/` and `scripts/`.

Historical provenance is documented in the project audit. The original 101-class source-checkpoint hash and the full epoch-by-epoch training log were not preserved with the available project files.
