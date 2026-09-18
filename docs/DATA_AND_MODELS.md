# Data and model sharing

## Baseline checkpoint

The verified model is `best_checkpoint.pt`, a DINOv2 ViT-B/14 food classifier with 121 classes. Its required SHA-256 is:

```text
662b398c7e377cca59b984ede3c2d6460c70eef5d7c74beff3da57cda8c6c469
```

Put it locally at `models/best_checkpoint.pt`. It is excluded from normal Git because it is approximately 1 GB. Before use, run the VS Code task **Verify baseline checkpoint** or the command in the root README.

## Dataset v2

The verified dataset shape is:

| Split | Images | Classes |
| --- | ---: | ---: |
| train | 76,198 | 121 |
| val | 3,630 | 121 |
| test | 25,853 | 121 |

Expected local layout:

```text
data/final_food_dataset_v2/
├── train/<class-name>/
├── val/<class-name>/
└── test/<class-name>/
```

The full image dataset is deliberately external to ordinary Git. Share it through the team-approved private storage location. The repository stores the version evidence in `dataset_reports/`, including source paths, split counts, and duplicate-audit results.

## Large-file policy

Do not commit images, checkpoints, or LLaVA weights directly to ordinary Git. If the team decides to use Git LFS, add it intentionally in a separate pull request after confirming storage and download limits. Until then, share large files through private storage and retain their hashes in this repository.
