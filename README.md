# Misbar — IT496

Misbar is a bilingual food-safety project. The repository preserves the verified 121-class food-recognition baseline and provides a clean place for future additions such as an `other` class and a LLaVA-assisted pipeline.

## What is in this repository?

| Folder | Purpose |
| --- | --- |
| `scripts/` | Training, evaluation, and local inference programs for the 121-class baseline. |
| `configs/` | Exact saved configuration for the final DINOv2 transfer run. |
| `reports/baseline_121/` | Immutable final-test metrics, per-class results, and confusion data. |
| `dataset_reports/` | Dataset-v2 counts, image manifest, and duplicate/leakage audit. |
| `docs/` | Dataset/model setup, baseline facts, and team workflow. |
| `models/` | Local-only location for model weights; weights are intentionally ignored by Git. |
| `data/` | Local-only location for the image dataset; images are intentionally ignored by Git. |

## Quick start in VS Code

1. Clone this private repository and open its folder in **Visual Studio Code**.
2. Install the **Python** extension when VS Code suggests it.
3. Create a Python 3.10+ virtual environment, then install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Ask a team member for the verified baseline checkpoint and place it at `models/best_checkpoint.pt`.
5. Verify it without running inference:

   ```bash
   python scripts/infer_final_food_v2_dinov2.py --checkpoint models/best_checkpoint.pt --verify-only
   ```

The expected SHA-256 is in `model_registry.json`. Do not use a checkpoint whose hash differs from that value as the baseline.

## Important boundaries

- The current baseline has **121 output classes**. Adding `other` creates a new 122-class experiment; it must not overwrite the baseline checkpoint.
- The training and final-test scripts require CUDA. Local inference supports Apple Silicon MPS when available, otherwise CPU.
- Dataset images and `.pt` weights are not committed to ordinary Git. See `docs/DATA_AND_MODELS.md` for the controlled sharing process.
- Never commit API keys, Hugging Face tokens, or private data.

## Team workflow

Keep `main` stable. Create one branch per focused change, for example:

```text
friend/add-other-class
friend/add-llava-pipeline
feature/improve-confidence-calibration
```

Open a Pull Request before merging. The baseline, model registry, dataset reports, and evaluation reports provide the reference point for review.

See `docs/TEAM_WORKFLOW.md` before adding a model, class, or dataset version.
