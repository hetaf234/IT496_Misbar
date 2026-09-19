#!/usr/bin/env python3
"""Export per-image baseline logits for calibration and uncertainty analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


AUDIT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AUDIT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from infer_final_food_v2_dinov2 import load_checkpoint, make_transform  # noqa: E402


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="ImageFolder-compatible split directory")
    parser.add_argument("--output", type=Path, required=True, help="Local .npz output; do not commit it")
    parser.add_argument("--mode", choices=["in-domain", "ood"], default="in-domain")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, classes, preprocessing, _ = load_checkpoint(args.checkpoint)
    dataset = ImageFolder(args.images, transform=make_transform(preprocessing))
    expected_mapping = {name: index for index, name in enumerate(classes)}
    if args.mode == "in-domain" and dataset.class_to_idx != expected_mapping:
        missing = sorted(set(expected_mapping) - set(dataset.class_to_idx))
        extra = sorted(set(dataset.class_to_idx) - set(expected_mapping))
        raise RuntimeError(f"Dataset class mapping differs from checkpoint; missing={missing}, extra={extra}")

    device = choose_device(args.device)
    model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    batches = []
    targets = []
    with torch.inference_mode():
        for images, labels in loader:
            batches.append(model(images.to(device, non_blocking=True)).cpu().numpy().astype(np.float32))
            targets.append(labels.numpy().astype(np.int64))
    logits = np.concatenate(batches) if batches else np.empty((0, len(classes)), dtype=np.float32)
    target_array = np.concatenate(targets) if targets else np.empty((0,), dtype=np.int64)
    relative_paths = np.asarray(
        [str(Path(path).resolve().relative_to(args.images.resolve())) for path, _ in dataset.samples]
    )
    metadata = {
        "mode": args.mode,
        "images_directory_name": args.images.resolve().name,
        "checkpoint_filename": args.checkpoint.name,
        "samples": len(dataset),
        "device": str(device),
        "note": "Paths are relative. This file contains model outputs and must remain outside normal Git.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        logits=logits,
        targets=target_array,
        relative_paths=relative_paths,
        checkpoint_classes=np.asarray(classes),
        source_classes=np.asarray(dataset.classes),
        metadata=np.asarray(json.dumps(metadata)),
    )
    predictions = logits.argmax(axis=1) if len(logits) else np.empty((0,), dtype=np.int64)
    if args.mode == "in-domain" and len(logits):
        print(f"Top-1 accuracy: {(predictions == target_array).mean():.6f}")
    print(f"Saved {len(dataset):,} predictions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
