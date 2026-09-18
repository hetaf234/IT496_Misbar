#!/usr/bin/env python3
"""Final, read-only test evaluation for the 121-class DINOv2 transfer model."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class DinoClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, outputs: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(int(getattr(backbone, "embed_dim", 768)), outputs)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


class MappedFolder(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], transform):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for name, index in sorted(class_to_idx.items(), key=lambda item: item[1]):
            directory = root / name
            if not directory.is_dir():
                raise RuntimeError(f"Missing test directory: {directory}")
            self.samples.extend((path, index) for path in sorted(directory.iterdir()) if path.is_file())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as source:
            return self.transform(source.convert("RGB")), label


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--original-classes", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mapping = checkpoint.get("class_to_idx")
    if not isinstance(mapping, dict) or len(mapping) != 121 or set(mapping.values()) != set(range(121)):
        raise RuntimeError("Checkpoint does not contain a valid 121-class class_to_idx mapping")
    classes = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    test_root = args.dataset / "test"
    folder_mapping = datasets.ImageFolder(test_root).class_to_idx
    if folder_mapping != mapping:
        mismatch = {name: [mapping.get(name), folder_mapping.get(name)] for name in sorted(set(mapping) | set(folder_mapping)) if mapping.get(name) != folder_mapping.get(name)}
        raise RuntimeError(f"Saved checkpoint class_to_idx does not match test mapping: {mismatch}")
    preprocessing = checkpoint.get("preprocessing", {})
    image_size = preprocessing.get("image_size", 336)
    eval_resize = preprocessing.get("eval_resize", 378)
    mean = preprocessing.get("mean", (0.485, 0.456, 0.406))
    std = preprocessing.get("std", (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize(eval_resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test = MappedFolder(test_root, mapping, transform)
    counts = [sum(1 for _, label in test.samples if label == index) for index in range(121)]
    if len(test) != sum(counts) or not all(counts):
        raise RuntimeError("Invalid test set")
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)

    backbone = torch.hub.load(str(args.dinov2_source), "dinov2_vitb14", source="local", pretrained=False)
    model = DinoClassifier(backbone, 121)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    confusion = torch.zeros(121, 121, dtype=torch.int64)
    loss_sum = correct1 = correct5 = total = 0
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits = model(images)
            batch_count = len(targets)
            loss_sum += F.cross_entropy(logits, targets).item() * batch_count
            guesses = logits.argmax(1)
            correct1 += guesses.eq(targets).sum().item()
            correct5 += (logits.topk(5, 1).indices == targets[:, None]).any(1).sum().item()
            total += batch_count
            confusion.index_put_((targets.cpu(), guesses.cpu()), torch.ones(batch_count, dtype=torch.int64), accumulate=True)
    torch.cuda.synchronize()
    recalls = confusion.diag().float() / confusion.sum(1).clamp_min(1)
    precisions = confusion.diag().float() / confusion.sum(0).clamp_min(1)
    f1 = 2 * precisions * recalls / (precisions + recalls).clamp_min(1e-12)
    per_class = [{"class_index": index, "class_name": name, "test_count": counts[index], "correct": int(confusion[index, index]),
                  "accuracy": float(recalls[index]), "recall": float(recalls[index]), "precision": float(precisions[index]), "f1": float(f1[index])}
                 for index, name in enumerate(classes)]
    original_names = set(json.loads(args.original_classes.read_text(encoding="utf-8"))["classes"])
    if len(original_names) != 101:
        raise RuntimeError("The original Food-101 class file must contain exactly 101 classes")
    if not original_names.issubset(mapping):
        raise RuntimeError("Original Food-101 classes are not a subset of the saved 121-class mapping")
    original_indices = [mapping[name] for name in classes if name in original_names]
    new_indices = [mapping[name] for name in classes if name not in original_names]
    def group(indices):
        group_total = int(confusion[indices].sum())
        return {"classes": len(indices), "images": group_total, "top1_accuracy": float(confusion[indices, indices].diag().sum() / group_total),
                "macro_f1": float(f1[indices].mean())}
    pairs = []
    for actual in range(121):
        for predicted in range(121):
            if actual != predicted and confusion[actual, predicted]:
                pairs.append({"true_class": classes[actual], "predicted_class": classes[predicted], "count": int(confusion[actual, predicted])})
    pairs.sort(key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))
    report = {"evaluation_type": "final_independent_test_only", "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
              "architecture": checkpoint.get("architecture"), "class_to_idx_verified_equal_to_test": True,
              "test_images": total, "images_per_class": {classes[i]: counts[i] for i in range(121)},
              "test_loss": loss_sum / total, "test_top1_accuracy": correct1 / total, "test_top5_accuracy": correct5 / total,
              "macro_f1": float(f1.mean()), "original_food101": group(original_indices), "new_20_classes": group(new_indices),
              "worst_10_classes": sorted(per_class, key=lambda row: (row["accuracy"], row["class_name"]))[:10],
              "most_confused_10_pairs": pairs[:10], "preprocessing": {"image_size": image_size, "eval_resize": eval_resize, "mean": mean, "std": std},
              "cuda_device": torch.cuda.get_device_name(0), "wall_seconds": time.time() - started}
    atomic_json(args.output / "final_test_metrics.json", report)
    with (args.output / "final_test_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class[0].keys()); writer.writeheader(); writer.writerows(per_class)
    with (args.output / "final_test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["true_class", *classes])
        for name, row in zip(classes, confusion.tolist(), strict=True): writer.writerow([name, *row])
    atomic_json(args.output / "final_test_confusion_pairs.json", {"pairs": pairs})
    print("FINAL_TEST_EVALUATION_COMPLETED", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
