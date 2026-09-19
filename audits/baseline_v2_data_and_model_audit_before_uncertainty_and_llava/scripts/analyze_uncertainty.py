#!/usr/bin/env python3
"""Calibrate baseline logits and evaluate a validation-selected abstention policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


AUDIT_DIR = Path(__file__).resolve().parents[1]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    probabilities = softmax(logits, temperature)
    return float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean())


def ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    total = len(targets)
    value = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            value += selected.sum() / total * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def metrics(logits: np.ndarray, targets: np.ndarray, temperature: float, threshold: float | None = None):
    probabilities = softmax(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    top_k = min(5, probabilities.shape[1])
    top5 = np.argpartition(probabilities, probabilities.shape[1] - top_k, axis=1)[:, -top_k:]
    result = {
        "samples": len(targets),
        "top1_accuracy": float(correct.mean()),
        "top5_accuracy": float((top5 == targets[:, None]).any(axis=1).mean()),
        "nll": nll(logits, targets, temperature),
        "ece_15_bins": ece(probabilities, targets),
        "mean_confidence": float(confidence.mean()),
    }
    if threshold is not None:
        accepted = confidence >= threshold
        result.update({
            "threshold": threshold,
            "coverage": float(accepted.mean()),
            "accepted_samples": int(accepted.sum()),
            "selective_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
            "error_detection_rate": float((~accepted & ~correct).sum() / max((~correct).sum(), 1)),
        })
    return result


def choose_threshold(logits: np.ndarray, targets: np.ndarray, temperature: float, target_accuracy: float) -> float:
    probabilities = softmax(logits, temperature)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence)
    ordered_correct = correct[order]
    cumulative_accuracy = np.cumsum(ordered_correct) / np.arange(1, len(correct) + 1)
    ordered_confidence = confidence[order]
    group_ends = np.r_[np.flatnonzero(ordered_confidence[:-1] != ordered_confidence[1:]), len(correct) - 1]
    eligible = group_ends[cumulative_accuracy[group_ends] >= target_accuracy]
    if not len(eligible):
        return 1.0
    last = eligible[-1]
    return float(ordered_confidence[last])


def load_npz(path: Path):
    data = np.load(path, allow_pickle=False)
    return data["logits"], data["targets"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--ood", type=Path)
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=AUDIT_DIR / "audit_results/uncertainty_analysis.json")
    args = parser.parse_args()

    val_logits, val_targets = load_npz(args.validation)
    temperatures = np.linspace(0.5, 5.0, 451)
    losses = np.asarray([nll(val_logits, val_targets, value) for value in temperatures])
    temperature = float(temperatures[losses.argmin()])
    threshold = choose_threshold(
        val_logits, val_targets, temperature, args.target_accepted_accuracy
    )
    result = {
        "status": "interim" if args.test is None or args.ood is None else "complete",
        "selection_rule": "Temperature and confidence threshold selected on validation only.",
        "temperature": temperature,
        "target_accepted_accuracy": args.target_accepted_accuracy,
        "confidence_threshold": threshold,
        "validation_uncalibrated": metrics(val_logits, val_targets, 1.0),
        "validation_calibrated_policy": metrics(val_logits, val_targets, temperature, threshold),
        "test_calibrated_policy": None,
        "ood_acceptance": None,
    }
    if args.test:
        test_logits, test_targets = load_npz(args.test)
        result["test_calibrated_policy"] = metrics(test_logits, test_targets, temperature, threshold)
    if args.ood:
        ood_logits, _ = load_npz(args.ood)
        ood_confidence = softmax(ood_logits, temperature).max(axis=1)
        result["ood_acceptance"] = {
            "samples": len(ood_logits),
            "accepted_as_known_rate": float((ood_confidence >= threshold).mean()),
            "rejected_or_escalated_rate": float((ood_confidence < threshold).mean()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
