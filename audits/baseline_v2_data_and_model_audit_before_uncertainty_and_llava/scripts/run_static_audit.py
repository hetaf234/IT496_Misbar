#!/usr/bin/env python3
"""Audit committed baseline reports and dataset metadata without loading images."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AUDIT_DIR.parents[1]
OUTPUT_DIR = AUDIT_DIR / "audit_results"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson interval for a binomial proportion."""
    if not total:
        return 0.0, 1.0
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> int:
    config = read_json(AUDIT_DIR / "configs" / "audit_config.json")
    expected = config["expected"]
    registry = read_json(REPO_ROOT / "model_registry.json")["baseline_121"]
    metrics = read_json(REPO_ROOT / "reports/baseline_121/final_test_metrics.json")
    pairs = read_json(REPO_ROOT / "reports/baseline_121/final_test_confusion_pairs.json")["pairs"]
    per_class = read_csv(REPO_ROOT / "reports/baseline_121/final_test_per_class_metrics.csv")
    manifest = read_csv(REPO_ROOT / "dataset_reports/final_dataset_manifest_v2.csv")
    leakage = read_csv(REPO_ROOT / "dataset_reports/final_dataset_leakage_report_v2.csv")
    build = read_json(REPO_ROOT / "dataset_reports/build_summary_v2.json")

    split_counts = Counter(row["split"] for row in manifest)
    class_names = {row["class_name"] for row in manifest}
    sha_groups = defaultdict(list)
    phash_groups = defaultdict(list)
    for row in manifest:
        sha_groups[row["sha256"]].append(row)
        phash_groups[row["phash"]].append(row)
    exact_duplicates = [rows for rows in sha_groups.values() if len(rows) > 1]
    exact_cross_split = [rows for rows in exact_duplicates if len({r["split"] for r in rows}) > 1]
    phash_cross_split = [
        rows for rows in phash_groups.values()
        if len(rows) > 1 and len({r["split"] for r in rows}) > 1
    ]
    tiny = [row for row in manifest if min(int(row["width"]), int(row["height"])) < 32]
    below_input = [
        row for row in manifest
        if int(row["width"]) < expected["image_size"] or int(row["height"]) < expected["image_size"]
    ]

    class_evidence = []
    for row in per_class:
        count = int(row["test_count"])
        low, high = wilson(int(row["correct"]), count)
        class_evidence.append({
            "class_name": row["class_name"],
            "test_count": count,
            "accuracy": float(row["accuracy"]),
            "f1": float(row["f1"]),
            "accuracy_ci95": [low, high],
            "ci95_width": high - low,
            "evidence": "insufficient" if count < 30 else "limited" if count < 100 else "adequate",
        })
    insufficient = [row for row in class_evidence if row["test_count"] < 30]
    limited = [row for row in class_evidence if 30 <= row["test_count"] < 100]
    weakest = sorted(class_evidence, key=lambda row: row["accuracy"])[:15]

    checks = {
        "class_count": len(class_names) == expected["classes"] == registry["classes"],
        "train_count": split_counts["train"] == expected["train_images"] == build["train_total"],
        "validation_count": split_counts["val"] == expected["validation_images"] == build["validation_total"],
        "test_count": split_counts["test"] == expected["test_images"] == metrics["test_images"],
        "architecture": metrics["architecture"] == expected["architecture"] == registry["architecture"],
        "input_size": metrics["preprocessing"]["image_size"] == expected["image_size"] == registry["input_size"],
        "top1": math.isclose(metrics["test_top1_accuracy"], registry["final_test_top1"], abs_tol=1e-12),
        "top5": math.isclose(metrics["test_top5_accuracy"], registry["final_test_top5"], abs_tol=1e-12),
    }

    findings = [
        {
            "severity": "high",
            "finding": f"{len(insufficient)} classes have fewer than 30 test images; perfect accuracy is not reliable evidence.",
            "action": "Build a separately sourced external test set with at least 50-100 diverse images per affected class.",
            "classes": [row["class_name"] for row in insufficient],
        },
        {
            "severity": "medium",
            "finding": f"{len(limited)} more classes have only 30-99 test images.",
            "action": "Expand independent evaluation coverage and retain confidence intervals in reports.",
            "classes": [row["class_name"] for row in limited],
        },
        {
            "severity": "high",
            "finding": "Committed results do not contain per-image logits or probabilities.",
            "action": "Re-run evaluation and save logits for calibration, risk-coverage, selective accuracy, and threshold selection. This is evaluation, not retraining.",
        },
        {
            "severity": "high",
            "finding": "No unknown/OOD evaluation set is present.",
            "action": "Create licensed non-food, unsupported-food, ambiguous/multi-dish, packaging, and poor-quality OOD groups before choosing abstention or an `other` class.",
        },
        {
            "severity": "medium",
            "finding": f"Largest error pair: {pairs[0]['true_class']} -> {pairs[0]['predicted_class']} ({pairs[0]['count']} images).",
            "action": "Visually review high-confusion pairs for label-policy overlap, improve data first, and fine-tune only if the error is learnable.",
        },
        {
            "severity": "high" if tiny else "info",
            "finding": f"The manifest records {len(tiny)} images with a side shorter than 32 pixels.",
            "action": "Inspect these files and replace or exclude unusable images only in a new dataset version.",
            "examples": [
                {
                    "split": row["split"], "class_name": row["class_name"],
                    "width": int(row["width"]), "height": int(row["height"]),
                }
                for row in tiny[:10]
            ],
        },
    ]

    result = {
        "status": "interim",
        "scope": "committed metadata and reports only; images were not loaded and model evaluation was not run",
        "consistency_checks": checks,
        "manifest": {
            "rows": len(manifest),
            "classes": len(class_names),
            "split_counts": dict(split_counts),
            "exact_duplicate_groups": len(exact_duplicates),
            "exact_cross_split_groups": len(exact_cross_split),
            "same_phash_cross_split_groups": len(phash_cross_split),
            "below_336_on_at_least_one_side": len(below_input),
            "below_32_on_at_least_one_side": len(tiny),
            "excluded_leakage_candidates": len(leakage),
        },
        "reported_performance": {
            "top1": metrics["test_top1_accuracy"],
            "top5": metrics["test_top5_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "top1_top5_gap": metrics["test_top5_accuracy"] - metrics["test_top1_accuracy"],
            "food101_top1": metrics["original_food101"]["top1_accuracy"],
            "new20_top1": metrics["new_20_classes"]["top1_accuracy"],
        },
        "class_evidence": class_evidence,
        "weakest_classes": weakest,
        "highest_confusion_pairs": pairs[:20],
        "priority_findings": findings,
        "next_runs": [
            "Verify external checkpoint hash and metadata.",
            "Verify all external dataset files against the manifest and decode every image.",
            "Export per-image logits from the unchanged baseline.",
            "Fit calibration and choose abstention thresholds on validation only.",
            "Evaluate the frozen policy once on test and separately on external/OOD data.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "static_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Interim Static Audit Findings", "",
        "Generated from committed metadata and reports. This does not replace image-level or model-level verification.", "",
        "## Verified now", "",
        f"- Manifest: {len(manifest):,} rows and {len(class_names)} classes.",
        f"- Splits: {split_counts['train']:,} train, {split_counts['val']:,} validation, {split_counts['test']:,} test.",
        f"- Exact duplicate groups: {len(exact_duplicates)}; cross-split: {len(exact_cross_split)}.",
        f"- Same-pHash cross-split groups: {len(phash_cross_split)}.",
        f"- All registry/report/config consistency checks pass: {all(checks.values())}.", "",
        "## Improvement-relevant findings", "",
    ]
    for item in findings:
        lines.extend([f"- **{item['severity'].upper()}** — {item['finding']}", f"  Action: {item['action']}"])
    lines.extend([
        "", "## Current gate decision", "",
        "Do not add LLaVA or train an `other` class yet. Obtain the complete images, export baseline logits, calibrate confidence, and test abstention on a designed OOD set. The current checkpoint remains protected.", "",
        "See `static_audit.json` for machine-readable class confidence intervals and details.",
    ])
    (OUTPUT_DIR / "STATIC_AUDIT_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote static audit reports to {OUTPUT_DIR}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
