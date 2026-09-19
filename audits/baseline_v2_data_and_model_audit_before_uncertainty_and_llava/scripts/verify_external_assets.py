#!/usr/bin/env python3
"""Verify external checkpoint and dataset assets without modifying them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AUDIT_DIR.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--verify-image-hashes", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=AUDIT_DIR / "audit_results/external_asset_verification.json",
    )
    args = parser.parse_args()
    expected = json.loads(
        (AUDIT_DIR / "configs/audit_config.json").read_text(encoding="utf-8")
    )["expected"]
    result = {"status": "interim", "checkpoint": None, "dataset": None}

    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
        actual_hash = sha256(checkpoint)
        result["checkpoint"] = {
            "filename": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": actual_hash,
            "sha256_matches_baseline": actual_hash == expected["checkpoint_sha256"],
        }

    if args.dataset:
        dataset = args.dataset.resolve()
        if not dataset.is_dir():
            parser.error(f"Dataset directory not found: {dataset}")
        with (REPO_ROOT / "dataset_reports/final_dataset_manifest_v2.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            manifest = list(csv.DictReader(stream))
        counts = Counter()
        missing = []
        mismatches = []
        for row in manifest:
            relative = Path(row["split"]) / row["class_name"] / Path(row["final_image_path"]).name
            image = dataset / relative
            if not image.is_file():
                missing.append(str(relative))
                continue
            counts[row["split"]] += 1
            if args.verify_image_hashes and sha256(image) != row["sha256"]:
                mismatches.append(str(relative))
        result["dataset"] = {
            "directory_name": dataset.name,
            "expected_files": len(manifest),
            "located_by_split": dict(counts),
            "missing_count": len(missing),
            "missing_examples": missing[:100],
            "hash_verification_enabled": args.verify_image_hashes,
            "hash_mismatch_count": len(mismatches),
            "hash_mismatch_examples": mismatches[:100],
            "complete": not missing and (not args.verify_image_hashes or not mismatches),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    checkpoint_ok = result["checkpoint"] is None or result["checkpoint"]["sha256_matches_baseline"]
    dataset_ok = result["dataset"] is None or result["dataset"]["complete"]
    return 0 if checkpoint_ok and dataset_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
