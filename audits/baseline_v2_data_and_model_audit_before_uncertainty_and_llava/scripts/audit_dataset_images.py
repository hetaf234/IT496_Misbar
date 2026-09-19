#!/usr/bin/env python3
"""Decode and validate every dataset image against the committed manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=AUDIT_DIR / "audit_results/dataset_image_audit.json",
    )
    args = parser.parse_args()
    root = args.dataset.resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory not found: {root}")

    manifest_path = REPO_ROOT / "dataset_reports/final_dataset_manifest_v2.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    problems = {
        "missing": [], "decode_error": [], "hash_mismatch": [],
        "dimension_mismatch": [], "tiny_side_below_32": [], "low_resolution": [],
    }
    found = Counter()
    formats = Counter()
    modes = Counter()
    for row in rows:
        relative = Path(row["split"]) / row["class_name"] / Path(row["final_image_path"]).name
        path = root / relative
        reference = str(relative)
        if not path.is_file():
            problems["missing"].append(reference)
            continue
        found[row["split"]] += 1
        if args.verify_hashes and sha256(path) != row["sha256"]:
            problems["hash_mismatch"].append(reference)
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                formats[str(image.format)] += 1
                modes[str(image.mode)] += 1
        except (OSError, ValueError, UnidentifiedImageError) as error:
            problems["decode_error"].append({"file": reference, "error": str(error)})
            continue
        expected_size = (int(row["width"]), int(row["height"]))
        if (width, height) != expected_size:
            problems["dimension_mismatch"].append({
                "file": reference, "manifest": expected_size, "actual": [width, height]
            })
        if min(width, height) < 32:
            problems["tiny_side_below_32"].append(reference)
        if width < 336 or height < 336:
            problems["low_resolution"].append(reference)

    summary = {key: len(value) for key, value in problems.items()}
    result = {
        "status": "complete" if not any(
            summary[key] for key in ["missing", "decode_error", "hash_mismatch", "dimension_mismatch"]
        ) else "failed",
        "dataset_reference": root.name,
        "manifest_rows": len(rows),
        "found_by_split": dict(found),
        "hash_verification_enabled": args.verify_hashes,
        "formats": dict(formats),
        "color_modes": dict(modes),
        "problem_counts": summary,
        "problems": {key: value[:200] for key, value in problems.items()},
        "note": "Tiny and low-resolution flags require visual review; they do not automatically delete an image.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "problem_counts": summary}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
