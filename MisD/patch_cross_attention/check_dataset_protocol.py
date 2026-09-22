#!/usr/bin/env python3
"""Validate the physical new detector split protocol without model inference."""

from __future__ import annotations

import json
from pathlib import Path

from teacher import DEFAULT_CLASS_MAP, MISD_ROOT, load_class_map


SPLITS = ("detector_train", "detector_calibration", "official_val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    expected = {"detector_train": 128130, "detector_calibration": 21355, "official_val": 42710}
    class_map = load_class_map(DEFAULT_CLASS_MAP)
    identities = {}
    counts = {}
    for split in SPLITS:
        root = MISD_ROOT / "data" / split
        classes = sorted(path.name for path in root.iterdir() if path.is_dir())
        if {name: i for i, name in enumerate(classes)} != class_map:
            raise RuntimeError(f"{split}: class mapping mismatch")
        relative = {
            path.relative_to(root).as_posix()
            for path in root.glob("*/*") if path.suffix.lower() in IMAGE_SUFFIXES
        }
        if len(relative) != expected[split]:
            raise RuntimeError(f"{split}: expected {expected[split]}, found {len(relative)}")
        identities[split] = relative
        counts[split] = {"samples": len(relative), "classes": len(classes)}
    overlaps = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            count = len(identities[left] & identities[right])
            overlaps[f"{left}__{right}"] = count
            if count:
                raise RuntimeError(f"{left}/{right}: {count} duplicate relative paths")
    report = {"passed": True, "counts": counts, "relative_path_overlaps": overlaps}
    output = MISD_ROOT / "output/patch_cross_attention/protocol/dataset_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
