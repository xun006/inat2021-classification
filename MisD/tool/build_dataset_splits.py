#!/usr/bin/env python3
"""Build leakage-free ImageFolder splits for classifier and detector training.

For each source-train class, randomly allocate 10 images to classifier_val,
30 to detector_train, 5 to detector_calibration, and all remaining images to
classifier_train. The complete official source val split becomes official_val.

Images are hard-linked by default: the outputs behave like ordinary files but
do not duplicate image bytes and never modify the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALLOCATIONS = {
    "classifier_val": 10,
    "detector_train": 30,
    "detector_calibration": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/mnt/hdd8t/Mingle/xyyy/data/inat2021/plants"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def stable_class_seed(seed: int, class_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{class_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def materialize(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    source_train = source / "train"
    source_val = source / "val"
    output = args.output.resolve()
    staging = output.with_name(output.name + ".building")

    if not source_train.is_dir() or not source_val.is_dir():
        raise FileNotFoundError(f"Expected source train/ and val/ under {source}")
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Refusing to overwrite it."
        )
    if staging.exists():
        raise FileExistsError(
            f"Staging directory already exists: {staging}. Inspect or remove it first."
        )
    if args.link_mode == "hardlink" and source.stat().st_dev != output.parent.stat().st_dev:
        raise RuntimeError("Hard links require source and output to be on the same filesystem")

    train_classes = sorted(p for p in source_train.iterdir() if p.is_dir())
    val_classes = sorted(p for p in source_val.iterdir() if p.is_dir())
    train_names = [p.name for p in train_classes]
    val_names = [p.name for p in val_classes]
    if train_names != val_names:
        raise RuntimeError("Source train and val class directories do not match exactly")
    required_per_class = sum(ALLOCATIONS.values())

    staging.mkdir(parents=True)
    split_names = ["classifier_train", *ALLOCATIONS, "official_val"]
    for split in split_names:
        (staging / split).mkdir()

    class_to_idx = {name: index for index, name in enumerate(train_names)}
    idx_to_class = {str(index): name for name, index in class_to_idx.items()}
    (staging / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (staging / "idx_to_class.json").write_text(
        json.dumps(idx_to_class, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_counts: Counter[str] = Counter()
    per_class_rows: list[dict[str, int | str]] = []
    manifest_path = staging / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("split", "class_index", "class_name", "filename", "source_path"),
        )
        writer.writeheader()

        for class_number, train_class_dir in enumerate(train_classes, start=1):
            class_name = train_class_dir.name
            train_images = list_images(train_class_dir)
            if len(train_images) < required_per_class:
                raise RuntimeError(
                    f"{class_name} has {len(train_images)} train images; "
                    f"at least {required_per_class} are required"
                )
            shuffled = train_images.copy()
            random.Random(stable_class_seed(args.seed, class_name)).shuffle(shuffled)

            cursor = 0
            allocated: dict[str, list[Path]] = {}
            for split, count in ALLOCATIONS.items():
                allocated[split] = shuffled[cursor:cursor + count]
                cursor += count
            allocated["classifier_train"] = shuffled[cursor:]

            val_images = list_images(source_val / class_name)
            if not val_images:
                raise RuntimeError(f"Official val class is empty: {class_name}")
            allocated["official_val"] = val_images

            row: dict[str, int | str] = {
                "class_index": class_to_idx[class_name],
                "class_name": class_name,
                "source_train": len(train_images),
            }
            seen_train: set[Path] = set()
            for split in split_names:
                images = allocated[split]
                destination_class = staging / split / class_name
                destination_class.mkdir()
                for image in images:
                    if split != "official_val":
                        if image in seen_train:
                            raise AssertionError(f"Train split overlap: {image}")
                        seen_train.add(image)
                    destination = destination_class / image.name
                    materialize(image, destination, args.link_mode)
                    writer.writerow(
                        {
                            "split": split,
                            "class_index": class_to_idx[class_name],
                            "class_name": class_name,
                            "filename": image.name,
                            "source_path": str(image.relative_to(source)),
                        }
                    )
                split_counts[split] += len(images)
                row[split] = len(images)
            if len(seen_train) != len(train_images):
                raise AssertionError(f"Not all source-train images allocated for {class_name}")
            per_class_rows.append(row)

            if class_number % 100 == 0 or class_number == len(train_classes):
                print(f"processed classes: {class_number}/{len(train_classes)}", flush=True)

    with (staging / "per_class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_index", "class_name", "source_train", "classifier_train",
            "classifier_val", "detector_train", "detector_calibration", "official_val",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)

    expected = {
        "classifier_val": len(train_classes) * 10,
        "detector_train": len(train_classes) * 30,
        "detector_calibration": len(train_classes) * 5,
        "classifier_train": sum(int(r["source_train"]) for r in per_class_rows)
        - len(train_classes) * required_per_class,
        "official_val": sum(int(r["official_val"]) for r in per_class_rows),
    }
    if dict(split_counts) != {name: expected[name] for name in split_names}:
        raise AssertionError(f"Unexpected split totals: {dict(split_counts)} vs {expected}")

    summary = {
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "randomization": "independent deterministic shuffle per class using SHA-256-derived seed",
        "link_mode": args.link_mode,
        "classes": len(train_classes),
        "source_train_images": sum(int(r["source_train"]) for r in per_class_rows),
        "source_official_val_images": expected["official_val"],
        "allocation_per_class_from_source_train": {
            "classifier_val": 10,
            "detector_train": 30,
            "detector_calibration": 5,
            "classifier_train": "all remaining images",
        },
        "split_counts": expected,
        "overlap_policy": "the four source-train-derived splits are mutually exclusive",
        "official_val_policy": "complete original val; reserved as final misclassification test",
    }
    (staging / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# Misclassification dataset splits\n\n"
        "Generated by `MisD/build_dataset_splits.py`. Images are hard links to the "
        "source by default, so deleting this output does not delete source images.\n\n"
        "- `classifier_train`: remaining source-train images after held-out allocations\n"
        "- `classifier_val`: 10 source-train images per class\n"
        "- `detector_train`: 30 source-train images per class\n"
        "- `detector_calibration`: 5 source-train images per class\n"
        "- `official_val`: complete official source val, reserved as final detector test\n",
        encoding="utf-8",
    )

    # Compatibility layout for the existing PlantCLEF2022 loader, which
    # expects <data_path>/train and <data_path>/val.
    classifier_layout = staging / "classifier"
    classifier_layout.mkdir()
    (classifier_layout / "train").symlink_to(Path("../classifier_train"))
    (classifier_layout / "val").symlink_to(Path("../classifier_val"))

    staging.rename(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
