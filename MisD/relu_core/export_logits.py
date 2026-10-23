#!/usr/bin/env python3
"""Export immutable full-logit caches for REL-U development splits only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protocol import (
    DEFAULT_CLASS_MAP,
    EXPECTED_SPLIT_COUNTS,
    MISD_ROOT,
    SCHEMA_VERSION,
    build_classifier_c_teacher,
    cache_paths,
    ensure_new_directory,
    load_class_map,
    make_plant_split,
    provenance,
    sha256_text_lines,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("detector_train", "detector_calibration"), required=True)
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-config", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ensure_new_directory(args.output_dir)
    paths = cache_paths(args.output_dir)
    class_map = load_class_map(args.class_map)
    dataset = make_plant_split(args.data_root, args.split, args.class_map)
    expected_count = EXPECTED_SPLIT_COUNTS[args.split]
    if len(dataset) != expected_count:
        raise RuntimeError(f"{args.split}: expected {expected_count} samples, found {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0,
    )
    teacher = build_classifier_c_teacher(
        args.checkpoint, args.class_map, args.classifier_config, args.device
    )
    logits_out = np.lib.format.open_memmap(
        paths["logits"], mode="w+", dtype=np.float32, shape=(len(dataset), len(class_map))
    )
    targets_out = np.lib.format.open_memmap(
        paths["targets"], mode="w+", dtype=np.int32, shape=(len(dataset),)
    )
    predictions_out = np.lib.format.open_memmap(
        paths["predictions"], mode="w+", dtype=np.int32, shape=(len(dataset),)
    )
    errors_out = np.lib.format.open_memmap(
        paths["errors"], mode="w+", dtype=np.uint8, shape=(len(dataset),)
    )
    sample_keys: list[str] = []
    cursor = 0
    with torch.inference_mode():
        for number, batch in enumerate(loader, start=1):
            images = batch["image"].to(args.device, non_blocking=True)
            labels = batch["target"].numpy().astype(np.int32, copy=False)
            # Do not autocast: the cached logits are the canonical float32 detector input.
            output = teacher(images)
            logits = output["logits"].float().cpu().numpy()
            predictions = output["prediction"].cpu().numpy().astype(np.int32, copy=False)
            stop = cursor + len(labels)
            logits_out[cursor:stop] = logits
            targets_out[cursor:stop] = labels
            predictions_out[cursor:stop] = predictions
            errors_out[cursor:stop] = predictions != labels
            sample_keys.extend(batch["sample_key"])
            cursor = stop
            if number % 50 == 0 or number == len(loader):
                print(f"{args.split}: {cursor}/{len(dataset)}", flush=True)
    for values in (logits_out, targets_out, predictions_out, errors_out):
        values.flush()
    if cursor != len(dataset) or sample_keys != dataset.sample_keys():
        raise RuntimeError("export order does not match the canonical ImageFolder order")
    paths["sample_keys"].write_text("\n".join(sample_keys) + "\n", encoding="utf-8")
    errors = int(np.asarray(errors_out).sum())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": args.split,
        "samples": len(dataset),
        "classes": len(class_map),
        "errors": errors,
        "error_rate": errors / len(dataset),
        "logits_dtype": "float32",
        "preprocessing": "Resize(256,bicubic), CenterCrop(224), ToTensor, ImageNet normalization",
        "ordering": "torchvision.datasets.ImageFolder canonical order; shuffle=False",
        "sample_keys_sha256": sha256_text_lines(sample_keys),
        "teacher_provenance": provenance(args.checkpoint, args.class_map, args.classifier_config),
        "official_val_used": False,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
