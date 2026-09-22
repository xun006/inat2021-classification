#!/usr/bin/env python3
"""Cache frozen-teacher TCP targets for a development split.

This program intentionally refuses ``official_val``.  Caches contain no full
4,271-class probability vector: only the quantities needed for supervision and
baseline comparisons are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import PlantSplit
from teacher import DEFAULT_CHECKPOINT, DEFAULT_CLASS_MAP, MISD_ROOT, build_models, file_sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument("--split", choices=("detector_train", "detector_calibration"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-dir", type=Path, default=MISD_ROOT / "output/confidnet_tcp/cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = PlantSplit(args.data_root, args.split, args.class_map)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0)
    teacher, _ = build_models(args.checkpoint, args.class_map, args.device)
    count = len(dataset)
    feature = torch.empty((count, 1024), dtype=torch.float16)
    tcp = torch.empty(count, dtype=torch.float32)
    error = torch.empty(count, dtype=torch.uint8)
    target = torch.empty(count, dtype=torch.int32)
    prediction = torch.empty(count, dtype=torch.int32)
    msp = torch.empty(count, dtype=torch.float32)
    margin = torch.empty(count, dtype=torch.float32)
    sample_id: list[str] = [""] * count
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            images = batch["image"].to(args.device, non_blocking=True)
            labels = batch["target"].to(args.device, non_blocking=True)
            index = batch["index"].long()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                output = teacher(images)
            probabilities = output["probabilities"]
            index_cpu = index.cpu()
            feature[index_cpu] = output["feature"].cpu().to(torch.float16)
            tcp[index_cpu] = probabilities.gather(1, labels[:, None]).squeeze(1).float().cpu()
            prediction[index_cpu] = output["prediction"].cpu().to(torch.int32)
            target[index_cpu] = labels.cpu().to(torch.int32)
            error[index_cpu] = output["prediction"].ne(labels).cpu().to(torch.uint8)
            msp[index_cpu] = output["msp"].float().cpu()
            margin[index_cpu] = output["margin"].float().cpu()
            for row_index, value in zip(index.tolist(), batch["sample_id"]):
                sample_id[row_index] = value
            if batch_number % 50 == 0 or batch_number == len(loader):
                print(f"{args.split}: {batch_number}/{len(loader)} batches", flush=True)
    if any(not value for value in sample_id):
        raise RuntimeError("some samples were not exported")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / f"{args.split}.pt"
    torch.save({
        "split": args.split, "sample_id": sample_id, "feature": feature, "tcp": tcp,
        "error": error, "target": target, "prediction": prediction, "msp": msp, "margin": margin,
    }, cache_path)
    metadata = {
        "split": args.split, "samples": count, "errors": int(error.sum()),
        "error_rate": float(error.float().mean()), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint), "class_map": str(args.class_map.resolve()),
        "official_val_used": False, "preprocessing": "Resize(256), CenterCrop(224), ImageNet normalization",
    }
    (args.output_dir / f"{args.split}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
