#!/usr/bin/env python3
"""Export lightweight teacher predictions for the new detector splits.

No global features or patch tokens are stored. Patch tokens will be obtained
online during detector training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from teacher import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CLASS_MAP,
    MISD_ROOT,
    build_frozen_teacher,
    load_class_map,
    validation_transform,
)


SPLITS = ("detector_train", "detector_calibration", "official_val")


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, self.samples[index][0], index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument(
        "--output-dir", type=Path,
        default=MISD_ROOT / "output" / "patch_cross_attention" / "teacher_predictions",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Debug only: process the first N samples per split")
    return parser.parse_args()


def export_split(args, teacher, split: str, class_map: dict[str, int]) -> None:
    split_root = (args.data_root / split).resolve()
    dataset = ImageFolderWithPath(split_root, transform=validation_transform())
    if dataset.class_to_idx != class_map:
        raise RuntimeError(f"{split}: ImageFolder class mapping differs from class_to_idx.json")
    full_size = len(dataset)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, full_size)))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    idx_to_class = {value: key for key, value in class_map.items()}
    output_path = args.output_dir / f"{split}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "split", "dataset_index", "image_path", "ground_truth_index",
        "ground_truth", "candidate_index", "candidate_class", "top2_index",
        "top2_class", "is_correct", "error_label", "teacher_top1_prob", "margin",
        "entropy", "max_logit", "energy", "teacher_topk_indices",
        "teacher_topk_classes", "teacher_topk_probs",
    ]
    total = correct = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for images, targets, paths, indices in loader:
            images = images.to(args.device, non_blocking=True)
            outputs = teacher(images)
            logits = outputs["logits"].float()
            probabilities = torch.softmax(logits, dim=1)
            top_probs, top_indices = probabilities.topk(args.topk, dim=1)
            predictions = top_indices[:, 0].cpu()
            targets = targets.cpu()
            matches = predictions.eq(targets)
            margin = (top_probs[:, 0] - top_probs[:, 1]).cpu()
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1).cpu()
            max_logit = logits.max(1).values.cpu()
            energy = -torch.logsumexp(logits, dim=1).cpu()
            top_indices = top_indices.cpu()
            top_probs = top_probs.cpu()
            for i, path_text in enumerate(paths):
                relative = Path(path_text).relative_to(split_root).as_posix()
                gt = int(targets[i])
                pred = int(predictions[i])
                second = int(top_indices[i, 1])
                ids = [int(x) for x in top_indices[i].tolist()]
                probs = [round(float(x), 8) for x in top_probs[i].tolist()]
                writer.writerow({
                    "sample_id": f"{split}/{relative}", "split": split,
                    "dataset_index": int(indices[i]), "image_path": relative,
                    "ground_truth_index": gt, "ground_truth": idx_to_class[gt],
                    "candidate_index": pred, "candidate_class": idx_to_class[pred],
                    "top2_index": second, "top2_class": idx_to_class[second],
                    "is_correct": int(matches[i]), "error_label": int(not bool(matches[i])),
                    "teacher_top1_prob": round(float(top_probs[i, 0]), 8),
                    "margin": round(float(margin[i]), 8),
                    "entropy": round(float(entropy[i]), 8),
                    "max_logit": round(float(max_logit[i]), 8),
                    "energy": round(float(energy[i]), 8),
                    "teacher_topk_indices": json.dumps(ids),
                    "teacher_topk_classes": json.dumps([idx_to_class[x] for x in ids], ensure_ascii=False),
                    "teacher_topk_probs": json.dumps(probs),
                })
            total += len(targets)
            correct += int(matches.sum())
            if total % (args.batch_size * 20) < len(targets):
                print(f"{split}: {total}/{len(dataset)} top1={correct/total:.4%}", flush=True)
    manifest = {
        "split": split, "split_root": str(split_root), "full_split_samples": full_size,
        "exported_samples": total, "debug_subset": args.max_samples is not None,
        "top1_accuracy": correct / total, "errors": total - correct,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "class_map": str(args.class_map.resolve()), "class_map_sha256": sha256(args.class_map),
        "predictions_csv": str(output_path.resolve()), "predictions_sha256": sha256(output_path),
        "transform": "Resize(256,bicubic)->CenterCrop(224)->ImageNet normalize",
        "patch_storage": "none; online extraction only",
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"finished {split}: samples={total}, errors={total-correct}, csv={output_path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.topk < 2:
        raise ValueError("invalid loader/top-k arguments")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    class_map = load_class_map(args.class_map)
    teacher = build_frozen_teacher(args.checkpoint, args.class_map, args.device)
    for split in args.splits:
        export_split(args, teacher, split, class_map)


if __name__ == "__main__":
    main()
