#!/usr/bin/env python3
"""Gate-0 validation for the new online Patch teacher and prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from export_teacher_predictions import ImageFolderWithPath, SPLITS
from teacher import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CLASS_MAP,
    MISD_ROOT,
    build_frozen_teacher,
    load_class_map,
    validation_transform,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument(
        "--predictions-dir", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/teacher_predictions",
    )
    parser.add_argument(
        "--output", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/protocol/online_teacher_validation.json",
    )
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def validate_split(args, split, teacher, class_map):
    csv_path = args.predictions_dir / f"{split}.csv"
    frame = pd.read_csv(csv_path)
    if len(frame) < args.samples_per_split:
        raise ValueError(f"{csv_path} has only {len(frame)} rows")
    sample = frame.sample(n=args.samples_per_split, random_state=args.seed).sort_values("dataset_index")
    root = (args.data_root / split).resolve()
    dataset = ImageFolderWithPath(root, transform=validation_transform())
    if dataset.class_to_idx != class_map:
        raise RuntimeError(f"{split}: class mapping mismatch")
    indices = sample.dataset_index.astype(int).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    sample_by_index = sample.set_index("dataset_index")
    checked = 0
    maxima = {
        "global_feature": 0.0,
        "logits": 0.0,
        "probability_margin": 0.0,
        "classifier_weight_logit_margin": 0.0,
    }
    all_top1_match = True
    all_shapes_match = True
    all_paths_match = True
    for images, targets, paths, dataset_indices in loader:
        images = images.to(args.device)
        out = teacher(images)
        direct_h = teacher.model.forward_features(images)
        direct_logits = teacher.model(images)
        rebuilt_logits = teacher.model.head[1](out["head_feature"])
        maxima["global_feature"] = max(maxima["global_feature"],
            float((out["global_feature"] - direct_h).abs().max().cpu()))
        maxima["logits"] = max(maxima["logits"],
            float((out["logits"] - direct_logits).abs().max().cpu()),
            float((out["logits"] - rebuilt_logits).abs().max().cpu()))
        probabilities = torch.softmax(out["logits"].float(), 1)
        top_probs, top_indices = probabilities.topk(2, 1)
        top_logits = out["logits"].gather(1, top_indices)
        linear = teacher.model.head[1]
        top1_weight = linear.weight[top_indices[:, 0]]
        top2_weight = linear.weight[top_indices[:, 1]]
        top1_bias = linear.bias[top_indices[:, 0]]
        top2_bias = linear.bias[top_indices[:, 1]]
        weight_margin = (
            out["head_feature"] * (top1_weight - top2_weight)
        ).sum(1) + top1_bias - top2_bias
        maxima["classifier_weight_logit_margin"] = max(
            maxima["classifier_weight_logit_margin"],
            float((weight_margin - (top_logits[:, 0] - top_logits[:, 1])).abs().max().cpu()),
        )
        for i, index_tensor in enumerate(dataset_indices):
            index = int(index_tensor)
            expected = sample_by_index.loc[index]
            relative = Path(paths[i]).relative_to(root).as_posix()
            all_paths_match &= relative == expected.image_path
            all_top1_match &= int(top_indices[i, 0]) == int(expected.candidate_index)
            all_top1_match &= int(targets[i]) == int(expected.ground_truth_index)
            online_margin = float((top_probs[i, 0] - top_probs[i, 1]).cpu())
            maxima["probability_margin"] = max(
                maxima["probability_margin"], abs(online_margin - float(expected.margin))
            )
        all_shapes_match &= tuple(out["p_raw"].shape[1:]) == (196, 1024)
        all_shapes_match &= tuple(out["p_norm"].shape[1:]) == (196, 1024)
        checked += len(targets)
    has_teacher_grad = any(parameter.grad is not None for parameter in teacher.parameters())
    passed = (
        checked == args.samples_per_split and all_top1_match and all_shapes_match
        and all_paths_match and not has_teacher_grad
        and maxima["global_feature"] <= args.atol and maxima["logits"] <= args.atol
        and maxima["probability_margin"] <= max(args.atol, 1e-7)
        and maxima["classifier_weight_logit_margin"] <= args.atol
    )
    return {
        "split": split, "checked": checked, "passed": bool(passed),
        "top1_and_target_match": bool(all_top1_match), "paths_match": bool(all_paths_match),
        "patch_shapes_match": bool(all_shapes_match), "teacher_has_gradient": has_teacher_grad,
        "max_absolute_errors": maxima,
    }


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    class_map = load_class_map(args.class_map)
    teacher = build_frozen_teacher(args.checkpoint, args.class_map, args.device)
    results = [validate_split(args, split, teacher, class_map) for split in SPLITS]
    report = {
        "gate": "Gate 0 online teacher consistency",
        "passed": all(row["passed"] for row in results),
        "checkpoint": str(args.checkpoint.resolve()), "samples_per_split": args.samples_per_split,
        "seed": args.seed, "atol": args.atol, "splits": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Gate 0 failed; do not start detector training")


if __name__ == "__main__":
    main()
