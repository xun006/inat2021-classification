#!/usr/bin/env python3
"""One-time online M2 inference on official_val; no model selection is done."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import DetectorDataset
from models import build_detector
from teacher import DEFAULT_CHECKPOINT, DEFAULT_CLASS_MAP, MISD_ROOT, build_frozen_teacher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    metadata_path = MISD_ROOT / "output/patch_cross_attention/teacher_predictions/official_val.csv"
    checkpoint_path = MISD_ROOT / "output/patch_cross_attention/ablation/m2_seed0/checkpoint_best.pth"
    output_dir = MISD_ROOT / "output/patch_cross_attention/final_fusion/margin_m2_seed0"
    output_path = output_dir / "official_val_m2_predictions.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite final-test predictions: {output_path}")
    dataset = DetectorDataset(
        MISD_ROOT / "data", "official_val", metadata_path, DEFAULT_CLASS_MAP
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    teacher = build_frozen_teacher(DEFAULT_CHECKPOINT, DEFAULT_CLASS_MAP, args.device)
    detector = build_detector("m2", teacher.model.head[1].weight).to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    detector.load_state_dict(checkpoint["model"], strict=True)
    detector.eval()
    amp = str(args.device).startswith("cuda")
    sample_ids, labels, scores = [], [], []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            images = batch["image"].to(args.device, non_blocking=True)
            candidates = batch["candidate_index"].to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                teacher_output = teacher(images)
                logits = detector(
                    teacher_output["p_norm"], candidates,
                    global_feature=teacher_output["global_feature"],
                )["error_logit"]
            sample_ids.extend(batch["sample_id"])
            labels.extend(batch["error_label"].numpy().astype(int).tolist())
            scores.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
            if (step + 1) % 20 == 0:
                print(f"official_val M2: {len(sample_ids)}/{len(dataset)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": sample_ids, "error_label": labels,
                  "m2_error_score": scores}).to_csv(output_path, index=False)
    report = {
        "split": "official_val", "samples": len(labels), "errors": int(sum(labels)),
        "model": "m2_seed0", "checkpoint": str(checkpoint_path.resolve()),
        "model_selection_performed": False, "output": str(output_path.resolve()),
    }
    (output_dir / "official_val_m2_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
