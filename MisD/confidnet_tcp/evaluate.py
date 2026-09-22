#!/usr/bin/env python3
"""Final, explicit official-val evaluation for a locked TCP-ConfiDNet checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PlantSplit
from metrics import failure_metrics
from model import TCPConfidenceHead
from teacher import DEFAULT_CHECKPOINT, DEFAULT_CLASS_MAP, MISD_ROOT, build_models, file_sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-official-val", action="store_true",
                        help="Required acknowledgement: this reads the held-out final test split.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.allow_official_val:
        raise PermissionError("pass --allow-official-val only after model selection is locked")
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    payload = torch.load(args.detector_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("phase") not in {"head", "finetune"}:
        raise ValueError("unsupported detector checkpoint")
    if payload.get("teacher_sha256") != file_sha256(args.checkpoint):
        raise ValueError("detector checkpoint was trained for a different teacher checkpoint")
    teacher, detector_encoder = build_models(args.checkpoint, args.class_map, args.device)
    head = TCPConfidenceHead().to(args.device); head.load_state_dict(payload["head"]); head.eval()
    if payload["phase"] == "finetune":
        detector_encoder.load_state_dict(payload["encoder"]); detector_encoder.eval()
    dataset = PlantSplit(args.data_root, "official_val", args.class_map, allow_official=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0)
    rows, errors, scores = [], [], []
    with torch.inference_mode():
        for number, batch in enumerate(loader, start=1):
            images = batch["image"].to(args.device, non_blocking=True)
            labels = batch["target"].to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                teacher_output = teacher(images)
                feature = teacher_output["feature"] if payload["phase"] == "head" else detector_encoder(images)
                confidence = head(feature)
            error = teacher_output["prediction"].ne(labels)
            error_score = 1 - confidence
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append({
                    "sample_id": sample_id, "true_label": int(labels[index]),
                    "teacher_prediction": int(teacher_output["prediction"][index]), "error_label": int(error[index]),
                    "tcp": float(teacher_output["probabilities"][index, labels[index]]),
                    "msp": float(teacher_output["msp"][index]), "margin": float(teacher_output["margin"][index]),
                    "confidence": float(confidence[index]), "error_score": float(error_score[index]),
                })
            errors.append(error.cpu().numpy()); scores.append(error_score.cpu().numpy())
            if number % 50 == 0 or number == len(loader): print(f"official_val: {number}/{len(loader)} batches", flush=True)
    metrics = failure_metrics(np.concatenate(errors), np.concatenate(scores))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    record = {
        "detector_checkpoint": str(args.detector_checkpoint.resolve()), "detector_phase": payload["phase"],
        "teacher_checkpoint": str(args.checkpoint.resolve()), "teacher_sha256": file_sha256(args.checkpoint),
        "evaluation_split": "official_val", "official_val_used": True, **metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__": main()
