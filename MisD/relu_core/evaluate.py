#!/usr/bin/env python3
"""One-time official_val evaluation of a calibration-locked REL-U-Core model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from math_core import failure_metrics, relative_uncertainty
from protocol import (
    DEFAULT_CLASS_MAP,
    EXPECTED_SPLIT_COUNTS,
    MISD_ROOT,
    build_classifier_c_teacher,
    ensure_new_directory,
    file_sha256,
    load_class_map,
    make_plant_split,
    provenance,
    validate_locked_distance,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-config", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-official-val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.allow_official_val:
        raise PermissionError("pass --allow-official-val only after selected_model.pt is locked")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = torch.load(args.selected_model, map_location="cpu", weights_only=False)
    if payload.get("method") != "REL-U-Core" or payload.get("official_val_used") is not False:
        raise ValueError("selected model is not a development-locked REL-U-Core artifact")
    expected_provenance = provenance(args.checkpoint, args.class_map, args.classifier_config)
    if payload.get("teacher_provenance") != expected_provenance:
        raise ValueError("selected REL-U model belongs to another teacher/config/class map")
    classes = len(load_class_map(args.class_map))
    distance_cpu = payload["distance"].float()
    validate_locked_distance(distance_cpu, classes)
    temperature = float(payload["temperature"])
    if temperature <= 0:
        raise ValueError("locked temperature must be positive")
    dataset = make_plant_split(
        args.data_root, "official_val", args.class_map, allow_official=True
    )
    if len(dataset) != EXPECTED_SPLIT_COUNTS["official_val"]:
        raise RuntimeError(
            f"official_val: expected {EXPECTED_SPLIT_COUNTS['official_val']} samples, found {len(dataset)}"
        )
    ensure_new_directory(args.output_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0,
    )
    teacher = build_classifier_c_teacher(
        args.checkpoint, args.class_map, args.classifier_config, args.device
    )
    distance = distance_cpu.to(args.device)
    errors: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    predictions_path = args.output_dir / "predictions.csv"
    fields = [
        "dataset_index", "sample_key", "true_label", "teacher_prediction",
        "error_label", "msp", "relu_error_score",
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        cursor = 0
        with torch.inference_mode():
            for number, batch in enumerate(loader, start=1):
                images = batch["image"].to(args.device, non_blocking=True)
                labels = batch["target"].to(args.device, non_blocking=True)
                output = teacher(images)
                original_logits = output["logits"].float()
                probabilities = torch.softmax(original_logits / temperature, dim=1)
                error = output["prediction"].ne(labels)
                score = relative_uncertainty(probabilities, distance)
                for offset, sample_key in enumerate(batch["sample_key"]):
                    writer.writerow({
                        "dataset_index": cursor + offset,
                        "sample_key": sample_key,
                        "true_label": int(labels[offset]),
                        "teacher_prediction": int(output["prediction"][offset]),
                        "error_label": int(error[offset]),
                        "msp": float(output["msp"][offset]),
                        "relu_error_score": float(score[offset]),
                    })
                cursor += len(labels)
                errors.append(error.cpu().numpy())
                scores.append(score.cpu().numpy())
                if number % 50 == 0 or number == len(loader):
                    print(f"official_val: {cursor}/{len(dataset)}", flush=True)
    if cursor != len(dataset):
        raise RuntimeError("official_val evaluation ended with an incomplete sample count")
    metrics = failure_metrics(np.concatenate(errors), np.concatenate(scores))
    record = {
        "method": "REL-U-Core",
        "selected_model": str(args.selected_model.resolve()),
        "selected_model_sha256": file_sha256(args.selected_model),
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "teacher_provenance": expected_provenance,
        "temperature": temperature,
        "lambda": float(payload["lambda"]),
        "epsilon": 0.0,
        "selection_split": "detector_calibration",
        "selection_metric": payload["selection_metric"],
        "evaluation_split": "official_val",
        "score_direction": "larger_is_more_likely_error",
        "official_val_used": True,
        **metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
