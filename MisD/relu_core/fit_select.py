#!/usr/bin/env python3
"""Fit exact REL-U moments and lock lambda/temperature on detector_calibration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from math_core import build_distance_matrix, failure_metrics, relative_uncertainty
from protocol import (
    DEFAULT_CLASS_MAP,
    MISD_ROOT,
    ensure_new_directory,
    load_cache,
    load_class_map,
    numpy_batches,
    provenance,
)


DEFAULT_LAMBDAS = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
DEFAULT_TEMPERATURES = "0.5,0.75,1.0,1.25,1.5,2.0"


def parse_grid(text: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or not all(np.isfinite(values)) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique finite comma-separated values")
    return values


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-config", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--temperatures", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def estimate_moments(
    logits: np.ndarray,
    errors: np.ndarray,
    temperature: float,
    classes: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    correct_sum = torch.zeros((classes, classes), dtype=torch.float32, device=device)
    error_sum = torch.zeros_like(correct_sum)
    correct_count = error_count = 0
    with torch.inference_mode():
        for _, _, values in numpy_batches(logits, batch_size):
            start = correct_count + error_count
            batch_error = torch.from_numpy(np.asarray(errors[start:start + len(values)], dtype=np.uint8)).to(device).bool()
            batch_logits = torch.from_numpy(values.copy()).to(device=device, dtype=torch.float32)
            probabilities = torch.softmax(batch_logits / temperature, dim=1)
            if (~batch_error).any():
                selected = probabilities[~batch_error]
                correct_sum.addmm_(selected.T, selected)
                correct_count += len(selected)
            if batch_error.any():
                selected = probabilities[batch_error]
                error_sum.addmm_(selected.T, selected)
                error_count += len(selected)
    if correct_count == 0 or error_count == 0:
        raise RuntimeError("REL-U fitting requires both correct and erroneous training samples")
    return (
        (correct_sum / correct_count).cpu(),
        (error_sum / error_count).cpu(),
        correct_count,
        error_count,
    )


def score_cache(
    logits: np.ndarray,
    temperature: float,
    distance_cpu: torch.Tensor,
    batch_size: int,
    device: str,
) -> np.ndarray:
    distance = distance_cpu.to(device=device, dtype=torch.float32)
    scores = np.empty(len(logits), dtype=np.float32)
    with torch.inference_mode():
        for start, stop, values in numpy_batches(logits, batch_size):
            batch_logits = torch.from_numpy(values.copy()).to(device=device, dtype=torch.float32)
            probabilities = torch.softmax(batch_logits / temperature, dim=1)
            scores[start:stop] = relative_uncertainty(probabilities, distance).cpu().numpy()
    return scores


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    lambdas = parse_grid(args.lambdas, "lambdas")
    temperatures = parse_grid(args.temperatures, "temperatures")
    if any(value < 0 or value > 1 for value in lambdas):
        raise ValueError("all lambdas must be in [0,1]")
    if any(value <= 0 for value in temperatures):
        raise ValueError("all temperatures must be positive")
    expected_provenance = provenance(args.checkpoint, args.class_map, args.classifier_config)
    train_manifest, train_logits, _, _, train_errors = load_cache(
        args.train_cache, "detector_train", expected_provenance
    )
    calibration_manifest, calibration_logits, calibration_targets, calibration_predictions, calibration_errors = load_cache(
        args.calibration_cache, "detector_calibration", expected_provenance
    )
    if train_manifest["classes"] != calibration_manifest["classes"]:
        raise ValueError("development caches have different class counts")
    classes = len(load_class_map(args.class_map))
    if classes != train_manifest["classes"]:
        raise ValueError("cache/class-map class count mismatch")
    ensure_new_directory(args.output_dir)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference_lambda = 1.0 - float(np.asarray(train_errors).mean())
    results: list[dict] = []
    best_key = None
    best_payload = None
    for temperature in temperatures:
        print(f"estimating exact moments for T={temperature}", flush=True)
        correct_moment, error_moment, correct_count, error_count = estimate_moments(
            train_logits, train_errors, temperature, classes, args.batch_size, args.device
        )
        for lbd in lambdas:
            distance, diagnostics = build_distance_matrix(correct_moment, error_moment, lbd)
            scores = score_cache(
                calibration_logits, temperature, distance, args.batch_size, args.device
            )
            metrics = failure_metrics(calibration_errors, scores)
            row = {
                "temperature": temperature,
                "lambda": lbd,
                "fallback_to_gini": diagnostics.fallback_to_gini,
                "distance_nonzero_entries": diagnostics.nonzero_entries,
                **metrics,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            # Locked selection: Error-AUPRC, AUROC, T closest to 1, lambda closest
            # to the paper's no-search class-balance recommendation.
            key = (
                float(metrics["error_auprc"]),
                float(metrics["auroc"]),
                -abs(temperature - 1.0),
                -abs(lbd - reference_lambda),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_payload = {
                    "distance": distance.clone(),
                    "scores": scores.copy(),
                    "row": row.copy(),
                    "train_correct": correct_count,
                    "train_errors": error_count,
                }
    assert best_payload is not None
    selected = best_payload["row"]
    checkpoint_payload = {
        "schema_version": 1,
        "method": "REL-U-Core",
        "distance": best_payload["distance"],
        "temperature": selected["temperature"],
        "lambda": selected["lambda"],
        "selection_metric": "detector_calibration.error_auprc",
        "selection_metrics": {key: selected[key] for key in (
            "auroc", "error_auprc", "fpr_at_95_error_tpr", "aurc", "eaurc",
            "error_rate", "samples", "errors",
        )},
        "fallback_to_gini": selected["fallback_to_gini"],
        "teacher_provenance": expected_provenance,
        "train_cache_manifest": train_manifest,
        "calibration_cache_manifest": calibration_manifest,
        "candidate_grid": {"lambdas": lambdas, "temperatures": temperatures, "epsilon": 0.0},
        "score_direction": "larger_is_more_likely_error",
        "official_val_used": False,
    }
    torch.save(checkpoint_payload, args.output_dir / "selected_model.pt")
    with (args.output_dir / "search_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    selected_record = {
        key: value for key, value in checkpoint_payload.items() if key != "distance"
    }
    (args.output_dir / "selected_config.json").write_text(
        json.dumps(selected_record, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "calibration_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset_index", "true_label", "teacher_prediction", "error_label", "relu_error_score"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, score in enumerate(best_payload["scores"]):
            writer.writerow({
                "dataset_index": index,
                "true_label": int(calibration_targets[index]),
                "teacher_prediction": int(calibration_predictions[index]),
                "error_label": int(calibration_errors[index]),
                "relu_error_score": float(score),
            })
    print(json.dumps(selected_record, indent=2))


if __name__ == "__main__":
    main()
