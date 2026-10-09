"""Recompute A/B/C/D metrics from exported logits using identical confidence scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from Classifier.metrics import evaluate_arrays
from Classifier.run import write_json


SCORES = ("max_softmax", "max_sigmoid", "sigmoid_logit_margin")
CLASSIFICATION_METRICS = ("top1", "top5", "macro_f1", "head_accuracy",
                          "medium_accuracy", "tail_accuracy")
FAILURE_METRICS = ("auroc_error", "aupr_error", "fpr_at_95_tpr", "aurc")


def read_samples(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "ground_truth"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid samples.csv: {path}")
    return [row["sample_id"] for row in rows], np.asarray(
        [int(row["ground_truth"]) for row in rows], dtype=np.int64
    )


def softmax_max(logits: np.ndarray):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return 1.0 / (exponent.sum(axis=1) / exponent.max(axis=1))


def stable_sigmoid(values: np.ndarray):
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def evaluate_export(export: Path, expected_ids=None, chunk_size=256):
    marker = json.loads((export / "complete.json").read_text(encoding="utf-8"))
    logits = np.load(export / "logits.npy", mmap_mode="r")
    sample_ids, labels = read_samples(export / "samples.csv")
    if logits.shape != (len(labels), marker["columns"]):
        raise ValueError(f"Shape/row mismatch in {export}")
    if expected_ids is not None and sample_ids != expected_ids:
        raise ValueError(f"Sample order/content differs: {export}")
    audit_path = export.parent / "data_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"Expected run data audit at {audit_path}")
    counts = json.loads(audit_path.read_text(encoding="utf-8"))["train_counts"]
    prediction = np.empty(len(labels), dtype=np.int64)
    top5 = np.empty(len(labels), dtype=bool)
    confidence = {name: np.empty(len(labels), dtype=np.float64) for name in SCORES}
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        values = np.asarray(logits[start:end], dtype=np.float64)
        topk = min(5, values.shape[1])
        order = np.argpartition(values, -topk, axis=1)[:, -topk:]
        top_two = np.partition(values, -2, axis=1)[:, -2:]
        prediction[start:end] = values.argmax(1)
        top5[start:end] = (order == labels[start:end, None]).any(1)
        confidence["max_softmax"][start:end] = softmax_max(values)
        maximum = values.max(1)
        confidence["max_sigmoid"][start:end] = stable_sigmoid(maximum)
        margin = top_two[:, 1] - top_two[:, 0]
        confidence["sigmoid_logit_margin"][start:end] = stable_sigmoid(margin)
    metrics = {}
    for score, values in confidence.items():
        metrics[score] = evaluate_arrays(
            labels, prediction, values, top5, counts, tau=0.5
        )[0]
    return sample_ids, metrics


def difference(left, right, keys):
    result = {}
    for key in keys:
        a, b = left.get(key), right.get(key)
        result[key] = None if a is None or b is None else a - b
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for group in "ABCD":
        parser.add_argument(f"--{group.lower()}", type=Path, required=True,
                            help=f"Group {group} export directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    exports = {group: getattr(args, group.lower()) for group in "ABCD"}
    all_metrics, reference_ids = {}, None
    for group, export in exports.items():
        ids, metrics = evaluate_export(export, reference_ids, args.chunk_size)
        reference_ids = ids if reference_ids is None else reference_ids
        all_metrics[group] = metrics

    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for group in "ABCD":
        for score in SCORES:
            metric = all_metrics[group][score]
            rows.append({"group": group, "score": score,
                         **{key: metric.get(key) for key in (*CLASSIFICATION_METRICS, *FAILURE_METRICS)}})
    with (args.output / "metrics_by_score.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Classification is score-independent; use one score as the common carrier.
    cls = {group: all_metrics[group]["max_softmax"] for group in "ABCD"}
    contrasts = {
        "D_minus_B_lora_effect_under_l2": difference(cls["D"], cls["B"], CLASSIFICATION_METRICS),
        "C_minus_A_lora_effect_under_ce": difference(cls["C"], cls["A"], CLASSIFICATION_METRICS),
        "B_minus_A_l2_effect_head_only": difference(cls["B"], cls["A"], CLASSIFICATION_METRICS),
        "D_minus_C_l2_effect_lora_head": difference(cls["D"], cls["C"], CLASSIFICATION_METRICS),
    }
    interaction = {}
    for key in CLASSIFICATION_METRICS:
        values = [cls[group].get(key) for group in "ABCD"]
        interaction[key] = None if any(value is None for value in values) else (
            cls["D"][key] - cls["B"][key] - cls["C"][key] + cls["A"][key]
        )
    contrasts["interaction_D_minus_B_minus_C_minus_A"] = interaction
    write_json(args.output / "contrasts.json", contrasts)
    write_json(args.output / "complete.json", {
        "groups": {group: str(path.resolve()) for group, path in exports.items()},
        "samples": len(reference_ids), "scores": list(SCORES),
        "prediction_rule": "argmax(logits)",
        "note": "single seed=42; contrasts are descriptive, not significance tests",
    })
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()

