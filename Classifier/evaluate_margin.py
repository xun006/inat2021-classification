"""Calibrate and evaluate sigmoid-margin misclassification detection.

The margin confidence is the difference between the largest and second-largest
sigmoid probabilities.  It is computed from the two logits already stored in
samples.csv, so this command does not load logits.npy or sigmoid.npy.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate_failure import (
    CALIBRATION_SPLIT,
    TEST_SPLIT,
    operating_point,
    public_metrics,
    select_threshold,
    validate_same_model,
    write_json,
    write_summary,
)


def sigmoid(values):
    """Numerically stable elementwise sigmoid for float64 NumPy arrays."""
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def load_margin_export(export_dir, expected_split):
    export_dir = Path(export_dir)
    required = ["samples.csv", "complete.json", "provenance.json"]
    missing = [name for name in required if not (export_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete export {}: missing {}".format(export_dir, missing))

    complete = json.loads((export_dir / "complete.json").read_text(encoding="utf-8"))
    provenance = json.loads((export_dir / "provenance.json").read_text(encoding="utf-8"))
    split_name = Path(provenance.get("split_dir", "")).name
    if split_name != expected_split:
        raise ValueError(
            "Expected {} export, got split_dir={!r}".format(
                expected_split, provenance.get("split_dir")
            )
        )

    errors, top1_logits, top2_logits, sample_ids = [], [], [], []
    with (export_dir / "samples.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        needed = {"sample_id", "correct", "top1_logit", "top2_logit"}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError("samples.csv is missing required columns: {}".format(sorted(needed)))
        for line_number, row in enumerate(reader, start=2):
            try:
                correct = int(row["correct"])
                top1 = float(row["top1_logit"])
                top2 = float(row["top2_logit"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid samples.csv row {}".format(line_number)) from exc
            if correct not in (0, 1):
                raise ValueError("correct must be 0 or 1 at samples.csv row {}".format(line_number))
            errors.append(1 - correct)
            top1_logits.append(top1)
            top2_logits.append(top2)
            sample_ids.append(row["sample_id"])

    errors = np.asarray(errors, dtype=bool)
    top1_logits = np.asarray(top1_logits, dtype=np.float64)
    top2_logits = np.asarray(top2_logits, dtype=np.float64)
    if len(errors) != int(complete.get("rows", -1)):
        raise ValueError(
            "samples.csv row count {} does not match complete.json {}".format(
                len(errors), complete.get("rows")
            )
        )
    if len(errors) == 0:
        raise ValueError("Export contains no samples")
    if not np.isfinite(top1_logits).all() or not np.isfinite(top2_logits).all():
        raise ValueError("top1_logit and top2_logit must contain only finite values")
    if (top1_logits < top2_logits).any():
        raise ValueError("top1_logit must be greater than or equal to top2_logit")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("samples.csv contains duplicate sample_id values")

    confidence = sigmoid(top1_logits) - sigmoid(top2_logits)
    # Guard against tiny floating-point excursions outside the public [0, 1] contract.
    confidence = np.clip(confidence, 0.0, 1.0)
    return {
        "path": export_dir,
        "error": errors,
        "confidence": confidence,
        "provenance": provenance,
        "split": split_name,
    }


def run(calibration_dir, test_dir, output, target_tpr=0.95):
    calibration = load_margin_export(calibration_dir, CALIBRATION_SPLIT)
    test = load_margin_export(test_dir, TEST_SPLIT)
    model_identity = validate_same_model(calibration, test)
    tau, boundary, calibration_point = select_threshold(
        calibration["error"], calibration["confidence"], target_tpr
    )
    metrics = public_metrics(test["error"], test["confidence"])
    test_point = operating_point(test["error"], test["confidence"], tau)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    decision_rule = (
        "sigmoid_margin < confidence_threshold means predicted error; "
        "equality means predicted correct"
    )
    write_json(
        output / "threshold.json",
        {
            "method": "sigmoid_margin",
            "confidence_definition": "sigmoid(top1_logit) - sigmoid(top2_logit)",
            "confidence_threshold": tau,
            "observed_boundary_confidence": boundary,
            "decision_rule": decision_rule,
            "target_error_tpr": target_tpr,
            "calibration_operating_point": calibration_point,
            "calibration_split": CALIBRATION_SPLIT,
            "calibration_export": str(calibration["path"].resolve()),
            "model_identity": model_identity,
        },
    )
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "operating_point.json",
        {
            "method": "sigmoid_margin",
            "confidence_definition": "sigmoid(top1_logit) - sigmoid(top2_logit)",
            "confidence_threshold": tau,
            "decision_rule": decision_rule,
            "threshold_source": CALIBRATION_SPLIT,
            "evaluation_split": TEST_SPLIT,
            **test_point,
        },
    )
    write_summary(output, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--calibration-export", type=Path, required=True)
    parser.add_argument("--test-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-error-tpr", type=float, default=0.95)
    args = parser.parse_args()
    metrics = run(
        args.calibration_export,
        args.test_export,
        args.output,
        args.target_error_tpr,
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
