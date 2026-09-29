"""Calibrate and evaluate max-Sigmoid misclassification detection.

The calibration split chooses the operating threshold.  The test split is
used only after that threshold has been frozen.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import failure_detection_metrics


CALIBRATION_SPLIT = "detector_calibration"
TEST_SPLIT = "official_val"


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mapping_digest(mapping):
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_export(export_dir, expected_split):
    export_dir = Path(export_dir)
    required = ["samples.csv", "complete.json", "provenance.json"]
    missing = [name for name in required if not (export_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete export {}: missing {}".format(export_dir, missing))

    complete = read_json(export_dir / "complete.json")
    provenance = read_json(export_dir / "provenance.json")
    split_name = Path(provenance.get("split_dir", "")).name
    if split_name != expected_split:
        raise ValueError(
            "Expected {} export, got split_dir={!r}".format(
                expected_split, provenance.get("split_dir")
            )
        )

    errors, confidence, sample_ids = [], [], []
    with (export_dir / "samples.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        needed = {"sample_id", "correct", "max_sigmoid_probability"}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError("samples.csv is missing required columns: {}".format(sorted(needed)))
        for line_number, row in enumerate(reader, start=2):
            try:
                correct = int(row["correct"])
                score = float(row["max_sigmoid_probability"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid samples.csv row {}".format(line_number)) from exc
            if correct not in (0, 1):
                raise ValueError("correct must be 0 or 1 at samples.csv row {}".format(line_number))
            errors.append(1 - correct)
            confidence.append(score)
            sample_ids.append(row["sample_id"])

    errors = np.asarray(errors, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    if len(errors) != int(complete.get("rows", -1)):
        raise ValueError(
            "samples.csv row count {} does not match complete.json {}".format(
                len(errors), complete.get("rows")
            )
        )
    if len(errors) == 0:
        raise ValueError("Export contains no samples")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("max_sigmoid_probability must contain finite values in [0, 1]")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("samples.csv contains duplicate sample_id values")
    return {
        "path": export_dir,
        "error": errors,
        "confidence": confidence,
        "provenance": provenance,
        "split": split_name,
    }


def identity(provenance):
    required = ("checkpoint_sha256", "pretrained_sha256", "class_to_idx")
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError("provenance.json is missing identity fields: {}".format(missing))
    return {
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "pretrained_sha256": provenance["pretrained_sha256"],
        "class_mapping_sha256": mapping_digest(provenance["class_to_idx"]),
    }


def validate_same_model(calibration, test):
    calibration_identity = identity(calibration["provenance"])
    test_identity = identity(test["provenance"])
    if calibration_identity != test_identity:
        differing = [
            key for key in calibration_identity
            if calibration_identity[key] != test_identity[key]
        ]
        raise ValueError("Calibration/test model identity mismatch: {}".format(differing))
    if calibration["path"].resolve() == test["path"].resolve():
        raise ValueError("Calibration and test exports must be different directories")
    return calibration_identity


def operating_point(error, confidence, tau):
    error = np.asarray(error, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    predicted_error = confidence < tau
    tp = int((predicted_error & error).sum())
    fp = int((predicted_error & ~error).sum())
    tn = int((~predicted_error & ~error).sum())
    fn = int((~predicted_error & error).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "error_recall": recall,
        "error_precision": precision,
        "error_f1": f1,
        "correct_false_positive_rate": fpr,
        "prediction_rejection_rate": float(predicted_error.mean()),
        "samples": int(len(error)),
        "errors": int(error.sum()),
        "error_rate": float(error.mean()),
    }


def select_threshold(error, confidence, target_tpr):
    """Choose the smallest strict confidence threshold reaching target error TPR.

    Candidate boundaries come only from observed confidence groups.  nextafter
    converts an inclusive observed boundary into the public strict rule c < tau,
    so every tied sample is included or excluded together.
    """
    if not 0 < target_tpr <= 1:
        raise ValueError("target-error-tpr must be in (0, 1]")
    error = np.asarray(error, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    total_errors = int(error.sum())
    if total_errors == 0 or total_errors == len(error):
        raise ValueError("Calibration requires both correct and erroneous samples")

    order = np.argsort(confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_error = error[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_confidence)) + 1, len(error)]
    cumulative_errors = np.cumsum(sorted_error)
    required_errors = int(np.ceil(target_tpr * total_errors))
    selected_end = next(end for end in group_ends if cumulative_errors[end - 1] >= required_errors)
    boundary = float(sorted_confidence[selected_end - 1])
    tau = float(np.nextafter(boundary, np.inf))
    result = operating_point(error, confidence, tau)
    if result["error_recall"] is None or result["error_recall"] + 1e-15 < target_tpr:
        raise AssertionError("Selected threshold did not reach target error TPR")
    return tau, boundary, result


def public_metrics(error, confidence):
    metrics, _ = failure_detection_metrics(error, confidence, target_tpr=0.95)
    return {
        "auroc": metrics["auroc_error"],
        "error_auprc": metrics["aupr_error"],
        "fpr_at_95_tpr": metrics["fpr_at_95_tpr"],
        "aurc": metrics["aurc"],
    }


def display(value):
    return "null" if value is None else "{:.6f}".format(value)


def write_summary(output, metrics):
    fields = ["auroc", "error_auprc", "fpr_at_95_tpr", "aurc"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(metrics)
    markdown = [
        "| AUROC ↑ | Error AUPRC ↑ | FPR@95TPR ↓ | AURC ↓ |",
        "|---:|---:|---:|---:|",
        "| {} | {} | {} | {} |".format(*(display(metrics[key]) for key in fields)),
        "",
    ]
    (output / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def run(calibration_dir, test_dir, output, target_tpr=0.95):
    calibration = load_export(calibration_dir, CALIBRATION_SPLIT)
    test = load_export(test_dir, TEST_SPLIT)
    model_identity = validate_same_model(calibration, test)
    tau, boundary, calibration_point = select_threshold(
        calibration["error"], calibration["confidence"], target_tpr
    )
    metrics = public_metrics(test["error"], test["confidence"])
    test_point = operating_point(test["error"], test["confidence"], tau)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    threshold = {
        "confidence_threshold": tau,
        "observed_boundary_confidence": boundary,
        "decision_rule": "max_sigmoid_probability < confidence_threshold means predicted error; equality means predicted correct",
        "target_error_tpr": target_tpr,
        "calibration_operating_point": calibration_point,
        "calibration_split": CALIBRATION_SPLIT,
        "calibration_export": str(calibration["path"].resolve()),
        "model_identity": model_identity,
    }
    write_json(output / "threshold.json", threshold)
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "operating_point.json",
        {
            "confidence_threshold": tau,
            "decision_rule": threshold["decision_rule"],
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
