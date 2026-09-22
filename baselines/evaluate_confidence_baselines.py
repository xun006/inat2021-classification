#!/usr/bin/env python3
"""Evaluate training-free misclassification-detection baselines.

The positive class is a wrong teacher prediction (error_label=1), and every
score is oriented so that a larger value means "more likely to be wrong".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


REQUIRED_COLUMNS = {
    "error_label",
    "teacher_top1_prob",
    "margin",
    "entropy",
    "max_logit",
    "energy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("output/data/misdetection_splits/test.csv"),
        help="Natural-prevalence detector test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/baselines/confidence"),
        help="Directory for metrics, scores, and the Risk-Coverage curve.",
    )
    return parser.parse_args()


def validate_input(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Test CSV is empty.")

    labels = pd.to_numeric(frame["error_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("error_label must contain only 0 and 1.")
    if labels.nunique() != 2:
        raise ValueError("Both correct and erroneous predictions are required.")

    for column in REQUIRED_COLUMNS - {"error_label"}:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains missing or non-finite values.")


def make_error_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return scores with a common direction: larger means more error-like."""
    return {
        "MSP": 1.0 - frame["teacher_top1_prob"].to_numpy(dtype=np.float64),
        "Margin": -frame["margin"].to_numpy(dtype=np.float64),
        "Entropy": frame["entropy"].to_numpy(dtype=np.float64),
        "Max Logit": -frame["max_logit"].to_numpy(dtype=np.float64),
        # Exported energy is the standard -logsumexp(logits), with T=1.
        "Energy": frame["energy"].to_numpy(dtype=np.float64),
    }


def fpr_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Linearly interpolate FPR at an error-detection TPR of 95%."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(0.95, tpr, fpr))


def risk_coverage_curve(
    labels: np.ndarray, error_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute empirical selective risk while retaining safest samples first.

    Coverage k/N retains the k samples with the lowest error scores. Risk is
    the teacher classification error rate among those retained samples. AURC
    is the standard empirical mean of risk at all non-zero coverage levels.
    """
    order = np.argsort(error_scores, kind="stable")
    sorted_errors = labels[order].astype(np.float64)
    retained = np.arange(1, len(labels) + 1, dtype=np.float64)
    coverage = retained / len(labels)
    risk = np.cumsum(sorted_errors) / retained
    aurc = float(risk.mean())
    return coverage, risk, aurc


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.test_csv)
    validate_input(frame)

    labels = frame["error_label"].to_numpy(dtype=np.int64)
    scores = make_error_scores(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | str]] = []
    curve_frames: list[pd.DataFrame] = []
    score_output = pd.DataFrame({"error_label": labels})
    for optional_id in ("image_id", "feature_row"):
        if optional_id in frame.columns:
            score_output.insert(len(score_output.columns) - 1, optional_id, frame[optional_id])

    for name, error_score in scores.items():
        coverage, risk, aurc = risk_coverage_curve(labels, error_score)
        metrics_rows.append(
            {
                "method": name,
                "auroc": float(roc_auc_score(labels, error_score)),
                "error_auprc": float(average_precision_score(labels, error_score)),
                "fpr_at_95_tpr": fpr_at_95_tpr(labels, error_score),
                "aurc": aurc,
            }
        )
        curve_frames.append(
            pd.DataFrame({"method": name, "coverage": coverage, "risk": risk})
        )
        score_output[name.lower().replace(" ", "_") + "_error_score"] = error_score

    metrics = pd.DataFrame(metrics_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    curves.to_csv(args.output_dir / "risk_coverage.csv", index=False)
    score_output.to_csv(args.output_dir / "test_error_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name in scores:
        curve = curves[curves["method"] == name]
        ax.plot(curve["coverage"], curve["risk"], label=name, linewidth=1.8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (classification error rate)")
    ax.set_title("Risk-Coverage: training-free confidence baselines")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "risk_coverage.png", dpi=200)
    plt.close(fig)

    run_info = {
        "test_csv": str(args.test_csv.resolve()),
        "samples": int(len(frame)),
        "errors": int(labels.sum()),
        "error_prevalence": float(labels.mean()),
        "positive_class": "error_label=1 (teacher top-1 prediction is wrong)",
        "score_direction": "higher means more likely to be wrong",
        "fpr_at_95_tpr": "linear interpolation on the test ROC curve",
        "aurc": "mean empirical selective risk over k/N for k=1..N",
        "methods": {
            "MSP": "1 - teacher_top1_prob",
            "Margin": "-margin",
            "Entropy": "entropy",
            "Max Logit": "-max_logit",
            "Energy": "energy = -logsumexp(logits), T=1",
        },
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Evaluated {len(frame)} samples ({labels.sum()} errors).")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
