#!/usr/bin/env python3
"""Final locked comparison of Margin and Margin+M2 on official_val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from teacher import MISD_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50, 50)))


def fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(0.95, tpr, fpr))


def aurc(labels, scores):
    order = np.argsort(scores, kind="stable")
    return float((np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)).mean())


def metrics(labels, scores):
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "error_auprc": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": fpr95(labels, scores), "aurc": aurc(labels, scores),
    }


def main():
    args = parse_args()
    root = MISD_ROOT / "output/patch_cross_attention/final_fusion/margin_m2_seed0"
    metadata = pd.read_csv(
        MISD_ROOT / "output/patch_cross_attention/teacher_predictions/official_val.csv"
    )
    m2 = pd.read_csv(root / "official_val_m2_predictions.csv")
    fusion = json.loads((root / "fusion_model.json").read_text())
    merged = metadata.merge(m2, on="sample_id", suffixes=("_metadata", "_m2"), validate="one_to_one")
    if len(merged) != len(metadata) or len(merged) != len(m2):
        raise ValueError("official metadata and M2 predictions do not match exactly")
    if not np.array_equal(merged.error_label_metadata, merged.error_label_m2):
        raise ValueError("official labels disagree")
    labels = merged.error_label_metadata.to_numpy(dtype=np.int64)
    margin = -merged.margin.to_numpy(dtype=np.float64)
    m2_score = merged.m2_error_score.to_numpy(dtype=np.float64)
    features = np.column_stack((margin, m2_score))
    mean = np.asarray(fusion["feature_mean"])
    scale = np.asarray(fusion["feature_scale"])
    coefficient = np.asarray(fusion["coefficient"])
    #fused = sigmoid(((features - mean) / scale).matmul(coefficient) + fusion["intercept"])
    fused = sigmoid(
        ((features - mean) / scale) @ coefficient
        + fusion["intercept"]
    )
    rows = [
        {"method": "Margin", **metrics(labels, margin)},
        {"method": "M2 Patch-only", **metrics(labels, m2_score)},
        {"method": "Margin + M2", **metrics(labels, fused)},
    ]
    metric_frame = pd.DataFrame(rows)
    metric_frame.to_csv(root / "official_val_metrics.csv", index=False)
    pd.DataFrame({"sample_id": merged.sample_id, "error_label": labels,
                  "margin_error_score": margin, "m2_error_score": m2_score,
                  "fusion_error_score": fused}).to_csv(
        root / "official_val_final_predictions.csv", index=False
    )
    base_metrics, fused_metrics = rows[0], rows[2]
    rng = np.random.default_rng(args.seed)
    deltas = []
    for _ in range(args.bootstrap_samples):
        index = rng.integers(0, len(labels), len(labels))
        y = labels[index]
        if np.unique(y).size != 2:
            continue
        base = metrics(y, margin[index]); combined = metrics(y, fused[index])
        deltas.append({key: combined[key] - base[key] for key in base})
    delta_frame = pd.DataFrame(deltas)
    summary = []
    for key in delta_frame:
        values = delta_frame[key].to_numpy()
        higher = key in {"auroc", "error_auprc"}
        summary.append({
            "metric": key, "point_delta_fusion_minus_margin": fused_metrics[key] - base_metrics[key],
            "bootstrap_mean_delta": float(values.mean()),
            "ci_2.5": float(np.quantile(values, 0.025)),
            "ci_97.5": float(np.quantile(values, 0.975)),
            "fusion_win_probability": float((values > 0).mean() if higher else (values < 0).mean()),
            "higher_is_better": higher,
        })
    pd.DataFrame(summary).to_csv(root / "official_val_bootstrap_summary.csv", index=False)
    print(metric_frame.to_string(index=False))
    print("\nPaired bootstrap: Margin+M2 minus Margin")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
