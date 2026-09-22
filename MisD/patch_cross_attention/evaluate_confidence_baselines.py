#!/usr/bin/env python3
"""Evaluate training-free confidence baselines on the new official_val CSV."""

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
    parser.add_argument(
        "--predictions", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/teacher_predictions/official_val.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/baselines/confidence",
    )
    return parser.parse_args()


def fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(0.95, tpr, fpr))


def aurc(labels, scores):
    order = np.argsort(scores, kind="stable")
    return float((np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)).mean())


def main():
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    required = {"error_label", "teacher_top1_prob", "margin", "entropy", "max_logit", "energy"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    labels = frame.error_label.to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("official_val must contain correct and erroneous predictions")
    methods = {
        "MSP": 1 - frame.teacher_top1_prob.to_numpy(float),
        "Margin": -frame.margin.to_numpy(float),
        "Entropy": frame.entropy.to_numpy(float),
        "Max Logit": -frame.max_logit.to_numpy(float),
        "Energy": frame.energy.to_numpy(float),
    }
    rows = []
    scores = frame[["sample_id", "error_label"]].copy()
    for name, values in methods.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite scores")
        rows.append({
            "method": name, "auroc": roc_auc_score(labels, values),
            "error_auprc": average_precision_score(labels, values),
            "fpr_at_95_tpr": fpr95(labels, values), "aurc": aurc(labels, values),
        })
        scores[name.lower().replace(" ", "_") + "_error_score"] = values
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    scores.to_csv(args.output_dir / "official_val_error_scores.csv", index=False)
    (args.output_dir / "run_info.json").write_text(json.dumps({
        "predictions": str(args.predictions.resolve()), "samples": len(frame),
        "errors": int(labels.sum()), "positive_class": "teacher top-1 error",
        "score_direction": "larger means more likely to be wrong",
        "selection_split_used": False, "evaluation_split": "official_val",
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
