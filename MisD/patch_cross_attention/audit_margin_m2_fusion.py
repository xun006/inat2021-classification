#!/usr/bin/env python3
"""Cross-validated complementarity audit: Margin versus Margin + M2.

This script uses detector_calibration only. It never reads official_val.
M2 was selected on this calibration split, so this is a development audit,
not a final unbiased performance estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from teacher import MISD_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/teacher_predictions/detector_calibration.csv",
    )
    parser.add_argument(
        "--m2-predictions", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/ablation/m2_seed0/calibration_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/fusion_audit/margin_m2_seed0",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(0.95, tpr, fpr))


def aurc(labels, scores):
    order = np.argsort(scores, kind="stable")
    risk = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    return float(risk.mean())


def metrics(labels, scores):
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "error_auprc": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": fpr95(labels, scores),
        "aurc": aurc(labels, scores),
    }


def cross_validated_logistic(features, labels, folds, seed):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    output = np.empty(len(labels), dtype=np.float64)
    coefficients = []
    for fold, (train_index, validation_index) in enumerate(splitter.split(features, labels)):
        scaler = StandardScaler().fit(features[train_index])
        train_x = scaler.transform(features[train_index])
        validation_x = scaler.transform(features[validation_index])
        model = LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000,
            random_state=seed + fold,
        )
        model.fit(train_x, labels[train_index])
        output[validation_index] = model.predict_proba(validation_x)[:, 1]
        coefficients.append({
            "fold": fold,
            **{f"coefficient_{i}": float(value) for i, value in enumerate(model.coef_[0])},
            "intercept": float(model.intercept_[0]),
        })
    return output, coefficients


def paired_bootstrap(labels, baseline, fusion, repetitions, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(repetitions):
        index = rng.integers(0, len(labels), len(labels))
        y = labels[index]
        if np.unique(y).size != 2:
            continue
        base = metrics(y, baseline[index])
        combined = metrics(y, fusion[index])
        rows.append({key: combined[key] - base[key] for key in base})
    frame = pd.DataFrame(rows)
    summary = []
    # For FPR and AURC, a negative difference is an improvement.
    for key in frame.columns:
        values = frame[key].to_numpy()
        higher_is_better = key in {"auroc", "error_auprc"}
        win_probability = float((values > 0).mean() if higher_is_better else (values < 0).mean())
        summary.append({
            "metric": key,
            "mean_delta_fusion_minus_margin": float(values.mean()),
            "ci_2.5": float(np.quantile(values, 0.025)),
            "ci_97.5": float(np.quantile(values, 0.975)),
            "fusion_win_probability": win_probability,
            "higher_is_better": higher_is_better,
        })
    return frame, pd.DataFrame(summary)


def main():
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    m2 = pd.read_csv(args.m2_predictions)[["sample_id", "error_label", "error_score"]]
    if not metadata.sample_id.is_unique or not m2.sample_id.is_unique:
        raise ValueError("sample_id must be unique in both inputs")
    merged = metadata.merge(m2, on="sample_id", suffixes=("_metadata", "_m2"), validate="one_to_one")
    if len(merged) != len(metadata) or len(merged) != len(m2):
        raise ValueError("metadata and M2 predictions do not contain identical samples")
    if not np.array_equal(merged.error_label_metadata, merged.error_label_m2):
        raise ValueError("error labels disagree after sample_id alignment")
    labels = merged.error_label_metadata.to_numpy(dtype=np.int64)
    margin_score = -merged.margin.to_numpy(dtype=np.float64)
    m2_score = merged.error_score.to_numpy(dtype=np.float64)
    margin_oof, margin_coefficients = cross_validated_logistic(
        margin_score[:, None], labels, args.folds, args.seed
    )
    fusion_oof, fusion_coefficients = cross_validated_logistic(
        np.column_stack((margin_score, m2_score)), labels, args.folds, args.seed
    )
    methods = {
        "Margin (raw)": margin_score,
        "Margin-only logistic (5-fold OOF)": margin_oof,
        "M2 Patch-only": m2_score,
        "Margin + M2 logistic (5-fold OOF)": fusion_oof,
    }
    metric_rows = [{"method": name, **metrics(labels, score)} for name, score in methods.items()]
    bootstrap_raw, bootstrap_summary = paired_bootstrap(
        labels, margin_score, fusion_oof, args.bootstrap_samples, args.seed
    )
    spearman = float(pd.Series(margin_score).corr(pd.Series(m2_score), method="spearman"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    bootstrap_summary.to_csv(args.output_dir / "bootstrap_summary.csv", index=False)
    bootstrap_raw.to_csv(args.output_dir / "bootstrap_deltas.csv", index=False)
    pd.DataFrame(margin_coefficients).to_csv(args.output_dir / "margin_fold_coefficients.csv", index=False)
    pd.DataFrame(fusion_coefficients).to_csv(args.output_dir / "fusion_fold_coefficients.csv", index=False)
    pd.DataFrame({
        "sample_id": merged.sample_id, "error_label": labels,
        "margin_error_score": margin_score, "m2_error_score": m2_score,
        "margin_oof_score": margin_oof, "fusion_oof_score": fusion_oof,
    }).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    report = {
        "purpose": "development-only complementarity audit",
        "split": "detector_calibration", "official_val_used": False,
        "samples": len(labels), "errors": int(labels.sum()), "folds": args.folds,
        "seed": args.seed, "bootstrap_samples": len(bootstrap_raw),
        "m2_model": "m2_seed0 original raw classifier-weight Query",
        "spearman_margin_m2": spearman,
        "caveat": "M2 checkpoint epoch was selected on this same calibration split",
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nPaired bootstrap: fusion OOF minus raw Margin")
    print(bootstrap_summary.to_string(index=False))
    print("\n", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
