"""Failure-detection metrics. Larger scores always mean more likely to be wrong."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def aurc(error: np.ndarray, error_score: np.ndarray) -> float:
    order = np.argsort(error_score, kind="stable")
    cumulative_risk = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    return float(cumulative_risk.mean())


def fpr_at_95_tpr(error: np.ndarray, error_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(error, error_score)
    return float(np.interp(0.95, tpr, fpr))


def failure_metrics(error: np.ndarray, error_score: np.ndarray) -> dict[str, float]:
    error = np.asarray(error, dtype=np.int64)
    error_score = np.asarray(error_score, dtype=np.float64)
    if len(error) == 0 or set(np.unique(error)) != {0, 1}:
        raise ValueError("metrics require both correct and erroneous samples")
    if not np.isfinite(error_score).all():
        raise ValueError("scores contain non-finite values")
    return {
        "auroc": float(roc_auc_score(error, error_score)),
        "error_auprc": float(average_precision_score(error, error_score)),
        "fpr_at_95_tpr": fpr_at_95_tpr(error, error_score),
        "aurc": aurc(error, error_score),
        "error_rate": float(error.mean()),
        "samples": int(len(error)),
        "errors": int(error.sum()),
    }
