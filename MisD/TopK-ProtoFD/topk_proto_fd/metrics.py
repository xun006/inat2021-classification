from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def aurc(error_labels: np.ndarray, error_scores: np.ndarray) -> float:
    order = np.argsort(error_scores)  # accept lowest predicted-error samples first
    errors = error_labels[order].astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    # Discrete area over all attainable coverage levels (selective prediction).
    return float(cumulative_risk.mean())


def detection_metrics(error_labels: np.ndarray, error_scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(error_labels, dtype=np.int64)
    scores = np.asarray(error_scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        raise ValueError("Metrics require at least one correct and one erroneous prediction")
    fpr, tpr, _ = roc_curve(labels, scores)
    indices = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[indices[0]]) if len(indices) else 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "fpr95": fpr95,
        "aupr_error": float(average_precision_score(labels, scores)),
        "aurc": aurc(labels, scores),
        "error_rate": float(labels.mean()),
        "n_samples": int(labels.size),
    }
