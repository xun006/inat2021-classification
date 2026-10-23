"""Numerically explicit REL-U mathematics and common failure-detection metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class DistanceDiagnostics:
    fallback_to_gini: bool
    nonzero_entries: int
    frobenius_norm: float


def build_distance_matrix(
    correct_moment: torch.Tensor,
    error_moment: torch.Tensor,
    lbd: float,
) -> tuple[torch.Tensor, DistanceDiagnostics]:
    """Build the exact symmetric REL-U matrix used by the authors' implementation.

    The lower triangle is retained and reflected to remove insignificant numerical
    asymmetry from matrix multiplication. The diagonal is always exactly zero.
    """
    if correct_moment.shape != error_moment.shape or correct_moment.ndim != 2:
        raise ValueError("correct/error moments must be equally shaped matrices")
    if correct_moment.shape[0] != correct_moment.shape[1]:
        raise ValueError("moments must be square")
    if not 0.0 <= lbd <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    if not torch.isfinite(correct_moment).all() or not torch.isfinite(error_moment).all():
        raise ValueError("moments contain non-finite values")

    raw = lbd * error_moment - (1.0 - lbd) * correct_moment
    distance = torch.tril(raw, diagonal=-1)
    distance = torch.relu(distance + distance.T)
    fallback = bool(torch.count_nonzero(distance).item() == 0)
    if fallback:
        distance = torch.ones_like(distance)
        distance.fill_diagonal_(0.0)
    norm = torch.linalg.vector_norm(distance)
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError("REL-U distance matrix has an invalid norm")
    distance = distance / norm
    diagnostics = DistanceDiagnostics(
        fallback_to_gini=fallback,
        nonzero_entries=int(torch.count_nonzero(distance).item()),
        frobenius_norm=float(torch.linalg.vector_norm(distance).item()),
    )
    return distance, diagnostics


def relative_uncertainty(probabilities: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Compute diag(P D P^T) without materializing a batch-by-batch matrix."""
    if probabilities.ndim != 2 or distance.ndim != 2:
        raise ValueError("probabilities and distance must be matrices")
    if probabilities.shape[1] != distance.shape[0] or distance.shape[0] != distance.shape[1]:
        raise ValueError("incompatible probability/distance shapes")
    return ((probabilities @ distance) * probabilities).sum(dim=1)


def expected_tie_aware_aurc(error: np.ndarray, error_score: np.ndarray) -> float:
    """Expected AURC under a uniformly random ordering inside equal-score groups."""
    error = np.asarray(error, dtype=np.float64)
    score = np.asarray(error_score, dtype=np.float64)
    order = np.argsort(score, kind="stable")
    error = error[order]
    score = score[order]
    total_risk = 0.0
    accepted = 0
    accepted_errors = 0.0
    start = 0
    while start < len(error):
        stop = start + 1
        while stop < len(error) and score[stop] == score[start]:
            stop += 1
        group_size = stop - start
        group_errors = float(error[start:stop].sum())
        error_fraction = group_errors / group_size
        for within_group in range(1, group_size + 1):
            expected_errors = accepted_errors + within_group * error_fraction
            total_risk += expected_errors / (accepted + within_group)
        accepted += group_size
        accepted_errors += group_errors
        start = stop
    return float(total_risk / len(error))


def optimal_aurc(error: np.ndarray) -> float:
    """AURC for the perfect ranking with all correct samples accepted first."""
    error = np.asarray(error, dtype=np.int64)
    ordered = np.sort(error)
    risk = np.cumsum(ordered, dtype=np.float64) / np.arange(1, len(ordered) + 1)
    return float(risk.mean())


def failure_metrics(error: np.ndarray, error_score: np.ndarray) -> dict[str, float | int]:
    """Metrics with error=1 and larger score meaning more likely to be wrong."""
    error = np.asarray(error, dtype=np.int64)
    score = np.asarray(error_score, dtype=np.float64)
    if error.ndim != 1 or score.ndim != 1 or len(error) != len(score) or len(error) == 0:
        raise ValueError("error and score must be non-empty, equally sized vectors")
    if set(np.unique(error)) != {0, 1}:
        raise ValueError("metrics require both correct and erroneous samples")
    if not np.isfinite(score).all():
        raise ValueError("scores contain non-finite values")
    fpr, tpr, _ = roc_curve(error, score)
    aurc = expected_tie_aware_aurc(error, score)
    return {
        "auroc": float(roc_auc_score(error, score)),
        "error_auprc": float(average_precision_score(error, score)),
        "fpr_at_95_error_tpr": float(np.interp(0.95, tpr, fpr)),
        "aurc": aurc,
        "eaurc": float(aurc - optimal_aurc(error)),
        "error_rate": float(error.mean()),
        "samples": int(len(error)),
        "errors": int(error.sum()),
    }
