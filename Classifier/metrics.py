import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def _as_failure_arrays(error, confidence):
    error = np.asarray(error)
    confidence = np.asarray(confidence, dtype=np.float64)
    if error.ndim != 1 or confidence.ndim != 1 or len(error) != len(confidence):
        raise ValueError("error and confidence must be non-empty 1-D arrays of equal length")
    if len(error) == 0:
        raise ValueError("failure-detection metrics require at least one sample")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence contains non-finite values")
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("confidence must be in [0, 1]")
    unique = set(np.unique(error).tolist())
    if not unique.issubset({False, True, 0, 1}):
        raise ValueError("error must be binary")
    return error.astype(bool), confidence


def risk_coverage_curve(error, confidence):
    """Expected selective risk at every coverage, including score ties fairly."""
    error, confidence = _as_failure_arrays(error, confidence)
    order = np.argsort(-confidence, kind="stable")
    sorted_score, sorted_error = confidence[order], error[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score)) + 1, len(error)]
    risk = np.empty(len(error), dtype=float)
    start, previous = 0, 0.0
    for end in ends:
        group_errors = sorted_error[start:end].sum()
        k = np.arange(1, end - start + 1)
        risk[start:end] = (previous + k * group_errors / (end - start)) / (start + k)
        previous += group_errors
        start = end
    return risk


def fpr_at_tpr(error, error_score, target_tpr=0.95):
    """Linearly interpolated FPR at a target error-detection TPR."""
    error_score = np.asarray(error_score, dtype=np.float64)
    if not 0 < target_tpr <= 1:
        raise ValueError("target_tpr must be in (0, 1]")
    error, _ = _as_failure_arrays(error, 1.0 - error_score)
    if len(np.unique(error)) != 2:
        return None
    fpr, tpr, _ = roc_curve(error, error_score)
    return float(np.interp(target_tpr, tpr, fpr))


def failure_detection_metrics(error, confidence, target_tpr=0.95):
    """Threshold-free failure metrics; errors are the positive class."""
    error, confidence = _as_failure_arrays(error, confidence)
    error_score = 1.0 - confidence
    has_both = len(np.unique(error)) == 2
    risk = risk_coverage_curve(error, confidence)
    return {
        "auroc_error": float(roc_auc_score(error, error_score)) if has_both else None,
        "aupr_error": float(average_precision_score(error, error_score)) if has_both else None,
        "fpr_at_95_tpr": fpr_at_tpr(error, error_score, target_tpr),
        "aurc": float(risk.mean()),
    }, risk


def evaluate_arrays(labels, predictions, confidence, top5, counts, tau=0.5):
    labels, predictions, confidence = map(np.asarray, (labels, predictions, confidence))
    if labels.ndim != 1 or predictions.ndim != 1 or len(labels) != len(predictions):
        raise ValueError("labels and predictions must be non-empty 1-D arrays of equal length")
    error = labels != predictions
    failure, risk = failure_detection_metrics(error, confidence)
    groups = np.where(np.asarray(counts) > 100, "head", np.where(np.asarray(counts) >= 20, "medium", "tail"))
    result = {"top1": float((~error).mean()), "top5": float(np.mean(top5)),
              "macro_f1": float(f1_score(labels, predictions, labels=np.arange(len(counts)), average="macro", zero_division=0)),
              **failure, "samples": len(labels), "tau": tau,
              "high_confidence_wrong": int((error & (confidence >= tau)).sum()),
              "low_confidence_correct": int((~error & (confidence < tau)).sum())}
    for group in ("head", "medium", "tail"):
        mask = groups[labels] == group
        result[group + "_accuracy"] = float((~error[mask]).mean()) if mask.any() else None
    for name, mask in (("correct", ~error), ("wrong", error)):
        values = confidence[mask]
        result[name + "_confidence_mean"] = float(values.mean()) if len(values) else None
        result[name + "_confidence_median"] = float(np.median(values)) if len(values) else None
        result[name + "_confidence_histogram"] = np.histogram(values, bins=np.linspace(0, 1, 21))[0].tolist()
    result["high_confidence_wrong_fraction_all"] = result["high_confidence_wrong"] / len(labels)
    result["low_confidence_correct_fraction_all"] = result["low_confidence_correct"] / len(labels)
    return result, risk, groups
