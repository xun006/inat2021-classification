import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def evaluate_arrays(labels, predictions, confidence, top5, counts, tau=0.5):
    labels, predictions, confidence = map(np.asarray, (labels, predictions, confidence))
    error = labels != predictions
    order = np.argsort(-confidence, kind="stable")
    # Expected risk inside a tied-confidence group, independent of input ordering.
    sorted_score, sorted_error = confidence[order], error[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score)) + 1, len(labels)]
    risk = np.empty(len(labels), dtype=float)
    start, previous = 0, 0.0
    for end in ends:
        group_errors = sorted_error[start:end].sum()
        k = np.arange(1, end - start + 1)
        risk[start:end] = (previous + k * group_errors / (end - start)) / (start + k)
        previous += group_errors
        start = end
    groups = np.where(np.asarray(counts) > 100, "head", np.where(np.asarray(counts) >= 20, "medium", "tail"))
    result = {"top1": float((~error).mean()), "top5": float(np.mean(top5)),
              "macro_f1": float(f1_score(labels, predictions, labels=np.arange(len(counts)), average="macro", zero_division=0)),
              "auroc_error": float(roc_auc_score(error, 1-confidence)) if len(np.unique(error)) == 2 else None,
              "aupr_error": float(average_precision_score(error, 1-confidence)) if error.any() else None,
              "aurc": float(risk.mean()), "samples": len(labels), "tau": tau,
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
