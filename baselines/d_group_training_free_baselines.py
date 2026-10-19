#!/usr/bin/env python3
"""Evaluate seven training-free failure scores with the trained D-group model.

The positive class is a wrong top-1 prediction. Every score is oriented so a
larger value means "more likely to be wrong". Final aggregate metrics are
reported as percentages (raw value * 100), rounded half-up to two decimals.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
from collections import OrderedDict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


METHODS = (
    "Max-Sigmoid",
    "Sigmoid Margin",
    "Raw-logit Margin",
    "OVR Self-BCE",
    "DOCTOR",
    "Energy",
    "L2Norm-MaxLogit",
)
METRIC_FIELDS = ("auroc", "error_auprc", "fpr_at_95_tpr", "aurc", "oscr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="D run config.json.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="D run best.pt. Non-best filenames are rejected.")
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="Labeled ImageFolder split to evaluate.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing directories are rejected.")
    parser.add_argument("--pretrained", type=Path,
                        help="Optional relocated PlantCLEF backbone checkpoint.")
    parser.add_argument("--class-map", type=Path,
                        help="Optional relocated class_to_idx.json.")
    parser.add_argument("--workers", type=int,
                        help="Optional inference DataLoader worker override.")
    parser.add_argument("--device", default="cuda",
                        help="Torch device, for example cuda, cuda:1, or cpu.")
    return parser.parse_args()


def validate_d_checkpoint_config(cfg: dict, saved: dict, checkpoint: Path) -> None:
    """Reject checkpoints that do not match the controlled D protocol."""
    if checkpoint.name != "best.pt":
        raise ValueError(
            f"Expected the validation-selected D checkpoint named best.pt, got {checkpoint.name!r}"
        )
    expected = {
        "loss": "l2", "seed": 42, "epochs": 20, "batch_size": 32,
        "rank": 8, "alpha": 16, "lora_dropout": 0.0, "bce_reduction": "mean",
    }
    saved_cfg = saved.get("config", {})
    for key, value in expected.items():
        if cfg.get(key) != value or saved_cfg.get(key) != value:
            raise ValueError(f"Not a valid D-group checkpoint: expected {key}={value!r}")
    if cfg.get("group") not in (None, "D") or saved_cfg.get("group") not in (None, "D"):
        raise ValueError("Explicit group metadata must identify group D")
    if cfg.get("tuning_mode") not in (None, "lora_head"):
        raise ValueError("D must use LoRA+Head tuning")
    if saved_cfg.get("tuning_mode") not in (None, "lora_head"):
        raise ValueError("D checkpoint must use LoRA+Head tuning")
    changed = {key for key in cfg if cfg[key] != saved_cfg.get(key)} - {
        "train_dir", "val_dir", "class_map", "pretrained", "workers"
    }
    if changed:
        raise ValueError(f"Checkpoint/config mismatch: {sorted(changed)}")


def verify_best_selection(checkpoint: Path, saved: dict) -> dict[str, object]:
    """Verify best-by-classifier_val selection when history.jsonl is available."""
    history_path = checkpoint.parent / "history.jsonl"
    if not history_path.exists():
        return {"verified": False, "reason": "history.jsonl not found"}
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"Empty training history: {history_path}")
    best_value = max(float(row["val"]["top1"]) for row in rows)
    best_epoch = next(int(row["epoch"]) for row in rows
                      if float(row["val"]["top1"]) == best_value)
    if int(saved["epoch"]) != best_epoch or not np.isclose(float(saved["best"]), best_value):
        raise ValueError(
            "best.pt does not match the earliest highest classifier_val Top-1 epoch"
        )
    return {
        "verified": True,
        "history": str(history_path.resolve()),
        "selected_epoch_one_based": best_epoch + 1,
        "selected_val_top1": best_value,
    }


def score_tensors(logits: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    """Return the seven D-group error scores; larger always means more error-like."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [N, K] with K >= 2")
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits encountered")

    top_logits = logits.topk(2, dim=1).values
    top1, top2 = top_logits[:, 0], top_logits[:, 1]
    top1_sigmoid, top2_sigmoid = top1.sigmoid(), top2.sigmoid()

    # D was trained with mean BCE over one positive and K-1 negative targets.
    # Evaluate that same loss using the model's own argmax as a pseudo-label.
    negative_terms = F.softplus(logits).sum(dim=1)
    ovr_self_bce = (
        negative_terms - F.softplus(top1) + F.softplus(-top1)
    ) / logits.shape[1]

    log_probabilities = logits.log_softmax(dim=1)
    probabilities = log_probabilities.exp()
    collision_probability = probabilities.square().sum(dim=1)
    doctor = (1.0 - collision_probability) / collision_probability.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )

    l2_norm = torch.linalg.vector_norm(logits, ord=2, dim=1)
    safe_l2_norm = l2_norm.clamp_min(torch.finfo(logits.dtype).tiny)
    l2_normalized_max_logit = (logits / safe_l2_norm[:, None]).max(dim=1).values

    return OrderedDict([
        ("Max-Sigmoid", 1.0 - top1_sigmoid),
        ("Sigmoid Margin", -(top1_sigmoid - top2_sigmoid)),
        ("Raw-logit Margin", -(top1 - top2)),
        ("OVR Self-BCE", ovr_self_bce),
        ("DOCTOR", doctor),
        ("Energy", -torch.logsumexp(logits, dim=1)),
        ("L2Norm-MaxLogit", -l2_normalized_max_logit),
    ])


def risk_coverage_curve(
    errors: np.ndarray, error_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Tie-aware expected selective risk while accepting safest samples first."""
    order = np.argsort(error_scores, kind="stable")
    sorted_scores = error_scores[order]
    sorted_errors = errors[order].astype(np.float64)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(errors)]
    risk = np.empty(len(errors), dtype=np.float64)
    start = 0
    previous_errors = 0.0
    for end in group_ends:
        group_errors = float(sorted_errors[start:end].sum())
        within_tie = np.arange(1, end - start + 1, dtype=np.float64)
        expected_errors = previous_errors + within_tie * group_errors / (end - start)
        risk[start:end] = expected_errors / np.arange(start + 1, end + 1)
        previous_errors += group_errors
        start = end
    coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
    return coverage, risk, float(risk.mean())


def closed_set_oscr_curve(
    errors: np.ndarray, error_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-set OSCR adaptation; incorrect predictions serve as negatives."""
    confidence = -error_scores
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_errors = errors[order].astype(bool)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_confidence)) + 1, len(errors)]
    total_wrong = int(errors.sum())
    if total_wrong == 0 or total_wrong == len(errors):
        raise ValueError("OSCR and binary failure metrics require correct and wrong predictions")
    fpr, ccr = [0.0], [0.0]
    accepted_wrong = accepted_correct = start = 0
    for end in group_ends:
        group = sorted_errors[start:end]
        accepted_wrong += int(group.sum())
        accepted_correct += int((~group).sum())
        fpr.append(accepted_wrong / total_wrong)
        ccr.append(accepted_correct / len(errors))
        start = end
    fpr_array = np.asarray(fpr, dtype=np.float64)
    ccr_array = np.asarray(ccr, dtype=np.float64)
    area = np.sum(np.diff(fpr_array) * (ccr_array[:-1] + ccr_array[1:]) * 0.5)
    return fpr_array, ccr_array, float(area)


def evaluate_scores(
    errors: np.ndarray, score_arrays: OrderedDict[str, np.ndarray]
) -> tuple[list[dict[str, float | str]], dict[str, tuple[np.ndarray, np.ndarray]],
           dict[str, tuple[np.ndarray, np.ndarray]]]:
    errors = np.asarray(errors, dtype=np.int64)
    if errors.ndim != 1 or len(errors) == 0 or set(np.unique(errors)) != {0, 1}:
        raise ValueError("Evaluation requires a non-empty error vector containing both 0 and 1")
    rows, risk_curves, oscr_curves = [], {}, {}
    for method in METHODS:
        scores = np.asarray(score_arrays[method], dtype=np.float64)
        if scores.shape != errors.shape or not np.isfinite(scores).all():
            raise ValueError(f"Invalid scores for {method}")
        fpr, tpr, _ = roc_curve(errors, scores)
        coverage, risk, aurc = risk_coverage_curve(errors, scores)
        oscr_fpr, ccr, oscr = closed_set_oscr_curve(errors, scores)
        rows.append({
            "method": method,
            "auroc": float(roc_auc_score(errors, scores)),
            "error_auprc": float(average_precision_score(errors, scores)),
            "fpr_at_95_tpr": float(np.interp(0.95, tpr, fpr)),
            "aurc": aurc,
            "oscr": oscr,
        })
        risk_curves[method] = (coverage, risk)
        oscr_curves[method] = (oscr_fpr, ccr)
    return rows, risk_curves, oscr_curves


def percent_string(value: float) -> str:
    """Multiply by 100 and use decimal ROUND_HALF_UP to exactly two places."""
    scaled = Decimal(str(float(value))) * Decimal("100")
    return format(scaled.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def write_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = ["method", *METRIC_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"method": row["method"], **{
                key: percent_string(float(row[key])) for key in METRIC_FIELDS
            }})


def write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = [
        "| Method | AUROC (%) ↑ | Error AUPRC (%) ↑ | FPR@95TPR (%) ↓ | AURC (%) ↓ | OSCR (%) ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [str(row["method"])] + [percent_string(float(row[key]))
                                                for key in METRIC_FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_score_table(
    path: Path, sample_ids: list[str], labels: np.ndarray, predictions: np.ndarray,
    score_arrays: OrderedDict[str, np.ndarray]
) -> None:
    score_fields = [method.lower().replace("-", "_").replace(" ", "_") + "_error_score"
                    for method in METHODS]
    fields = ["row", "sample_id", "ground_truth", "prediction", "error_label", *score_fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, sample_id in enumerate(sample_ids):
            writer.writerow([
                index, sample_id, int(labels[index]), int(predictions[index]),
                int(labels[index] != predictions[index]),
                *[float(score_arrays[method][index]) for method in METHODS],
            ])


def write_curve_tables(
    output: Path, risk_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    oscr_curves: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    with (output / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "coverage_percent", "selective_risk_percent"])
        for method in METHODS:
            coverage, risk = risk_curves[method]
            writer.writerows((method, percent_string(x), percent_string(y))
                             for x, y in zip(coverage, risk))
    with (output / "oscr_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "false_accept_rate_among_errors_percent",
            "correct_classification_rate_percent",
        ])
        for method in METHODS:
            fpr, ccr = oscr_curves[method]
            writer.writerows((method, percent_string(x), percent_string(y))
                             for x, y in zip(fpr, ccr))


def plot_curves(
    output: Path, risk_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    oscr_curves: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method in METHODS:
        coverage, risk = risk_curves[method]
        ax.plot(coverage * 100, risk * 100, label=method, linewidth=1.5)
    ax.set(xlabel="Coverage (%)", ylabel="Selective risk (%)", xlim=(0, 100), ylim=(0, None))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "risk_coverage.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method in METHODS:
        fpr, ccr = oscr_curves[method]
        ax.plot(fpr * 100, ccr * 100, label=method, linewidth=1.5)
    ax.set(xlabel="False acceptance rate among errors (%)",
           ylabel="Correct classification rate (%)", xlim=(0, 100), ylim=(0, 100))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "oscr_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    # Local imports keep numerical unit tests independent of timm/torchvision.
    from Classifier.data import Images, load_mapping
    from Classifier.model import build_model
    from Classifier.run import loader, restore_adapter, seed_all, sha256, write_json

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.pretrained is not None:
        cfg["pretrained"] = str(args.pretrained.resolve())
    if args.class_map is not None:
        cfg["class_map"] = str(args.class_map.resolve())
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("workers must be non-negative")
        cfg["workers"] = args.workers

    # Checkpoints contain RNG objects and must only come from a trusted local run.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_d_checkpoint_config(cfg, saved, args.checkpoint)
    best_selection = verify_best_selection(args.checkpoint, saved)
    mapping = load_mapping(cfg)
    if mapping != saved["mapping"]:
        raise ValueError("Class mapping differs from the D checkpoint")
    backbone_sha256 = sha256(cfg["pretrained"])
    if backbone_sha256 != saved["pretrained_sha256"]:
        raise ValueError("PlantCLEF backbone SHA256 differs from the D checkpoint")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_all(cfg["seed"])
    model, _ = build_model(cfg)
    restore_adapter(model, saved)
    model.to(device).eval()
    dataset = Images(args.split_dir, mapping, cfg["image_size"])

    labels_all: list[int] = []
    predictions_all: list[int] = []
    sample_ids: list[str] = []
    score_lists: OrderedDict[str, list[float]] = OrderedDict((name, []) for name in METHODS)
    with torch.inference_mode():
        for images, labels, ids in loader(dataset, cfg):
            # D's established evaluation path is FP32; no autocast is used here.
            logits = model(images.to(device, non_blocking=True)).float()
            predictions = logits.argmax(dim=1).cpu()
            scores = score_tensors(logits)
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())
            sample_ids.extend(ids)
            for method, values in scores.items():
                score_lists[method].extend(values.cpu().tolist())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    predictions_np = np.asarray(predictions_all, dtype=np.int64)
    errors = (labels_np != predictions_np).astype(np.int64)
    score_arrays = OrderedDict(
        (method, np.asarray(values, dtype=np.float64)) for method, values in score_lists.items()
    )
    rows, risk_curves, oscr_curves = evaluate_scores(errors, score_arrays)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_metrics(args.output_dir / "metrics.csv", rows)
    write_markdown(args.output_dir / "metrics.md", rows)
    write_score_table(args.output_dir / "sample_scores.csv", sample_ids, labels_np,
                      predictions_np, score_arrays)
    write_curve_tables(args.output_dir, risk_curves, oscr_curves)
    plot_curves(args.output_dir, risk_curves, oscr_curves)

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    accuracy = float((errors == 0).mean())
    run_info = {
        "protocol": "D-group training-free misclassification-detection baselines",
        "resolved_config": cfg,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch_one_based": int(saved["epoch"]) + 1,
        "checkpoint_best_val_top1_raw": float(saved["best"]),
        "checkpoint_best_val_top1_percent": percent_string(float(saved["best"])),
        "pretrained": str(Path(cfg["pretrained"]).resolve()),
        "pretrained_sha256": backbone_sha256,
        "class_map": str(Path(cfg["class_map"]).resolve()),
        "split_dir": str(args.split_dir.resolve()),
        "samples": int(len(errors)),
        "correct": int((errors == 0).sum()),
        "errors": int(errors.sum()),
        "accuracy_raw": accuracy,
        "accuracy_percent": percent_string(accuracy),
        "error_prevalence_raw": float(errors.mean()),
        "error_prevalence_percent": percent_string(float(errors.mean())),
        "positive_class": "error_label=1 (D-group top-1 prediction is wrong)",
        "score_direction": "larger means more likely to be wrong",
        "prediction_rule": "argmax(logits)",
        "inference_precision": "FP32",
        "reported_metric_scale": "metrics.csv/metrics.md and curve CSVs use raw value * 100",
        "reported_metric_rounding": "Decimal ROUND_HALF_UP to exactly two decimal places",
        "sample_score_scale": "raw method-specific values; deliberately not multiplied by 100",
        "energy_temperature": 1.0,
        "fpr_at_95_tpr": "linear interpolation on the error-detection ROC curve",
        "aurc": "mean tie-aware expected selective risk at coverages k/N, k=1..N",
        "oscr": (
            "closed-set adaptation: area under CCR versus false acceptance rate among wrong "
            "predictions; not standard open-set OSCR because no OOD set is used"
        ),
        "oscr_identity_check": "OSCR = D accuracy * AUROC (before percentage scaling)",
        "git_revision": revision,
        "best_selection_audit": best_selection,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "timm": importlib.metadata.version("timm"),
            "torchvision": importlib.metadata.version("torchvision"),
        },
        "methods": {
            "Max-Sigmoid": "1 - sigmoid(top1_logit)",
            "Sigmoid Margin": "-(sigmoid(top1_logit) - sigmoid(top2_logit))",
            "Raw-logit Margin": "-(top1_logit - top2_logit)",
            "OVR Self-BCE": (
                "mean one-vs-rest BCE using argmax(logits) as the pseudo-label; "
                "softplus(-top1) + sum_{j != top1} softplus(logit_j), divided by K"
            ),
            "DOCTOR": "(1 - sum_k softmax(logits)_k^2) / sum_k softmax(logits)_k^2",
            "Energy": "-logsumexp(logits), temperature=1",
            "L2Norm-MaxLogit": "-max_k(logit_k / ||logits||_2)",
        },
    }
    write_json(args.output_dir / "run_info.json", run_info)

    print(
        f"Evaluated {len(errors)} samples: accuracy={percent_string(accuracy)}%, "
        f"errors={errors.sum()}"
    )
    print((args.output_dir / "metrics.md").read_text(encoding="utf-8"))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
