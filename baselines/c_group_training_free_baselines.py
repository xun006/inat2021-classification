#!/usr/bin/env python3
"""Evaluate seven training-free failure scores with the trained C-group model.

The positive class is a wrong top-1 prediction. Every exported score is oriented
so that a larger value means "more likely to be wrong". Run this module from the
repository root so that imports from ``Classifier`` resolve consistently.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METHODS = (
    "MSP",
    "Probability Margin",
    "Entropy",
    "DOCTOR",
    "Raw-logit Margin",
    "MaxLogit",
    "Energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="C run config.json.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="C run best.pt. Non-best filenames are rejected.")
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


def validate_checkpoint_config(cfg: dict, saved: dict, checkpoint: Path) -> None:
    if checkpoint.name != "best.pt":
        raise ValueError(
            f"Expected the validation-selected C checkpoint named best.pt, got {checkpoint.name!r}"
        )
    expected = {"group": "C", "tuning_mode": "lora_head", "loss": "ce",
                "native_confidence": "max_softmax", "seed": 42}
    for key, value in expected.items():
        if cfg.get(key) != value or saved.get("config", {}).get(key) != value:
            raise ValueError(f"Not a valid C-group checkpoint: expected {key}={value!r}")
    changed = {key for key in cfg if cfg[key] != saved["config"].get(key)} - {
        "train_dir", "val_dir", "class_map", "pretrained", "workers"
    }
    if changed:
        raise ValueError(f"Checkpoint/config mismatch: {sorted(changed)}")


def verify_best_selection(checkpoint: Path, saved: dict) -> dict[str, object]:
    """Verify best-by-validation selection when the run history is available."""
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
            "best.pt does not match the earliest highest classifier_val Top-1 epoch in history.jsonl"
        )
    return {
        "verified": True,
        "history": str(history_path.resolve()),
        "selected_epoch_one_based": best_epoch + 1,
        "selected_val_top1": best_value,
    }


def score_tensors(logits: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    """Return training-free error scores; larger always means more error-like."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [N, K] with K >= 2")
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits encountered")
    log_probabilities = logits.log_softmax(dim=1)
    probabilities = log_probabilities.exp()
    top_probability = probabilities.topk(2, dim=1).values
    top_logit = logits.topk(2, dim=1).values
    collision_probability = probabilities.square().sum(dim=1)
    doctor = (1.0 - collision_probability) / collision_probability.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    return OrderedDict([
        ("MSP", 1.0 - top_probability[:, 0]),
        ("Probability Margin", -(top_probability[:, 0] - top_probability[:, 1])),
        ("Entropy", -(probabilities * log_probabilities).sum(dim=1)),
        ("DOCTOR", doctor),
        ("Raw-logit Margin", -(top_logit[:, 0] - top_logit[:, 1])),
        ("MaxLogit", -top_logit[:, 0]),
        ("Energy", -torch.logsumexp(logits, dim=1)),
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
        ranks_within_tie = np.arange(1, end - start + 1, dtype=np.float64)
        expected_errors = previous_errors + ranks_within_tie * group_errors / (end - start)
        risk[start:end] = expected_errors / np.arange(start + 1, end + 1)
        previous_errors += group_errors
        start = end
    coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
    return coverage, risk, float(risk.mean())


def closed_set_oscr_curve(
    errors: np.ndarray, error_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-set OSCR adaptation: incorrect predictions serve as negatives.

    FPR is the fraction of wrong predictions accepted at a confidence threshold.
    CCR is the fraction of all samples that are both accepted and correctly
    classified. This is not the standard open-set OSCR, which requires OOD data.
    """
    confidence = -error_scores
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_errors = errors[order].astype(bool)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_confidence)) + 1, len(errors)]
    total_wrong = int(errors.sum())
    if total_wrong == 0 or total_wrong == len(errors):
        raise ValueError("OSCR and binary failure metrics require correct and wrong predictions")
    fpr = [0.0]
    ccr = [0.0]
    accepted_wrong = 0
    accepted_correct = 0
    start = 0
    for end in group_ends:
        group = sorted_errors[start:end]
        accepted_wrong += int(group.sum())
        accepted_correct += int((~group).sum())
        fpr.append(accepted_wrong / total_wrong)
        ccr.append(accepted_correct / len(errors))
        start = end
    fpr_array = np.asarray(fpr, dtype=np.float64)
    ccr_array = np.asarray(ccr, dtype=np.float64)
    widths = np.diff(fpr_array)
    area = np.sum(widths * (ccr_array[:-1] + ccr_array[1:]) * 0.5)
    return fpr_array, ccr_array, float(area)


def evaluate_scores(
    errors: np.ndarray, score_arrays: OrderedDict[str, np.ndarray]
) -> tuple[list[dict[str, float | str]], dict[str, tuple[np.ndarray, np.ndarray]],
           dict[str, tuple[np.ndarray, np.ndarray]]]:
    errors = np.asarray(errors, dtype=np.int64)
    if errors.ndim != 1 or len(errors) == 0 or set(np.unique(errors)) != {0, 1}:
        raise ValueError("Evaluation requires a non-empty error vector containing both 0 and 1")
    rows = []
    risk_curves = {}
    oscr_curves = {}
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


def write_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = ["method", "auroc", "error_auprc", "fpr_at_95_tpr", "aurc", "oscr"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = [
        "| Method | AUROC ↑ | Error AUPRC ↑ | FPR@95TPR ↓ | AURC ↓ | OSCR ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [str(row["method"])] + [f"{float(row[key]):.6f}" for key in (
            "auroc", "error_auprc", "fpr_at_95_tpr", "aurc", "oscr"
        )]
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
        writer.writerow(["method", "coverage", "selective_risk"])
        for method in METHODS:
            coverage, risk = risk_curves[method]
            writer.writerows((method, float(x), float(y)) for x, y in zip(coverage, risk))
    with (output / "oscr_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "false_accept_rate_among_errors", "correct_classification_rate"])
        for method in METHODS:
            fpr, ccr = oscr_curves[method]
            writer.writerows((method, float(x), float(y)) for x, y in zip(fpr, ccr))


def plot_curves(
    output: Path, risk_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    oscr_curves: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method in METHODS:
        ax.plot(*risk_curves[method], label=method, linewidth=1.5)
    ax.set(xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1), ylim=(0, None))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "risk_coverage.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method in METHODS:
        ax.plot(*oscr_curves[method], label=method, linewidth=1.5)
    ax.set(xlabel="False acceptance rate among errors", ylabel="Correct classification rate",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "oscr_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    # Keep model/dataset imports local so the metric unit tests do not require
    # timm or torchvision merely to validate the numerical definitions.
    from Classifier.abc_comparison.model import build_model
    from Classifier.abc_comparison.run import resolve_config
    from Classifier.data import Images, load_mapping
    from Classifier.run import loader, restore_adapter, seed_all, sha256, write_json

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    cfg = resolve_config(args.config, "C")
    if args.pretrained is not None:
        cfg["pretrained"] = str(args.pretrained.resolve())
    if args.class_map is not None:
        cfg["class_map"] = str(args.class_map.resolve())
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("workers must be non-negative")
        cfg["workers"] = args.workers

    # Checkpoints contain Python RNG objects and therefore require trusted pickle loading.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_config(cfg, saved, args.checkpoint)
    best_selection = verify_best_selection(args.checkpoint, saved)
    mapping = load_mapping(cfg)
    if mapping != saved["mapping"]:
        raise ValueError("Class mapping differs from the C checkpoint")
    backbone_sha256 = sha256(cfg["pretrained"])
    if backbone_sha256 != saved["pretrained_sha256"]:
        raise ValueError("PlantCLEF backbone SHA256 differs from the C checkpoint")

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
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda",
                                dtype=torch.float16):
                logits = model(images)
            logits = logits.float()
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
    accuracy = float((~errors.astype(bool)).mean())
    run_info = {
        "protocol": "C-group training-free misclassification-detection baselines",
        "resolved_config": cfg,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch_one_based": int(saved["epoch"]) + 1,
        "checkpoint_best_val_top1": float(saved["best"]),
        "pretrained": str(Path(cfg["pretrained"]).resolve()),
        "pretrained_sha256": backbone_sha256,
        "class_map": str(Path(cfg["class_map"]).resolve()),
        "split_dir": str(args.split_dir.resolve()),
        "samples": int(len(errors)),
        "correct": int((errors == 0).sum()),
        "errors": int(errors.sum()),
        "accuracy": accuracy,
        "error_prevalence": float(errors.mean()),
        "positive_class": "error_label=1 (C-group top-1 prediction is wrong)",
        "score_direction": "larger means more likely to be wrong",
        "prediction_rule": "argmax(logits)",
        "energy_temperature": 1.0,
        "fpr_at_95_tpr": "linear interpolation on the error-detection ROC curve",
        "aurc": "mean tie-aware expected selective risk at coverages k/N, k=1..N",
        "oscr": (
            "closed-set adaptation: area under CCR versus false acceptance rate among wrong "
            "predictions; not standard open-set OSCR because no OOD set is used"
        ),
        "oscr_identity_check": "for this definition OSCR = accuracy * AUROC (up to rounding)",
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
            "MSP": "1 - max(softmax(logits))",
            "Probability Margin": "-(top1_probability - top2_probability)",
            "Entropy": "-sum_k p_k log(p_k)",
            "DOCTOR": "(1 - sum_k p_k^2) / sum_k p_k^2 (DOCTOR-alpha)",
            "Raw-logit Margin": "-(top1_logit - top2_logit)",
            "MaxLogit": "-top1_logit",
            "Energy": "-logsumexp(logits), temperature=1",
        },
    }
    write_json(args.output_dir / "run_info.json", run_info)

    print(f"Evaluated {len(errors)} samples: accuracy={accuracy:.6f}, errors={errors.sum()}")
    print((args.output_dir / "metrics.md").read_text(encoding="utf-8"))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
