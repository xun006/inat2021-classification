#!/usr/bin/env python3
"""Train one B1/B2/B3/M1/M2 detector with online frozen Patch extraction.

This development-stage script only accepts detector_train and
detector_calibration. It never loads official_val.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import DetectorDataset
from models import (
    MODEL_NAMES, build_detector, effective_classifier_parameters, trainable_parameter_count,
)
from teacher import DEFAULT_CHECKPOINT, DEFAULT_CLASS_MAP, MISD_ROOT, build_frozen_teacher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument(
        "--predictions-dir", type=Path,
        default=MISD_ROOT / "output/patch_cross_attention/teacher_predictions",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-calibration-samples", type=int, default=None)
    parser.add_argument("--balanced-debug-train", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, drop_last=False,
    )


def evaluate(teacher, detector, loader, device, amp):
    detector.eval()
    labels, scores, sample_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            candidates = batch["candidate_index"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                teacher_output = teacher(images)
                logits = detector(
                    teacher_output["p_norm"], candidates,
                    global_feature=teacher_output["global_feature"],
                )["error_logit"]
            labels.append(batch["error_label"].numpy())
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    labels = np.concatenate(labels).astype(np.int64)
    scores = np.concatenate(scores)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "error_auprc": float(average_precision_score(labels, scores)),
    }, labels, scores, sample_ids


def main():
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or (
        MISD_ROOT / f"output/patch_cross_attention/ablation/{args.model}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = DetectorDataset(
        args.data_root, "detector_train", args.predictions_dir / "detector_train.csv",
        args.class_map, args.max_train_samples, args.balanced_debug_train, args.seed,
    )
    calibration_set = DetectorDataset(
        args.data_root, "detector_calibration",
        args.predictions_dir / "detector_calibration.csv", args.class_map,
        args.max_calibration_samples, False, args.seed,
    )
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers, args.seed)
    calibration_loader = make_loader(
        calibration_set, args.batch_size * 2, False, args.num_workers, args.seed
    )
    teacher = build_frozen_teacher(args.checkpoint, args.class_map, device)
    uses_effective_weights = args.model in {"m1_effective", "m2_effective"}
    if uses_effective_weights:
        classifier_weights, effective_bias = effective_classifier_parameters(
            teacher.model.head[0], teacher.model.head[1]
        )
    else:
        classifier_weights = teacher.model.head[1].weight.detach()
        effective_bias = None
    detector = build_detector(args.model, classifier_weights, dropout=args.dropout).to(device)
    parameters = trainable_parameter_count(detector)
    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_labels = train_set.frame.error_label.to_numpy(dtype=np.int64)
    errors = int(train_labels.sum())
    if errors == 0 or errors == len(train_labels):
        raise ValueError("training subset must contain both error labels")
    pos_weight = (len(train_labels) - errors) / errors
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_auprc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        detector.train()
        loss_sum = seen = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            candidates = batch["candidate_index"].to(device, non_blocking=True)
            targets = batch["error_label"].to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                teacher_output = teacher(images)
                error_logits = detector(
                    teacher_output["p_norm"], candidates,
                    global_feature=teacher_output["global_feature"],
                )["error_logit"]
                loss = criterion(error_logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss) * len(targets); seen += len(targets)
        metrics, _, _, _ = evaluate(teacher, detector, calibration_loader, device, amp)
        row = {"epoch": epoch, "train_loss": loss_sum / seen,
               "calibration_auroc": metrics["auroc"],
               "calibration_error_auprc": metrics["error_auprc"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["error_auprc"] > best_auprc + 1e-6:
            best_auprc, best_epoch = metrics["error_auprc"], epoch
            best_state = copy.deepcopy(detector.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no best checkpoint was selected")
    detector.load_state_dict(best_state)
    final_metrics, labels, scores, sample_ids = evaluate(
        teacher, detector, calibration_loader, device, amp
    )
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    pd.DataFrame({"sample_id": sample_ids, "error_label": labels,
                  "error_score": scores}).to_csv(
        output_dir / "calibration_predictions.csv", index=False
    )
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "output_dir": str(output_dir.resolve()), "train_split": "detector_train",
        "selection_split": "detector_calibration", "official_val_used": False,
        "selection_metric": "calibration_error_auprc", "best_epoch": best_epoch,
        "best_calibration_error_auprc": best_auprc,
        "trainable_parameters": parameters, "pos_weight": pos_weight,
        "patch_definition": "fc_norm(final_block_tokens_without_cls)",
        "global_feature_definition": "fc_norm(mean(final_block_tokens_without_cls))",
        "classifier_query_weight": (
            "BN-effective weight W*rsqrt(running_var+eps)"
            if uses_effective_weights else "raw head.1.weight"
        ),
        "query_residual": False, "score_direction": "larger means more likely error",
    })
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    pd.DataFrame([{**final_metrics, "best_epoch": best_epoch,
                   "trainable_parameters": parameters}]).to_csv(
        output_dir / "calibration_metrics.csv", index=False
    )
    torch.save({"model": best_state, "config": config}, output_dir / "checkpoint_best.pth")
    print(f"best epoch={best_epoch}; calibration={final_metrics}; output={output_dir}")


if __name__ == "__main__":
    main()
