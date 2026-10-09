"""Train/export controlled groups A, B, and C with seed fixed to 42.

Run from repository root, for example:
python -m Classifier.abc_comparison.run train --group A --output Classifier/outputs/abc_seed42/A
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Classifier.abc_comparison.model import build_model
from Classifier.data import Images, audit, load_mapping
from Classifier.losses import sigmoid_bce_loss, true_and_wrong
from Classifier.metrics import evaluate_arrays
from Classifier.run import (
    checkpoint, grad_norm, learning_rate_factor, loader, optimizer_step,
    restore_adapter, seed_all, sha256, write_json,
)


GROUPS = {
    "A": {"tuning_mode": "head_only", "loss": "ce", "native_confidence": "max_softmax"},
    "B": {"tuning_mode": "head_only", "loss": "l2", "native_confidence": "max_sigmoid"},
    "C": {"tuning_mode": "lora_head", "loss": "ce", "native_confidence": "max_softmax"},
}
FAILURE_KEYS = (
    "auroc_error", "aupr_error", "fpr_at_95_tpr", "aurc",
    "high_confidence_wrong", "low_confidence_correct",
    "correct_confidence_mean", "wrong_confidence_mean",
)


def resolve_config(path: Path, group: str) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.update(GROUPS[group], group=group)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["group"] not in GROUPS or cfg["seed"] != 42:
        raise ValueError("This protocol supports only groups A/B/C with seed=42")
    if any(cfg[key] <= 0 for key in ("epochs", "batch_size", "image_size", "num_classes")):
        raise ValueError("epochs, batch_size, image_size and num_classes must be positive")
    if cfg["epochs"] != 20 or cfg["batch_size"] != 32 or cfg["lr_warmup"] != 2:
        raise ValueError("Controlled protocol is fixed at epochs=20, batch_size=32, lr_warmup=2")
    if cfg["loss"] not in ("ce", "l2") or cfg["bce_reduction"] != "mean":
        raise ValueError("Only CE and mean-over-elements L2 are allowed")
    if cfg["workers"] < 0:
        raise ValueError("workers must be non-negative")


def loss_values(logits: torch.Tensor, labels: torch.Tensor, cfg: dict):
    ce = F.cross_entropy(logits.float(), labels)
    l2 = sigmoid_bce_loss(logits, labels, cfg["bce_reduction"])
    selected = ce if cfg["loss"] == "ce" else l2
    return selected, ce, l2


def confidence_values(logits: torch.Tensor):
    top = logits.topk(min(5, logits.shape[1]), dim=1)
    max_softmax = logits.softmax(1).max(1).values
    max_sigmoid = logits.sigmoid().max(1).values
    sigmoid_margin = (top.values[:, 0] - top.values[:, 1]).sigmoid()
    return top, {
        "max_softmax": max_softmax,
        "max_sigmoid": max_sigmoid,
        "sigmoid_logit_margin": sigmoid_margin,
    }


def make_optimizer(model, cfg):
    head_ids = {id(parameter) for parameter in model.head.parameters()}
    groups = []
    adaptation = [parameter for parameter in model.parameters()
                  if parameter.requires_grad and id(parameter) not in head_ids]
    if cfg["tuning_mode"] == "head_only" and adaptation:
        raise RuntimeError("Head-only group contains trainable backbone parameters")
    if cfg["tuning_mode"] == "lora_head":
        if not adaptation:
            raise RuntimeError("LoRA group has no trainable adaptation parameters")
        groups.append({"params": adaptation, "lr": cfg["lr"], "initial_lr": cfg["lr"],
                       "name": "lora"})
    groups.append({"params": list(model.head.parameters()), "lr": cfg["head_lr"],
                   "initial_lr": cfg["head_lr"], "name": "head"})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def train_epoch(model, dataset, optimizer, scaler, device, cfg, epoch, output):
    model.train()  # Matches the existing D implementation exactly.
    batches = loader(dataset, cfg, shuffle=True, epoch=epoch)
    totals = {"loss": 0.0, "ce": 0.0, "l2": 0.0, "correct": 0, "samples": 0}
    skipped_steps = consecutive_skips = 0
    for step, (images, labels, _) in enumerate(batches):
        progress = epoch + (step + 1) / len(batches)
        factor = learning_rate_factor(cfg, progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * factor
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda",
                            dtype=torch.float16):
            logits = model(images)
        selected, ce, l2 = loss_values(logits.float(), labels, cfg)
        if not torch.isfinite(selected):
            raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, step={step}")
        if step < cfg["diagnostic_batches"]:
            row = {
                "epoch": epoch, "step": step, "selected_loss": selected.item(),
                "ce": ce.item(), "l2": l2.item(),
                "head_grad_ce": grad_norm(ce, tuple(model.head.parameters())),
                "head_grad_l2": grad_norm(l2, tuple(model.head.parameters())),
                "correct": int(logits.argmax(1).eq(labels).sum()), "batch": len(labels),
            }
            with (output / "diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        scaler.scale(selected).backward()
        update = optimizer_step(model, optimizer, scaler, cfg["grad_clip"])
        if update["skipped"]:
            skipped_steps += 1
            consecutive_skips += 1
            with (output / "amp_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "step": step, **update}) + "\n")
            print(f"AMP overflow: epoch={epoch+1} step={step}; update skipped, "
                  f"scale {update['scale_before']} -> {update['scale_after']}", flush=True)
            if consecutive_skips >= 20:
                raise FloatingPointError("20 consecutive AMP overflows")
        else:
            consecutive_skips = 0
        n = len(labels)
        totals["loss"] += selected.item() * n
        totals["ce"] += ce.item() * n
        totals["l2"] += l2.item() * n
        totals["correct"] += int(logits.argmax(1).eq(labels).sum())
        totals["samples"] += n
        if step % 50 == 0:
            print(f"group {cfg['group']} epoch {epoch+1} step {step}/{len(batches)} "
                  f"loss={selected.item():.5f}", flush=True)
    if skipped_steps == len(batches):
        raise FloatingPointError("No successful optimizer updates in this epoch")
    return {
        **{key: totals[key] / totals["samples"] for key in ("loss", "ce", "l2", "correct")},
        "amp_skipped_steps": skipped_steps,
        "optimizer_steps": len(batches) - skipped_steps,
        "amp_scale": scaler.get_scale(),
    }


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, counts, export=None):
    model.eval()
    labels_all, predictions_all, top5_all = [], [], []
    confidence_all = {name: [] for name in ("max_softmax", "max_sigmoid", "sigmoid_logit_margin")}
    loss_sums = {"loss": 0.0, "ce": 0.0, "l2": 0.0}
    offset = 0
    logits_store = probabilities_store = sample_handle = None
    if export is not None:
        export.mkdir(parents=True, exist_ok=False)
        shape = (len(dataset), cfg["num_classes"])
        logits_store = np.lib.format.open_memmap(
            export / "logits.npy", mode="w+", dtype="float32", shape=shape
        )
        probabilities_store = np.lib.format.open_memmap(
            export / "probabilities.npy", mode="w+", dtype="float32", shape=shape
        )
        sample_handle = (export / "samples.csv").open("w", newline="", encoding="utf-8")
        writer = csv.writer(sample_handle)
        writer.writerow([
            "row", "sample_id", "ground_truth", "prediction", "correct",
            "max_softmax", "max_sigmoid", "sigmoid_logit_margin",
            "top1_logit", "top2_logit", "top1_top2_logit_margin",
            "ground_truth_logit", "max_wrong_class_logit",
            "ground_truth_vs_max_wrong_margin", "class_frequency_group",
        ])
    try:
        for images, labels, ids in loader(dataset, cfg):
            images, labels = images.to(device), labels.to(device)
            logits = model(images).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            selected, ce, l2 = loss_values(logits, labels, cfg)
            top, scores = confidence_values(logits)
            prediction = logits.argmax(1)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(prediction.cpu().tolist())
            top5_all.extend(top.indices.eq(labels[:, None]).any(1).cpu().tolist())
            for name, score in scores.items():
                confidence_all[name].extend(score.cpu().tolist())
            n = len(labels)
            for name, value in (("loss", selected), ("ce", ce), ("l2", l2)):
                loss_sums[name] += value.item() * n
            if export is not None:
                logits_store[offset:offset+n] = logits.cpu().numpy()
                native_probabilities = logits.softmax(1) if cfg["loss"] == "ce" else logits.sigmoid()
                probabilities_store[offset:offset+n] = native_probabilities.cpu().numpy()
                true, wrong = true_and_wrong(logits, labels)
                for index, sample_id in enumerate(ids):
                    label = int(labels[index])
                    frequency = counts[label]
                    group = "head" if frequency > 100 else "medium" if frequency >= 20 else "tail"
                    writer.writerow([
                        offset + index, sample_id, label, int(prediction[index]),
                        int(prediction[index] == labels[index]),
                        float(scores["max_softmax"][index]),
                        float(scores["max_sigmoid"][index]),
                        float(scores["sigmoid_logit_margin"][index]),
                        float(top.values[index, 0]), float(top.values[index, 1]),
                        float(top.values[index, 0] - top.values[index, 1]),
                        float(true[index]), float(wrong[index]), float(true[index] - wrong[index]), group,
                    ])
                offset += n
    finally:
        if sample_handle is not None:
            sample_handle.close()
            logits_store.flush()
            probabilities_store.flush()

    score_metrics = {}
    risks = {}
    groups = None
    for name, confidence in confidence_all.items():
        metric, risk, group_labels = evaluate_arrays(
            labels_all, predictions_all, confidence, top5_all, counts, cfg["tau"]
        )
        score_metrics[name] = metric
        risks[name] = risk
        groups = group_labels
    native = score_metrics[cfg["native_confidence"]]
    metrics = {**native, **{name: value / len(dataset) for name, value in loss_sums.items()},
               "group": cfg["group"], "tuning_mode": cfg["tuning_mode"],
               "training_loss": cfg["loss"], "native_confidence": cfg["native_confidence"],
               "confidence_metrics": {name: {key: metric.get(key) for key in FAILURE_KEYS}
                                      for name, metric in score_metrics.items()}}
    if export is not None:
        np.savez(
            export / "risk_coverage.npz",
            coverage=np.arange(1, len(dataset)+1) / len(dataset),
            **{name: risk for name, risk in risks.items()},
        )
        labels_np, predictions_np = np.asarray(labels_all), np.asarray(predictions_all)
        supports = np.bincount(labels_np, minlength=len(counts))
        hits = np.bincount(labels_np[labels_np == predictions_np], minlength=len(counts))
        with (export / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_index", "train_count", "group", "support", "accuracy"])
            for index in range(len(counts)):
                accuracy = hits[index] / supports[index] if supports[index] else ""
                writer.writerow([index, counts[index], groups[index], supports[index], accuracy])
        write_json(export / "metrics.json", metrics)
        write_json(export / "complete.json", {
            "rows": offset, "columns": cfg["num_classes"], "dtype": "float32",
            "probability_transform": "softmax" if cfg["loss"] == "ce" else "sigmoid",
            "prediction_rule": "argmax(logits)",
        })
    return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("train", "export"))
    parser.add_argument("--group", required=True, choices=tuple(GROUPS))
    parser.add_argument("--config", type=Path,
                        default=Path("Classifier/abc_comparison/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process/GPU per experiment")
    cfg = resolve_config(args.config, args.group)
    seed_all(cfg["seed"])
    mapping = load_mapping(cfg)
    saved = None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        changed = {key for key in cfg if cfg[key] != saved["config"].get(key)} - {
            "train_dir", "val_dir", "class_map", "pretrained", "workers"
        }
        if changed or mapping != saved["mapping"]:
            raise ValueError(f"Checkpoint/config mismatch: {sorted(changed)}")

    if args.mode == "export":
        if saved is None or args.split_dir is None:
            parser.error("export requires --checkpoint and --split-dir")
        stats = saved["stats"]
    else:
        train = Images(cfg["train_dir"], mapping, cfg["image_size"], train=True)
        val = Images(cfg["val_dir"], mapping, cfg["image_size"])
        stats = audit(train, val)
        if saved and stats != saved["stats"]:
            raise ValueError("Dataset changed since checkpoint")

    weight_hash = sha256(cfg["pretrained"])
    if saved and saved["pretrained_sha256"] != weight_hash:
        raise ValueError("Pretrained weight SHA256 mismatch")
    model, report = build_model(cfg)
    device = torch.device(args.device)
    model.to(device)
    if saved:
        restore_adapter(model, saved)

    if args.mode == "export":
        dataset = Images(args.split_dir, mapping, cfg["image_size"])
        metrics = evaluate(model, dataset, cfg, device, stats["train_counts"], args.output)
        write_json(args.output / "provenance.json", {
            "config": cfg, "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": weight_hash,
            "split_dir": str(args.split_dir.resolve()), "class_to_idx": mapping,
        })
        print(json.dumps(metrics, indent=2))
        return

    print(
        f"Group {cfg['group']}: tuning={cfg['tuning_mode']}, loss={cfg['loss']}, "
        f"confidence={cfg['native_confidence']}, seed=42, epochs={cfg['epochs']}, "
        f"batch_size={cfg['batch_size']}, train_samples={stats['train_samples']}, "
        f"batches_per_epoch={math.ceil(stats['train_samples']/cfg['batch_size'])}",
        flush=True,
    )
    if saved is None:
        args.output.mkdir(parents=True, exist_ok=False)
    elif args.checkpoint.resolve() != (args.output / "last.pt").resolve():
        raise ValueError("Resume only from output/last.pt in the original run directory")
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    start, best = 0, -1.0
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best = saved["epoch"] + 1, saved["best"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        for name in ("history.jsonl", "diagnostics.jsonl", "amp_events.jsonl"):
            path = args.output / name
            if path.exists():
                rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                        if json.loads(line)["epoch"] < start]
                path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    else:
        import timm
        import torchvision
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        write_json(args.output / "config.json", cfg)
        write_json(args.output / "data_audit.json", stats)
        write_json(args.output / "class_to_idx.json", mapping)
        write_json(args.output / "model_report.json", {
            **report, "pretrained_sha256": weight_hash,
            "trainable_fraction": report["trainable_parameters"] / report["total_parameters"],
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "git_revision": revision,
        })

    for epoch in range(start, cfg["epochs"]):
        train_metrics = train_epoch(
            model, train, optimizer, scaler, device, cfg, epoch, args.output
        )
        val_metrics = evaluate(model, val, cfg, device, stats["train_counts"])
        improved = val_metrics["top1"] > best
        best = max(best, val_metrics["top1"])
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": epoch, "train": train_metrics,
                "val": val_metrics, "best_top1": best,
            }, allow_nan=False) + "\n")
        checkpoint(
            args.output / "last.pt", model, optimizer, scaler,
            cfg, epoch, best, mapping, stats, weight_hash,
        )
        if improved:
            checkpoint(
                args.output / "best.pt", model, optimizer, scaler,
                cfg, epoch, best, mapping, stats, weight_hash,
            )
        print(f"group {cfg['group']} epoch {epoch+1}: val top1={val_metrics['top1']:.5f} "
              f"best={best:.5f}", flush=True)


if __name__ == "__main__":
    main()

