"""One training/evaluation engine for the complete formal classifier matrix."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .checkpoint import load as load_checkpoint
from .checkpoint import restore_model, restore_rng, save as save_checkpoint
from .data import Images, audit, load_mapping
from .losses import sigmoid_bce_loss, true_and_wrong
from .metrics import evaluate_arrays
from .model import build_model
from .protocol import GROUPS, apply_group, validate_fixed_protocol


FAILURE_KEYS = ("auroc_error", "aupr_error", "fpr_at_95_tpr", "aurc",
                "high_confidence_wrong", "low_confidence_correct",
                "correct_confidence_mean", "wrong_confidence_mean")
TRAINING_SIGNATURE_KEYS = (
    "model", "global_pool", "image_size", "num_classes", "rank", "alpha",
    "lora_dropout", "epochs", "batch_size", "lr", "head_lr", "weight_decay",
    "lr_warmup", "grad_clip", "amp", "bce_reduction", "group", "seed",
    "initialization", "tuning_mode", "loss", "native_confidence", "protocol",
)


def write_json(path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(_) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, cfg, shuffle=False, epoch=0):
    return DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=shuffle,
        num_workers=cfg["workers"], pin_memory=True, worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(cfg["seed"] + epoch),
    )


def resolve_config(path: Path, group: str, seed: int | None = None) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    selected_seed = raw.get("seed", 42) if seed is None else seed
    if "protocols" in raw:
        cfg = apply_group(raw, group, selected_seed)
    else:
        cfg = dict(raw)
        cfg.update(GROUPS[group], group=group, seed=selected_seed)
        if cfg["initialization"] == "plantclef":
            cfg["pretrained"] = cfg.get("plantclef_pretrained", cfg.get("pretrained"))
        elif cfg["initialization"] == "imagenet":
            cfg["pretrained"] = cfg.get("imagenet_pretrained", cfg.get("pretrained"))
        else:
            cfg["pretrained"] = None
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg.get("group") not in GROUPS:
        raise ValueError(f"Unknown group: {cfg.get('group')}")
    expected = GROUPS[cfg["group"]]
    altered = {key for key, value in expected.items() if cfg.get(key) != value}
    if altered:
        raise ValueError(f"Group definition changed: {sorted(altered)}")
    validate_fixed_protocol(cfg)
    if not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if cfg["workers"] < 0 or cfg["diagnostic_batches"] < 0:
        raise ValueError("workers and diagnostic_batches must be non-negative")
    if cfg["initialization"] != "random" and not cfg.get("pretrained"):
        raise ValueError("Pretrained initialization requires a checkpoint path")


def loss_values(logits, labels, cfg):
    ce = F.cross_entropy(logits.float(), labels)
    bce = sigmoid_bce_loss(logits, labels, cfg["bce_reduction"])
    return (ce if cfg["loss"] == "ce" else bce), ce, bce


def confidence_values(logits):
    top = logits.topk(min(5, logits.shape[1]), dim=1)
    return top, {
        "max_softmax": logits.softmax(1).max(1).values,
        "max_sigmoid": logits.sigmoid().max(1).values,
        "sigmoid_logit_margin": (top.values[:, 0] - top.values[:, 1]).sigmoid(),
    }


def learning_rate_factor(cfg, progress):
    if progress < cfg["lr_warmup"]:
        return progress / max(1, cfg["lr_warmup"])
    return 0.5 * (1 + math.cos(
        math.pi * (progress - cfg["lr_warmup"]) / (cfg["epochs"] - cfg["lr_warmup"])
    ))


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum(
        (grad.detach().float().square().sum() for grad in grads if grad is not None),
        loss.new_zeros(()),
    )).item())


def optimizer_step(model, optimizer, scaler, max_norm):
    scaler.unscale_(optimizer)
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad and parameter.grad is not None]
    if not parameters:
        raise RuntimeError("No gradients found")
    finite = bool(torch.stack([torch.isfinite(parameter.grad).all()
                               for parameter in parameters]).all())
    before = scaler.get_scale()
    if not finite and not scaler.is_enabled():
        raise FloatingPointError("Nonfinite gradients without GradScaler")
    if finite:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return {"skipped": not finite, "scale_before": before,
            "scale_after": scaler.get_scale()}


def make_optimizer(model, cfg):
    head_ids = {id(parameter) for parameter in model.head.parameters()}
    adaptation = [parameter for parameter in model.parameters()
                  if parameter.requires_grad and id(parameter) not in head_ids]
    groups = []
    if cfg["tuning_mode"] == "head_only" and adaptation:
        raise RuntimeError("Head-only group contains trainable backbone parameters")
    if cfg["tuning_mode"] != "head_only":
        if not adaptation:
            raise RuntimeError("No trainable adapter/backbone parameters")
        groups.append({"params": adaptation, "lr": cfg["lr"],
                       "initial_lr": cfg["lr"],
                       "name": "lora" if cfg["tuning_mode"] == "lora_head" else "backbone"})
    groups.append({"params": list(model.head.parameters()), "lr": cfg["head_lr"],
                   "initial_lr": cfg["head_lr"], "name": "head"})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def train_epoch(model, dataset, optimizer, scaler, device, cfg, epoch, output,
                max_batches=None):
    model.train()
    batches = loader(dataset, cfg, shuffle=True, epoch=epoch)
    totals = {"loss": 0.0, "ce": 0.0, "l2": 0.0, "correct": 0, "samples": 0}
    skipped = consecutive = processed = 0
    for step, (images, labels, _) in enumerate(batches):
        factor = learning_rate_factor(cfg, epoch + (step + 1) / len(batches))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = parameter_group["initial_lr"] * factor
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda",
                            dtype=torch.float16):
            logits = model(images)
        selected, ce, bce = loss_values(logits.float(), labels, cfg)
        if not torch.isfinite(selected):
            raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, step={step}")
        if step < cfg["diagnostic_batches"]:
            row = {
                "epoch": epoch, "step": step, "selected_loss": selected.item(),
                "ce": ce.item(), "l2": bce.item(),
                "head_grad_ce": grad_norm(ce, tuple(model.head.parameters())),
                "head_grad_l2": grad_norm(bce, tuple(model.head.parameters())),
                "correct": int(logits.argmax(1).eq(labels).sum()), "batch": len(labels),
            }
            with (output / "diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        scaler.scale(selected).backward()
        update = optimizer_step(model, optimizer, scaler, cfg["grad_clip"])
        if update["skipped"]:
            skipped += 1
            consecutive += 1
            with (output / "amp_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "step": step, **update}) + "\n")
            if consecutive >= 20:
                raise FloatingPointError("20 consecutive AMP overflows")
        else:
            consecutive = 0
        n = len(labels)
        totals["loss"] += selected.item() * n
        totals["ce"] += ce.item() * n
        totals["l2"] += bce.item() * n
        totals["correct"] += int(logits.argmax(1).eq(labels).sum())
        totals["samples"] += n
        processed += 1
        if step % 50 == 0:
            print(f"group {cfg['group']} epoch {epoch+1} step {step}/{len(batches)} "
                  f"loss={selected.item():.5f}", flush=True)
        if max_batches is not None and processed >= max_batches:
            break
    if skipped == processed:
        raise FloatingPointError("No successful optimizer updates in this epoch")
    return {**{key: totals[key] / totals["samples"]
               for key in ("loss", "ce", "l2", "correct")},
            "amp_skipped_steps": skipped, "optimizer_steps": processed - skipped,
            "processed_steps": processed, "amp_scale": scaler.get_scale()}


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, counts, export=None):
    model.eval()
    labels_all, predictions_all, top5_all = [], [], []
    confidence_all = {name: [] for name in
                      ("max_softmax", "max_sigmoid", "sigmoid_logit_margin")}
    loss_sums = {"loss": 0.0, "ce": 0.0, "l2": 0.0}
    offset = 0
    logits_store = probabilities_store = sample_handle = None
    if export is not None:
        export.mkdir(parents=True, exist_ok=False)
        shape = (len(dataset), cfg["num_classes"])
        logits_store = np.lib.format.open_memmap(
            export / "logits.npy", mode="w+", dtype="float32", shape=shape)
        probabilities_store = np.lib.format.open_memmap(
            export / "probabilities.npy", mode="w+", dtype="float32", shape=shape)
        sample_handle = (export / "samples.csv").open("w", newline="", encoding="utf-8")
        writer = csv.writer(sample_handle)
        writer.writerow([
            "row", "sample_id", "ground_truth", "prediction", "correct",
            "max_softmax", "max_sigmoid", "sigmoid_logit_margin", "top1_logit",
            "top2_logit", "top1_top2_logit_margin", "ground_truth_logit",
            "max_wrong_class_logit", "ground_truth_vs_max_wrong_margin",
            "class_frequency_group",
        ])
    try:
        for images, labels, ids in loader(dataset, cfg):
            images, labels = images.to(device), labels.to(device)
            logits = model(images).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            selected, ce, bce = loss_values(logits, labels, cfg)
            top, scores = confidence_values(logits)
            prediction = logits.argmax(1)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(prediction.cpu().tolist())
            top5_all.extend(top.indices.eq(labels[:, None]).any(1).cpu().tolist())
            for name, score in scores.items():
                confidence_all[name].extend(score.cpu().tolist())
            n = len(labels)
            for name, value in (("loss", selected), ("ce", ce), ("l2", bce)):
                loss_sums[name] += value.item() * n
            if export is not None:
                logits_store[offset:offset+n] = logits.cpu().numpy()
                probabilities = logits.softmax(1) if cfg["loss"] == "ce" else logits.sigmoid()
                probabilities_store[offset:offset+n] = probabilities.cpu().numpy()
                true, wrong = true_and_wrong(logits, labels)
                for index, sample_id in enumerate(ids):
                    label = int(labels[index])
                    frequency = counts[label]
                    frequency_group = "head" if frequency > 100 else "medium" if frequency >= 20 else "tail"
                    writer.writerow([
                        offset + index, sample_id, label, int(prediction[index]),
                        int(prediction[index] == labels[index]),
                        float(scores["max_softmax"][index]), float(scores["max_sigmoid"][index]),
                        float(scores["sigmoid_logit_margin"][index]), float(top.values[index, 0]),
                        float(top.values[index, 1]), float(top.values[index, 0] - top.values[index, 1]),
                        float(true[index]), float(wrong[index]), float(true[index] - wrong[index]),
                        frequency_group,
                    ])
                offset += n
    finally:
        if sample_handle is not None:
            sample_handle.close()
            logits_store.flush()
            probabilities_store.flush()
    score_metrics, risks, groups = {}, {}, None
    for name, confidence in confidence_all.items():
        metric, risk, group_labels = evaluate_arrays(
            labels_all, predictions_all, confidence, top5_all, counts, cfg["tau"])
        score_metrics[name], risks[name], groups = metric, risk, group_labels
    native = score_metrics[cfg["native_confidence"]]
    metrics = {**native, **{name: value / len(dataset) for name, value in loss_sums.items()},
               "group": cfg["group"], "tuning_mode": cfg["tuning_mode"],
               "training_loss": cfg["loss"], "native_confidence": cfg["native_confidence"],
               "confidence_metrics": {name: {key: metric.get(key) for key in FAILURE_KEYS}
                                      for name, metric in score_metrics.items()}}
    if export is not None:
        np.savez(export / "risk_coverage.npz",
                 coverage=np.arange(1, len(dataset)+1) / len(dataset), **risks)
        labels_np, predictions_np = np.asarray(labels_all), np.asarray(predictions_all)
        supports = np.bincount(labels_np, minlength=len(counts))
        hits = np.bincount(labels_np[labels_np == predictions_np], minlength=len(counts))
        with (export / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_index", "train_count", "group", "support", "accuracy"])
            for index in range(len(counts)):
                writer.writerow([index, counts[index], groups[index], supports[index],
                                 hits[index] / supports[index] if supports[index] else ""])
        write_json(export / "metrics.json", metrics)
        write_json(export / "complete.json", {
            "rows": offset, "columns": cfg["num_classes"], "dtype": "float32",
            "probability_transform": "softmax" if cfg["loss"] == "ce" else "sigmoid",
            "prediction_rule": "argmax(logits)",
        })
    return metrics


def initialization_id(cfg) -> str:
    if cfg["initialization"] in ("plantclef", "imagenet"):
        return f"{cfg['initialization']}_sha256:{sha256(cfg['pretrained'])}"
    return f"random_full_state:seed={cfg['seed']}"


def _normalized_saved_config(saved, group):
    cfg = dict(saved["config"])
    cfg.update(GROUPS[group], group=group)
    cfg.setdefault("protocol", GROUPS[group]["protocol"])
    cfg.setdefault("initialization", GROUPS[group]["initialization"])
    cfg.setdefault("tuning_mode", GROUPS[group]["tuning_mode"])
    cfg.setdefault("native_confidence", GROUPS[group]["native_confidence"])
    return cfg


def _check_saved(saved, cfg, mapping, stats, identity):
    old_cfg = _normalized_saved_config(saved, cfg["group"])
    changed = {key for key in TRAINING_SIGNATURE_KEYS
               if old_cfg.get(key) != cfg.get(key)}
    if changed:
        raise ValueError(f"Checkpoint/config mismatch: {sorted(changed)}")
    if saved["mapping"] != mapping or saved["stats"] != stats:
        raise ValueError("Checkpoint data identity mismatch")
    if saved.get("format_version") == 1:
        expected_hash = identity.split(":", 1)[1]
        if saved.get("pretrained_sha256") != expected_hash:
            raise ValueError("Legacy checkpoint pretrained SHA256 mismatch")
    elif saved["initialization_id"] != identity:
        raise ValueError("Checkpoint initialization identity mismatch")


def _trim_logs(output: Path, start: int):
    for name in ("history.jsonl", "diagnostics.jsonl", "amp_events.jsonl"):
        path = output / name
        if path.exists():
            rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                    if json.loads(line)["epoch"] < start]
            path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def train(cfg, output: Path, device_name="cuda", resume=False, smoke_batches=None):
    seed_all(cfg["seed"])
    mapping = load_mapping(cfg)
    train_set = Images(cfg["train_dir"], mapping, cfg["image_size"], train=True)
    val_set = Images(cfg["val_dir"], mapping, cfg["image_size"])
    stats = audit(train_set, val_set)
    expected_counts = {
        "train_samples": cfg.get("expected_train_samples"),
        "val_samples": cfg.get("expected_val_samples"),
    }
    mismatched_counts = {
        key: {"expected": expected, "actual": stats[key]}
        for key, expected in expected_counts.items()
        if expected is not None and stats[key] != expected
    }
    if mismatched_counts:
        raise ValueError(f"Dataset sample-count mismatch: {mismatched_counts}")
    checkpoint_path = output / "last.pt"
    saved = load_checkpoint(checkpoint_path) if resume else None
    model, report = build_model(cfg)
    identity = initialization_id(cfg)
    if saved:
        _check_saved(saved, cfg, mapping, stats, identity)
        restore_model(model, saved)
    device = torch.device(device_name)
    model.to(device)
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    start, best = 0, -1.0
    if saved:
        if saved.get("kind") != "last":
            raise ValueError("Resume requires last.pt")
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best = saved["epoch"] + 1, saved["best"]
        restore_rng(saved, device)
        _trim_logs(output, start)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name in ("history.jsonl", "diagnostics.jsonl", "amp_events.jsonl"):
            (output / name).touch()
        import timm
        import torchvision
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
        write_json(output / "config.json", cfg)
        write_json(output / "data_audit.json", stats)
        write_json(output / "class_to_idx.json", mapping)
        write_json(output / "model_report.json", {
            **report, "initialization_id": identity,
            "trainable_fraction": report["trainable_parameters"] / report["total_parameters"],
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "git_revision": revision,
        })
    print(f"Group {cfg['group']}: seed={cfg['seed']} epochs={cfg['epochs']} "
          f"warmup={cfg['lr_warmup']} initialization={cfg['initialization']} "
          f"tuning={cfg['tuning_mode']} loss={cfg['loss']}", flush=True)
    if smoke_batches is not None:
        train_metrics = train_epoch(model, train_set, optimizer, scaler, device, cfg, 0,
                                    output, max_batches=smoke_batches)
        rows = min(len(val_set), cfg["batch_size"] * smoke_batches)
        val_metrics = evaluate(model, Subset(val_set, range(rows)), cfg, device,
                               stats["train_counts"])
        write_json(output / "smoke.json", {"status": "complete", "group": cfg["group"],
                   "seed": cfg["seed"], "train": train_metrics, "validation": val_metrics,
                   "formal_checkpoint_created": False})
        return
    for epoch in range(start, cfg["epochs"]):
        train_metrics = train_epoch(model, train_set, optimizer, scaler, device, cfg, epoch, output)
        val_metrics = evaluate(model, val_set, cfg, device, stats["train_counts"])
        improved = val_metrics["top1"] > best
        best = max(best, val_metrics["top1"])
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": train_metrics,
                                     "val": val_metrics, "best_top1": best},
                                    allow_nan=False) + "\n")
        save_checkpoint(output / "last.pt", "last", model, cfg, epoch, best,
                        mapping, stats, identity, optimizer, scaler)
        if improved:
            save_checkpoint(output / "best.pt", "best", model, cfg, epoch, best,
                            mapping, stats, identity)
        print(f"group {cfg['group']} epoch {epoch+1}: val top1={val_metrics['top1']:.5f} "
              f"best={best:.5f}", flush=True)


def export_run(group: str, run: Path, split_dir: Path, output: Path,
               device_name="cuda", expected_samples=None):
    saved = load_checkpoint(run / "best.pt")
    cfg = _normalized_saved_config(saved, group)
    validate_config(cfg)
    seed_all(cfg["seed"])
    mapping = saved["mapping"]
    model, _ = build_model(cfg)
    restore_model(model, saved)
    device = torch.device(device_name)
    model.to(device)
    dataset = Images(split_dir, mapping, cfg["image_size"])
    if expected_samples is not None and len(dataset) != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, found {len(dataset)}")
    metrics = evaluate(model, dataset, cfg, device, saved["stats"]["train_counts"], output)
    write_json(output / "provenance.json", {
        "group": group, "config": cfg, "checkpoint": str((run / "best.pt").resolve()),
        "checkpoint_sha256": sha256(run / "best.pt"), "split_dir": str(split_dir.resolve()),
        "class_to_idx": mapping, "metrics": metrics,
    })
