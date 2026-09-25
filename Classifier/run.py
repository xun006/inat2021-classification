"""Single-device reproducible training and bounded-memory prediction export.

Run from repository root: python -m Classifier.run --help
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Images, audit, image_audit, load_mapping
from .losses import components, ranking_weight, true_and_wrong
from .metrics import evaluate_arrays
from .model import build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, cfg, shuffle=False, epoch=0):
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["workers"], pin_memory=True, worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(cfg["seed"] + epoch))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path, model, optimizer, scaler, cfg, epoch, best, mapping, stats, weight_hash):
    # Frozen backbone is referenced by content hash, not duplicated each epoch.
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}
    payload = dict(adapter=state, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                   config=cfg, epoch=epoch, best=best, mapping=mapping, stats=stats,
                   pretrained_sha256=weight_hash, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   python_rng=random.getstate(), numpy_rng=np.random.get_state())
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_adapter(model, saved):
    expected = {k for k, p in model.named_parameters() if p.requires_grad}
    if set(saved["adapter"]) != expected:
        raise ValueError("Adapter parameter names do not match model")
    model.load_state_dict(saved["adapter"], strict=False)


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.detach().float().square().sum() for g in grads if g is not None),
                                loss.new_zeros(()))).item())


def train_epoch(model, dataset, optimizer, scaler, device, cfg, epoch, out):
    model.train()
    batches = loader(dataset, cfg, True, epoch)
    weight = ranking_weight(cfg, epoch)
    totals = dict(loss=0., l1=0., l2=0., l3=0., correct=0, pairs=0, samples=0)
    for step, (images, labels, _) in enumerate(batches):
        progress = epoch + (step + 1) / len(batches)
        factor = min(1., progress / max(1, cfg["lr_warmup"])) if progress < cfg["lr_warmup"] else (
            0.5 * (1 + math.cos(math.pi * (progress - cfg["lr_warmup"]) / (cfg["epochs"] - cfg["lr_warmup"]))))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda", dtype=torch.float16):
            logits = model(images)
        logits = logits.float()
        l1, l2, l3, pairs = components(logits, labels, cfg)
        base = l1 if cfg["loss"].startswith("l1") else l2
        total = base + weight * l3
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, step={step}")
        if step < cfg["diagnostic_batches"]:
            true, wrong = true_and_wrong(logits, labels)
            with torch.no_grad():
                positive = F.softplus(-true).mean()
                negative = (F.softplus(logits).sum(1) - F.softplus(true)).mean()
            row = dict(epoch=epoch, step=step, l1=l1.item(), l2=l2.item(), l3=l3.item(),
                       lambda3=weight, pairs=pairs, true_logit=true.mean().item(), wrong_logit=wrong.mean().item(),
                       positive_bce=positive.item(), negative_bce_class_sum=negative.item(),
                       correct=int(logits.argmax(1).eq(labels).sum()), batch=len(labels),
                       sigmoid_saturated_fraction=float(((logits.sigmoid() < 1e-5) | (logits.sigmoid() > 1-1e-5)).float().mean()),
                       head_grad_l1=grad_norm(l1, tuple(model.head.parameters())),
                       head_grad_l2=grad_norm(l2, tuple(model.head.parameters())),
                       head_grad_l3=grad_norm(l3, tuple(model.head.parameters())))
            with (out / "diagnostics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        n = len(labels)
        for key, value in (("loss", total), ("l1", l1), ("l2", l2), ("l3", l3)):
            totals[key] += value.item() * n
        totals["samples"] += n
        totals["correct"] += int(logits.argmax(1).eq(labels).sum())
        totals["pairs"] += pairs
        if step % 50 == 0:
            print(f"epoch {epoch+1} step {step}/{len(batches)} loss={total.item():.5f} pairs={pairs}", flush=True)
    return {**{k: totals[k] / totals["samples"] for k in ("loss", "l1", "l2", "l3", "correct")},
            "pairs": totals["pairs"], "lambda3": weight}


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, counts, epoch, export=None):
    model.eval()
    labels_all, pred_all, conf_all, top5_all = [], [], [], []
    sums = np.zeros(3)
    offset = 0
    logits_store = probs_store = None
    handle = None
    if export is not None:
        export.mkdir(parents=True, exist_ok=False)
        shape = (len(dataset), cfg["num_classes"])
        logits_store = np.lib.format.open_memmap(export / "logits.npy", mode="w+", dtype="float32", shape=shape)
        probs_store = np.lib.format.open_memmap(export / "sigmoid.npy", mode="w+", dtype="float32", shape=shape)
        handle = (export / "samples.csv").open("w", newline="", encoding="utf-8")
        writer = csv.writer(handle)
        writer.writerow(["row", "sample_id", "ground_truth", "prediction", "correct", "max_sigmoid_probability",
                         "top1_logit", "top2_logit", "top1_top2_logit_margin", "ground_truth_logit",
                         "max_wrong_class_logit", "ground_truth_vs_max_wrong_margin", "class_frequency_group"])
    try:
        for images, labels, ids in loader(dataset, cfg):
            images, labels = images.to(device), labels.to(device)
            # FP32 evaluation preserves confidence comparisons across runs.
            logits = model(images).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            l1, l2, l3, _ = components(logits, labels, cfg)
            sums += np.array([l1.item(), l2.item(), l3.item()]) * len(labels)
            top = logits.topk(min(5, logits.shape[1]), dim=1)
            pred = logits.argmax(1)
            probs = logits.sigmoid()
            confidence = probs.max(1).values
            labels_all.extend(labels.cpu().tolist())
            pred_all.extend(pred.cpu().tolist())
            conf_all.extend(confidence.cpu().tolist())
            top5_all.extend(top.indices.eq(labels[:, None]).any(1).cpu().tolist())
            if export is not None:
                n = len(labels)
                logits_store[offset:offset+n] = logits.cpu().numpy()
                probs_store[offset:offset+n] = probs.cpu().numpy()
                true, wrong = true_and_wrong(logits, labels)
                rows = torch.stack([labels, pred, pred.eq(labels), confidence, top.values[:, 0], top.values[:, 1],
                                    top.values[:, 0] - top.values[:, 1], true, wrong, true-wrong], 1).cpu().tolist()
                for index, (sample_id, row) in enumerate(zip(ids, rows)):
                    for j in range(3):
                        row[j] = int(row[j])
                    frequency = counts[row[0]]
                    writer.writerow([offset+index, sample_id, *row, "head" if frequency > 100 else "medium" if frequency >= 20 else "tail"])
                offset += n
    finally:
        if handle is not None:
            handle.close()
            logits_store.flush()
            probs_store.flush()
    metrics, risk, groups = evaluate_arrays(labels_all, pred_all, conf_all, top5_all, counts, cfg["tau"])
    sums /= len(dataset)
    metrics.update(l1=float(sums[0]), l2=float(sums[1]), l3=float(sums[2]))
    metrics["loss"] = float(sums[0 if cfg["loss"].startswith("l1") else 1] + ranking_weight(cfg, epoch) * sums[2])
    if export is not None:
        np.savez(export / "risk_coverage.npz", coverage=np.arange(1, len(risk)+1)/len(risk), risk=risk)
        labels_np, pred_np = np.asarray(labels_all), np.asarray(pred_all)
        supports = np.bincount(labels_np, minlength=len(counts))
        hits = np.bincount(labels_np[labels_np == pred_np], minlength=len(counts))
        with (export / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_index", "train_count", "group", "support", "accuracy"])
            for c in range(len(counts)):
                writer.writerow([c, counts[c], groups[c], supports[c], hits[c]/supports[c] if supports[c] else ""])
        write_json(export / "metrics.json", metrics)
        write_json(export / "complete.json", {"rows": offset, "columns": cfg["num_classes"],
                   "dtype": "float32", "prediction_rule": "argmax(logits); avoids sigmoid saturation ties"})
    return metrics


def validate_config(cfg):
    for key in ("epochs", "batch_size", "image_size", "rank", "l3_ramp", "num_classes"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["num_classes"] < 2 or not 0 <= cfg["lr_warmup"] < cfg["epochs"]:
        raise ValueError("Need >=2 classes and 0 <= lr_warmup < epochs")
    if cfg["loss"] not in ("l1", "l2", "l1_l3", "l2_l3"):
        raise ValueError("Unknown loss")
    if cfg["workers"] < 0 or cfg["l3_warmup"] < 0 or cfg["lambda3"] < 0:
        raise ValueError("Invalid workers/warmup/lambda3")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["audit", "train", "export"])
    parser.add_argument("--config", default="Classifier/config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss", choices=["l1", "l2", "l1_l3", "l2_l3"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Trusted locally generated checkpoint for resume/export")
    parser.add_argument("--split-dir", help="Export only: explicit ImageFolder (val/detector/final test)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-images", action="store_true", help="Audit only: decode all train/val images and record dimensions")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process/GPU per experiment; L3 uses the physical batch")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("loss", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    validate_config(cfg)
    seed_all(cfg["seed"])
    mapping = load_mapping(cfg)
    saved = None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        changed = {k for k in cfg if cfg[k] != saved["config"].get(k)} - {"train_dir", "val_dir", "class_map", "pretrained", "workers"}
        if changed or mapping != saved["mapping"]:
            raise ValueError(f"Checkpoint/config mismatch: {changed}")
    if args.mode == "export":
        if saved is None or not args.split_dir:
            parser.error("export requires --checkpoint and --split-dir")
        stats = saved["stats"]
    else:
        train = Images(cfg["train_dir"], mapping, cfg["image_size"], True)
        val = Images(cfg["val_dir"], mapping, cfg["image_size"])
        stats = audit(train, val)
        if saved and stats != saved["stats"]:
            raise ValueError("Dataset changed since checkpoint")
    if args.mode == "audit":
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "data_audit.json", stats)
        write_json(args.output / "class_to_idx.json", mapping)
        if args.verify_images:
            write_json(args.output / "image_audit.json", {"train": image_audit(train), "val": image_audit(val)})
        print(json.dumps({k: v for k, v in stats.items() if k != "train_counts"}, indent=2))
        return
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
        metrics = evaluate(model, dataset, cfg, device, stats["train_counts"], saved["epoch"], args.output)
        write_json(args.output / "provenance.json", dict(config=cfg, checkpoint_sha256=sha256(args.checkpoint),
                   pretrained_sha256=weight_hash, split_dir=str(Path(args.split_dir).resolve()), class_to_idx=mapping))
        print(json.dumps(metrics, indent=2))
        return
    if saved is None:
        args.output.mkdir(parents=True, exist_ok=False)
    elif args.checkpoint.resolve() != (args.output / "last.pt").resolve():
        raise ValueError("Resume from output/last.pt in the original run directory")
    head_ids = {id(p) for p in model.head.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], "lr": cfg["lr"], "initial_lr": cfg["lr"]},
        {"params": list(model.head.parameters()), "lr": cfg["head_lr"], "initial_lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    start, best = 0, -1.
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best = saved["epoch"] + 1, saved["best"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        # Remove rows from a failed/uncommitted epoch before continuing.
        for name in ("history.jsonl", "diagnostics.jsonl"):
            path = args.output / name
            if path.exists():
                rows = [line for line in path.read_text().splitlines() if json.loads(line)["epoch"] < start]
                path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    else:
        import timm, torchvision
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        write_json(args.output / "config.json", cfg)
        write_json(args.output / "data_audit.json", stats)
        write_json(args.output / "class_to_idx.json", mapping)
        write_json(args.output / "model_report.json", {**report, "pretrained_sha256": weight_hash,
                   "trainable_fraction": report["trainable_parameters"]/report["total_parameters"],
                   "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__, "git_revision": revision})
    for epoch in range(start, cfg["epochs"]):
        train_metrics = train_epoch(model, train, optimizer, scaler, device, cfg, epoch, args.output)
        val_metrics = evaluate(model, val, cfg, device, stats["train_counts"], epoch)
        improved = val_metrics["top1"] > best
        best = max(best, val_metrics["top1"])
        row = dict(epoch=epoch, train=train_metrics, val=val_metrics, best_top1=best)
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        checkpoint(args.output / "last.pt", model, optimizer, scaler, cfg, epoch, best, mapping, stats, weight_hash)
        if improved:
            checkpoint(args.output / "best.pt", model, optimizer, scaler, cfg, epoch, best, mapping, stats, weight_hash)
        print(f"epoch {epoch+1}: val top1={val_metrics['top1']:.5f} best={best:.5f}", flush=True)


if __name__ == "__main__":
    main()
