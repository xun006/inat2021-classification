#!/usr/bin/env python3
"""Stage-1 classification training with LoRA PEFT + Sigmoid head.

Supports three loss configurations (To_do_list section 23):
    l1       : L1  (independent Sigmoid BCE)
    l1_l3    : L1 + L3  (BCE + SupCon, requires PK-sampler)
    l2_l3    : L2 + L3  (Margin Ranking + SupCon, requires PK-sampler)

Usage examples:
    python train.py --loss-config l1       --output-dir output/exp_l1
    python train.py --loss-config l1_l3    --output-dir output/exp_l1_l3
    python train.py --loss-config l2_l3    --output-dir output/exp_l2_l3
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import build_model          # noqa: E402
from losses import CombinedLoss          # noqa: E402
from data import (                       # noqa: E402
    validate_dataset,
    build_standard_loaders,
    build_pk_loaders,
)

PROJECT_ROOT = HERE.parent
DEFAULT_DATA = PROJECT_ROOT / "MisD" / "data" / "classifier"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "MisD" / "data" / "class_to_idx.json"
DEFAULT_PRETRAINED = (
    PROJECT_ROOT
    / "models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
)
DEFAULT_OUTPUT = HERE / "output"

LOSS_LAMBDAS = {
    "l1":    dict(lambda1=1.0, lambda2=0.0, lambda3=0.0),
    "l1_l3": dict(lambda1=1.0, lambda2=0.0, lambda3=1.0),
    "l2_l3": dict(lambda1=0.0, lambda2=1.0, lambda3=1.0),
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 LoRA + Sigmoid training")
    # data
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    p.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    p.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    # model
    p.add_argument("--model", default="vit_large_patch16")
    p.add_argument("--num-classes", type=int, default=4271)
    p.add_argument("--global-pool", action="store_true", default=True)
    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-targets", nargs="+", default=["qkv", "proj"])
    # loss
    p.add_argument("--loss-config", choices=list(LOSS_LAMBDAS), default="l1")
    p.add_argument("--margin", type=float, default=0.5, help="L2 margin m")
    p.add_argument("--temperature", type=float, default=0.07, help="L3 temperature tau")
    # PK-sampler (only used when lambda3 > 0)
    p.add_argument("--pk-P", type=int, default=32, help="classes per batch")
    p.add_argument("--pk-K", type=int, default=4, help="samples per class")
    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128, help="for L1 (non-PK)")
    p.add_argument("--accum-iter", type=int, default=1,
                   help="Gradient accumulation steps (effective_batch = batch_size * accum_iter)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, progress, total_epochs, warmup_epochs, base_lr, min_lr):
    if progress < warmup_epochs:
        lr = base_lr * progress / max(1, warmup_epochs)
    else:
        span = max(1, total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (progress - warmup_epochs) / span)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def autocast_ctx(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    epoch, args, base_lr):
    model.train()

    optimizer.zero_grad(set_to_none=True)
    loss_sum = sample_count = 0.0
    component_sums = {}
    started = time.time()
    use_amp = not args.no_amp and device.type == "cuda"

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_ctx(use_amp):
            logits, features = model.forward_with_features(images)
            if args.loss_config in ("l1_l3", "l2_l3"):
                feat_norm = F.normalize(features, dim=1)
                loss, comps = criterion(logits, feat_norm, targets)
            else:
                loss, comps = criterion(logits, features, targets)

        scaler.scale(loss / args.accum_iter).backward()
        should_step = (step + 1) % args.accum_iter == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch = images.size(0)
        loss_sum += loss.item() * batch
        sample_count += batch
        for k, v in comps.items():
            component_sums[k] = component_sums.get(k, 0.0) + v.item() * batch

        if step % 100 == 0 or step + 1 == len(loader):
            elapsed = time.time() - started
            print(
                f"epoch {epoch+1:03d} train [{step+1:05d}/{len(loader):05d}] "
                f"loss={loss.item():.5f} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"elapsed={elapsed/60:.1f}m",
                flush=True,
            )

    avg_loss = loss_sum / sample_count
    avg_comps = {k: v / sample_count for k, v in component_sums.items()}
    return avg_loss, avg_comps


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct1 = correct5 = sample_count = 0.0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_ctx(True):
            logits, _ = model.forward_with_features(images)
            probs = torch.sigmoid(logits)
        batch = images.size(0)
        predictions = probs.topk(5, dim=1).indices
        correct = predictions.eq(targets[:, None])
        correct1 += correct[:, :1].sum().item()
        correct5 += correct.sum().item()
        sample_count += batch
        if step % 100 == 0:
            print(f"validation [{step+1:04d}/{len(loader):04d}]", flush=True)

    top1 = 100.0 * correct1 / sample_count
    top5 = 100.0 * correct5 / sample_count
    return top1, top5


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scaler, epoch, best_top1, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_top1": best_top1,
        "args": vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # data
    validate_dataset(args.data_path, args.class_map, args.num_classes)
    needs_pk = args.loss_config in ("l1_l3", "l2_l3")
    if needs_pk:
        train_loader, val_loader, train_set, val_set = build_pk_loaders(
            args.data_path, P=args.pk_P, K=args.pk_K,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda", seed=args.seed,
        )
        effective_batch = args.pk_P * args.pk_K
    else:
        train_loader, val_loader, train_set, val_set = build_standard_loaders(
            args.data_path, batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        effective_batch = args.batch_size

    # model
    model, info = build_model(
        pretrained_path=args.pretrained,
        model_name=args.model,
        num_classes=args.num_classes,
        global_pool=args.global_pool,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
    )
    model.to(device)

    # loss
    lambdas = LOSS_LAMBDAS[args.loss_config]
    criterion = CombinedLoss(
        margin=args.margin, temperature=args.temperature, **lambdas
    )

    # optimizer (only trainable params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp and device.type == "cuda")

    # output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir / "tensorboard")

    # save config
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "effective_batch": effective_batch,
        "trainable_params": info["trainable_params"],
        "total_params": info["total_params"],
        "trainable_ratio": info["trainable_ratio"],
        "lora_num_modules": info["lora_num_modules"],
        "lora_params": info["lora_params"],
        "loss_lambdas": lambdas,
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"loss_config={args.loss_config}, lambdas={lambdas}")
    print(f"train={len(train_set):,}, val={len(val_set):,}, classes={args.num_classes}")
    print(f"LoRA modules={info['lora_num_modules']}, "
          f"LoRA params={info['lora_params']:,}")
    print(f"trainable={info['trainable_params']:,} / total={info['total_params']:,} "
          f"({info['trainable_ratio']:.4%})")
    print(f"effective batch={effective_batch}, lr={args.lr:g}")
    print(f"output={args.output_dir}", flush=True)

    # resume
    start_epoch, best_top1 = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt["best_val_top1"]
        print(f"resumed from epoch {start_epoch}, best_top1={best_top1:.3f}%")

    # training loop
    csv_path = args.output_dir / "metrics.csv"
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs,
                       args.lr, args.min_lr)
        train_loss, train_comps = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            device, epoch, args, args.lr
        )
        top1, top5 = evaluate(model, val_loader, device)

        improved = top1 > best_top1
        best_top1 = max(best_top1, top1)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_top1": top1,
            "val_top5": top5,
            "lr": optimizer.param_groups[0]["lr"],
            "best_val_top1": best_top1,
        }
        row.update({f"train_{k}": v for k, v in train_comps.items()})

        new_csv = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            out = csv.DictWriter(f, fieldnames=row.keys())
            if new_csv:
                out.writeheader()
            out.writerow(row)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch + 1)
        writer.flush()

        save_checkpoint(args.output_dir / "checkpoint_last.pth",
                        model, optimizer, scaler, epoch, best_top1, args)
        if improved:
            save_checkpoint(args.output_dir / "checkpoint_best.pth",
                            model, optimizer, scaler, epoch, best_top1, args)

        comp_str = " ".join(f"{k}={v:.5f}" for k, v in train_comps.items() if k != "total")
        print(
            f"epoch {epoch+1:03d}: train_loss={train_loss:.5f} [{comp_str}] "
            f"top1={top1:.3f}% top5={top5:.3f}% best={best_top1:.3f}%",
            flush=True,
        )

    writer.close()
    print(f"Training complete. best_val_top1={best_top1:.3f}%")


if __name__ == "__main__":
    main()
