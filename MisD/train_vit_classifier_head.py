#!/usr/bin/env python3
"""Linear-probe PlantCLEF ViT on classifier/{train,val} only.

The official_val split is deliberately not accepted or referenced by this script.
It remains reserved for the final misclassification-detection evaluation.
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REFERENCE_DIR = PROJECT_ROOT / "PlantCLEF2022"
sys.path.insert(0, str(REFERENCE_DIR))

import models_vit  # noqa: E402
from timm.models.layers import trunc_normal_  # noqa: E402
from util.lars import LARS  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402


DEFAULT_DATA = HERE / "data" / "classifier"
DEFAULT_MAPPING = HERE / "data" / "class_to_idx.json"
DEFAULT_PRETRAINED = (
    PROJECT_ROOT
    / "models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
)
DEFAULT_OUTPUT = HERE / "output" / "vit_large_linear_probe_4271"


def parse_args():
    p = argparse.ArgumentParser(description="Train only the 4,271-class ViT classification head")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    p.add_argument("--class-map", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", default="vit_large_patch16")
    p.add_argument("--num-classes", type=int, default=4271)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size")
    p.add_argument("--accum-iter", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="Absolute LR; default uses blr scaling")
    p.add_argument("--blr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--global-pool", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--save-every", type=int, default=0,
                   help="Also keep epoch_NNN.pth every N epochs (0 disables)")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def distributed_setup():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world, local_rank = 0, 1, 0
    return distributed, rank, world, local_rank


def reduce_sums(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


class ExactDistributedEvalSampler(Sampler):
    """Disjoint rank-strided validation indices, without padding or duplication."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size


def validate_dataset(data_path, class_map_path, num_classes):
    train_dir, val_dir = data_path / "train", data_path / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageFolder splits: {train_dir} and {val_dir}")
    with class_map_path.open(encoding="utf-8") as f:
        expected = json.load(f)
    if len(expected) != num_classes or sorted(expected.values()) != list(range(num_classes)):
        raise ValueError("class_to_idx.json must contain a contiguous 0..4270 mapping")
    folder_classes = sorted(x.name for x in train_dir.iterdir() if x.is_dir())
    val_classes = sorted(x.name for x in val_dir.iterdir() if x.is_dir())
    imagefolder_mapping = {name: idx for idx, name in enumerate(folder_classes)}
    if imagefolder_mapping != expected:
        raise ValueError("train ImageFolder alphabetical indices do not match class_to_idx.json")
    if val_classes != folder_classes:
        raise ValueError("classifier/val class folders differ from classifier/train")
    return expected


def build_model(args):
    model = models_vit.__dict__[args.model](
        num_classes=args.num_classes, global_pool=args.global_pool
    )
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    # state = {k.removeprefix("module."): v for k, v in state.items()}
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    target = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in state and state[key].shape != target[key].shape:
            del state[key]
    interpolate_pos_embed(model, state)
    message = model.load_state_dict(state, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    if args.global_pool:
        allowed_missing |= {"fc_norm.weight", "fc_norm.bias"}
    bad_missing = set(message.missing_keys) - allowed_missing
    if bad_missing or message.unexpected_keys:
        raise RuntimeError(
            f"Unsafe checkpoint mismatch; missing={sorted(bad_missing)}, "
            f"unexpected={sorted(message.unexpected_keys)}"
        )

    # Always discard any existing classifier and initialize a fresh 4,271-way head.
    in_features = model.head.in_features
    classifier = torch.nn.Linear(in_features, args.num_classes)
    trunc_normal_(classifier.weight, std=0.01)
    torch.nn.init.zeros_(classifier.bias)
    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(in_features, affine=False, eps=1e-6), classifier
    )
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    return model, message


def cosine_lr(optimizer, progress, args, base_lr):
    if progress < args.warmup_epochs:
        lr = base_lr * progress / max(1, args.warmup_epochs)
    else:
        span = max(1, args.epochs - args.warmup_epochs)
        lr = args.min_lr + (base_lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (progress - args.warmup_epochs) / span)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def autocast_context(device, enabled):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()


def train_epoch(model, loader, optimizer, scaler, criterion, device, epoch, args, base_lr, rank):
    # Frozen backbone remains deterministic; only the classification head is in train mode.
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.head.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = sample_count = 0.0
    started = time.time()
    for step, (images, targets) in enumerate(loader):
        if step % args.accum_iter == 0:
            lr = cosine_lr(optimizer, epoch + step / len(loader), args, base_lr)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, not args.no_amp):
            loss = criterion(model(images), targets)
        scaler.scale(loss / args.accum_iter).backward()
        should_step = (step + 1) % args.accum_iter == 0 or step + 1 == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch = images.size(0)
        loss_sum += loss.item() * batch
        sample_count += batch
        if rank == 0 and (step % 100 == 0 or step + 1 == len(loader)):
            elapsed = time.time() - started
            print(f"epoch {epoch + 1:03d} train [{step + 1:05d}/{len(loader):05d}] "
                  f"loss={loss.item():.5f} lr={lr:.6g} elapsed={elapsed / 60:.1f}m", flush=True)
    loss_sum, sample_count = reduce_sums((loss_sum, sample_count), device)
    return loss_sum / sample_count, lr


@torch.inference_mode()
def evaluate(model, loader, criterion, device, rank):
    model.eval()
    loss_sum = correct1 = correct5 = sample_count = 0.0
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, True):
            logits = model(images)
            loss = criterion(logits, targets)
        batch = images.size(0)
        predictions = logits.topk(5, dim=1).indices
        correct = predictions.eq(targets[:, None])
        loss_sum += loss.item() * batch
        correct1 += correct[:, :1].sum().item()
        correct5 += correct.sum().item()
        sample_count += batch
        if rank == 0 and step % 100 == 0:
            print(f"validation [{step + 1:04d}/{len(loader):04d}]", flush=True)
    loss_sum, correct1, correct5, sample_count = reduce_sums(
        (loss_sum, correct1, correct5, sample_count), device
    )
    return loss_sum / sample_count, 100.0 * correct1 / sample_count, 100.0 * correct5 / sample_count


def save_checkpoint(path, model, optimizer, scaler, epoch, best_top1, args):
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "best_val_top1": best_top1,
        "args": vars(args), "selection_split": "classifier/val",
    }, path)


def main():
    args = parse_args()
    distributed, rank, world, local_rank = distributed_setup()
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = args.seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    validate_dataset(args.data_path, args.class_map, args.num_classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = datasets.ImageFolder(args.data_path / "train", transform=train_tf)
    val_set = datasets.ImageFolder(args.data_path / "val", transform=val_tf)
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    # Every validation image is evaluated exactly once. DDP ranks take disjoint strided subsets.
    val_sampler = (ExactDistributedEvalSampler(val_set, rank, world)
                   if distributed else SequentialSampler(val_set))
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=train_sampler is None,
                              drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, shuffle=False, drop_last=False, **loader_args)

    model, load_message = build_model(args)
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if isinstance(model, DDP) else model
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in raw_model.named_parameters() if p.requires_grad]
    if not trainable_names or any(not n.startswith("head.") for n in trainable_names):
        raise RuntimeError(f"Freeze invariant failed: {trainable_names}")
    effective_batch = args.batch_size * args.accum_iter * world
    base_lr = args.lr if args.lr is not None else args.blr * effective_batch / 256
    # Match the optimizer used by the reference MAE linear-probing recipe.
    optimizer = LARS(trainable, lr=base_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    criterion = torch.nn.CrossEntropyLoss()
    start_epoch, best_top1 = 0, -1.0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        raw_model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch, best_top1 = resume["epoch"] + 1, resume["best_val_top1"]

    writer = SummaryWriter(args.output_dir / "tensorboard") if rank == 0 else None
    if rank == 0:
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump({**config,
                       "selection_split": "classifier/val", "official_val_used": False}, f, indent=2)
        print(f"pretrained load: {load_message}")
        print(f"train={len(train_set):,}, classifier_val={len(val_set):,}, classes={len(train_set.classes):,}")
        print(f"trainable parameters={sum(p.numel() for p in trainable):,}; names={trainable_names}")
        print(f"effective batch={effective_batch}; lr={base_lr:g}; output={args.output_dir}", flush=True)

    csv_path = args.output_dir / "metrics.csv"
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, lr = train_epoch(model, train_loader, optimizer, scaler, criterion,
                                     device, epoch, args, base_lr, rank)
        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device, rank)
        improved = top1 > best_top1
        best_top1 = max(best_top1, top1)
        if rank == 0:
            row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                   "val_top1": top1, "val_top5": top5, "lr": lr, "best_val_top1": best_top1}
            new_csv = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                out = csv.DictWriter(f, fieldnames=row.keys())
                if new_csv: out.writeheader()
                out.writerow(row)
            with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            for key, value in row.items():
                if key != "epoch": writer.add_scalar(key, value, epoch + 1)
            writer.flush()
            save_checkpoint(args.output_dir / "checkpoint_last.pth", model, optimizer, scaler,
                            epoch, best_top1, args)
            if improved:
                save_checkpoint(args.output_dir / "checkpoint_best.pth", model, optimizer, scaler,
                                epoch, best_top1, args)
            if args.save_every and (epoch + 1) % args.save_every == 0:
                save_checkpoint(args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth",
                                model, optimizer, scaler, epoch, best_top1, args)
            print(f"epoch {epoch + 1:03d}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                  f"top1={top1:.3f}% top5={top5:.3f}% best={best_top1:.3f}%", flush=True)
    if writer: writer.close()
    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
