#!/usr/bin/env python3
"""Train the two phases of TCP-ConfiDNet without accessing official_val."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data import PlantSplit
from metrics import failure_metrics
from model import TCPConfidenceHead
from teacher import DEFAULT_CHECKPOINT, DEFAULT_CLASS_MAP, MISD_ROOT, build_models, file_sha256


class CachedFeatureDataset(Dataset):
    def __init__(self, values: dict, expected_split: str) -> None:
        self.values = values
        if self.values.get("split") != expected_split:
            raise ValueError(f"cache is not {expected_split}")
        self.length = len(self.values["tcp"])
        if len(self.values["feature"]) != self.length:
            raise ValueError("feature cache is malformed")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return {key: self.values[key][index] for key in ("feature", "tcp", "error")}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("head", "finetune"), required=True)
    parser.add_argument("--data-root", type=Path, default=MISD_ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=MISD_ROOT / "output/confidnet_tcp/cache")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--init-checkpoint", type=Path, help="Required for phase=finetune")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_cache(path: Path, split: str) -> dict:
    # values = torch.load(path, map_location="cpu", weights_only=False)
    values = torch.load(path, map_location="cpu")
    if values.get("split") != split:
        raise ValueError(f"{path} does not contain {split}")
    return values


def evaluate_head(head: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    head.eval()
    errors, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            confidence = head(batch["feature"].to(device, non_blocking=True).float())
            scores.append((1 - confidence).cpu().numpy())
            errors.append(batch["error"].numpy())
    return failure_metrics(np.concatenate(errors), np.concatenate(scores))


def evaluate_model(encoder: nn.Module, head: nn.Module, loader: DataLoader, cache: dict,
                   device: str) -> dict[str, float]:
    encoder.eval(); head.eval()
    errors, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            index = batch["index"].long()
            feature = encoder(batch["image"].to(device, non_blocking=True))
            confidence = head(feature)
            scores.append((1 - confidence).cpu().numpy())
            errors.append(cache["error"][index].numpy())
    return failure_metrics(np.concatenate(errors), np.concatenate(scores))


def save_checkpoint(path: Path, phase: str, encoder: nn.Module | None, head: nn.Module,
                    epoch: int, metrics: dict[str, float], args: argparse.Namespace) -> None:
    torch.save({
        "phase": phase, "encoder": None if encoder is None else encoder.state_dict(),
        "head": head.state_dict(), "epoch": epoch, "calibration_metrics": metrics,
        "teacher_checkpoint": str(args.checkpoint.resolve()), "teacher_sha256": file_sha256(args.checkpoint),
        "selection_split": "detector_calibration", "official_val_used": False,
    }, path)


def train_head(args: argparse.Namespace, train_cache: dict, calibration_cache: dict) -> None:
    # Reuse the already loaded tensors instead of mapping each cache into RAM twice.
    train_set = CachedFeatureDataset(train_cache, "detector_train")
    calibration_set = CachedFeatureDataset(calibration_cache, "detector_calibration")
    train_loader = DataLoader(train_set, batch_size=args.batch_size or 512, shuffle=True, num_workers=args.num_workers,
                              pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0)
    calibration_loader = DataLoader(calibration_set, batch_size=1024, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0)
    head = TCPConfidenceHead().to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best, stale, history = -np.inf, 0, []
    for epoch in range(1, (args.epochs or 40) + 1):
        head.train(); loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            confidence = head(batch["feature"].to(args.device, non_blocking=True).float())
            loss = loss_fn(confidence, batch["tcp"].to(args.device, non_blocking=True).float())
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
            loss_sum += loss.item() * len(confidence)
        metrics = evaluate_head(head, calibration_loader, args.device)
        row = {"epoch": epoch, "train_mse": loss_sum / len(train_set), **metrics}; history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["error_auprc"] > best:
            best, stale = metrics["error_auprc"], 0
            save_checkpoint(args.output_dir / "checkpoint_best.pth", "head", None, head, epoch, metrics, args)
        else:
            stale += 1
            if stale >= args.patience: break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def train_finetune(args: argparse.Namespace, train_cache: dict, calibration_cache: dict) -> None:
    if args.init_checkpoint is None:
        raise ValueError("--init-checkpoint is required for phase=finetune")
    teacher, encoder = build_models(args.checkpoint, args.class_map, args.device)
    del teacher  # TCP targets are frozen in cache; retaining teacher wastes GPU memory.
    # initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    initial = torch.load(args.init_checkpoint, map_location="cpu")
    if initial.get("phase") != "head" or initial.get("encoder") is not None:
        raise ValueError("finetune must initialize from a phase=head checkpoint")
    head = TCPConfidenceHead().to(args.device); head.load_state_dict(initial["head"])
    # Paper's fine-tuning phase disables dropout. eval() keeps ViT stochastic layers disabled while gradients flow.
    head.set_dropout(0.0); encoder.eval()
    parameters = [{"params": encoder.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-4}]
    optimizer = torch.optim.AdamW(parameters, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_set = PlantSplit(args.data_root, "detector_train", args.class_map)
    calibration_set = PlantSplit(args.data_root, "detector_calibration", args.class_map)
    train_loader = DataLoader(train_set, batch_size=args.batch_size or 16, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    calibration_loader = DataLoader(calibration_set, batch_size=args.batch_size or 16, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    best, stale, history = -np.inf, 0, []
    amp_enabled = args.device.startswith("cuda")
    # Use the established CUDA AMP API for compatibility with the project runtime.
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    for epoch in range(1, (args.epochs or 20) + 1):
        encoder.eval(); head.eval(); loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            index = batch["index"].long()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                feature = encoder(batch["image"].to(args.device, non_blocking=True))
                confidence = head(feature)
                tcp = train_cache["tcp"][index].to(args.device, non_blocking=True)
                loss = loss_fn(confidence, tcp)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
            scaler.step(optimizer); scaler.update(); loss_sum += loss.item() * len(confidence)
        metrics = evaluate_model(encoder, head, calibration_loader, calibration_cache, args.device)
        row = {"epoch": epoch, "train_mse": loss_sum / len(train_set), **metrics}; history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["error_auprc"] > best:
            best, stale = metrics["error_auprc"], 0
            save_checkpoint(args.output_dir / "checkpoint_best.pth", "finetune", encoder, head, epoch, metrics, args)
        else:
            stale += 1
            if stale >= args.patience: break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    set_seed(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(args.cache_dir / "detector_train.pt", "detector_train")
    calibration_cache = load_cache(args.cache_dir / "detector_calibration.pt", "detector_calibration")
    metadata = {"phase": args.phase, "seed": args.seed, "official_val_used": False,
                "teacher_checkpoint": str(args.checkpoint.resolve()), "teacher_sha256": file_sha256(args.checkpoint),
                "train_samples": len(train_cache["tcp"]), "train_errors": int(train_cache["error"].sum()),
                "calibration_samples": len(calibration_cache["tcp"]), "calibration_errors": int(calibration_cache["error"].sum())}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.phase == "head": train_head(args, train_cache, calibration_cache)
    else: train_finetune(args, train_cache, calibration_cache)


if __name__ == "__main__": main()
