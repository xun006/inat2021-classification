"""Train or smoke-test the controlled F/G/H comparison.

Run from the repository root, for example:
python -m Classifier.fgh_comparison.run train --group F --output Classifier/outputs/fgh_seed42/F
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from Classifier.abc_comparison.run import evaluate, train_epoch
from Classifier.data import Images, audit, load_mapping
from Classifier.fgh_comparison.checkpoint_utils import (
    load as load_checkpoint,
    restore_model,
    restore_rng,
    save_best,
    save_last,
)
from Classifier.fgh_comparison.model import build_model
from Classifier.run import seed_all, sha256, write_json


GROUPS = {
    "F": {
        "initialization": "random",
        "tuning_mode": "lora_head",
        "loss": "l2",
        "native_confidence": "max_sigmoid",
    },
    "G": {
        "initialization": "plantclef",
        "tuning_mode": "full",
        "loss": "l2",
        "native_confidence": "max_sigmoid",
    },
    "H": {
        "initialization": "random",
        "tuning_mode": "full",
        "loss": "l2",
        "native_confidence": "max_sigmoid",
    },
}


def resolve_config(path: Path, group: str) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.update(GROUPS[group], group=group)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["group"] not in GROUPS:
        raise ValueError(f"Unknown group: {cfg['group']}")
    expected = GROUPS[cfg["group"]]
    changed = {key for key, value in expected.items() if cfg.get(key) != value}
    if changed:
        raise ValueError(f"Group protocol was changed: {sorted(changed)}")
    fixed = {
        "model": "vit_large_patch16",
        "global_pool": True,
        "image_size": 224,
        "num_classes": 4271,
        "rank": 8,
        "alpha": 16,
        "lora_dropout": 0.0,
        "epochs": 20,
        "batch_size": 32,
        "lr": 0.0001,
        "head_lr": 0.001,
        "weight_decay": 0.05,
        "lr_warmup": 2,
        "grad_clip": 1.0,
        "seed": 42,
        "bce_reduction": "mean",
    }
    altered = {key for key, value in fixed.items() if cfg.get(key) != value}
    if altered:
        raise ValueError(f"Controlled F/G/H protocol was changed: {sorted(altered)}")
    if cfg["workers"] < 0 or cfg["diagnostic_batches"] < 0:
        raise ValueError("workers and diagnostic_batches must be non-negative")


def validate_data_audit(stats: dict, cfg: dict) -> None:
    expected = {
        "train_samples": cfg.get("expected_train_samples"),
        "val_samples": cfg.get("expected_val_samples"),
    }
    mismatch = {
        key: {"expected": value, "actual": stats[key]}
        for key, value in expected.items()
        if value is not None and stats[key] != value
    }
    if mismatch:
        raise ValueError(f"Dataset sample-count mismatch: {mismatch}")
    if len(stats["train_counts"]) != cfg["num_classes"] or min(stats["train_counts"]) <= 0:
        raise ValueError("Training split must contain every configured class")


def make_optimizer(model, cfg):
    head_ids = {id(parameter) for parameter in model.head.parameters()}
    backbone_or_lora = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    if not backbone_or_lora:
        raise RuntimeError("No trainable backbone/LoRA parameters")
    return torch.optim.AdamW([
        {
            "params": backbone_or_lora,
            "lr": cfg["lr"],
            "initial_lr": cfg["lr"],
            "name": "lora" if cfg["tuning_mode"] == "lora_head" else "backbone",
        },
        {
            "params": list(model.head.parameters()),
            "lr": cfg["head_lr"],
            "initial_lr": cfg["head_lr"],
            "name": "head",
        },
    ], weight_decay=cfg["weight_decay"])


def frozen_parameter_fingerprint(model) -> str:
    """Protect F against rebuilding a different random frozen backbone at export."""
    digest = hashlib.sha256()
    parameters = dict(model.named_parameters())
    for name, value in model.state_dict().items():
        parameter = parameters.get(name)
        if parameter is None or parameter.requires_grad:
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def initialization_id(model, cfg) -> str:
    if cfg["initialization"] == "plantclef":
        return f"plantclef_sha256:{sha256(cfg['pretrained'])}"
    if cfg["tuning_mode"] == "lora_head":
        return f"random_frozen_sha256:{frozen_parameter_fingerprint(model)}"
    return f"random_full_state_saved:seed={cfg['seed']}"


def check_saved(saved, cfg, mapping, stats, identity) -> None:
    allowed_path_changes = {
        "train_dir", "val_dir", "official_val_dir", "class_map", "pretrained", "workers"
    }
    changed = {
        key for key, value in cfg.items()
        if saved["config"].get(key) != value
    } - allowed_path_changes
    if changed:
        raise ValueError(f"Checkpoint/config mismatch: {sorted(changed)}")
    if saved["mapping"] != mapping:
        raise ValueError("Checkpoint class mapping mismatch")
    if saved["stats"] != stats:
        raise ValueError("Dataset changed since checkpoint creation")
    if saved["initialization_id"] != identity:
        raise ValueError("Checkpoint initialization identity mismatch")


def _trim_uncommitted_logs(output: Path, start: int) -> None:
    for name in ("history.jsonl", "diagnostics.jsonl", "amp_events.jsonl"):
        path = output / name
        if not path.exists():
            continue
        rows = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["epoch"] < start
        ]
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("train", "smoke"))
    parser.add_argument("--group", required=True, choices=tuple(GROUPS))
    parser.add_argument(
        "--config", type=Path,
        default=Path("Classifier/fgh_comparison/config.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Resume only from this run's last.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-batches", type=int, default=2)
    args = parser.parse_args()
    if args.smoke_batches <= 0:
        parser.error("--smoke-batches must be positive")
    if args.mode == "smoke" and args.checkpoint is not None:
        parser.error("smoke always starts fresh and cannot resume")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process/GPU per F/G/H run")

    cfg = resolve_config(args.config, args.group)
    seed_all(cfg["seed"])
    mapping = load_mapping(cfg)
    train = Images(cfg["train_dir"], mapping, cfg["image_size"], train=True)
    val = Images(cfg["val_dir"], mapping, cfg["image_size"])
    stats = audit(train, val)
    validate_data_audit(stats, cfg)

    saved = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if saved is not None and saved.get("kind") != "last":
        raise ValueError("Training can resume only from last.pt, not best.pt")

    model, report = build_model(cfg)
    identity = initialization_id(model, cfg)
    if saved is not None:
        check_saved(saved, cfg, mapping, stats, identity)
        restore_model(model, saved)

    device = torch.device(args.device)
    model.to(device)
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg["amp"] and device.type == "cuda"
    )
    start, best = 0, -1.0
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best = saved["epoch"] + 1, saved["best"]
        restore_rng(saved, device)
        if args.checkpoint.resolve() != (args.output / "last.pt").resolve():
            raise ValueError("Resume only from output/last.pt in the original run directory")
        _trim_uncommitted_logs(args.output, start)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        import timm
        import torchvision
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        write_json(args.output / "config.json", cfg)
        write_json(args.output / "data_audit.json", stats)
        write_json(args.output / "class_to_idx.json", mapping)
        write_json(args.output / "model_report.json", {
            **report,
            "initialization_id": identity,
            "trainable_fraction": report["trainable_parameters"] / report["total_parameters"],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "git_revision": revision,
        })

    print(
        f"Group {cfg['group']}: initialization={cfg['initialization']}, "
        f"tuning={cfg['tuning_mode']}, loss=BCEWithLogits(one-vs-rest mean), "
        f"seed={cfg['seed']}, epochs={cfg['epochs']}, batch_size={cfg['batch_size']}, "
        f"train_samples={stats['train_samples']}, "
        f"batches_per_epoch={math.ceil(stats['train_samples'] / cfg['batch_size'])}",
        flush=True,
    )

    if args.mode == "smoke":
        train_metrics = train_epoch(
            model, train, optimizer, scaler, device, cfg, 0, args.output,
            max_batches=args.smoke_batches,
        )
        rows = min(len(val), cfg["batch_size"] * args.smoke_batches)
        val_metrics = evaluate(
            model, Subset(val, range(rows)), cfg, device, stats["train_counts"]
        )
        smoke = {
            "status": "complete",
            "formal_checkpoint_created": False,
            "group": cfg["group"],
            "seed": cfg["seed"],
            "train": train_metrics,
            "validation": val_metrics,
            "validation_samples": rows,
            "note": "Infrastructure smoke only; not a formal experiment result",
        }
        write_json(args.output / "smoke.json", smoke)
        print(json.dumps(smoke, indent=2))
        return

    for epoch in range(start, cfg["epochs"]):
        train_metrics = train_epoch(
            model, train, optimizer, scaler, device, cfg, epoch, args.output
        )
        val_metrics = evaluate(model, val, cfg, device, stats["train_counts"])
        improved = val_metrics["top1"] > best
        best = max(best, val_metrics["top1"])
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_top1": best,
            }, allow_nan=False) + "\n")
        save_last(
            args.output / "last.pt", model, optimizer, scaler, cfg,
            epoch, best, mapping, stats, identity,
        )
        if improved:
            save_best(
                args.output / "best.pt", model, cfg, epoch, best,
                mapping, stats, identity,
            )
        print(
            f"group {cfg['group']} epoch {epoch + 1}: "
            f"val top1={val_metrics['top1']:.5f} best={best:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
