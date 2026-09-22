from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from topk_proto_fd.checkpoint import build_frozen_vit, load_mean_prototypes
from topk_proto_fd.data import build_loaders
from topk_proto_fd.engine import run_epoch
from topk_proto_fd.model import FrozenViTWithFailureDetector, TopKPrototypeFailureDetector

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Top-K prototype failure detector")
    parser.add_argument("--vit-checkpoint", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/output/vit_large_linear_probe_4271/checkpoint_best.pth"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/mean_proto_bottleneck_k5")
    parser.add_argument("--prototype-path", type=Path, default=PROJECT_DIR / "prototypes/classifier_train_mean.pth")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--architecture", choices=("legacy", "bottleneck"), default="bottleneck")
    parser.add_argument("--feature-bottleneck-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders, classes = build_loaders(args.data_root, args.batch_size, args.num_workers)
    vit, classifier, vit_meta = build_frozen_vit(args.vit_checkpoint, args.model_name)
    if len(classes) != vit_meta["num_classes"]:
        raise ValueError(f"Dataset has {len(classes)} classes, checkpoint has {vit_meta['num_classes']}")
    prototypes, prototype_meta = load_mean_prototypes(args.prototype_path, classes)
    expected_shape = (vit_meta["num_classes"], vit_meta["feature_dim"])
    if tuple(prototypes.shape) != expected_shape:
        raise ValueError(f"Mean prototypes have shape {tuple(prototypes.shape)}, expected {expected_shape}")
    detector = TopKPrototypeFailureDetector(
        vit_meta["feature_dim"], args.attention_dim, args.top_k, args.dropout,
        args.architecture, args.feature_bottleneck_dim,
    )
    model = FrozenViTWithFailureDetector(vit, detector, classifier, prototypes).to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    best_auroc, stale_epochs = -float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, train_loss = run_epoch(
            model, loaders["detector_train"], device, optimizer, not args.no_amp, args.max_grad_norm
        )
        calibration_metrics, calibration_loss = run_epoch(model, loaders["detector_calibration"], device, amp=not args.no_amp)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": current_lr, "train_loss": train_loss, "calibration_loss": calibration_loss,
               "train": train_metrics, "calibration": calibration_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        scheduler.step(calibration_metrics["auroc"])
        if calibration_metrics["auroc"] > best_auroc + args.min_delta:
            best_auroc = calibration_metrics["auroc"]
            stale_epochs = 0
            torch.save({
                "detector": detector.state_dict(),
                "detector_config": {"feature_dim": vit_meta["feature_dim"], "attention_dim": args.attention_dim,
                                    "top_k": args.top_k, "dropout": args.dropout,
                                    "architecture": args.architecture,
                                    "feature_bottleneck_dim": args.feature_bottleneck_dim},
                "vit": {**vit_meta, "checkpoint": str(args.vit_checkpoint)},
                "prototypes": model.prototypes.detach().cpu(),
                "prototype_meta": {
                    "source_split": prototype_meta.get("source_split"),
                    "normalization": prototype_meta.get("normalization"),
                    "path": str(args.prototype_path),
                },
                "class_names": classes,
                "epoch": epoch,
                "calibration_metrics": calibration_metrics,
            }, args.output_dir / "detector_best.pth")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best calibration AUROC={best_auroc:.6f}")
                break

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    best = torch.load(args.output_dir / "detector_best.pth", map_location="cpu", weights_only=False)
    detector.load_state_dict(best["detector"])
    official_metrics, official_loss = run_epoch(model, loaders["official_val"], device, amp=not args.no_amp)
    result = {"checkpoint_epoch": best["epoch"], "loss": official_loss, **official_metrics}
    (args.output_dir / "official_val_metrics.json").write_text(json.dumps(result, indent=2))
    print("official_val", json.dumps(result))


if __name__ == "__main__":
    main()
