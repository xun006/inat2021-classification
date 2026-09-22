from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOPK_ROOT = ROOT.parent / "TopK-ProtoFD"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOPK_ROOT))

import numpy as np
import torch

from formal_topk_fd.data import build_dataset, build_loader
from formal_topk_fd.engine import run_epoch
from formal_topk_fd.model import build_detector, effective_classifier_weights
from formal_topk_fd.runtime import seed_everything, torch_load, write_json
from topk_proto_fd.checkpoint import build_frozen_vit, load_mean_prototypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the formal competition-aware Top-K failure detector")
    parser.add_argument("--vit-checkpoint", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/output/vit_large_linear_probe_4271/checkpoint_best.pth"))
    parser.add_argument("--prototype-path", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/TopK-ProtoFD/prototypes/classifier_train_mean.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/output/formal_topk_fd/seed0"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--architecture", choices=("matching", "direct", "probability_only"), default="matching")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pair-hidden-dim", type=int, default=256)
    parser.add_argument("--pair-bottleneck-dim", type=int, default=64)
    parser.add_argument("--aggregator-hidden-dim", type=int, default=32)
    parser.add_argument("--pair-loss-weight", type=float, default=0.25)
    parser.add_argument("--error-pos-weight", type=float, default=10.403524385902456)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command in the GPU environment.")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Deliberately instantiate only development splits. official_val is never
    # opened by this training program.
    train_dataset = build_dataset(args.data_root, "detector_train")
    calibration_dataset = build_dataset(args.data_root, "detector_calibration")
    if train_dataset.class_to_idx != calibration_dataset.class_to_idx:
        raise ValueError("Class mappings differ between detector splits")
    train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    calibration_loader = build_loader(calibration_dataset, args.batch_size, args.num_workers, shuffle=False)

    vit, _, vit_meta = build_frozen_vit(args.vit_checkpoint, args.model_name)
    if vit_meta["num_classes"] != len(train_dataset.classes):
        raise ValueError("Teacher output dimension does not match the detector dataset")
    means, prototype_meta = load_mean_prototypes(args.prototype_path, train_dataset.classes)
    if tuple(means.shape) != (vit_meta["num_classes"], vit_meta["feature_dim"]):
        raise ValueError("Mean-prototype shape does not match the teacher")
    vit.eval().requires_grad_(False)
    weights = effective_classifier_weights(vit)
    detector_config = dict(
        feature_dim=vit_meta["feature_dim"],
        embedding_dim=args.embedding_dim,
        top_k=args.top_k,
        dropout=args.dropout,
        pair_hidden_dim=args.pair_hidden_dim,
        pair_bottleneck_dim=args.pair_bottleneck_dim,
        aggregator_hidden_dim=args.aggregator_hidden_dim,
    )
    detector = build_detector(args.architecture, **detector_config)
    vit, detector = vit.to(device), detector.to(device)
    weights, means = weights.to(device), means.to(device)

    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )
    config = {
        **vars(args),
        "vit_checkpoint": str(args.vit_checkpoint),
        "prototype_path": str(args.prototype_path),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "selection_split": "detector_calibration",
        "selection_metric": "error_auprc",
        "official_val_used": False,
        "score_direction": "larger means more likely error",
        "teacher": vit_meta,
        "prototype_source": prototype_meta.get("source_split"),
        "train_samples": len(train_dataset),
        "calibration_samples": len(calibration_dataset),
        "trainable_parameters": sum(p.numel() for p in detector.parameters() if p.requires_grad),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    write_json(args.output_dir / "config.json", config)

    history, best_ap, stale = [], -np.inf, 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, train_losses, _ = run_epoch(
            vit, detector, weights, means, train_loader, device,
            args.error_pos_weight, args.pair_loss_weight,
            optimizer=optimizer, amp=not args.no_amp, max_grad_norm=args.max_grad_norm,
        )
        calibration_metrics, calibration_losses, _ = run_epoch(
            vit, detector, weights, means, calibration_loader, device,
            args.error_pos_weight, args.pair_loss_weight, amp=not args.no_amp,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": {**train_losses, **train_metrics},
            "calibration": {**calibration_losses, **calibration_metrics},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        current_ap = calibration_metrics["aupr_error"]
        scheduler.step(current_ap)
        if current_ap > best_ap + args.min_delta:
            best_ap, stale = current_ap, 0
            torch.save(
                {
                    "detector": detector.state_dict(),
                    "detector_config": {
                        **detector_config,
                    },
                    "architecture": args.architecture,
                    "vit": {**vit_meta, "checkpoint": str(args.vit_checkpoint)},
                    "mean_prototypes": means.detach().cpu(),
                    "effective_weights": weights.detach().cpu(),
                    "class_names": train_dataset.classes,
                    "epoch": epoch,
                    "pair_loss_weight": args.pair_loss_weight,
                    "error_pos_weight": args.error_pos_weight,
                    "calibration_metrics": calibration_metrics,
                },
                args.output_dir / "checkpoint_best.pth",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best calibration Error AUPRC={best_ap:.6f}")
                break
    write_json(args.output_dir / "history.json", history)

    # Export predictions from the selected checkpoint, not the final epoch.
    saved = torch_load(args.output_dir / "checkpoint_best.pth")
    detector.load_state_dict(saved["detector"])
    metrics, losses, predictions = run_epoch(
        vit, detector, weights, means, calibration_loader, device,
        args.error_pos_weight, args.pair_loss_weight, amp=not args.no_amp,
        collect_predictions=True,
    )
    write_json(args.output_dir / "calibration_metrics.json", {**losses, **metrics})
    with (args.output_dir / "calibration_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    main()
