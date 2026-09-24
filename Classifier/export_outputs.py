#!/usr/bin/env python3
"""Export full model outputs for every sample in a given split.

For each image the script saves:
    sample_path, ground_truth, prediction, p_max, full 4271-dim sigmoid probs

Outputs are stored as:
    {output_dir}/predictions.npz   (arrays: paths, gt, pred, pmax, probs)
    {output_dir}/predictions.csv   (per-sample summary)

This data is required by Stage-2 (misclassification detection).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import build_model  # noqa: E402

PROJECT_ROOT = HERE.parent
DEFAULT_DATA = PROJECT_ROOT / "MisD" / "data" / "classifier"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "MisD" / "data" / "class_to_idx.json"
DEFAULT_PRETRAINED = (
    PROJECT_ROOT
    / "models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
)


def parse_args():
    p = argparse.ArgumentParser(description="Export full sigmoid outputs")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    p.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    p.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Trained model checkpoint (checkpoint_best.pth)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val"],
                   help="Which ImageFolder split to export")
    p.add_argument("--model", default="vit_large_patch16")
    p.add_argument("--num-classes", type=int, default=4271)
    p.add_argument("--global-pool", action="store_true", default=True)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-targets", nargs="+", default=["qkv", "proj"])
    p.add_argument("--batch-size", type=int, default=64,
                   help="Smaller batch for full output storage")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-full-probs", action="store_true", default=True,
                   help="Save full 4271-dim probability vectors")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model and load trained checkpoint
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
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # data
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(args.data_path / args.split, transform=val_tf)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_gt, all_pred, all_pmax = [], [], []
    all_paths = []
    all_probs = [] if args.save_full_probs else None

    total = 0
    correct = 0
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits, _ = model.forward_with_features(images)
                probs = torch.sigmoid(logits).float()

            pmax, pred = probs.max(dim=1)
            batch = images.size(0)
            all_gt.append(targets.numpy())
            all_pred.append(pred.cpu().numpy())
            all_pmax.append(pmax.cpu().numpy())
            correct += (pred.cpu() == targets).sum().item()
            total += batch

            # record sample paths
            start = step * args.batch_size
            for i in range(batch):
                path, _ = dataset.samples[start + i]
                all_paths.append(path)

            if args.save_full_probs:
                all_probs.append(probs.cpu().numpy())

            if step % 50 == 0:
                print(f"export [{step+1:04d}/{len(loader):04d}]", flush=True)

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    all_pmax = np.concatenate(all_pmax)
    if all_probs is not None:
        all_probs = np.concatenate(all_probs, axis=0)

    accuracy = 100.0 * correct / total
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {accuracy:.3f}%")

    # save npz
    save_dict = {
        "ground_truth": all_gt,
        "prediction": all_pred,
        "p_max": all_pmax,
        "sample_paths": np.array(all_paths, dtype=object),
    }
    if all_probs is not None:
        save_dict["probs"] = all_probs
    np_path = args.output_dir / f"predictions_{args.split}.npz"
    np.savez_compressed(str(np_path), **save_dict)
    print(f"Saved {np_path} ({np_path.stat().st_size / 1e6:.1f} MB)")

    # save csv summary
    csv_path = args.output_dir / f"predictions_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_path", "ground_truth", "prediction",
                         "p_max", "correct"])
        for i in range(total):
            writer.writerow([
                all_paths[i], int(all_gt[i]), int(all_pred[i]),
                float(all_pmax[i]), int(all_pred[i] == all_gt[i]),
            ])
    print(f"Saved {csv_path}")

    # save summary json
    summary = {
        "split": args.split,
        "num_samples": total,
        "num_classes": args.num_classes,
        "top1_accuracy": accuracy,
        "checkpoint": str(args.checkpoint),
        "has_full_probs": all_probs is not None,
    }
    with (args.output_dir / f"export_summary_{args.split}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved export summary")


if __name__ == "__main__":
    main()
