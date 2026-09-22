from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from topk_proto_fd.checkpoint import build_frozen_vit
from topk_proto_fd.data import build_transform
from topk_proto_fd.model import extract_classifier_features

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build class-mean ViT feature prototypes")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/data/classifier/train"))
    parser.add_argument("--vit-checkpoint", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/output/vit_large_linear_probe_4271/checkpoint_best.pth"))
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "prototypes/classifier_train_mean.pth")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = ImageFolder(args.data_dir, transform=build_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    vit, _, meta = build_frozen_vit(args.vit_checkpoint)
    if len(dataset.classes) != meta["num_classes"]:
        raise ValueError(f"Dataset has {len(dataset.classes)} classes, checkpoint has {meta['num_classes']}")
    vit.requires_grad_(False).eval().to(device)

    # Accumulate on CPU in float64 for stable class means without retaining features.
    sums = torch.zeros(meta["num_classes"], meta["feature_dim"], dtype=torch.float64)
    counts = torch.zeros(meta["num_classes"], dtype=torch.long)
    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=not args.no_amp and device.type == "cuda"):
            features = extract_classifier_features(vit, images)
        features = F.normalize(features.float(), dim=-1).cpu().to(torch.float64)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels))
        if step % 100 == 0 or step == len(loader):
            print(json.dumps({"batch": step, "batches": len(loader), "images": int(counts.sum())}))

    missing = torch.nonzero(counts == 0).flatten().tolist()
    if missing:
        raise RuntimeError(f"Classes without training samples: {missing[:20]}")
    prototypes = F.normalize((sums / counts[:, None]).float(), dim=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prototypes": prototypes,
        "counts": counts,
        "class_names": dataset.classes,
        "source_split": str(args.data_dir),
        "vit_checkpoint": str(args.vit_checkpoint),
        "normalization": "mean(L2(feature)), then L2(mean)",
    }, args.output)
    print(f"Saved {tuple(prototypes.shape)} mean prototypes to {args.output}")


if __name__ == "__main__":
    main()
