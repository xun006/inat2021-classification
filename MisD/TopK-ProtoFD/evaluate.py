from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from topk_proto_fd.checkpoint import build_frozen_vit
from topk_proto_fd.data import build_loaders
from topk_proto_fd.engine import run_epoch
from topk_proto_fd.model import FrozenViTWithFailureDetector, TopKPrototypeFailureDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained failure detector")
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vit-checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("detector_train", "detector_calibration", "official_val"), default="official_val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    saved = torch.load(args.detector_checkpoint, map_location="cpu", weights_only=False)
    vit_path = args.vit_checkpoint or Path(saved["vit"]["checkpoint"])
    vit, classifier, meta = build_frozen_vit(vit_path, saved["vit"]["model_name"])
    detector_config = dict(saved["detector_config"])
    # Checkpoints produced before competition_v2 used the original direct-concat model.
    detector_config.setdefault("architecture", "legacy")
    detector = TopKPrototypeFailureDetector(**detector_config)
    detector.load_state_dict(saved["detector"])
    prototypes = saved.get("prototypes")
    if prototypes is None:
        raise ValueError("Detector checkpoint has no mean prototypes; retrain with --prototype-path")
    model = FrozenViTWithFailureDetector(vit, detector, classifier, prototypes).to(args.device)
    loaders, classes = build_loaders(args.data_root, args.batch_size, args.num_workers)
    if classes != saved["class_names"]:
        raise ValueError("Dataset class mapping differs from the detector training mapping")
    metrics, loss = run_epoch(model, loaders[args.split], torch.device(args.device), amp=not args.no_amp)
    print(json.dumps({"split": args.split, "loss": loss, **metrics}, indent=2))


if __name__ == "__main__":
    main()
