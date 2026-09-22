from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOPK_ROOT = ROOT.parent / "TopK-ProtoFD"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOPK_ROOT))

import torch

from formal_topk_fd.data import build_dataset, build_loader
from formal_topk_fd.engine import run_epoch
from formal_topk_fd.model import build_detector
from formal_topk_fd.runtime import torch_load, write_json
from topk_proto_fd.checkpoint import build_frozen_vit


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a locked formal Top-K failure detector")
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/data"))
    parser.add_argument("--split", choices=("detector_calibration", "official_val"), default="detector_calibration")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command in the GPU environment.")

    saved = torch_load(args.detector_checkpoint)
    vit, _, meta = build_frozen_vit(saved["vit"]["checkpoint"], saved["vit"]["model_name"])
    detector = build_detector(saved.get("architecture", "matching"), **saved["detector_config"])
    detector.load_state_dict(saved["detector"])
    dataset = build_dataset(args.data_root, args.split)
    if dataset.classes != saved["class_names"]:
        raise ValueError("Evaluation class mapping differs from the locked checkpoint")
    if meta["num_classes"] != len(dataset.classes):
        raise ValueError("Teacher dimension differs from evaluation dataset")

    device = torch.device(args.device)
    vit.eval().requires_grad_(False).to(device)
    detector.to(device)
    loader = build_loader(dataset, args.batch_size, args.num_workers, shuffle=False)
    metrics, losses, predictions = run_epoch(
        vit, detector, saved["effective_weights"].to(device), saved["mean_prototypes"].to(device),
        loader, device, saved["error_pos_weight"], saved["pair_loss_weight"],
        amp=not args.no_amp, collect_predictions=True,
    )
    output_dir = args.output_dir or args.detector_checkpoint.parent / f"eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"split": args.split, "checkpoint_epoch": saved["epoch"], **losses, **metrics}
    write_json(output_dir / "metrics.json", result)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
