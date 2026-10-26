"""Evaluate a frozen F/G/H best checkpoint on a deterministic ImageFolder split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from Classifier.abc_comparison.run import evaluate
from Classifier.data import Images, load_mapping
from Classifier.fgh_comparison.checkpoint_utils import (
    load as load_checkpoint,
    restore_model,
)
from Classifier.fgh_comparison.model import build_model
from Classifier.fgh_comparison.run import (
    initialization_id,
    validate_config,
)
from Classifier.run import seed_all, sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True, help="F, G or H run directory")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, help="Smoke only: deterministic prefix")
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    checkpoint_path = args.run / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    saved = load_checkpoint(checkpoint_path)
    if saved.get("kind") != "best":
        raise ValueError("Evaluation requires the best.pt checkpoint")
    cfg = saved["config"]
    validate_config(cfg)
    seed_all(cfg["seed"])
    mapping = load_mapping(cfg)
    if mapping != saved["mapping"]:
        raise ValueError("Current class map differs from the checkpoint class map")

    model, _ = build_model(cfg)
    identity = initialization_id(model, cfg)
    if identity != saved["initialization_id"]:
        raise ValueError("Checkpoint initialization identity mismatch")
    restore_model(model, saved)
    device = torch.device(args.device)
    model.to(device)

    full_dataset = Images(args.split_dir, mapping, cfg["image_size"])
    if args.expected_samples is not None and len(full_dataset) != args.expected_samples:
        raise ValueError(
            f"Split sample count mismatch: expected {args.expected_samples}, "
            f"found {len(full_dataset)}"
        )
    dataset = full_dataset
    if args.max_samples is not None:
        dataset = Subset(full_dataset, range(min(args.max_samples, len(full_dataset))))

    metrics = evaluate(
        model, dataset, cfg, device, saved["stats"]["train_counts"], args.output
    )
    write_json(args.output / "provenance.json", {
        "group": cfg["group"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "initialization_id": identity,
        "split_dir": str(args.split_dir.resolve()),
        "full_split_samples": len(full_dataset),
        "evaluated_samples": len(dataset),
        "max_samples": args.max_samples,
        "class_to_idx": mapping,
        "preprocessing": "Classifier.data.transform(train=False)",
        "prediction_rule": "argmax(logits)",
        "confidence": "max(sigmoid(logits))",
        "loss": "one-vs-rest BCEWithLogitsLoss, mean over batch and classes",
    })
    print(metrics)


if __name__ == "__main__":
    main()
