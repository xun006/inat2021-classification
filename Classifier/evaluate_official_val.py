"""Evaluate the frozen A/B/C/D(/E) classifiers on the same official_val rows.

This is intentionally an evaluation-only entry point.  It never constructs a
training loader and never modifies checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from Classifier.abc_comparison.model import build_model as build_abc_model
from Classifier.data import Images
from Classifier.losses import sigmoid_bce_loss
from Classifier.model import build_model as build_d_model
from Classifier.run import restore_adapter, seed_all, sha256, worker_seed, write_json


GROUPS = {
    "A": {"tuning_mode": "head_only", "loss": "ce", "confidence": "max_softmax"},
    "B": {"tuning_mode": "head_only", "loss": "l2", "confidence": "max_sigmoid"},
    "C": {"tuning_mode": "lora_head", "loss": "ce", "confidence": "max_softmax"},
    "D": {"tuning_mode": "lora_head", "loss": "l2", "confidence": "max_sigmoid"},
    "E": {"tuning_mode": "dora_head", "loss": "l2", "confidence": "max_sigmoid"},
}
PROTOCOL_KEYS = (
    "model", "global_pool", "image_size", "num_classes", "rank", "alpha",
    "lora_dropout", "epochs", "batch_size", "lr", "head_lr", "weight_decay",
    "lr_warmup", "grad_clip", "seed", "bce_reduction",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mapping_hash(mapping: dict[str, int]) -> str:
    canonical = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_mapping(path: Path, num_classes: int) -> dict[str, int]:
    mapping = read_json(path)
    if len(mapping) != num_classes or sorted(mapping.values()) != list(range(num_classes)):
        raise ValueError(
            f"Invalid class mapping {path}: expected {num_classes} contiguous indices, "
            f"found {len(mapping)} entries"
        )
    return mapping


def load_run(group: str, run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"Group {group} config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Group {group} best checkpoint not found: {checkpoint_path}")
    config = read_json(config_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config != saved.get("config"):
        differing = sorted(
            key for key in set(config) | set(saved.get("config", {}))
            if config.get(key) != saved.get("config", {}).get(key)
        )
        raise ValueError(f"Group {group} config.json/checkpoint config mismatch: {differing}")
    expected = GROUPS[group]
    actual_tuning = config.get("tuning_mode", "lora_head" if group == "D" else None)
    for key, value in (("tuning_mode", expected["tuning_mode"]), ("loss", expected["loss"])):
        actual = actual_tuning if key == "tuning_mode" else config.get(key)
        if actual != value:
            raise ValueError(f"Group {group} must use {key}={value!r}, found {actual!r}")
    if config.get("seed") != 42:
        raise ValueError(f"Group {group} checkpoint seed must be 42, found {config.get('seed')}")
    if saved.get("mapping") is None or saved.get("stats") is None:
        raise ValueError(f"Group {group} checkpoint lacks mapping or dataset audit metadata")
    return {
        "group": group, "run_dir": run_dir, "config": config, "saved": saved,
        "checkpoint": checkpoint_path, "checkpoint_sha256": sha256(checkpoint_path),
    }


def validate_runs(runs: list[dict], mapping: dict[str, int], pretrained: Path) -> None:
    reference = runs[0]
    pretrained_digest = sha256(pretrained)
    for run in runs:
        group, config, saved = run["group"], run["config"], run["saved"]
        if saved["mapping"] != mapping:
            raise ValueError(f"Group {group} checkpoint mapping differs from --class-map")
        if saved.get("pretrained_sha256") != pretrained_digest:
            raise ValueError(f"Group {group} pretrained SHA256 mismatch")
        if saved["stats"] != reference["saved"]["stats"]:
            raise ValueError(f"Group {group} training split audit differs from group A")
        differences = [
            key for key in PROTOCOL_KEYS
            if config.get(key) != reference["config"].get(key)
        ]
        if differences:
            raise ValueError(f"Group {group} controlled-protocol mismatch: {differences}")
    if pretrained_digest != reference["saved"]["pretrained_sha256"]:
        raise ValueError("Pretrained weight does not match the checkpoints")


def dataset_audit(dataset: Images, expected_samples: int) -> dict:
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes))
    if len(dataset) != expected_samples:
        raise ValueError(
            f"official_val sample count mismatch: expected {expected_samples}, found {len(dataset)}"
        )
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"official_val has empty mapped classes: {missing[:20]}")
    digest = hashlib.sha256()
    for path, label in dataset.samples:
        relative = Path(path).relative_to(dataset.root).as_posix()
        digest.update(f"{relative}:{label}\n".encode("utf-8"))
    return {
        "path": str(Path(dataset.root).resolve()),
        "samples": len(dataset),
        "classes": len(dataset.classes),
        "min_samples_per_class": int(counts.min()),
        "max_samples_per_class": int(counts.max()),
        "ordered_sample_fingerprint": digest.hexdigest(),
    }


def make_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(42),
    )


@torch.inference_mode()
def evaluate_group(run: dict, dataset, mapping: dict[str, int], pretrained: Path,
                   device: torch.device, batch_size: int, workers: int,
                   output: Path) -> dict:
    group = run["group"]
    config = dict(run["config"])
    config["pretrained"] = str(pretrained)
    seed_all(42)
    model, _ = (build_d_model(config) if group == "D" else build_abc_model(config))
    restore_adapter(model, run["saved"])
    model.to(device).eval()
    index_to_class = {index: name for name, index in mapping.items()}
    correct = total = 0
    loss_sum = 0.0
    sample_path = output / f"{group}_samples.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "row", "path", "true_label", "true_class", "predicted_label",
            "predicted_class", "confidence", "correct", "misclassified",
        ])
        for images, labels, paths in make_loader(dataset, batch_size, workers):
            images, labels = images.to(device), labels.to(device)
            logits = model(images).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Group {group} produced non-finite logits")
            if run["config"]["loss"] == "ce":
                loss = F.cross_entropy(logits, labels)
                confidence = logits.softmax(1).max(1).values
            else:
                loss = sigmoid_bce_loss(logits, labels, run["config"]["bce_reduction"])
                confidence = logits.sigmoid().max(1).values
            predictions = logits.argmax(1)
            matches = predictions.eq(labels)
            n = len(labels)
            loss_sum += float(loss) * n
            correct += int(matches.sum())
            for offset in range(n):
                truth = int(labels[offset])
                prediction = int(predictions[offset])
                is_correct = bool(matches[offset])
                writer.writerow([
                    total + offset, paths[offset], truth, index_to_class[truth], prediction,
                    index_to_class[prediction], float(confidence[offset]),
                    int(is_correct), int(not is_correct),
                ])
            total += n
    metrics = {
        "group": group,
        "correct": correct,
        "total": total,
        "loss": loss_sum / total,
        "top1_accuracy": correct / total,
        "loss_implementation": (
            "torch.nn.functional.cross_entropy"
            if run["config"]["loss"] == "ce"
            else "one-vs-rest torch.nn.functional.binary_cross_entropy_with_logits; mean over batch/classes"
        ),
        "confidence": GROUPS[group]["confidence"],
        "checkpoint": str(run["checkpoint"].resolve()),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "samples_csv": sample_path.name,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for group in "ABCD":
        parser.add_argument(f"--{group.lower()}-run", type=Path, required=True,
                            help=f"Group {group} run directory containing config.json and best.pt")
    parser.add_argument("--e-run", type=Path,
                        help="Optional group E DoRA run containing config.json and best.pt")
    parser.add_argument("--official-val", type=Path,
                        default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/data/official_val"))
    parser.add_argument("--class-map", type=Path,
                        default=Path("/mnt/hdd8t/Mingle/xyyy/MisD/data/class_to_idx.json"))
    parser.add_argument("--pretrained", type=Path, default=Path(
        "/mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/"
        "PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=42710)
    parser.add_argument("--max-samples", type=int,
                        help="Smoke test only: evaluate the first N rows after auditing the full split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    group_order = list("ABCD") + (["E"] if args.e_run is not None else [])
    runs = [load_run(group, getattr(args, f"{group.lower()}_run")) for group in group_order]
    num_classes = runs[0]["config"]["num_classes"]
    mapping = load_mapping(args.class_map, num_classes)
    validate_runs(runs, mapping, args.pretrained)
    dataset = Images(args.official_val, mapping, runs[0]["config"]["image_size"], train=False)
    audit = dataset_audit(dataset, args.expected_samples)
    evaluation_dataset = dataset
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(dataset):
            raise ValueError("max-samples must be between 1 and the full dataset size")
        evaluation_dataset = Subset(dataset, range(args.max_samples))
    args.output.mkdir(parents=True)
    metrics = [
        evaluate_group(run, evaluation_dataset, mapping, args.pretrained,
                       torch.device(args.device), args.batch_size, args.workers, args.output)
        for run in runs
    ]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "status": "complete", "seed": 42, "smoke_test": args.max_samples is not None,
        "evaluated_samples": len(evaluation_dataset), "official_val_audit": audit,
        "class_mapping_sha256": mapping_hash(mapping), "preprocess": (
            f"Resize({round(runs[0]['config']['image_size'] * 256 / 224)}, bicubic), "
            f"CenterCrop({runs[0]['config']['image_size']}), ToTensor, ImageNet normalization"
        ), "groups": metrics,
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "complete.json", {
        "status": "complete", "groups": group_order, "rows_per_group": len(evaluation_dataset),
        "official_val_fingerprint": audit["ordered_sample_fingerprint"],
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
