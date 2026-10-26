"""Summarize matched F/G/H exports, optionally alongside the existing D export."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from Classifier.run import write_json


METRICS = ("top1", "top5", "loss", "samples", "correct")


def read_export(path: Path, expected_rows=None):
    complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    with (path / "samples.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)
    rows = [(row["sample_id"], int(row["ground_truth"])) for row in records]
    metrics["correct"] = sum(int(row["correct"]) for row in records)
    if complete["rows"] != len(rows) or metrics["samples"] != len(rows):
        raise ValueError(f"Incomplete export at {path}")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Sample identity/order mismatch at {path}")
    return rows, metrics


def difference(left, right):
    return {
        metric: left[metric] - right[metric]
        for metric in ("top1", "top5", "loss")
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--f", type=Path, required=True)
    parser.add_argument("--g", type=Path, required=True)
    parser.add_argument("--h", type=Path, required=True)
    parser.add_argument("--d", type=Path, help="Optional existing D export")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison output: {args.output}")

    paths = {"F": args.f, "G": args.g, "H": args.h}
    if args.d is not None:
        paths["D"] = args.d
    rows = None
    metrics = {}
    for group, path in paths.items():
        current_rows, metrics[group] = read_export(path, rows)
        rows = current_rows

    contrasts = {
        "G_minus_H_pretraining_effect_under_full_training": difference(
            metrics["G"], metrics["H"]
        ),
        "H_minus_F_full_training_effect_under_random_init": difference(
            metrics["H"], metrics["F"]
        ),
    }
    if "D" in metrics:
        contrasts.update({
            "F_minus_D_random_init_effect_under_lora": difference(
                metrics["F"], metrics["D"]
            ),
            "G_minus_D_full_training_effect_under_plantclef": difference(
                metrics["G"], metrics["D"]
            ),
        })

    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group",) + METRICS)
        writer.writeheader()
        for group in paths:
            writer.writerow({"group": group, **{
                key: metrics[group].get(key) for key in METRICS
            }})
    write_json(args.output / "summary.json", {
        "groups": {group: str(path.resolve()) for group, path in paths.items()},
        "metrics": metrics,
        "contrasts": contrasts,
        "sample_count": len(rows),
        "interpretation": {
            "F": "Random initialization + frozen backbone + LoRA + head; strict negative control",
            "G": "PlantCLEF initialization + full fine-tuning",
            "H": "Random initialization + full training; true scratch baseline",
        },
    })
    write_json(args.output / "complete.json", {
        "groups": list(paths), "rows": len(rows), "status": "complete"
    })


if __name__ == "__main__":
    main()
