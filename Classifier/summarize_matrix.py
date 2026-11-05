"""Aggregate the formal A/B/C/D/G/H/I x 42/43/44 validation matrix."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .protocol import FORMAL_SEEDS, GROUPS, comparable_config, run_dir


METRICS = ("top1", "top5", "macro_f1", "auroc_error", "aupr_error",
           "fpr_at_95_tpr", "aurc")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path("/mnt/hdd8t/Mingle/xyyy/Classifier/outputs"))
    parser.add_argument("--export-name", default="val_export")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    audits = []
    for group in GROUPS:
        configs = []
        for seed in FORMAL_SEEDS:
            run = run_dir(args.output_root, group, seed)
            config_path = run / "config.json"
            metrics_path = run / args.export_name / "metrics.json"
            audit_path = run / "data_audit.json"
            history_path = run / "history.jsonl"
            for path in (config_path, metrics_path, audit_path, history_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Formal matrix is incomplete: {path}")
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                       if line]
            if len(history) != cfg["epochs"] or history[-1]["epoch"] != cfg["epochs"] - 1:
                raise ValueError(f"Formal run is incomplete: {run}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            configs.append(comparable_config(cfg))
            audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
            rows.append({"group": group, "seed": seed,
                         **{metric: metrics.get(metric) for metric in METRICS}})
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError(f"Group {group} has cross-seed config mismatch")
    if any(audit != audits[0] for audit in audits[1:]):
        raise ValueError("Formal runs do not share the same data audit")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group", "seed", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        summaries[group] = {"seeds": list(FORMAL_SEEDS), "n": len(selected)}
        for metric in METRICS:
            values = [row[metric] for row in selected if row[metric] is not None]
            summaries[group][metric] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
            }
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "complete.json").write_text(
        json.dumps({"status": "complete", "groups": list(GROUPS),
                    "seeds": list(FORMAL_SEEDS), "export_name": args.export_name}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
