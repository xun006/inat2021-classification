"""Aggregate exported validation results, keeping seed replicates visible."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, grouped = [], defaultdict(list)
    for path in sorted(args.root.glob("*/val_export/metrics.json")):
        provenance = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
        cfg = provenance["config"]
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {"run": path.parent.parent.name, "loss": cfg["loss"], "seed": cfg["seed"],
               **{k: v for k, v in metrics.items() if not isinstance(v, list)}}
        rows.append(row)
        grouped[cfg["loss"]].append(row)
    if not rows:
        raise ValueError("No */val_export/metrics.json found")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for loss, values in grouped.items():
        summaries[loss] = {"n_runs": len(values), "seeds": [v["seed"] for v in values]}
        for metric in ("top1", "top5", "macro_f1", "auroc_error", "aupr_error", "aurc"):
            a = [v[metric] for v in values if v[metric] is not None]
            summaries[loss][metric] = {"mean": float(np.mean(a)) if a else None,
                                       "std": float(np.std(a, ddof=1)) if len(a)>1 else None}
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
