from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/hdd8t/Mingle/xyyy/MisD")


def read_csv_rows(path: Path, name_key: str = "method") -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        result.append(
            {
                "method": row[name_key],
                "auroc": float(row["auroc"]),
                "error_auprc": float(row.get("error_auprc", row.get("aupr_error"))),
                "fpr_at_95_tpr": float(row.get("fpr_at_95_tpr", row.get("fpr95"))),
                "aurc": float(row["aurc"]),
            }
        )
    return result


def read_json_method(path: Path, method: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": method,
        "auroc": float(value["auroc"]),
        "error_auprc": float(value.get("error_auprc", value.get("aupr_error"))),
        "fpr_at_95_tpr": float(value.get("fpr_at_95_tpr", value.get("fpr95"))),
        "aurc": float(value["aurc"]),
    }


def main() -> None:
    rows = read_csv_rows(ROOT / "output/patch_cross_attention/baselines/confidence/metrics.csv")
    rows += read_csv_rows(ROOT / "output/patch_cross_attention/baselines/probability_shape_mlp/metrics.csv")
    rows.append(read_json_method(
        ROOT / "TopK-ProtoFD/outputs/mean_proto_bottleneck_k5/official_val_metrics.json",
        "Mean-prototype bottleneck K=5",
    ))
    rows.append(read_json_method(
        ROOT / "TopK-ProtoFD/outputs/competition_v2_k5/official_val_metrics.json",
        "Competition-v2 K=5",
    ))
    selected = ROOT / "output/formal_topk_fd/final/s4_direct_seed0_official_val/metrics.json"
    if not selected.exists():
        raise FileNotFoundError(f"Run official evaluation first: {selected}")
    rows.append(read_json_method(selected, "Direct correctness MLP (selected, seed 0)"))

    output_dir = ROOT / "output/formal_topk_fd/final"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "auroc", "error_auprc", "fpr_at_95_tpr", "aurc"]
    with (output_dir / "official_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Official-val comparison",
        "",
        "All methods use the same frozen teacher and the same 42,710 official-val images.",
        "",
        "| Method | AUROC ↑ | Error AUPRC ↑ | FPR@95TPR ↓ | AURC ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['auroc']:.6f} | {row['error_auprc']:.6f} | "
            f"{row['fpr_at_95_tpr']:.6f} | {row['aurc']:.6f} |"
        )
    (output_dir / "official_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
