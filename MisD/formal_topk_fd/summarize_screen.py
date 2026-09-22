from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("/mnt/hdd8t/Mingle/xyyy/MisD/output/formal_topk_fd")
    runs = [root / "seed0"] + sorted((root / "screen").glob("s*"))
    columns = ("run", "parameters", "best_epoch", "auroc", "aupr_error", "fpr95", "aurc")
    print(",".join(columns))
    for run in runs:
        config_path = run / "config.json"
        metrics_path = run / "calibration_metrics.json"
        checkpoint_path = run / "checkpoint_best.pth"
        if not config_path.exists() or not metrics_path.exists() or not checkpoint_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        best_epoch = max(history, key=lambda row: row["calibration"]["aupr_error"])["epoch"]
        values = (
            run.name,
            config["trainable_parameters"],
            best_epoch,
            metrics["auroc"],
            metrics["aupr_error"],
            metrics["fpr95"],
            metrics["aurc"],
        )
        print(",".join(map(str, values)))


if __name__ == "__main__":
    main()
