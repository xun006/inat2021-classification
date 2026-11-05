"""Audit completed runs before multi-seed aggregation or deciding to retune."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import FORMAL_SEEDS, GROUPS, comparable_config, run_dir


REQUIRED = ("config.json", "data_audit.json", "class_to_idx.json", "model_report.json",
            "history.jsonl", "diagnostics.jsonl", "amp_events.jsonl", "last.pt", "best.pt")


def inspect_run(path: Path) -> dict:
    missing = [name for name in REQUIRED if not (path / name).exists()]
    result = {"run": str(path), "missing": missing, "status": "incomplete" if missing else "complete"}
    if not (path / "config.json").is_file() or not (path / "history.jsonl").is_file():
        return result
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if (path / "data_audit.json").is_file():
        result["data_audit"] = json.loads((path / "data_audit.json").read_text(encoding="utf-8"))
    if (path / "class_to_idx.json").is_file():
        result["class_to_idx"] = json.loads((path / "class_to_idx.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / "history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected_epochs = cfg.get("epochs")
    result.update(group=cfg.get("group"), seed=cfg.get("seed"), epochs=len(rows),
                  expected_epochs=expected_epochs, config=cfg)
    if expected_epochs is not None and (
            len(rows) != expected_epochs or
            (rows and rows[-1].get("epoch") != expected_epochs - 1)):
        result["status"] = "incomplete"
        result.setdefault("warnings", []).append(
            f"history has {len(rows)} committed epochs; expected {expected_epochs}"
        )
    if rows:
        scores = [row["val"]["top1"] for row in rows]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        result.update(best_epoch=best_index + 1, best_top1=scores[best_index],
                      best_at_last_epoch=best_index == len(scores) - 1,
                      late_improvement=(scores[-1] - scores[max(0, len(scores) - 4)]))
        if best_index == len(scores) - 1 and len(rows) == expected_epochs:
            result.setdefault("warnings", []).append("best checkpoint is the final epoch; horizon may be too short")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/hdd8t/Mingle/xyyy/Classifier/outputs"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = [inspect_run(run_dir(args.output_root, group, seed)) for group in GROUPS for seed in FORMAL_SEEDS]
    fingerprints = {}
    data_audits = []
    mappings = []
    for item in reports:
        if "config" in item:
            fingerprints.setdefault(item["group"], []).append(comparable_config(item.pop("config")))
        if "data_audit" in item:
            data_audits.append(item.pop("data_audit"))
        if "class_to_idx" in item:
            mappings.append(item.pop("class_to_idx"))
    mismatches = {group: values for group, values in fingerprints.items()
                  if any(value != values[0] for value in values[1:])}
    data_match = bool(data_audits) and all(value == data_audits[0] for value in data_audits[1:])
    mapping_match = bool(mappings) and all(value == mappings[0] for value in mappings[1:])
    payload = {"runs": reports, "within_group_config_mismatches": sorted(mismatches),
               "data_audits_match": data_match, "class_mappings_match": mapping_match,
               "aggregation_allowed": not mismatches and data_match and mapping_match and
               all(r["status"] == "complete" for r in reports)}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
