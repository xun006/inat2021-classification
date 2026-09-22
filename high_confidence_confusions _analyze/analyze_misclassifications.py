#!/usr/bin/env python3
"""Analyze and materialize a reproducible manual review of classifier errors.

This exploration tool is intentionally restricted to train_natural.csv.  It
produces aggregate CSV tables, a fixed four-group review queue, and an offline
HTML annotation gallery that can export the completed labels as CSV.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "image_path", "image_id", "ground_truth", "candidate_class",
    "is_correct", "teacher_top1_prob", "margin", "error_label",
}
CONFIDENCE_BINS = [-0.000001, 0.5, 0.7, 0.8, 0.9, 1.000001]
CONFIDENCE_LABELS = ["0.0-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
REASONS = [
    "同属相似物种", "同科相似物种", "缺少关键鉴别部位", "主体过小",
    "模糊或低分辨率", "遮挡", "多植物或背景干扰", "非典型生长阶段",
    "花、叶、果实等器官差异", "标签可能错误或有歧义", "跨类别明显误判", "无法判断",
]


def taxonomy(class_name: str) -> dict[str, str]:
    """Parse iNat folder names, preserving underscores in the species name."""
    parts = str(class_name).split("_", 7)
    if len(parts) != 8:
        raise ValueError(f"Cannot parse taxonomy from class name: {class_name!r}")
    keys = ("class_id", "kingdom", "phylum", "tax_class", "order", "family", "genus", "species")
    return dict(zip(keys, parts))


def relationship(true_name: str, predicted_name: str) -> str:
    true, pred = taxonomy(true_name), taxonomy(predicted_name)
    if true_name == predicted_name:
        return "same_species"
    if true["genus"] == pred["genus"] and true["family"] == pred["family"]:
        return "same_genus"
    if true["family"] == pred["family"]:
        return "same_family_different_genus"
    if true["order"] == pred["order"]:
        return "same_order_different_family"
    return "different_order"


def percent_table(series: pd.Series, name: str) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    counts["proportion"] = counts["count"] / counts["count"].sum()
    return counts


def select_review_groups(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    specs = [
        ("A_high_confidence_error", 1, False),
        ("B_low_confidence_error", 1, True),
        ("C_low_confidence_correct", 0, True),
        ("D_high_confidence_correct", 0, False),
    ]
    groups = []
    for group, error_label, ascending in specs:
        candidates = frame[frame["error_label"] == error_label]
        selected = candidates.sort_values(
            ["teacher_top1_prob", "image_id"], ascending=[ascending, True], kind="stable"
        ).head(count).copy()
        selected.insert(0, "review_group", group)
        selected.insert(1, "rank_in_group", range(1, len(selected) + 1))
        groups.append(selected)
    return pd.concat(groups, ignore_index=True)


def write_gallery(queue: pd.DataFrame, output: Path) -> None:
    records = queue.to_dict(orient="records")
    cards = []
    for index, row in enumerate(records):
        reason_boxes = "".join(
            f'<label><input type="checkbox" value="{html.escape(reason)}">{html.escape(reason)}</label>'
            for reason in REASONS
        )
        cards.append(f"""
<article class="card" data-index="{index}">
  <img src="{html.escape(row['review_image'])}" loading="lazy" alt="review image {row['image_id']}">
  <div class="body"><div class="group">{html.escape(row['review_group'])} #{row['rank_in_group']}</div>
  <div><b>ID</b> {row['image_id']} &nbsp; <b>Top-1</b> {row['teacher_top1_prob']:.4f} &nbsp; <b>Margin</b> {row['margin']:.4f}</div>
  <div><b>真实</b> {html.escape(row['ground_truth'])}</div>
  <div><b>预测</b> {html.escape(row['candidate_class'])}</div>
  <div><b>层级</b> {html.escape(row['taxonomy_relation'])}</div>
  <div class="reasons">{reason_boxes}</div>
  <textarea placeholder="简短备注"></textarea></div>
</article>""")

    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>错图人工审查</title>
<style>
body{{margin:0;font:14px Arial,sans-serif;background:#f4f5f3;color:#172018}}header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #ccd2ca;padding:12px 20px;display:flex;gap:14px;align-items:center}}button{{padding:8px 13px;border:1px solid #176b3a;background:#176b3a;color:#fff;cursor:pointer}}main{{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}}.card{{background:#fff;border:1px solid #d8ddd6;border-radius:6px;overflow:hidden}}img{{width:100%;height:300px;object-fit:contain;background:#202420}}.body{{padding:12px}}.body>div{{margin:5px 0;overflow-wrap:anywhere}}.group{{font-weight:bold;color:#176b3a}}.reasons{{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:10px!important}}label{{font-size:13px}}textarea{{box-sizing:border-box;width:100%;height:55px;margin-top:8px;border:1px solid #aeb7ac}}
</style></head><body><header><strong>错图人工审查</strong><span>{len(records)} 张</span><button onclick="exportCsv()">导出标注 CSV</button></header><main>{''.join(cards)}</main>
<script>const rows={payload};
function quote(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}
function exportCsv(){{const fields=['review_group','rank_in_group','image_id','image_path','ground_truth','candidate_class','teacher_top1_prob','margin','taxonomy_relation','error_reason_labels','review_note'];let lines=[fields.map(quote).join(',')];document.querySelectorAll('.card').forEach((card,i)=>{{let r={{...rows[i]}};r.error_reason_labels=[...card.querySelectorAll('input:checked')].map(x=>x.value).join(';');r.review_note=card.querySelector('textarea').value;lines.push(fields.map(f=>quote(r[f])).join(','))}});let blob=new Blob(['\\ufeff'+lines.join('\\n')],{{type:'text/csv;charset=utf-8'}});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='manual_review_completed.csv';a.click();URL.revokeObjectURL(a.href)}}
</script></body></html>"""
    output.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/data/misdetection_splits/train_natural.csv")
    parser.add_argument("--image-root", default="data/inat2021/plants/val")
    parser.add_argument("--output-dir", default="output/analysis/misclassification_review")
    parser.add_argument("--per-group", type=int, default=50)
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of symlink review images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if input_path.name != "train_natural.csv":
        raise ValueError("Exploratory analysis is restricted to train_natural.csv; test.csv is forbidden.")
    if args.per_group < 1:
        raise ValueError("--per-group must be positive")
    frame = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame["image_id"].duplicated().any():
        raise ValueError("Input must be non-empty and image_id must be unique")
    for column in ("is_correct", "error_label"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0/1")
    if not (frame["error_label"] == 1 - frame["is_correct"]).all():
        raise ValueError("error_label must equal 1 - is_correct")
    if not frame["teacher_top1_prob"].between(0, 1).all():
        raise ValueError("teacher_top1_prob must be in [0, 1]")

    frame = frame.copy()
    frame["taxonomy_relation"] = [
        relationship(t, p) for t, p in zip(frame["ground_truth"], frame["candidate_class"])
    ]
    frame["confidence_bin"] = pd.cut(
        frame["teacher_top1_prob"], CONFIDENCE_BINS, labels=CONFIDENCE_LABELS
    )
    errors = frame[frame["error_label"] == 1].copy()
    output = Path(args.output_dir)
    tables = output / "tables"
    images = output / "review_images"
    tables.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    summary = {
        "input": str(input_path.resolve()), "samples": len(frame), "errors": len(errors),
        "error_rate": len(errors) / len(frame), "per_review_group": args.per_group,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    percent_table(errors["ground_truth"], "ground_truth").to_csv(tables / "errors_by_true_class.csv", index=False)
    percent_table(errors["candidate_class"], "candidate_class").to_csv(tables / "errors_by_predicted_class.csv", index=False)
    percent_table(errors["taxonomy_relation"], "taxonomy_relation").to_csv(tables / "error_taxonomy_relations.csv", index=False)

    confusion = (errors.groupby(["ground_truth", "candidate_class", "taxonomy_relation"], observed=True)
                 .agg(count=("image_id", "size"), mean_confidence=("teacher_top1_prob", "mean"), mean_margin=("margin", "mean"))
                 .reset_index().sort_values(["count", "mean_confidence"], ascending=[False, False]))
    confusion.to_csv(tables / "confusion_pairs.csv", index=False)
    class_stats = (frame.groupby("ground_truth", observed=True)
                   .agg(samples=("image_id", "size"), errors=("error_label", "sum"), mean_confidence=("teacher_top1_prob", "mean"))
                   .reset_index())
    class_stats["error_rate"] = class_stats["errors"] / class_stats["samples"]
    class_stats.sort_values(["errors", "error_rate"], ascending=False).to_csv(tables / "class_error_rates.csv", index=False)
    bin_stats = (frame.groupby(["confidence_bin", "error_label"], observed=False)
                 .agg(samples=("image_id", "size"), mean_margin=("margin", "mean"), same_genus=("taxonomy_relation", lambda x: (x == "same_genus").mean()), same_family=("taxonomy_relation", lambda x: x.isin(["same_genus", "same_family_different_genus"]).mean()))
                 .reset_index())
    bin_stats["proportion_of_all_errors"] = bin_stats["samples"].where(bin_stats["error_label"] == 1, 0) / max(len(errors), 1)
    bin_stats.to_csv(tables / "confidence_bins.csv", index=False)

    queue = select_review_groups(frame, args.per_group)
    root = Path(args.image_root)
    review_paths, missing_images = [], []
    for row in queue.itertuples():
        source = root / row.image_path
        group_dir = images / row.review_group
        group_dir.mkdir(exist_ok=True)
        destination = group_dir / f"{row.rank_in_group:03d}_{row.image_id}{source.suffix.lower()}"
        if not source.is_file():
            missing_images.append(str(source))
            review_paths.append("")
            continue
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if args.copy_images:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(source.resolve())
        review_paths.append(destination.relative_to(output).as_posix())
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} review images are missing; first: {missing_images[0]}")
    queue["review_image"] = review_paths
    queue["error_reason_labels"] = ""
    queue["review_note"] = ""
    queue.to_csv(output / "manual_review_queue.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_gallery(queue, output / "review_gallery.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Review gallery: {(output / 'review_gallery.html').resolve()}")


if __name__ == "__main__":
    main()
