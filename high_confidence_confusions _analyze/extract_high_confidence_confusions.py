#!/usr/bin/env python3
"""Extract high-confidence confusion pairs and reference images for review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "image_path", "image_id", "ground_truth", "candidate_class",
    "is_correct", "teacher_top1_prob", "margin",
}


def taxonomy_relation(true_name: str, predicted_name: str) -> str:
    true_parts = str(true_name).split("_", 7)
    pred_parts = str(predicted_name).split("_", 7)
    if len(true_parts) != 8 or len(pred_parts) != 8:
        return "unparsed"
    if true_parts[5:7] == pred_parts[5:7]:
        return "same_genus"
    if true_parts[5] == pred_parts[5]:
        return "same_family_different_genus"
    if true_parts[4] == pred_parts[4]:
        return "same_order_different_family"
    return "different_order"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/data/teacher_val_predictions.csv")
    parser.add_argument("--image-root", default="data/inat2021/plants/val")
    parser.add_argument("--output-dir", default="output/analysis/high_confidence_confusions")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--references-per-class", type=int, default=3)
    return parser.parse_args()


def safe_link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())
    return destination.name


def choose_references(frame: pd.DataFrame, class_name: str, count: int) -> tuple[pd.DataFrame, str]:
    correct = frame[(frame["ground_truth"] == class_name) & (frame["is_correct"] == 1)]
    if not correct.empty:
        pool, source = correct, "correct_prediction"
    else:
        pool = frame[frame["ground_truth"] == class_name]
        source = "fallback_any_labeled_image"
    return pool.sort_values(["teacher_top1_prob", "image_id"], ascending=[False, True]).head(count), source


def image_strip(title: str, paths: list[str], labels: list[str]) -> str:
    items = "".join(
        f'<figure><img loading="lazy" src="{html.escape(path)}"><figcaption>{html.escape(label)}</figcaption></figure>'
        for path, label in zip(paths, labels)
    )
    return f'<section><h3>{html.escape(title)}</h3><div class="strip">{items}</div></section>'


def write_gallery(path: Path, threshold: float, pair_rows: list[dict]) -> None:
    blocks = []
    for row in pair_rows:
        blocks.append(f"""
<article class="pair">
 <header><strong>#{row['pair_rank']} · {row['error_count']} 个错误 · {html.escape(row['taxonomy_relation'])}</strong><span>平均置信度 {row['mean_confidence']:.4f} · 最高 {row['max_confidence']:.4f}</span></header>
 <div class="names"><b>真实：</b>{html.escape(row['ground_truth'])}<br><b>误判为：</b>{html.escape(row['candidate_class'])}</div>
 <div class="columns">
  {image_strip('高置信度错图', row['error_paths'], row['error_labels'])}
  {image_strip('真实类别参考图', row['true_paths'], row['true_labels'])}
  {image_strip('预测类别参考图', row['pred_paths'], row['pred_labels'])}
 </div>
</article>""")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>高置信度混淆对 ≥ {threshold:.2f}</title><style>
body{{margin:0;background:#f3f5f2;color:#172018;font:14px Arial,sans-serif}}body>header{{position:sticky;top:0;z-index:2;padding:14px 20px;background:#fff;border-bottom:1px solid #cbd2c9}}main{{padding:16px;max-width:1600px;margin:auto}}.pair{{background:#fff;border:1px solid #d3d9d1;border-radius:6px;margin-bottom:16px;overflow:hidden}}.pair>header{{display:flex;justify-content:space-between;gap:12px;padding:10px 14px;background:#eaf1e9}}.names{{padding:10px 14px;overflow-wrap:anywhere;line-height:1.6}}.columns{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:0 14px 14px}}h3{{font-size:14px;margin:5px 0 8px}}.strip{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}}figure{{margin:0;min-width:0}}img{{width:100%;aspect-ratio:1/1;object-fit:contain;background:#202420}}figcaption{{font-size:11px;overflow-wrap:anywhere;margin-top:3px}}@media(max-width:900px){{.columns{{grid-template-columns:1fr}}}}
</style></head><body><header><strong>Top-1 ≥ {threshold:.2f} 的错误混淆对</strong> · {len(pair_rows)} 对</header><main>{''.join(blocks)}</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.references_per_class < 1:
        raise ValueError("--references-per-class must be positive")
    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Thresholds must be in [0, 1]")

    frame = pd.read_csv(args.input)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["is_correct"] = pd.to_numeric(frame["is_correct"], errors="raise").astype(int)
    root, output = Path(args.image_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"input": str(Path(args.input).resolve()), "thresholds": {}, "sets_are_cumulative": True}

    for threshold in thresholds:
        threshold_name = f"ge_{threshold:.2f}"
        threshold_dir = output / threshold_name
        image_dir = threshold_dir / "images"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        errors = frame[(frame["is_correct"] == 0) & (frame["teacher_top1_prob"] >= threshold)].copy()
        errors = errors.sort_values(["teacher_top1_prob", "image_id"], ascending=[False, True])
        grouped = errors.groupby(["ground_truth", "candidate_class"], sort=False)
        pair_summary = (grouped.agg(
            error_count=("image_id", "size"),
            mean_confidence=("teacher_top1_prob", "mean"),
            max_confidence=("teacher_top1_prob", "max"),
            mean_margin=("margin", "mean"),
        ).reset_index().sort_values(["error_count", "max_confidence", "ground_truth", "candidate_class"], ascending=[False, False, True, True]))
        pair_summary["taxonomy_relation"] = [
            taxonomy_relation(true_name, pred_name)
            for true_name, pred_name in zip(pair_summary["ground_truth"], pair_summary["candidate_class"])
        ]
        pair_summary.insert(0, "pair_rank", range(1, len(pair_summary) + 1))
        pair_summary.to_csv(threshold_dir / "confusion_pairs.csv", index=False)
        errors.to_csv(threshold_dir / "high_confidence_errors.csv", index=False)

        gallery_rows, ref_records, missing_images = [], [], []
        for pair in pair_summary.itertuples(index=False):
            pair_errors = errors[(errors["ground_truth"] == pair.ground_truth) & (errors["candidate_class"] == pair.candidate_class)]
            pair_folder = image_dir / f"pair_{pair.pair_rank:04d}"
            row = pair._asdict()
            for kind, selected, reference_source in (
                ("error", pair_errors, "high_confidence_error"),
                ("true", *choose_references(frame, pair.ground_truth, args.references_per_class)),
                ("pred", *choose_references(frame, pair.candidate_class, args.references_per_class)),
            ):
                paths, labels = [], []
                for number, sample in enumerate(selected.itertuples(index=False), 1):
                    source = root / sample.image_path
                    if not source.is_file():
                        missing_images.append(str(source))
                        continue
                    destination = pair_folder / f"{kind}_{number:02d}_{sample.image_id}{source.suffix.lower()}"
                    safe_link(source, destination)
                    relative = destination.relative_to(threshold_dir).as_posix()
                    paths.append(relative)
                    labels.append(f"ID {sample.image_id} · p={sample.teacher_top1_prob:.4f} · {reference_source}")
                    ref_records.append({
                        "pair_rank": pair.pair_rank, "kind": kind, "reference_source": reference_source,
                        "image_id": sample.image_id, "image_path": sample.image_path,
                        "ground_truth": sample.ground_truth, "candidate_class": sample.candidate_class,
                        "teacher_top1_prob": sample.teacher_top1_prob, "linked_image": relative,
                    })
                row[f"{kind}_paths"] = paths
                row[f"{kind}_labels"] = labels
            gallery_rows.append(row)
        if missing_images:
            raise FileNotFoundError(f"{len(missing_images)} images missing; first: {missing_images[0]}")
        pd.DataFrame(ref_records).to_csv(threshold_dir / "review_images.csv", index=False)
        write_gallery(threshold_dir / "confusion_gallery.html", threshold, gallery_rows)
        manifest["thresholds"][f"{threshold:.2f}"] = {
            "errors": len(errors), "confusion_pairs": len(pair_summary),
            "review_image_links": len(ref_records),
        }

    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    links = "".join(
        f'<li><a href="ge_{value:.2f}/confusion_gallery.html">Top-1 ≥ {value:.2f}</a></li>'
        for value in thresholds
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>高置信度混淆分析</title>'
        '<style>body{font:16px Arial,sans-serif;max-width:720px;margin:50px auto;line-height:2;color:#172018}'
        'a{color:#176b3a}</style></head><body><h1>高置信度混淆分析</h1><p>三个阈值为累计集合。</p><ul>'
        + links + '</ul></body></html>', encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
