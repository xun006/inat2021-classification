#!/usr/bin/env python3
"""Build reproducible error-analysis tables and image galleries for detector_train."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


HIGH_THRESHOLD = 0.90
LOW_THRESHOLD = 0.50
PAIR_THRESHOLDS = (0.90, 0.95, 0.99)
REQUIRED = {
    "sample_id", "image_path", "ground_truth", "candidate_class",
    "is_correct", "error_label", "teacher_top1_prob", "margin",
}

GROUP_LABELS = {
    "random_high_confidence_error": "高置信度错误（随机 50）",
    "random_high_confidence_correct": "高置信度正确（随机 50）",
    "random_low_confidence_error": "低置信度错误（随机 50）",
    "random_low_confidence_correct": "低置信度正确（随机 50）",
    "extreme_highest_confidence_error": "置信度最高的错误（Top 50）",
    "extreme_lowest_confidence_correct": "置信度最低的正确样本（Top 50）",
}
RELATION_LABELS = {
    "same_genus": "同属",
    "same_family_different_genus": "同科不同属",
    "same_order_different_family": "同目不同科",
    "different_order": "跨目",
    "unparsed": "层级未知",
}


def taxon_display(class_name: str) -> dict[str, str]:
    parts = str(class_name).split("_", 7)
    if len(parts) != 8:
        return {
            "scientific": str(class_name), "hierarchy": "", "full": str(class_name),
            "ranks": [],
        }
    species = parts[7].replace("_", " ")
    return {
        "scientific": f"{parts[6]} {species}",
        "hierarchy": f"{parts[4]} / {parts[5]}",
        "full": str(class_name),
        "ranks": list(zip(
            ("界", "门", "纲", "目", "科", "属", "种"),
            (parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], species),
        )),
    }


def taxonomy_html(taxon: dict[str, object]) -> str:
    ranks = taxon.get("ranks", [])
    if not ranks:
        return f'<div class="taxonomy fallback">{html.escape(str(taxon["full"]))}</div>'
    return '<div class="taxonomy">' + ''.join(
        f'<span class="rank">{html.escape(label)}</span><span class="value">{html.escape(value)}</span>'
        for label, value in ranks
    ) + '</div>'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("MisD/output/patch_cross_attention/teacher_predictions/detector_train.csv"),
    )
    parser.add_argument("--image-root", type=Path, default=Path("MisD/data"))
    parser.add_argument("--output", type=Path, default=Path("MisD/analyse_data/results"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--references-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def relation(true_name: str, pred_name: str) -> str:
    true, pred = str(true_name).split("_", 7), str(pred_name).split("_", 7)
    if len(true) != 8 or len(pred) != 8:
        return "unparsed"
    if true[5:7] == pred[5:7]:
        return "same_genus"
    if true[5] == pred[5]:
        return "same_family_different_genus"
    if true[4] == pred[4]:
        return "same_order_different_family"
    return "different_order"


def link_image(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())
    return destination.name


def write_sample_gallery(rows: pd.DataFrame, output: Path, title: str) -> None:
    sections = []
    for group, group_rows in rows.groupby("analysis_group", sort=False):
        cards = []
        for row in group_rows.itertuples(index=False):
            true, pred = taxon_display(row.ground_truth), taxon_display(row.candidate_class)
            outcome = "错误" if row.error_label else "正确"
            outcome_class = "bad" if row.error_label else "good"
            cards.append(
                f'<article><img loading="lazy" src="{html.escape(row.review_image)}">'
                f'<div class="card-body"><div class="metrics"><span class="badge {outcome_class}">{outcome}</span>'
                f'<strong>MSP {row.teacher_top1_prob:.4f}</strong><span>Margin {row.margin:.4f}</span></div>'
                f'<dl><dt>真实类别</dt><dd><strong class="folder-name">{html.escape(true["full"])}</strong></dd>'
                f'<dt>预测类别</dt><dd><strong class="folder-name">{html.escape(pred["full"])}</strong></dd></dl>'
                f'<details><summary>查看完整类别名与文件 ID</summary><p>真实：{html.escape(true["full"])}<br>'
                f'预测：{html.escape(pred["full"])}<br>文件：{html.escape(row.sample_id)}</p></details></div></article>'
            )
        sections.append(
            f'<section class="review-group"><h2>{html.escape(GROUP_LABELS.get(group, group))}</h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    output.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;font:16px/1.55 Arial,"Noto Sans SC",sans-serif;background:#f4f6f3;color:#172018}}'
        '.page-head{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid #cbd3c9;padding:16px 24px}.page-head h1{font-size:24px;margin:0}.page-head p{margin:3px 0 0;color:#526052}'
        'main{max-width:1500px;margin:auto;padding:20px}.review-group{margin-bottom:36px}.review-group h2{font-size:21px;margin:0 0 12px;padding-bottom:8px;border-bottom:2px solid #176b3a}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}article{background:#fff;border:1px solid #cfd7cd;border-radius:6px;overflow:hidden;box-shadow:0 2px 8px #17201810}'
        'article>img{display:block;width:100%;height:360px;object-fit:contain;background:#202420}.card-body{padding:14px 16px}.metrics{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}.metrics strong{font-size:17px}.metrics>span:last-child{color:#526052}'
        '.badge{padding:3px 8px;border-radius:4px;font-size:13px;font-weight:bold}.badge.bad{background:#fde7e4;color:#9f251c}.badge.good{background:#e2f3e7;color:#176b3a}'
        'dl{display:grid;grid-template-columns:74px minmax(0,1fr);gap:14px 10px;margin:0}dt{color:#526052;font-weight:bold;padding-top:3px}dd{margin:0;min-width:0}.folder-name{display:block;font-size:15px;line-height:1.5;overflow-wrap:anywhere;word-break:break-word;color:#172018}'
        'details{margin-top:12px;border-top:1px solid #e2e7e0;padding-top:9px}summary{cursor:pointer;color:#176b3a;font-weight:bold}details p{font-size:12px;color:#596359;overflow-wrap:anywhere;margin:8px 0 0}'
        '@media(max-width:520px){main{padding:12px}.grid{grid-template-columns:1fr}article>img{height:310px}.page-head{padding:12px 14px}.page-head h1{font-size:20px}}'
        f'</style></head><body><header class="page-head"><h1>{html.escape(title)}</h1><p>重点比较高置信度错误与低置信度正确样本</p></header><main>{"".join(sections)}</main></body></html>',
        encoding="utf-8",
    )


def choose_references(frame: pd.DataFrame, class_name: str, count: int) -> tuple[pd.DataFrame, str]:
    correct = frame[(frame.ground_truth == class_name) & (frame.is_correct == 1)]
    if len(correct) >= count:
        pool, source = correct, "correct_prediction"
    else:
        pool = frame[frame.ground_truth == class_name]
        source = "fallback_labeled_image"
    return pool.sort_values(["teacher_top1_prob", "sample_id"], ascending=[False, True]).head(count), source


def strip(title: str, paths: list[str], captions: list[str]) -> str:
    figures = "".join(
        f'<figure><img loading="lazy" src="{html.escape(path)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        for path, caption in zip(paths, captions)
    )
    return f'<section><h3>{html.escape(title)}</h3><div class="strip">{figures}</div></section>'


def write_pair_gallery(rows: list[dict], output: Path, threshold: float) -> None:
    blocks = []
    for row in rows:
        true, pred = taxon_display(row["ground_truth"]), taxon_display(row["candidate_class"])
        relation_name = RELATION_LABELS.get(row["taxonomy_relation"], row["taxonomy_relation"])
        blocks.append(
            f'<article><header><div><b>混淆对 #{row["pair_rank"]}</b><span class="badge">{html.escape(relation_name)}</span>'
            f'<span class="badge count">{row["error_count"]} 个错误</span></div>'
            f'<span>平均 MSP <b>{row["mean_confidence"]:.4f}</b> · 最高 <b>{row["max_confidence"]:.4f}</b></span></header>'
            f'<div class="pair-name"><div><small>真实类别（数据集文件夹名）</small><strong>{html.escape(true["full"])}</strong><span>{html.escape(true["scientific"])} · {html.escape(true["hierarchy"])}</span></div>'
            f'<div class="arrow">→</div><div><small>误判为（数据集文件夹名）</small><strong>{html.escape(pred["full"])}</strong><span>{html.escape(pred["scientific"])} · {html.escape(pred["hierarchy"])}</span></div></div>'
            '<div class="cols">'
            + strip("高置信度错图", row["error_paths"], row["error_captions"])
            + strip("真实类别参考图", row["true_paths"], row["true_captions"])
            + strip("预测类别参考图", row["pred_paths"], row["pred_captions"])
            + '</div></article>'
        )
    output.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>混淆对 MSP ≥ {threshold:.2f}</title><style>*{{box-sizing:border-box}}body{{font:16px/1.5 Arial,"Noto Sans SC",sans-serif;background:#f3f5f2;color:#172018;margin:0}}'
        'body>header{position:sticky;top:0;background:white;padding:16px 24px;z-index:2;border-bottom:1px solid #ccd2ca;font-size:18px}'
        'main{max-width:1700px;margin:auto;padding:18px}article{background:white;border:1px solid #d3d9d1;border-radius:6px;margin-bottom:20px;overflow:hidden;box-shadow:0 2px 8px #17201810}'
        'article>header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;background:#eaf1e9}article>header>div{display:flex;align-items:center;gap:8px;flex-wrap:wrap}'
        '.badge{display:inline-block;padding:3px 8px;border-radius:4px;background:#d8eadc;color:#155f34;font-size:13px}.badge.count{background:#fff;color:#384438}'
        '.pair-name{display:grid;grid-template-columns:minmax(0,1fr) 36px minmax(0,1fr);align-items:center;gap:10px;padding:16px;border-bottom:1px solid #e0e5de}.pair-name>div:not(.arrow){min-width:0}.pair-name small,.pair-name span{display:block;color:#667066}.pair-name strong{display:block;font-size:16px;line-height:1.45;overflow-wrap:anywhere;word-break:break-word;margin:3px 0 5px}.arrow{text-align:center;font-size:24px;color:#176b3a}'
        '.cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;padding:10px 16px 18px}.cols h3{font-size:17px;margin:8px 0}.strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}'
        'figure{margin:0;min-width:0}img{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#202420}figcaption{font-size:13px;line-height:1.4;color:#495449;margin-top:6px;overflow-wrap:anywhere}'
        '@media(max-width:1050px){.cols{grid-template-columns:1fr}.strip{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:600px){main{padding:10px}.strip{grid-template-columns:repeat(2,minmax(0,1fr))}.pair-name strong{font-size:18px}article>header{align-items:flex-start;flex-direction:column}}'
        f'</style></head><body><header><b>预测混淆对：MSP ≥ {threshold:.2f}</b> · {len(rows)} 对</header><main>{"".join(blocks)}</main></body></html>',
        encoding="utf-8",
    )


def build_samples(frame: pd.DataFrame, output: Path, size: int, seed: int, image_root: Path) -> dict:
    specs = {
        "random_high_confidence_error": frame[(frame.error_label == 1) & (frame.teacher_top1_prob >= HIGH_THRESHOLD)],
        "random_high_confidence_correct": frame[(frame.error_label == 0) & (frame.teacher_top1_prob >= HIGH_THRESHOLD)],
        "random_low_confidence_error": frame[(frame.error_label == 1) & (frame.teacher_top1_prob <= LOW_THRESHOLD)],
        "random_low_confidence_correct": frame[(frame.error_label == 0) & (frame.teacher_top1_prob <= LOW_THRESHOLD)],
    }
    selected = []
    counts = {}
    for offset, (name, pool) in enumerate(specs.items()):
        if len(pool) < size:
            raise ValueError(f"{name} has only {len(pool)} rows; cannot sample {size}")
        sample = pool.sample(n=size, random_state=seed + offset).copy()
        sample.insert(0, "analysis_group", name)
        counts[name] = {"pool": len(pool), "sampled": len(sample)}
        selected.append(sample)

    extremes = [
        ("extreme_highest_confidence_error", frame[frame.error_label == 1], False),
        ("extreme_lowest_confidence_correct", frame[frame.error_label == 0], True),
    ]
    for name, pool, ascending in extremes:
        sample = pool.sort_values(["teacher_top1_prob", "sample_id"], ascending=[ascending, True]).head(size).copy()
        sample.insert(0, "analysis_group", name)
        counts[name] = {"pool": len(pool), "sampled": len(sample)}
        selected.append(sample)

    queue = pd.concat(selected, ignore_index=True)
    review_paths = []
    for group, group_rows in queue.groupby("analysis_group", sort=False):
        for rank, row in enumerate(group_rows.itertuples(index=False), 1):
            source = image_root / row.sample_id
            destination = output / "images" / group / f"{rank:03d}_{row.sample_id.replace('/', '__')}"
            link_image(source, destination)
            review_paths.append(destination.relative_to(output).as_posix())
    queue["review_image"] = review_paths
    queue.to_csv(output / "review_samples.csv", index=False)
    write_sample_gallery(queue, output / "review_samples.html", "随机四组与 MSP 极端样本")
    return counts


def build_pairs(frame: pd.DataFrame, output: Path, refs: int, image_root: Path) -> dict:
    results = {}
    for threshold in PAIR_THRESHOLDS:
        folder = output / f"msp_ge_{threshold:.2f}"
        folder.mkdir(parents=True, exist_ok=True)
        errors = frame[(frame.error_label == 1) & (frame.teacher_top1_prob >= threshold)].copy()
        errors = errors.sort_values(["teacher_top1_prob", "sample_id"], ascending=[False, True])
        pairs = (errors.groupby(["ground_truth", "candidate_class"], sort=False)
                 .agg(error_count=("sample_id", "size"), mean_confidence=("teacher_top1_prob", "mean"),
                      max_confidence=("teacher_top1_prob", "max"), mean_margin=("margin", "mean"))
                 .reset_index()
                 .sort_values(["error_count", "max_confidence", "ground_truth"], ascending=[False, False, True]))
        pairs.insert(0, "pair_rank", range(1, len(pairs) + 1))
        pairs["taxonomy_relation"] = [relation(a, b) for a, b in zip(pairs.ground_truth, pairs.candidate_class)]
        pairs.to_csv(folder / "confusion_pairs.csv", index=False)
        errors.to_csv(folder / "high_confidence_errors.csv", index=False)

        gallery, image_records = [], []
        for pair in pairs.itertuples(index=False):
            pair_errors = errors[(errors.ground_truth == pair.ground_truth) & (errors.candidate_class == pair.candidate_class)]
            true_refs, true_source = choose_references(frame, pair.ground_truth, refs)
            pred_refs, pred_source = choose_references(frame, pair.candidate_class, refs)
            item = pair._asdict()
            for kind, rows, source_type in (
                ("error", pair_errors, "high_confidence_error"),
                ("true", true_refs, true_source),
                ("pred", pred_refs, pred_source),
            ):
                paths, captions = [], []
                for number, row in enumerate(rows.itertuples(index=False), 1):
                    source = image_root / row.sample_id
                    destination = folder / "images" / f"pair_{pair.pair_rank:04d}" / f"{kind}_{number:02d}_{row.sample_id.replace('/', '__')}"
                    link_image(source, destination)
                    relative = destination.relative_to(folder).as_posix()
                    paths.append(relative)
                    source_label = {
                        "high_confidence_error": "高置信度错误",
                        "correct_prediction": "正确参考图",
                        "fallback_labeled_image": "标签参考图",
                    }.get(source_type, source_type)
                    captions.append(f"MSP {row.teacher_top1_prob:.4f} · {source_label}")
                    image_records.append({
                        "pair_rank": pair.pair_rank, "kind": kind, "source_type": source_type,
                        "sample_id": row.sample_id, "ground_truth": row.ground_truth,
                        "candidate_class": row.candidate_class, "teacher_top1_prob": row.teacher_top1_prob,
                        "linked_image": relative,
                    })
                item[f"{kind}_paths"], item[f"{kind}_captions"] = paths, captions
            gallery.append(item)
        pd.DataFrame(image_records).to_csv(folder / "review_images.csv", index=False)
        write_pair_gallery(gallery, folder / "confusion_gallery.html", threshold)
        results[f"{threshold:.2f}"] = {
            "errors": len(errors), "confusion_pairs": len(pairs), "review_image_links": len(image_records),
            "taxonomy_relations": pairs.taxonomy_relation.value_counts().to_dict(),
        }
    return results


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if set(frame["split"].unique()) != {"detector_train"}:
        raise ValueError("This exploratory analysis must use detector_train only")
    if not (frame.error_label == 1 - frame.is_correct).all():
        raise ValueError("error_label must equal 1 - is_correct")

    args.output.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame["taxonomy_relation"] = [relation(a, b) if a != b else "same_species" for a, b in zip(frame.ground_truth, frame.candidate_class)]
    errors = frame[frame.error_label == 1]

    tables = args.output / "tables"
    tables.mkdir(exist_ok=True)
    class_stats = (frame.groupby("ground_truth").agg(samples=("sample_id", "size"), errors=("error_label", "sum"),
                                                       mean_msp=("teacher_top1_prob", "mean"), mean_margin=("margin", "mean")).reset_index())
    class_stats["error_rate"] = class_stats.errors / class_stats.samples
    class_stats.sort_values(["errors", "error_rate"], ascending=False).to_csv(tables / "class_error_rates.csv", index=False)
    (errors.groupby(["ground_truth", "candidate_class", "taxonomy_relation"])
     .agg(count=("sample_id", "size"), mean_msp=("teacher_top1_prob", "mean"), max_msp=("teacher_top1_prob", "max"))
     .reset_index().sort_values(["count", "max_msp"], ascending=False).to_csv(tables / "all_error_confusion_pairs.csv", index=False))
    errors.taxonomy_relation.value_counts().rename_axis("taxonomy_relation").reset_index(name="count").to_csv(
        tables / "error_taxonomy_relations.csv", index=False
    )

    samples = build_samples(frame, args.output / "sample_review", args.sample_size, args.seed, args.image_root)
    pairs = build_pairs(frame, args.output / "confusion_review", args.references_per_class, args.image_root)
    summary = {
        "source": str(args.input.resolve()), "split": "detector_train", "seed": args.seed,
        "samples": len(frame), "correct": int((frame.error_label == 0).sum()), "errors": len(errors),
        "accuracy": float(frame.is_correct.mean()), "high_threshold": HIGH_THRESHOLD,
        "low_threshold": LOW_THRESHOLD, "sample_groups": samples, "pair_thresholds": pairs,
        "protocol_note": "Use detector_train for exploration/training, detector_calibration for thresholds, official_val once for final evaluation.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    links = ''.join(
        f'<li><a href="confusion_review/msp_ge_{threshold:.2f}/confusion_gallery.html">混淆对 MSP ≥ {threshold:.2f}</a></li>'
        for threshold in PAIR_THRESHOLDS
    )
    (args.output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>新版错图分析</title>'
        '<style>body{font:16px Arial;max-width:850px;margin:50px auto;line-height:2;color:#172018}a{color:#176b3a}</style>'
        '</head><body><h1>新版 detector_train 错图分析</h1><ul>'
        '<li><a href="sample_review/review_samples.html">随机四组与极端样本</a></li>' + links
        + '<li><a href="summary.json">统计摘要 JSON</a></li></ul></body></html>', encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
