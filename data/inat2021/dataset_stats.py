#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def is_missing(value):
    return value is None or value == ""


def describe(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def aspect_ratio_bins(ratios):
    bins = {
        "portrait_<0.75": 0,
        "portrait_0.75_to_<1.0": 0,
        "square_1.0_to_<1.1": 0,
        "landscape_1.1_to_<1.5": 0,
        "wide_>=1.5": 0,
    }

    for ratio in ratios:
        if ratio < 0.75:
            bins["portrait_<0.75"] += 1
        elif ratio < 1.0:
            bins["portrait_0.75_to_<1.0"] += 1
        elif ratio < 1.1:
            bins["square_1.0_to_<1.1"] += 1
        elif ratio < 1.5:
            bins["landscape_1.1_to_<1.5"] += 1
        else:
            bins["wide_>=1.5"] += 1

    return bins


def analyze_split(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {c["id"]: c for c in data["categories"]}
    images = data["images"]
    annotations = data["annotations"]

    class_counts = Counter(a["category_id"] for a in annotations)
    samples_per_class = list(class_counts.values())

    widths = [img["width"] for img in images if not is_missing(img.get("width"))]
    heights = [img["height"] for img in images if not is_missing(img.get("height"))]
    ratios = [
        round(img["width"] / img["height"], 6)
        for img in images
        if img.get("width") and img.get("height")
    ]

    gps_missing = sum(
        is_missing(img.get("latitude")) or is_missing(img.get("longitude"))
        for img in images
    )
    gps_both_missing = sum(
        is_missing(img.get("latitude")) and is_missing(img.get("longitude"))
        for img in images
    )
    date_missing = sum(is_missing(img.get("date")) for img in images)
    license_missing = sum(is_missing(img.get("license")) for img in images)
    rights_holder_missing = sum(is_missing(img.get("rights_holder")) for img in images)

    taxonomy = {}
    for field in ("genus", "family", "order"):
        taxonomy[field] = len({
            c.get(field) for c in categories.values()
            if not is_missing(c.get(field))
        })

    license_counts = Counter(
        str(img.get("license"))
        for img in images
        if not is_missing(img.get("license"))
    )

    total = len(images)
    summary = {
        "source_file": str(json_path),
        "images": total,
        "annotations": len(annotations),
        "categories_total": len(categories),
        "categories_with_samples": len(class_counts),
        "taxonomy_unique_counts": taxonomy,
        "samples_per_class": describe(samples_per_class),
        "image_width": describe(widths),
        "image_height": describe(heights),
        "aspect_ratio_width_div_height": describe(ratios),
        "aspect_ratio_distribution": aspect_ratio_bins(ratios),
        "missing_fields": {
            "gps_any_coordinate_missing": {
                "count": gps_missing,
                "rate_percent": round(gps_missing / total * 100, 4),
            },
            "gps_both_coordinates_missing": {
                "count": gps_both_missing,
                "rate_percent": round(gps_both_missing / total * 100, 4),
            },
            "date": {
                "count": date_missing,
                "rate_percent": round(date_missing / total * 100, 4),
            },
            "license": {
                "count": license_missing,
                "rate_percent": round(license_missing / total * 100, 4),
            },
            "rights_holder": {
                "count": rights_holder_missing,
                "rate_percent": round(rights_holder_missing / total * 100, 4),
            },
        },
        "license_distribution": dict(sorted(license_counts.items())),
    }

    class_rows = []
    for category_id, category in sorted(categories.items()):
        class_rows.append({
            "category_id": category_id,
            "scientific_name": category.get("name", ""),
            "common_name": category.get("common_name", ""),
            "genus": category.get("genus", ""),
            "family": category.get("family", ""),
            "order": category.get("order", ""),
            "image_dir_name": category.get("image_dir_name", ""),
            "image_count": class_counts.get(category_id, 0),
        })

    return summary, class_rows


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_json")
    parser.add_argument("val_json")
    parser.add_argument("--output-dir", default="dataset_statistics")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary, train_rows = analyze_split(args.train_json)
    val_summary, val_rows = analyze_split(args.val_json)

    report = {
        "train": train_summary,
        "val": val_summary,
        "notes": {
            "aspect_ratio": "width / height；小于 1 为竖图，大于 1 为横图",
            "gps_missing": "纬度或经度任一缺失即计为 GPS 缺失",
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_csv(train_rows, output_dir / "train_class_counts.csv")
    write_csv(val_rows, output_dir / "val_class_counts.csv")

    print(f"统计完成，结果目录：{output_dir}")
    print(f"训练集：{train_summary['images']} 张，{train_summary['categories_with_samples']} 类")
    print(f"验证集：{val_summary['images']} 张，{val_summary['categories_with_samples']} 类")
    print("训练集每类样本数：", train_summary["samples_per_class"])
    print("验证集每类样本数：", val_summary["samples_per_class"])


if __name__ == "__main__":
    main()