#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) != 4:
    print(f"用法: {sys.argv[0]} 输入json split(train或val) 输出目录")
    sys.exit(1)

json_path = Path(sys.argv[1])
split = sys.argv[2]
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

plant_categories = [
    c for c in data["categories"]
    if c.get("supercategory") == "Plants"
]
plant_ids = {c["id"] for c in plant_categories}

plant_image_ids = {
    ann["image_id"]
    for ann in data["annotations"]
    if ann["category_id"] in plant_ids
}

plant_images = [
    image for image in data["images"]
    if image["id"] in plant_image_ids
]

plant_annotations = [
    ann for ann in data["annotations"]
    if ann["category_id"] in plant_ids
]

subset = dict(data)
subset["categories"] = plant_categories
subset["images"] = plant_images
subset["annotations"] = plant_annotations

with (out_dir / f"{split}_plants.json").open("w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False)

with (out_dir / f"{split}_plants_paths.txt").open("w", encoding="utf-8") as f:
    for image in plant_images:
        name = image["file_name"].lstrip("/")
        if not name.startswith(f"{split}/"):
            name = f"{split}/{name}"
        f.write(name + "\n")

print(f"{split}:")
print(f"  类别数: {len(plant_categories)}")
print(f"  图片数: {len(plant_images)}")
print(f"  标注数: {len(plant_annotations)}")
