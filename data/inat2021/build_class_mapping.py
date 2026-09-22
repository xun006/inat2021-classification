# build_class_mapping.py
import json
from pathlib import Path
from torchvision.datasets import ImageFolder

data_root = Path("/mnt/hdd8t/Mingle/xyyy/data/inat2021/plants")

train_set = ImageFolder(data_root / "train")
val_set = ImageFolder(data_root / "val")

# 教师模型训练时的映射：ImageFolder 按 train 类别文件夹名排序生成
class_to_idx = train_set.class_to_idx
idx_to_class = {str(idx): class_name for class_name, idx in class_to_idx.items()}

print(f"训练类别数: {len(class_to_idx)}")
print(f"验证类别数: {len(val_set.class_to_idx)}")
print("前 10 个类别映射：")
for class_name, idx in list(class_to_idx.items())[:10]:
    print(f"{idx} -> {class_name}")

# 若 val 中每个类别都有图片，两者映射应完全一致
if val_set.class_to_idx != class_to_idx:
    raise RuntimeError(
        "train 与 val 的 ImageFolder 类别映射不一致。"
        "后续推理必须始终使用 train 生成的 class_to_idx。"
    )

with open(data_root / "class_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

with open(data_root / "idx_to_class.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

print("已生成 class_to_idx.json 与 idx_to_class.json")