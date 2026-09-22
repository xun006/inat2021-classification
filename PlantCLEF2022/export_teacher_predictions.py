#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models_vit


class ImageFolderWithPath(datasets.ImageFolder):
    """ImageFolder，同时返回图片绝对路径。"""

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image_path, _ = self.samples[index]
        return image, target, image_path


def normalize_path(path_str: str) -> str:
    return str(path_str).replace("\\", "/").lstrip("./")


def load_image_id_lookup(json_path: Path):
    """
    支持 iNat 常见 COCO 格式：
    {"images": [{"id": ..., "file_name": ...}, ...], ...}

    返回：
    - 按相对路径匹配的 image_id
    - 按唯一文件名匹配的 image_id
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise ValueError(
            f"{json_path} 不是预期的 COCO/iNat JSON 格式，"
            "应包含顶层字段 images。"
        )

    by_path = {}
    by_name_candidates = defaultdict(list)

    for item in data["images"]:
        if "id" not in item or "file_name" not in item:
            raise ValueError("images 中每条记录必须同时包含 id 和 file_name。")

        file_name = normalize_path(item["file_name"])
        image_id = item["id"]

        by_path[file_name] = image_id
        by_name_candidates[Path(file_name).name].append(image_id)

    # 仅当一个文件名唯一时才允许 basename 匹配，防止同名图像匹配错误
    by_unique_name = {
        name: ids[0]
        for name, ids in by_name_candidates.items()
        if len(ids) == 1
    }
    return by_path, by_unique_name


def find_image_id(image_path: Path, val_root: Path, by_path, by_unique_name):
    """根据 val 文件路径匹配 val_plants.json 中的 image_id。"""
    rel_path = normalize_path(image_path.relative_to(val_root))

    candidates = [
        rel_path,
        f"val/{rel_path}",
        normalize_path(str(image_path)),
    ]

    for candidate in candidates:
        if candidate in by_path:
            return by_path[candidate]

    filename = image_path.name
    if filename in by_unique_name:
        return by_unique_name[filename]

    raise KeyError(
        "无法从 val_plants.json 中找到 image_id：\n"
        f"  图片: {image_path}\n"
        f"  相对路径: {rel_path}\n"
        "请检查 JSON 中 images[].file_name 与 val 文件夹内路径的对应关系。"
    )


def load_checkpoint(model, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    '''
    # 兼容部分 DDP 保存的 module. 前缀
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    '''
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    message = model.load_state_dict(state_dict, strict=True)
    print(f"checkpoint 已加载: {checkpoint_path}")
    print(message)


def main(args):
    device = torch.device(args.device)
    val_root = Path(args.val_root).resolve()
    mapping_path = Path(args.mapping_path).resolve()
    val_json_path = Path(args.val_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_features = Path(args.output_features).resolve()

    if not val_root.is_dir():
        raise FileNotFoundError(f"val 目录不存在: {val_root}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"类别映射不存在: {mapping_path}")
    if not val_json_path.is_file():
        raise FileNotFoundError(f"val JSON 不存在: {val_json_path}")

    # 读取“训练时生成”的权威映射
    with open(mapping_path, "r", encoding="utf-8") as f:
        train_class_to_idx = {
            class_name: int(index)
            for class_name, index in json.load(f).items()
        }

    idx_to_class = {
        int(index): class_name
        for class_name, index in train_class_to_idx.items()
    }

    if len(train_class_to_idx) != args.num_classes:
        raise RuntimeError(
            f"映射类别数为 {len(train_class_to_idx)}，"
            f"但 --num-classes 为 {args.num_classes}。"
        )

    # 与 PlantCLEF2022/main_linprobe.py 完全一致的验证预处理
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ImageFolderWithPath(val_root, transform=transform_val)

    # 硬校验：验证目录的 ImageFolder 索引必须等于训练目录的索引
    if dataset.class_to_idx != train_class_to_idx:
        raise RuntimeError(
            "val 的类别映射与训练映射不一致，已停止。\n"
            "请勿继续推理或手动修改映射。"
        )

    print(f"验证图片数: {len(dataset)}")
    print(f"类别数: {len(dataset.classes)}")

    # 对应 main_linprobe.py：
    # vit_large_patch16 + global_pool=False（默认 CLS token）
    model = models_vit.vit_large_patch16(
        num_classes=args.num_classes,
        global_pool=True,
    )

    # 对应线性探测保存 checkpoint 时的分类头结构
    original_head = model.head
    model.head = nn.Sequential(
        nn.BatchNorm1d(
            original_head.in_features,
            affine=False,
            eps=1e-6,
        ),
        original_head,
    )

    if model.head[-1].out_features != args.num_classes:
        raise RuntimeError("模型分类头输出维度与 --num-classes 不一致。")

    load_checkpoint(model, Path(args.checkpoint).resolve())
    model.to(device)
    model.eval()

    by_path, by_unique_name = load_image_id_lookup(val_json_path)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    topk = min(args.topk, args.num_classes)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_features.parent.mkdir(parents=True, exist_ok=True)

    # The trained checkpoint uses global average pooling, so this is the
    # 1024-D pre-head image representation (not a CLS token).  A memmap keeps
    # memory usage bounded while exporting the complete validation set.
    feature_dim = model.head[-1].in_features
    feature_store = np.lib.format.open_memmap(
        output_features,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), feature_dim),
    )

    fieldnames = [
        "image_path",
        "image_id",
        "ground_truth",
        "candidate_class",
        "is_correct",
        "teacher_top1_prob",
        "margin",
        "entropy",
        "max_logit",
        "energy",
        "feature_row",
        "teacher_topk_classes",
        "teacher_topk_probs",
    ]

    total = 0
    correct = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.inference_mode():
            for batch_index, (images, targets, paths) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                image_features = model.forward_features(images)
                logits = model.head(image_features)
                probabilities = torch.softmax(logits, dim=1)

                top_probs, top_indices = torch.topk(
                    probabilities, k=topk, dim=1
                )

                candidate_indices = top_indices[:, 0]
                is_correct = candidate_indices.eq(targets)

                # Top-1 与 Top-2 的概率差
                margin = top_probs[:, 0] - top_probs[:, 1]

                # Shannon entropy: -sum(p * log(p))
                entropy = -(
                    probabilities * probabilities.clamp_min(1e-12).log()
                ).sum(dim=1)
                max_logit = logits.max(dim=1).values
                # Standard energy score with temperature T=1.
                energy = -torch.logsumexp(logits, dim=1)

                batch_start = total
                feature_store[batch_start:batch_start + targets.size(0)] = (
                    image_features.detach().cpu().float().numpy()
                )

                for i, image_path_str in enumerate(paths):
                    image_path = Path(image_path_str)
                    image_id = find_image_id(
                        image_path,
                        val_root,
                        by_path,
                        by_unique_name,
                    )

                    gt_index = int(targets[i].item())
                    candidate_index = int(candidate_indices[i].item())

                    topk_indices_i = top_indices[i].tolist()
                    topk_probs_i = [
                        round(float(value), 8)
                        for value in top_probs[i].tolist()
                    ]

                    writer.writerow({
                        "image_path": str(image_path.relative_to(val_root)),
                        "image_id": image_id,
                        "ground_truth": idx_to_class[gt_index],
                        "candidate_class": idx_to_class[candidate_index],
                        "is_correct": int(is_correct[i].item()),
                        "teacher_top1_prob": round(
                            float(top_probs[i, 0].item()), 8
                        ),
                        "margin": round(float(margin[i].item()), 8),
                        "entropy": round(float(entropy[i].item()), 8),
                        "max_logit": round(float(max_logit[i].item()), 8),
                        "energy": round(float(energy[i].item()), 8),
                        "feature_row": batch_start + i,
                        "teacher_topk_classes": json.dumps([
                            idx_to_class[int(index)]
                            for index in topk_indices_i
                        ], ensure_ascii=False),
                        "teacher_topk_probs": json.dumps(topk_probs_i),
                    })

                total += targets.size(0)
                correct += int(is_correct.sum().item())

                if (batch_index + 1) % 20 == 0:
                    print(
                        f"[{total}/{len(dataset)}] "
                        f"当前 Top-1: {correct / total:.4%}"
                    )

    feature_store.flush()
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation_root": str(val_root),
        "predictions_csv": str(output_csv),
        "features_npy": str(output_features),
        "samples": total,
        "feature_shape": [total, feature_dim],
        "feature_dtype": "float32",
        "feature_definition": "ViT pre-head global-average-pooled representation",
        "label_definition": "is_correct=1 iff candidate_class equals ground_truth",
        "top1_accuracy": correct / total,
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n导出完成")
    print(f"CSV: {output_csv}")
    print(f"总图片数: {total}")
    print(f"教师 Top-1 正确数: {correct}")
    print(f"教师 Top-1 错误数: {total - correct}")
    print(f"教师 Top-1 准确率: {correct / total:.4%}")
    print(f"图像特征: {output_features}")
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="导出 PlantCLEF MAE ViT 教师模型在 iNat2021 Plants val 上的预测 CSV。"
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/hdd8t/Mingle/xyyy/output/inat2021_plants_linprobe/checkpoint-24.pth",
    )
    parser.add_argument(
        "--val-root",
        default="/mnt/hdd8t/Mingle/xyyy/data/inat2021/plants/val",
    )
    parser.add_argument(
        "--val-json",
        default="/mnt/hdd8t/Mingle/xyyy/data/inat2021/plants/val_plants.json",
    )
    parser.add_argument(
        "--mapping-path",
        default="/mnt/hdd8t/Mingle/xyyy/data/inat2021/plants/class_to_idx.json",
    )
    parser.add_argument(
        "--output-csv",
        default="/mnt/hdd8t/Mingle/xyyy/output/data/teacher_val_predictions.csv",
    )
    parser.add_argument(
        "--output-features",
        default="/mnt/hdd8t/Mingle/xyyy/output/data/teacher_val_features.npy",
    )
    parser.add_argument("--num-classes", type=int, default=4271)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    main(args)
