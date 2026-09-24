"""Data loading for Stage-1 training.

Provides:
  * Standard ImageFolder loaders for L1 (no contrastive loss).
  * PK-batch-sampler + loader for L3 (supervised contrastive loss), ensuring
    every batch contains P classes x K samples so each anchor has K-1
    positives.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms():
    """Train / val transforms.

    Train: RandomResizedCrop(224) + HFlip.
    Val:   Resize(256) + CenterCrop(224)  (matches teacher eval protocol).
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def validate_dataset(data_path: Path, class_map_path: Path, num_classes: int):
    """Ensure ImageFolder structure matches class_to_idx.json."""
    train_dir, val_dir = data_path / "train", data_path / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected ImageFolder splits: {train_dir} and {val_dir}"
        )
    with class_map_path.open(encoding="utf-8") as f:
        expected = json.load(f)
    if len(expected) != num_classes or sorted(expected.values()) != list(range(num_classes)):
        raise ValueError("class_to_idx.json must contain a contiguous 0..N-1 mapping")
    folder_classes = sorted(x.name for x in train_dir.iterdir() if x.is_dir())
    imagefolder_mapping = {name: idx for idx, name in enumerate(folder_classes)}
    if imagefolder_mapping != expected:
        raise ValueError(
            "train ImageFolder alphabetical indices do not match class_to_idx.json"
        )
    val_classes = sorted(x.name for x in val_dir.iterdir() if x.is_dir())
    if val_classes != folder_classes:
        raise ValueError("classifier/val class folders differ from classifier/train")
    return expected


# ---------------------------------------------------------------------------
# PK-BatchSampler
# ---------------------------------------------------------------------------

class PKBatchSampler(Sampler):
    """PK batch sampling: each batch draws P classes, K samples per class.

    Yields lists of indices (one list per batch).  Batch size = P * K.
    Guarantees every anchor has at least K-1 in-batch positives (required
    by SupCon).
    """

    def __init__(self, labels, P: int, K: int, seed: int = 0):
        self.labels = np.asarray(labels)
        self.P = P
        self.K = K
        self.rng = np.random.RandomState(seed)

        self.class_to_indices = {}
        for c in np.unique(self.labels):
            idx = np.where(self.labels == c)[0]
            self.class_to_indices[int(c)] = idx

        self.num_classes = len(self.class_to_indices)
        self.num_batches = max(1, self.num_classes // P)

    def __iter__(self):
        classes = list(self.class_to_indices.keys())
        for _ in range(self.num_batches):
            selected = self.rng.choice(classes, size=self.P, replace=False)
            batch = []
            for c in selected:
                pool = self.class_to_indices[int(c)]
                if len(pool) >= self.K:
                    chosen = self.rng.choice(pool, size=self.K, replace=False)
                else:
                    chosen = self.rng.choice(pool, size=self.K, replace=True)
                batch.extend(chosen.tolist())
            yield batch

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Loader builders
# ---------------------------------------------------------------------------

def build_standard_loaders(
    data_path: Path,
    batch_size: int = 128,
    num_workers: int = 12,
    pin_memory: bool = True,
):
    """Standard shuffled train + sequential val loaders (for L1)."""
    train_tf, val_tf = build_transforms()
    train_set = datasets.ImageFolder(data_path / "train", transform=train_tf)
    val_set = datasets.ImageFolder(data_path / "val", transform=val_tf)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_set, val_set


def build_pk_loaders(
    data_path: Path,
    P: int = 32,
    K: int = 4,
    num_workers: int = 12,
    pin_memory: bool = True,
    seed: int = 0,
):
    """PK-sampled train + sequential val loaders (for L1+L3, L2+L3).

    Effective batch size = P * K.
    """
    train_tf, val_tf = build_transforms()
    train_set = datasets.ImageFolder(data_path / "train", transform=train_tf)
    val_set = datasets.ImageFolder(data_path / "val", transform=val_tf)

    batch_sampler = PKBatchSampler(train_set.targets, P=P, K=K, seed=seed)
    train_loader = DataLoader(
        train_set, batch_sampler=batch_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=P * K, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_set, val_set
