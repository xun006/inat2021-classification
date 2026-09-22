"""Strict image/metadata alignment for the new detector splits."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from teacher import load_class_map, validation_transform


REQUIRED = {"sample_id", "split", "dataset_index", "image_path", "ground_truth_index",
            "candidate_index", "error_label"}


class DetectorDataset(Dataset):
    def __init__(self, data_root: Path, split: str, predictions_csv: Path,
                 class_map_path: Path, max_samples: Optional[int] = None,
                 balanced_subset: bool = False, subset_seed: int = 42):
        self.root = (data_root / split).resolve()
        self.split = split
        self.frame = pd.read_csv(predictions_csv)
        if missing := REQUIRED - set(self.frame.columns):
            raise ValueError(f"{predictions_csv} missing columns: {sorted(missing)}")
        if set(self.frame["split"].unique()) != {split}:
            raise ValueError(f"{predictions_csv} is not exclusively split={split}")
        if not self.frame["sample_id"].is_unique or not self.frame["dataset_index"].is_unique:
            raise ValueError(f"{predictions_csv} has duplicate identities")
        if max_samples is not None:
            if balanced_subset:
                per_label = max_samples // 2
                parts = [group.sample(n=min(per_label, len(group)), random_state=subset_seed)
                         for _, group in self.frame.groupby("error_label")]
                self.frame = pd.concat(parts).sort_values("dataset_index").reset_index(drop=True)
            else:
                self.frame = self.frame.sample(
                    n=min(max_samples, len(self.frame)), random_state=subset_seed
                ).sort_values("dataset_index").reset_index(drop=True)
        class_map = load_class_map(class_map_path)
        for row in self.frame.itertuples(index=False):
            path = self.root / row.image_path
            class_name = Path(row.image_path).parts[0]
            if not path.is_file() or class_map.get(class_name) != int(row.ground_truth_index):
                raise ValueError(f"metadata/image mismatch: {path}")
        self.transform = validation_transform()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.root / row.image_path
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return {
            "image": image,
            "error_label": float(row.error_label),
            "candidate_index": int(row.candidate_index),
            "sample_id": row.sample_id,
        }
