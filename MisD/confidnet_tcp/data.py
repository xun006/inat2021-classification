"""Data loading and split guards for the TCP-ConfiDNet protocol."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from teacher import load_class_map, validation_transform


ALLOWED_DEVELOPMENT_SPLITS = {"detector_train", "detector_calibration"}


class PlantSplit(Dataset):
    def __init__(self, data_root: Path, split: str, class_map_path: Path, allow_official: bool = False) -> None:
        if split == "official_val" and not allow_official:
            raise PermissionError("official_val is blocked outside final evaluation")
        if split not in ALLOWED_DEVELOPMENT_SPLITS | {"official_val"}:
            raise ValueError(f"unsupported detector split: {split}")
        root = data_root / split
        self.dataset = ImageFolder(root, transform=validation_transform())
        expected = load_class_map(class_map_path)
        if self.dataset.class_to_idx != expected:
            raise ValueError(f"class mapping mismatch in {root}")
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        image, target = self.dataset[index]
        path, imagefolder_target = self.dataset.samples[index]
        if target != imagefolder_target:
            raise RuntimeError("ImageFolder target mismatch")
        return {"image": image, "target": target, "index": index, "sample_id": str(Path(path).resolve())}
