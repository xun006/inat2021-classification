from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader


def build_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(int(image_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def build_dataset(data_root: str | Path, split: str, image_size: int = 224):
    from torchvision.datasets import ImageFolder

    allowed = {"detector_train", "detector_calibration", "official_val"}
    if split not in allowed:
        raise ValueError(f"Unknown split {split!r}")
    return ImageFolder(Path(data_root) / split, transform=build_transform(image_size))


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
