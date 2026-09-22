from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader


def build_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(int(image_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_loaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
) -> tuple[dict[str, DataLoader], list[str]]:
    from torchvision.datasets import ImageFolder

    root = Path(data_root)
    transform = build_transform(image_size)
    split_names = ("detector_train", "detector_calibration", "official_val")
    datasets = {name: ImageFolder(root / name, transform=transform) for name in split_names}
    reference = datasets["detector_train"].class_to_idx
    for name, dataset in datasets.items():
        if dataset.class_to_idx != reference:
            raise ValueError(f"Class-to-index mapping differs in split {name}")
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "detector_train",
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        for name, dataset in datasets.items()
    }
    return loaders, datasets["detector_train"].classes
