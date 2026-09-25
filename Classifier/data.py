import hashlib
import json
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import datasets, transforms


def transform(size, train):
    ops = [transforms.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
           transforms.RandomHorizontalFlip()] if train else [
               transforms.Resize(round(size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
               transforms.CenterCrop(size)]
    return transforms.Compose(ops + [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Images(datasets.ImageFolder):
    def __init__(self, root, mapping, size, train=False):
        super().__init__(root, transform=transform(size, train))
        if self.class_to_idx != mapping:
            raise ValueError(f"Class mapping mismatch at {root}; all class folders must exist")

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, str(Path(self.samples[index][0]).relative_to(self.root)).replace("\\", "/")


def load_mapping(cfg):
    mapping = json.loads(Path(cfg["class_map"]).read_text(encoding="utf-8"))
    if len(mapping) != cfg["num_classes"] or sorted(mapping.values()) != list(range(cfg["num_classes"])):
        raise ValueError("class_map must be contiguous and match num_classes")
    return mapping


def audit(train, val):
    # Detect equal class/file identifiers and hardlink/symlink reuse without reading TB of pixels.
    ids, files = set(), set()
    digest = hashlib.sha256()
    counts = np.bincount(train.targets, minlength=len(train.classes))
    for split, dataset in (("train", train), ("val", val)):
        for filename, label in dataset.samples:
            path = Path(filename)
            relative = path.relative_to(dataset.root).as_posix()
            st = path.stat()
            inode = (st.st_dev, st.st_ino)
            if split == "val" and (relative in ids or inode in files):
                raise ValueError(f"Train/validation overlap: {filename}")
            if split == "train":
                ids.add(relative)
                files.add(inode)
            digest.update(f"{split}:{relative}:{label}:{st.st_size}\n".encode())
    return {"train_samples": len(train), "val_samples": len(val), "train_counts": counts.tolist(),
            "split_fingerprint": digest.hexdigest(), "content_hash_audit": False,
            "min_class_count": int(counts.min()), "max_class_count": int(counts.max()),
            "imbalance_ratio": float(counts.max() / counts.min())}


def image_audit(dataset):
    """Explicit expensive audit: decode every image, fail on corrupt files."""
    dimensions = []
    for filename, _ in dataset.samples:
        with Image.open(filename) as image:
            dimensions.append(image.size)
            image.load()
    sizes = np.asarray(dimensions)
    return {"decoded": len(sizes), "width_quantiles": np.quantile(sizes[:, 0], [0, .25, .5, .75, 1]).tolist(),
            "height_quantiles": np.quantile(sizes[:, 1], [0, .25, .5, .75, 1]).tolist()}
