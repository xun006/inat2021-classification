"""Shared immutable-cache and frozen-teacher protocol helpers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


RELU_DIR = Path(__file__).resolve().parent
MISD_ROOT = RELU_DIR.parent
PROJECT_ROOT = MISD_ROOT.parent
CONFIDNET_DIR = MISD_ROOT / "confidnet_tcp"
for import_root in (PROJECT_ROOT, CONFIDNET_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

SCHEMA_VERSION = 1
DEFAULT_CLASS_MAP = MISD_ROOT / "data/class_to_idx.json"
EXPECTED_SPLIT_COUNTS = {
    "detector_train": 128130,
    "detector_calibration": 21355,
    "official_val": 42710,
}


def sha256_text_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_class_map(path: Path = DEFAULT_CLASS_MAP) -> dict[str, int]:
    mapping = {name: int(index) for name, index in json.loads(path.read_text(encoding="utf-8")).items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("class map must contain contiguous indices")
    return mapping


def ensure_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def build_classifier_c_teacher(
    checkpoint: Path,
    class_map: Path,
    classifier_config: Path,
    device: str,
) -> torch.nn.Module:
    """Load only the frozen Classifier C teacher; REL-U owns no image encoder."""
    # Delay torchvision/timm imports so cache-only fitting remains independent
    # from the image inference environment.
    from teacher import FrozenTimmTeacher, _load_classifier_c

    model = _load_classifier_c(checkpoint, class_map, classifier_config)
    teacher = FrozenTimmTeacher(model)
    teacher.requires_grad_(False).eval().to(device)
    return teacher


def make_plant_split(
    data_root: Path,
    split: str,
    class_map: Path,
    allow_official: bool = False,
) -> Any:
    from data import PlantSplit

    return PlantSplit(data_root, split, class_map, allow_official=allow_official)


def provenance(
    checkpoint: Path,
    class_map: Path,
    classifier_config: Path,
) -> dict[str, str]:
    return {
        "teacher_kind": "classifier_c",
        "teacher_checkpoint_sha256": file_sha256(checkpoint),
        "class_map_sha256": file_sha256(class_map),
        "classifier_config_sha256": file_sha256(classifier_config),
    }


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "manifest": cache_dir / "manifest.json",
        "logits": cache_dir / "logits.npy",
        "targets": cache_dir / "targets.npy",
        "predictions": cache_dir / "predictions.npy",
        "errors": cache_dir / "errors.npy",
        "sample_keys": cache_dir / "sample_keys.txt",
    }


def load_cache(
    cache_dir: Path,
    expected_split: str,
    expected_provenance: dict[str, str],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = cache_paths(cache_dir)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema in {cache_dir}")
    if manifest.get("split") != expected_split:
        raise ValueError(f"{cache_dir} is not a {expected_split} cache")
    if manifest.get("teacher_provenance") != expected_provenance:
        raise ValueError(f"{cache_dir} belongs to another teacher/config/class map")
    logits = np.load(paths["logits"], mmap_mode="r")
    targets = np.load(paths["targets"], mmap_mode="r")
    predictions = np.load(paths["predictions"], mmap_mode="r")
    errors = np.load(paths["errors"], mmap_mode="r")
    expected_shape = (int(manifest["samples"]), int(manifest["classes"]))
    if logits.shape != expected_shape:
        raise ValueError(f"malformed logits shape in {cache_dir}: {logits.shape}")
    for name, values in (("targets", targets), ("predictions", predictions), ("errors", errors)):
        if values.shape != (expected_shape[0],):
            raise ValueError(f"malformed {name} shape in {cache_dir}")
    keys = paths["sample_keys"].read_text(encoding="utf-8").splitlines()
    if len(keys) != expected_shape[0] or sha256_text_lines(keys) != manifest["sample_keys_sha256"]:
        raise ValueError(f"sample identity mismatch in {cache_dir}")
    if not np.array_equal(errors, predictions != targets):
        raise ValueError(f"cached error labels are inconsistent in {cache_dir}")
    return manifest, logits, targets, predictions, errors


def numpy_batches(values: np.ndarray, batch_size: int) -> Iterator[tuple[int, int, np.ndarray]]:
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        yield start, stop, np.asarray(values[start:stop])


def validate_locked_distance(distance: torch.Tensor, classes: int) -> None:
    if distance.shape != (classes, classes):
        raise ValueError("locked distance matrix has the wrong shape")
    if not torch.isfinite(distance).all() or torch.any(distance < 0):
        raise ValueError("locked distance matrix is non-finite or negative")
    if not torch.equal(torch.diagonal(distance), torch.zeros(classes, dtype=distance.dtype)):
        raise ValueError("locked distance matrix diagonal is not exactly zero")
    if not torch.allclose(distance, distance.T, atol=1e-7, rtol=1e-6):
        raise ValueError("locked distance matrix is not symmetric")
    if not torch.isclose(torch.linalg.vector_norm(distance), torch.tensor(1.0), atol=1e-5):
        raise ValueError("locked distance matrix is not Frobenius-normalized")
