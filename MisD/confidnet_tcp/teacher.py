"""Exact frozen teacher and an independently initialized detector encoder."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import transforms


MISD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MISD_ROOT.parent
REFERENCE_DIR = PROJECT_ROOT / "PlantCLEF2022"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))
import models_vit  # noqa: E402


DEFAULT_CHECKPOINT = MISD_ROOT / "output/vit_large_linear_probe_4271/checkpoint_best.pth"
DEFAULT_CLASS_MAP = MISD_ROOT / "data/class_to_idx.json"


def validation_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_class_map(path: Path = DEFAULT_CLASS_MAP) -> dict[str, int]:
    mapping = {name: int(index) for name, index in json.loads(path.read_text(encoding="utf-8")).items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("class map must contain contiguous indices")
    return mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ViTGlobalEncoder(nn.Module):
    """ViT feature extractor ending exactly before the classifier BatchNorm/head."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.blocks = model.blocks
        # self.norm = model.norm
        self.fc_norm = model.fc_norm
        self.global_pool = bool(getattr(model, "global_pool", True))
        if not self.global_pool:
            raise ValueError("this experiment requires the checkpoint's global-pool ViT")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        tokens = self.patch_embed(images)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = self.pos_drop(torch.cat((cls, tokens), dim=1) + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        # MAE global-pool ViT uses fc_norm after mean pooling patch tokens.
        return self.fc_norm(tokens[:, 1:, :].mean(dim=1))


class FrozenTeacher(nn.Module):
    """Frozen classifier: only this object supplies TCP targets and class predictions."""

    def __init__(self, encoder: ViTGlobalEncoder, classifier: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.encoder(images)
        logits = self.classifier(feature)
        probabilities = logits.softmax(dim=-1)
        top2_probability, top2_index = probabilities.topk(2, dim=-1)
        return {
            "feature": feature,
            "logits": logits,
            "probabilities": probabilities,
            "prediction": top2_index[:, 0],
            "msp": top2_probability[:, 0],
            "margin": top2_probability[:, 0] - top2_probability[:, 1],
        }


class TimmGlobalEncoder(nn.Module):
    """C-group timm ViT up to the representation immediately before its head."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(images)
        return self.model.forward_head(tokens, pre_logits=True)


class FrozenTimmTeacher(nn.Module):
    """Frozen Classifier C model with its LoRA adapter and linear head restored."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.model.forward_features(images)
        feature = self.model.forward_head(tokens, pre_logits=True)
        logits = self.model.head(feature)
        probabilities = logits.softmax(dim=-1)
        top2_probability, top2_index = probabilities.topk(2, dim=-1)
        return {
            "feature": feature,
            "logits": logits,
            "probabilities": probabilities,
            "prediction": top2_index[:, 0],
            "msp": top2_probability[:, 0],
            "margin": top2_probability[:, 0] - top2_probability[:, 1],
        }


def _load_classifier(checkpoint_path: Path, class_map_path: Path) -> nn.Module:
    class_map = load_class_map(class_map_path)
    model = models_vit.vit_large_patch16(num_classes=len(class_map), global_pool=True)
    original_head = model.head
    model.head = nn.Sequential(nn.BatchNorm1d(original_head.in_features, affine=False, eps=1e-6), original_head)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    state = {(key.removeprefix("module.")): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {result}")
    return model


def _load_classifier_c(checkpoint_path: Path, class_map_path: Path,
                       classifier_config_path: Path | None) -> nn.Module:
    if classifier_config_path is None:
        raise ValueError("--classifier-config is required for teacher-kind=classifier_c")
    if checkpoint_path.name != "best.pt":
        raise ValueError("Classifier C teacher must use the validation-selected best.pt")
    from Classifier.abc_comparison.model import build_model
    from Classifier.abc_comparison.run import resolve_config
    from Classifier.run import restore_adapter

    cfg = resolve_config(classifier_config_path, "C")
    cfg["class_map"] = str(class_map_path.resolve())
    saved: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "group": "C", "tuning_mode": "lora_head", "loss": "ce",
        "native_confidence": "max_softmax", "seed": 42,
    }
    for key, value in expected.items():
        if cfg.get(key) != value or saved.get("config", {}).get(key) != value:
            raise ValueError(f"not a valid Classifier C checkpoint: expected {key}={value!r}")
    changed = {key for key in cfg if cfg[key] != saved["config"].get(key)} - {
        "train_dir", "val_dir", "class_map", "pretrained", "workers",
    }
    if changed:
        raise ValueError(f"Classifier C checkpoint/config mismatch: {sorted(changed)}")
    class_map = load_class_map(class_map_path)
    if class_map != saved.get("mapping"):
        raise ValueError("class map differs from the Classifier C checkpoint")
    if saved.get("pretrained_sha256") != file_sha256(Path(cfg["pretrained"])):
        raise ValueError("Classifier C MAE backbone SHA256 mismatch")
    model, _ = build_model(cfg)
    restore_adapter(model, saved)
    return model


def teacher_provenance(checkpoint_path: Path, class_map_path: Path,
                       teacher_kind: str = "legacy",
                       classifier_config_path: Path | None = None) -> dict[str, str]:
    """Content identity used to reject stale targets and mismatched checkpoints."""
    if teacher_kind not in {"legacy", "classifier_c"}:
        raise ValueError(f"unsupported teacher kind: {teacher_kind}")
    result = {
        "teacher_kind": teacher_kind,
        "teacher_checkpoint_sha256": file_sha256(checkpoint_path),
        "class_map_sha256": file_sha256(class_map_path),
    }
    if teacher_kind == "classifier_c":
        if classifier_config_path is None:
            raise ValueError("classifier_config_path is required for classifier_c")
    return result


def build_models(checkpoint_path: Path = DEFAULT_CHECKPOINT, class_map_path: Path = DEFAULT_CLASS_MAP,
                 device: str | torch.device = "cpu", teacher_kind: str = "legacy",
                 classifier_config_path: Path | None = None) -> tuple[nn.Module, nn.Module]:
    """Build a frozen teacher and a parameter-independent, identically initialized encoder."""
    if teacher_kind == "classifier_c":
        classifier_model = _load_classifier_c(
            checkpoint_path, class_map_path, classifier_config_path
        )
        detector_encoder = TimmGlobalEncoder(copy.deepcopy(classifier_model))
        # Classifier training freezes the MAE base. ConfidNet phase B instead
        # fine-tunes its independent full encoder, as prescribed by the paper.
        detector_encoder.requires_grad_(True).to(device)
        teacher = FrozenTimmTeacher(classifier_model)
        teacher.requires_grad_(False).eval().to(device)
        return teacher, detector_encoder
    if teacher_kind != "legacy":
        raise ValueError(f"unsupported teacher kind: {teacher_kind}")
    classifier_model = _load_classifier(checkpoint_path, class_map_path)
    teacher_encoder = ViTGlobalEncoder(classifier_model)
    # Deep-copy is deliberate: phase B updates only this detector-specific encoder.
    detector_encoder = copy.deepcopy(teacher_encoder)
    teacher = FrozenTeacher(teacher_encoder, classifier_model.head)
    teacher.requires_grad_(False).eval().to(device)
    detector_encoder.to(device)
    return teacher, detector_encoder
