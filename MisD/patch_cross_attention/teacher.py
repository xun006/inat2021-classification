"""Frozen PlantCLEF teacher with online access to patch tokens."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms


MISD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MISD_ROOT.parent
REFERENCE_DIR = PROJECT_ROOT / "PlantCLEF2022"
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

import models_vit  # noqa: E402


DEFAULT_CHECKPOINT = MISD_ROOT / "output" / "vit_large_linear_probe_4271" / "checkpoint_best.pth"
DEFAULT_CLASS_MAP = MISD_ROOT / "data" / "class_to_idx.json"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validation_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_class_map(path: Path = DEFAULT_CLASS_MAP) -> dict[str, int]:
    mapping = {k: int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("class_to_idx values must be contiguous from zero")
    return mapping


class OnlinePatchTeacher(nn.Module):
    """Exact frozen classifier forward plus final-block patch representations."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    # no_grad (rather than inference_mode) keeps returned tensors usable as
    # inputs to trainable detector layers whose backward pass saves activations.
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        model = self.model
        batch = images.shape[0]
        tokens = model.patch_embed(images)
        cls = model.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = model.pos_drop(tokens + model.pos_embed)
        for block in model.blocks:
            tokens = block(tokens)

        p_raw = tokens[:, 1:, :]
        # The trained global-pool model applies fc_norm after spatial averaging.
        global_feature = model.fc_norm(p_raw.mean(dim=1))
        p_norm = model.fc_norm(p_raw)
        head_feature = model.head[0](global_feature)
        logits = model.head[1](head_feature)
        return {
            "p_raw": p_raw,
            "p_norm": p_norm,
            "global_feature": global_feature,
            "head_feature": head_feature,
            "logits": logits,
        }


def build_frozen_teacher(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    class_map_path: Path = DEFAULT_CLASS_MAP,
    device: str | torch.device = "cpu",
) -> OnlinePatchTeacher:
    class_map = load_class_map(class_map_path)
    model = models_vit.vit_large_patch16(num_classes=len(class_map), global_pool=True)
    original_head = model.head
    model.head = nn.Sequential(
        nn.BatchNorm1d(original_head.in_features, affine=False, eps=1e-6),
        original_head,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    state = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {result}")
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return OnlinePatchTeacher(model)
