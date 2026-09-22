from __future__ import annotations

from pathlib import Path
from typing import Any
import types

import torch
from torch import nn


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("ViT checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def _strip_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.")
    result = state
    for prefix in prefixes:
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix):]: value for key, value in result.items()}
    return result


def build_frozen_vit(checkpoint_path: str | Path, model_name: str | None = None) -> tuple[nn.Module, nn.Linear, dict]:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required; install dependencies from requirements.txt") from exc

    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    state = _strip_prefix(_state_dict(checkpoint))
    classifier_key = "head.1.weight" if "head.1.weight" in state else "head.weight"
    if classifier_key not in state:
        raise KeyError("Could not find classifier weights (head.1.weight or head.weight)")
    num_classes, feature_dim = state[classifier_key].shape
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    inferred_name = saved_args.get("model", "vit_large_patch16") if isinstance(saved_args, dict) else "vit_large_patch16"
    name = model_name or inferred_name
    aliases = {"vit_large_patch16": "vit_large_patch16_224"}
    resolved_name = aliases.get(name, name)
    try:
        vit = timm.create_model(resolved_name, pretrained=False, num_classes=num_classes, global_pool=True)
    except TypeError as exc:
        # timm 0.4.x predates its global_pool argument. Reproduce the MAE
        # implementation used to train this checkpoint: mean patch tokens,
        # followed by fc_norm, with the original token norm removed.
        if "global_pool" not in str(exc):
            raise
        vit = timm.create_model(resolved_name, pretrained=False, num_classes=num_classes)
        vit.global_pool = True
        vit.fc_norm = nn.LayerNorm(feature_dim)
        vit.norm = nn.Identity()

        def forward_features_global_pool(self, x):
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            x = self.blocks(x)
            return self.fc_norm(x[:, 1:, :].mean(dim=1))

        vit.forward_features = types.MethodType(forward_features_global_pool, vit)
    if classifier_key == "head.1.weight":
        vit.head = nn.Sequential(nn.BatchNorm1d(feature_dim, affine=False, eps=1e-6), nn.Linear(feature_dim, num_classes))
    incompat = vit.load_state_dict(state, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch. Missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}"
        )
    classifier = vit.head[1] if isinstance(vit.head, nn.Sequential) else vit.head
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Resolved classification head is not Linear")
    metadata = {"model_name": name, "num_classes": num_classes, "feature_dim": feature_dim}
    return vit, classifier, metadata


def load_mean_prototypes(path: str | Path, expected_classes: list[str] | None = None) -> tuple[torch.Tensor, dict]:
    try:
        saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        saved = torch.load(Path(path), map_location="cpu")
    if not isinstance(saved, dict) or "prototypes" not in saved:
        raise ValueError("Prototype file must be a dictionary containing 'prototypes'")
    prototypes = saved["prototypes"].float()
    if prototypes.ndim != 2 or not torch.isfinite(prototypes).all():
        raise ValueError("Prototypes must be a finite [num_classes, feature_dim] tensor")
    if expected_classes is not None and saved.get("class_names") != expected_classes:
        raise ValueError("Mean-prototype class mapping differs from the detector dataset mapping")
    return prototypes, saved
