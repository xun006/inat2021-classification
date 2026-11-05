"""Unified ViT factory for all formal classifier groups."""
from __future__ import annotations

import math
from argparse import Namespace
from contextlib import contextmanager
from functools import partial

import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer


SIZES = {
    "vit_base_patch16": (768, 12, 12),
    "vit_large_patch16": (1024, 24, 16),
    "vit_tiny_patch16": (192, 12, 3),
}


@contextmanager
def allow_mae_namespace():
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([Namespace]):
            yield
    else:
        previous = torch.serialization.get_safe_globals()
        try:
            torch.serialization.add_safe_globals([Namespace])
            yield
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(previous)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.b(self.a(self.dropout(inputs))) * self.scale


def _load_mae(model: nn.Module, path: str, global_pool: bool, source: str) -> dict:
    with allow_mae_namespace():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {(key[7:] if key.startswith("module.") else key): value
             for key, value in state.items()}
    discarded = [key for key in state
                 if key.startswith("decoder_") or key == "mask_token" or key.startswith("head.")]
    for key in discarded:
        del state[key]
    if global_pool and "norm.weight" in state:
        state["fc_norm.weight"] = state.pop("norm.weight")
        state["fc_norm.bias"] = state.pop("norm.bias")
    message = model.load_state_dict(state, strict=False)
    bad_missing = set(message.missing_keys) - {"head.weight", "head.bias"}
    if bad_missing or message.unexpected_keys:
        raise RuntimeError(
            f"Unsafe {source} MAE mismatch; missing={sorted(bad_missing)}, "
            f"unexpected={sorted(message.unexpected_keys)}"
        )
    return {"missing": message.missing_keys, "discarded": discarded,
            "pretrained_source": source}


def build_model(cfg: dict, initialize: bool = True):
    """Build Random/ImageNet/PlantCLEF x Head/LoRA/Full without silent fallback."""
    if cfg["initialization"] not in ("random", "imagenet", "plantclef"):
        raise ValueError(f"Unknown initialization: {cfg['initialization']}")
    if cfg["tuning_mode"] not in ("head_only", "lora_head", "full"):
        raise ValueError(f"Unknown tuning mode: {cfg['tuning_mode']}")
    dim, depth, heads = SIZES[cfg["model"]]
    # This matches the previous A/B/C/D constructor. timm's default qkv_bias is True.
    model = VisionTransformer(
        img_size=cfg["image_size"], patch_size=16, embed_dim=dim, depth=depth,
        num_heads=heads, num_classes=cfg["num_classes"],
        global_pool="avg" if cfg["global_pool"] else "token",
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    report = {}
    if initialize and cfg["initialization"] != "random":
        report.update(_load_mae(
            model, cfg["pretrained"], cfg["global_pool"], cfg["initialization"]
        ))

    nn.init.trunc_normal_(model.head.weight, std=0.01)
    nn.init.zeros_(model.head.bias)
    targets = []
    if cfg["tuning_mode"] == "full":
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)
        if cfg["tuning_mode"] == "lora_head":
            for index, block in enumerate(model.blocks):
                block.attn.qkv = LoRALinear(
                    block.attn.qkv, cfg["rank"], cfg["alpha"], cfg["lora_dropout"]
                )
                targets.append(f"blocks.{index}.attn.qkv")
        model.head.requires_grad_(True)

    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    if cfg["tuning_mode"] == "head_only" and any(
            not name.startswith("head.") for name in trainable_names):
        raise RuntimeError("Head-only group exposed trainable backbone parameters")
    if cfg["tuning_mode"] == "lora_head" and (
            not targets or not any(".a.weight" in name for name in trainable_names)):
        raise RuntimeError("LoRA group did not expose adapter parameters")
    if cfg["tuning_mode"] == "full" and len(trainable_names) != len(list(model.named_parameters())):
        raise RuntimeError("Full-training group contains frozen parameters")

    report.update(
        initialization=cfg["initialization"], tuning_mode=cfg["tuning_mode"],
        patch_size=16, embedding_dim=dim, depth=depth, attention_heads=heads,
        mlp_ratio=4.0, qkv_bias=True,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()
                                 if parameter.requires_grad),
        trainable_names=trainable_names, target_modules=targets,
    )
    return model, report
