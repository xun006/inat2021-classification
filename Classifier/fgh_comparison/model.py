"""Exact ViT-L/16 construction for the controlled F/G/H experiment."""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from Classifier.model import LoRALinear, allow_mae_namespace


SIZES = {
    "vit_base_patch16": (768, 12, 12),
    "vit_large_patch16": (1024, 24, 16),
    "vit_tiny_patch16": (192, 12, 3),
}


def _load_plantclef(model: nn.Module, path: str, global_pool: bool) -> dict:
    with allow_mae_namespace():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    discarded = [
        key for key in state
        if key.startswith("decoder_") or key == "mask_token" or key.startswith("head.")
    ]
    for key in discarded:
        del state[key]
    if global_pool and "norm.weight" in state:
        state["fc_norm.weight"] = state.pop("norm.weight")
        state["fc_norm.bias"] = state.pop("norm.bias")
    message = model.load_state_dict(state, strict=False)
    bad_missing = set(message.missing_keys) - {"head.weight", "head.bias"}
    if bad_missing or message.unexpected_keys:
        raise RuntimeError(
            "Unsafe PlantCLEF checkpoint mismatch; "
            f"missing={sorted(bad_missing)}, unexpected={sorted(message.unexpected_keys)}"
        )
    return {"missing": message.missing_keys, "discarded": discarded}


def build_model(cfg: dict):
    """Build one of F/G/H without silently substituting another initialization."""
    if cfg["initialization"] not in ("random", "plantclef"):
        raise ValueError(f"Unknown initialization: {cfg['initialization']}")
    if cfg["tuning_mode"] not in ("lora_head", "full"):
        raise ValueError(f"Unknown tuning_mode: {cfg['tuning_mode']}")
    dim, depth, heads = SIZES[cfg["model"]]
    model = VisionTransformer(
        img_size=cfg["image_size"], patch_size=16, embed_dim=dim,
        depth=depth, num_heads=heads, mlp_ratio=4.0, qkv_bias=True,
        num_classes=cfg["num_classes"],
        global_pool="avg" if cfg["global_pool"] else "token",
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    report = {}
    if cfg["initialization"] == "plantclef":
        report.update(_load_plantclef(model, cfg["pretrained"], cfg["global_pool"]))

    # All groups get the same freshly initialized linear downstream head.
    nn.init.trunc_normal_(model.head.weight, std=0.01)
    nn.init.zeros_(model.head.bias)

    targets = []
    if cfg["tuning_mode"] == "lora_head":
        model.requires_grad_(False)
        for index, block in enumerate(model.blocks):
            block.attn.qkv = LoRALinear(
                block.attn.qkv, cfg["rank"], cfg["alpha"], cfg["lora_dropout"]
            )
            targets.append(f"blocks.{index}.attn.qkv")
        model.head.requires_grad_(True)
    else:
        model.requires_grad_(True)

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if cfg["tuning_mode"] == "lora_head":
        if not targets or not any(".a.weight" in name for name in trainable_names):
            raise RuntimeError("F must expose trainable LoRA parameters")
        if any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if ".base." in name
        ):
            raise RuntimeError("F has trainable frozen-base parameters")
    elif len(trainable_names) != len(list(model.named_parameters())):
        raise RuntimeError("Full fine-tuning groups must train every model parameter")

    report.update(
        initialization=cfg["initialization"],
        tuning_mode=cfg["tuning_mode"],
        patch_size=16,
        embedding_dim=dim,
        depth=depth,
        attention_heads=heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        trainable_names=trainable_names,
        target_modules=targets,
    )
    return model, report
