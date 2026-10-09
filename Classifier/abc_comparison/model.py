"""ViT construction for the controlled Head-only versus LoRA+Head comparison."""
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


def build_model(cfg: dict, initialize: bool = True):
    if cfg["tuning_mode"] not in ("head_only", "lora_head"):
        raise ValueError(f"Unknown tuning_mode: {cfg['tuning_mode']}")
    dim, depth, heads = SIZES[cfg["model"]]
    model = VisionTransformer(
        img_size=cfg["image_size"], patch_size=16, embed_dim=dim,
        depth=depth, num_heads=heads, num_classes=cfg["num_classes"],
        global_pool="avg" if cfg["global_pool"] else "token",
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    report = {}
    if initialize:
        with allow_mae_namespace():
            checkpoint = torch.load(cfg["pretrained"], map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {(key[7:] if key.startswith("module.") else key): value
                 for key, value in state.items()}
        discarded = [key for key in state
                     if key.startswith("decoder_") or key == "mask_token" or key.startswith("head.")]
        for key in discarded:
            del state[key]
        if cfg["global_pool"] and "norm.weight" in state:
            state["fc_norm.weight"] = state.pop("norm.weight")
            state["fc_norm.bias"] = state.pop("norm.bias")
        message = model.load_state_dict(state, strict=False)
        bad_missing = set(message.missing_keys) - {"head.weight", "head.bias"}
        if bad_missing or message.unexpected_keys:
            raise RuntimeError(
                f"Unsafe pretrained mismatch; missing={sorted(bad_missing)}, "
                f"unexpected={sorted(message.unexpected_keys)}"
            )
        report = {"missing": message.missing_keys, "discarded": discarded}

    # Identical fresh linear head for all A/B/C and the existing D run.
    nn.init.trunc_normal_(model.head.weight, std=0.01)
    nn.init.zeros_(model.head.bias)
    model.requires_grad_(False)
    targets = []
    if cfg["tuning_mode"] == "lora_head":
        for index, block in enumerate(model.blocks):
            block.attn.qkv = LoRALinear(
                block.attn.qkv, cfg["rank"], cfg["alpha"], cfg["lora_dropout"]
            )
            targets.append(f"blocks.{index}.attn.qkv")
    model.head.requires_grad_(True)

    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    if cfg["tuning_mode"] == "head_only":
        if not trainable_names or any(not name.startswith("head.") for name in trainable_names):
            raise RuntimeError(f"Head-only freeze invariant failed: {trainable_names[:10]}")
    else:
        if not targets or not any(".a.weight" in name for name in trainable_names):
            raise RuntimeError("LoRA parameters were not made trainable")

    report.update(
        tuning_mode=cfg["tuning_mode"],
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()
                                 if parameter.requires_grad),
        trainable_names=trainable_names,
        target_modules=targets,
        embedding_dim=dim,
        depth=depth,
        attention_heads=heads,
    )
    return model, report
