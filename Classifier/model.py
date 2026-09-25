"""Modern timm ViT with strict PlantCLEF MAE weights and fused-QKV LoRA."""
import math
from functools import partial
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(self.dropout(x))) * self.scale


def build_model(cfg, initialize=True):
    sizes = {"vit_base_patch16": (768, 12, 12), "vit_large_patch16": (1024, 24, 16),
             "vit_tiny_patch16": (192, 12, 3)}
    dim, depth, heads = sizes[cfg["model"]]
    model = VisionTransformer(img_size=cfg["image_size"], patch_size=16, embed_dim=dim,
                              depth=depth, num_heads=heads, num_classes=cfg["num_classes"],
                              global_pool="avg" if cfg["global_pool"] else "token",
                              norm_layer=partial(nn.LayerNorm, eps=1e-6))
    report = {}
    if initialize:
        checkpoint = torch.load(cfg["pretrained"], map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        # An MAE pretraining checkpoint can contain the reconstruction decoder.
        discarded = [k for k in state if k.startswith("decoder_") or k == "mask_token" or k.startswith("head.")]
        for key in discarded:
            del state[key]
        # MAE token norm -> downstream average-pool norm, as in the reference.
        if cfg["global_pool"] and "norm.weight" in state:
            state["fc_norm.weight"] = state.pop("norm.weight")
            state["fc_norm.bias"] = state.pop("norm.bias")
        msg = model.load_state_dict(state, strict=False)
        if set(msg.missing_keys) - {"head.weight", "head.bias"} or msg.unexpected_keys:
            raise RuntimeError(f"Unsafe pretrained mismatch: {msg}")
        report = {"missing": msg.missing_keys, "discarded": discarded}
    nn.init.trunc_normal_(model.head.weight, std=0.01)
    nn.init.zeros_(model.head.bias)
    model.requires_grad_(False)
    targets = []
    for index, block in enumerate(model.blocks):
        block.attn.qkv = LoRALinear(block.attn.qkv, cfg["rank"], cfg["alpha"], cfg["lora_dropout"])
        targets.append(f"blocks.{index}.attn.qkv")
    model.head.requires_grad_(True)
    report.update(total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  target_modules=targets, embedding_dim=dim, depth=depth, attention_heads=heads)
    return model, report
