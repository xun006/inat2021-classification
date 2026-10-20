"""Modern timm ViT with strict PlantCLEF MAE weights and fused-QKV LoRA."""
import math
from argparse import Namespace
from contextlib import contextmanager
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


@contextmanager
def allow_mae_namespace():
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([Namespace]):
            yield
    else:
        # PyTorch 2.4 exposes allowlist functions without the context manager.
        previous = torch.serialization.get_safe_globals()
        try:
            torch.serialization.add_safe_globals([Namespace])
            yield
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(previous)


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


class DoRALinear(nn.Module):
    """DoRA for a PyTorch Linear layer, with a factored detached weight norm.

    The base weight has shape [out_features, in_features].  Each output row has
    one trainable magnitude, while LoRA A/B updates its direction.  Computing
    the row norm from low-rank factors avoids retaining a dense B@A tensor per
    adapted ViT layer.
    """
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"DoRALinear requires nn.Linear, found {type(base).__name__}")
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.magnitude = nn.Parameter(base.weight.detach().float().norm(dim=1).to(base.weight.dtype))

    def _weight_norm(self):
        # ||W+sBA||^2 = ||W||^2 + 2s<W,BA> + s^2||BA||^2, row by row.
        # The DoRA paper recommends detaching this dynamic denominator.
        with torch.no_grad():
            weight = self.base.weight.float()
            a = self.a.weight.float()
            b = self.b.weight.float()
            base_squared = weight.square().sum(dim=1)
            cross = (b * (weight @ a.t())).sum(dim=1)
            gram = a @ a.t()
            update_squared = ((b @ gram) * b).sum(dim=1)
            squared = base_squared + 2 * self.scale * cross + self.scale**2 * update_squared
            return squared.clamp_min(torch.finfo(squared.dtype).tiny).sqrt()

    def forward(self, x):
        base_result = F.linear(x, self.base.weight, bias=None)
        low_rank_result = self.b(self.a(self.dropout(x))) * self.scale
        norm_scale = (self.magnitude.float() / self._weight_norm()).to(base_result.dtype)
        result = (base_result + low_rank_result) * norm_scale
        if self.base.bias is not None:
            result = result + self.base.bias
        return result


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
        # Original MAE checkpoints include training args alongside tensor weights.
        with allow_mae_namespace():
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
