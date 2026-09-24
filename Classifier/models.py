"""Model construction: PlantCLEF ViT-L/16 + LoRA + 4271-class sigmoid head.

The backbone is loaded from the PlantCLEF2022 MAE pre-trained checkpoint.
LoRA adapters are injected into every attention block's ``qkv`` and ``proj``
linear layers.  The classification head is
``BatchNorm1d(affine=False) + Linear(1024, 4271)`` producing raw logits;
sigmoid is applied externally so that BCEWithLogits can be used for stable
training.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REFERENCE_DIR = PROJECT_ROOT / "PlantCLEF2022"
sys.path.insert(0, str(REFERENCE_DIR))

import models_vit  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402
from timm.models.layers import trunc_normal_  # noqa: E402


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` with a low-rank additive update.

    forward(x) = W x + b + (alpha / r) * B (A (dropout(x)))

    The original weight/bias are frozen; only A and B are trainable.
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.original.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def inject_lora(model: nn.Module, target_modules: list[str],
                rank: int = 16, alpha: int = 32, dropout: float = 0.1):
    """Replace every ``nn.Linear`` whose parent-name contains one of
    ``target_modules`` with a ``LoRALinear`` wrapper (in-place).

    Returns a list of (parent_module, attr_name, original_linear) tuples
    so the LoRA parameters can be enumerated.
    """
    replacements = []

    def _replace_recursive(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                replacements.append((full_name, lora_layer))
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(model)
    return replacements


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    pretrained_path: Path,
    model_name: str = "vit_large_patch16",
    num_classes: int = 4271,
    global_pool: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_targets: list[str] | None = None,
):
    """Build ViT + LoRA + fresh 4271-class head.

    Returns:
        model, info_dict
    """
    if lora_targets is None:
        lora_targets = ["qkv", "proj"]

    model = models_vit.__dict__[model_name](
        num_classes=num_classes, global_pool=global_pool
    )

    # --- load pre-trained backbone ----------------------------------------
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    target = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in state and state[key].shape != target[key].shape:
            del state[key]
    interpolate_pos_embed(model, state)
    message = model.load_state_dict(state, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    if global_pool:
        allowed_missing |= {"fc_norm.weight", "fc_norm.bias"}
    bad_missing = set(message.missing_keys) - allowed_missing
    if bad_missing or message.unexpected_keys:
        raise RuntimeError(
            f"Unsafe checkpoint mismatch; missing={sorted(bad_missing)}, "
            f"unexpected={sorted(message.unexpected_keys)}"
        )

    # --- fresh 4271-class head --------------------------------------------
    in_features = model.head.in_features
    classifier = nn.Linear(in_features, num_classes)
    trunc_normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)
    model.head = nn.Sequential(
        nn.BatchNorm1d(in_features, affine=False, eps=1e-6), classifier
    )

    # --- freeze everything, then inject LoRA ------------------------------
    model.requires_grad_(False)
    lora_replacements = inject_lora(
        model, target_modules=lora_targets,
        rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
    )
    model.head.requires_grad_(True)

    # --- attach helper for dual output ------------------------------------
    def forward_with_features(x):
        feat = model.forward_features(x)
        logits = model.head(feat)
        return logits, feat

    model.forward_with_features = forward_with_features

    # --- count parameters --------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel() for name, lora in lora_replacements
        for p in lora.lora_A.parameters()
    ) + sum(
        p.numel() for name, lora in lora_replacements
        for p in lora.lora_B.parameters()
    )

    info = {
        "load_message": message,
        "lora_replacements": lora_replacements,
        "lora_num_modules": len(lora_replacements),
        "lora_params": lora_params,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
    }
    return model, info
