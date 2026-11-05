"""Single source of truth for the formal classifier experiment matrix."""
from __future__ import annotations

from pathlib import Path


FORMAL_SEEDS = (42, 43, 44)
GROUPS = {
    "A": dict(initialization="plantclef", tuning_mode="head_only", loss="ce",
              native_confidence="max_softmax", protocol="abcd"),
    "B": dict(initialization="plantclef", tuning_mode="head_only", loss="l2",
              native_confidence="max_sigmoid", protocol="abcd"),
    "C": dict(initialization="plantclef", tuning_mode="lora_head", loss="ce",
              native_confidence="max_softmax", protocol="abcd"),
    "D": dict(initialization="plantclef", tuning_mode="lora_head", loss="l2",
              native_confidence="max_sigmoid", protocol="abcd"),
    "G": dict(initialization="plantclef", tuning_mode="full", loss="l2",
              native_confidence="max_sigmoid", protocol="ghi"),
    "H": dict(initialization="random", tuning_mode="full", loss="l2",
              native_confidence="max_sigmoid", protocol="ghi"),
    "I": dict(initialization="imagenet", tuning_mode="full", loss="l2",
              native_confidence="max_sigmoid", protocol="ghi"),
}

# These are controlled protocol values, not claims that one LR is optimal for
# every trainable scope.  Scope-specific LR groups are part of the treatment.
FIXED_PROTOCOL = {
    "model": "vit_large_patch16",
    "global_pool": True,
    "image_size": 224,
    "num_classes": 4271,
    "rank": 8,
    "alpha": 16,
    "lora_dropout": 0.0,
    "batch_size": 32,
    "lr": 1e-4,
    "head_lr": 1e-3,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "bce_reduction": "mean",
    "amp": True,
    "diagnostic_batches": 5,
    "tau": 0.5,
    "expected_train_samples": 956507,
    "expected_val_samples": 42710,
}


def validate_group_seed(group: str, seed: int, formal: bool = True) -> None:
    if group not in GROUPS:
        raise ValueError(f"Unknown or retired group {group!r}; choose from {tuple(GROUPS)}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if formal and seed not in FORMAL_SEEDS:
        raise ValueError(f"Formal runs use seeds {FORMAL_SEEDS}; got {seed}")


def validate_fixed_protocol(cfg: dict) -> None:
    altered = {key: {"expected": value, "actual": cfg.get(key)}
               for key, value in FIXED_PROTOCOL.items() if cfg.get(key) != value}
    if altered:
        raise ValueError(f"Controlled protocol fields changed: {altered}")
    expected_schedule = {"abcd": (20, 2), "ghi": (30, 3)}[cfg["protocol"]]
    if (cfg.get("epochs"), cfg.get("lr_warmup")) != expected_schedule:
        raise ValueError(
            f"{cfg['protocol']} requires epochs/warmup={expected_schedule}, found "
            f"{(cfg.get('epochs'), cfg.get('lr_warmup'))}"
        )


def run_dir(output_root: str | Path, group: str, seed: int) -> Path:
    validate_group_seed(group, seed)
    return Path(output_root) / group / f"seed_{seed}"


def apply_group(cfg: dict, group: str, seed: int) -> dict:
    validate_group_seed(group, seed, formal=False)
    resolved = dict(cfg)
    protocols = resolved.pop("protocols")
    plantclef_pretrained = resolved.pop("plantclef_pretrained")
    imagenet_pretrained = resolved.pop("imagenet_pretrained")
    resolved.update(GROUPS[group], group=group, seed=seed)
    resolved.update(protocols[resolved["protocol"]])
    if resolved["initialization"] == "plantclef":
        resolved["pretrained"] = plantclef_pretrained
    elif resolved["initialization"] == "imagenet":
        resolved["pretrained"] = imagenet_pretrained
    else:
        resolved["pretrained"] = None
    validate_fixed_protocol(resolved)
    return resolved


def comparable_config(cfg: dict) -> dict:
    """Fields that must match before results may be aggregated over seeds."""
    ignored = {
        "seed", "train_dir", "val_dir", "official_val_dir", "class_map",
        "workers", "pretrained", "plantclef_pretrained", "imagenet_pretrained",
    }
    return {key: value for key, value in cfg.items() if key not in ignored}
