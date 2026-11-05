"""Unified, atomic checkpoint I/O with legacy A/B/C/D compatibility."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


FORMAT_VERSION = 2


def trainable_state(model) -> dict:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if name in names}


def _atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save(path: Path, kind: str, model, cfg, epoch: int, best: float,
         mapping: dict, stats: dict, initialization_id: str,
         optimizer=None, scaler=None) -> None:
    payload = {
        "format_version": FORMAT_VERSION, "kind": kind,
        "model_state": trainable_state(model), "config": cfg, "epoch": epoch,
        "best": best, "mapping": mapping, "stats": stats,
        "initialization_id": initialization_id,
    }
    if kind == "last":
        payload.update(
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        )
    _atomic_save(Path(path), payload)


def load(path: Path) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") == FORMAT_VERSION:
        return saved
    # Previous A/B/C/D checkpoints stored the same trainable tensors as `adapter`.
    if "adapter" in saved and "config" in saved:
        saved = dict(saved)
        saved["format_version"] = 1
        saved["kind"] = "last" if "optimizer" in saved else "best"
        saved["model_state"] = saved["adapter"]
        saved.setdefault(
            "initialization_id",
            f"plantclef_sha256:{saved.get('pretrained_sha256', 'unknown')}",
        )
        return saved
    raise ValueError(f"Unsupported checkpoint format in {path}")


def restore_model(model, saved: dict) -> None:
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = set(saved["model_state"])
    if actual != expected:
        raise ValueError(
            "Checkpoint/model trainable-state mismatch; "
            f"missing={sorted(expected - actual)[:10]}, unexpected={sorted(actual - expected)[:10]}"
        )
    message = model.load_state_dict(saved["model_state"], strict=False)
    if message.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint parameters: {message.unexpected_keys}")


def restore_rng(saved: dict, device: torch.device) -> None:
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
