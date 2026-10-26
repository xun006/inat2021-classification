"""Checkpoint I/O for both PEFT and full-model F/G/H runs."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


FORMAT_VERSION = 1


def _trainable_state(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in names
    }


def _atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_best(path, model, cfg, epoch, best, mapping, stats, initialization_id):
    """Save inference state only; optimizer tensors belong only in last.pt."""
    _atomic_save(Path(path), {
        "format_version": FORMAT_VERSION,
        "kind": "best",
        "model_state": _trainable_state(model),
        "config": cfg,
        "epoch": epoch,
        "best": best,
        "mapping": mapping,
        "stats": stats,
        "initialization_id": initialization_id,
    })


def save_last(path, model, optimizer, scaler, cfg, epoch, best, mapping, stats,
              initialization_id):
    """Save exact training-resume state."""
    _atomic_save(Path(path), {
        "format_version": FORMAT_VERSION,
        "kind": "last",
        "model_state": _trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "best": best,
        "mapping": mapping,
        "stats": stats,
        "initialization_id": initialization_id,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    })


def load(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported F/G/H checkpoint format: {saved.get('format_version')}")
    return saved


def restore_model(model, saved):
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = set(saved["model_state"])
    if actual != expected:
        missing, unexpected = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(
            f"Checkpoint/model trainable-state mismatch; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    message = model.load_state_dict(saved["model_state"], strict=False)
    if message.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint parameters: {message.unexpected_keys}")


def restore_rng(saved, device):
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
