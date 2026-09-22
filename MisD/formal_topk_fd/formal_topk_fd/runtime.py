from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: str | Path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(Path(path), map_location="cpu")


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
