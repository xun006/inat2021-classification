"""The small confidence head appended to the detector-specific ViT encoder."""

from __future__ import annotations

import torch
from torch import nn


class TCPConfidenceHead(nn.Module):
    def __init__(self, feature_dim: int = 1024, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 512), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def set_dropout(self, probability: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = probability

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature).squeeze(-1).sigmoid()
