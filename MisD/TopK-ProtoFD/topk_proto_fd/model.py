from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TopKPrototypeFailureDetector(nn.Module):
    """Single-head cross-attention over the classifier's Top-K prototypes."""

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int = 256,
        top_k: int = 5,
        dropout: float = 0.1,
        architecture: str = "legacy",
        feature_bottleneck_dim: int = 128,
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.top_k = top_k
        self.architecture = architecture
        self.feature_bottleneck_dim = feature_bottleneck_dim
        if architecture not in {"legacy", "bottleneck", "competition_v2"}:
            raise ValueError("architecture must be 'legacy', 'bottleneck', or 'competition_v2'")
        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, attention_dim)
        if architecture in {"bottleneck", "competition_v2"}:
            # Prevent the 1024-D raw feature/prototype pair from bypassing the
            # attention path. Separate encoders avoid forcing both modalities
            # into exactly the same representation.
            self.image_bottleneck = nn.Sequential(
                nn.Linear(feature_dim, feature_bottleneck_dim),
                nn.LayerNorm(feature_bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.prototype_bottleneck = nn.Sequential(
                nn.Linear(feature_dim, feature_bottleneck_dim),
                nn.LayerNorm(feature_bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            mlp_input_dim = 2 * feature_bottleneck_dim + attention_dim
            if architecture == "competition_v2":
                # Kept only to load/compare checkpoints from the previous experiment.
                mlp_input_dim += 3 * top_k
            self.input_dropout = nn.Dropout(dropout)
        else:
            mlp_input_dim = 2 * feature_dim + attention_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, cls_features: Tensor, logits: Tensor, prototypes: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.top_k > logits.shape[1]:
            raise ValueError(f"top_k={self.top_k} exceeds num_classes={logits.shape[1]}")

        features = F.normalize(cls_features.float(), dim=-1)
        prototypes = F.normalize(prototypes.float(), dim=-1)
        topk_indices = logits.float().topk(self.top_k, dim=-1).indices
        topk_prototypes = prototypes[topk_indices]
        predicted_indices = topk_indices[:, 0]
        predicted_prototypes = prototypes[predicted_indices]

        q = self.query(features).unsqueeze(1)
        k = self.key(topk_prototypes)
        v = self.value(topk_prototypes)
        attention = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.attention_dim), dim=-1)
        context = torch.matmul(attention, v).squeeze(1)
        if self.architecture in {"bottleneck", "competition_v2"}:
            parts = [
                self.image_bottleneck(features),
                self.prototype_bottleneck(predicted_prototypes),
                context,
            ]
            if self.architecture == "competition_v2":
                parts.extend([
                    attention.squeeze(1),
                    logits.float().softmax(dim=-1).gather(1, topk_indices),
                    torch.einsum("bd,bkd->bk", features, topk_prototypes),
                ])
            detector_input = torch.cat(parts, dim=-1)
            detector_input = self.input_dropout(detector_input)
        else:
            detector_input = torch.cat((features, predicted_prototypes, context), dim=-1)
        error_logits = self.mlp(detector_input).squeeze(-1)
        auxiliary = {
            "attention": attention.squeeze(1),
            "topk_indices": topk_indices,
            "predicted_indices": predicted_indices,
            "context": context,
        }
        if self.architecture == "competition_v2":
            auxiliary["topk_probabilities"] = logits.float().softmax(dim=-1).gather(1, topk_indices)
            auxiliary["cosine_similarities"] = torch.einsum("bd,bkd->bk", features, topk_prototypes)
        return error_logits, auxiliary

    @torch.no_grad()
    def predict_proba(self, cls_features: Tensor, logits: Tensor, prototypes: Tensor) -> Tensor:
        """Return P(classifier prediction is wrong) for each sample."""
        error_logits, _ = self(cls_features, logits, prototypes)
        return error_logits.sigmoid()


class FrozenViTWithFailureDetector(nn.Module):
    """Keeps the classifier frozen while training only the detector."""

    def __init__(
        self,
        vit: nn.Module,
        detector: TopKPrototypeFailureDetector,
        classifier: nn.Linear,
        prototypes: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.detector = detector
        self.classifier = classifier
        prototype_tensor = classifier.weight.detach().clone() if prototypes is None else prototypes.detach().clone()
        expected_shape = tuple(classifier.weight.shape)
        if tuple(prototype_tensor.shape) != expected_shape:
            raise ValueError(f"Prototype shape {tuple(prototype_tensor.shape)} != classifier shape {expected_shape}")
        self.register_buffer("prototypes", F.normalize(prototype_tensor.float(), dim=-1))
        self.vit.requires_grad_(False)
        self.vit.eval()

    def train(self, mode: bool = True) -> "FrozenViTWithFailureDetector":
        super().train(mode)
        self.vit.eval()  # also freezes BatchNorm running statistics in the linear-probe head
        self.detector.train(mode)
        return self

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        with torch.no_grad():
            features = extract_classifier_features(self.vit, images)
            logits = self.vit.head(features)
        error_logits, auxiliary = self.detector(features, logits, self.prototypes)
        return logits, error_logits, auxiliary


def extract_classifier_features(vit: nn.Module, images: Tensor) -> Tensor:
    """Return exactly the vector consumed by the trained classification head."""
    features: Any = vit.forward_features(images)
    if hasattr(vit, "forward_head"):
        return vit.forward_head(features, pre_logits=True)
    if isinstance(features, dict):
        features = features.get("x_norm_clstoken", features.get("x"))
    if features.ndim == 3:
        features = features[:, 0]
    return features
