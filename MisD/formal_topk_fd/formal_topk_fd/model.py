from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _entropy(probabilities: Tensor, eps: float = 1e-8) -> Tensor:
    probabilities = probabilities.clamp_min(eps)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probabilities * probabilities.log()).sum(dim=-1)


class CompetitionAwareFailureDetector(nn.Module):
    """Shared image-class matching followed by a competition aggregator.

    The class prediction is always supplied by the frozen teacher. This module
    only returns an error logit and auxiliary pair logits.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        embedding_dim: int = 128,
        top_k: int = 5,
        dropout: float = 0.3,
        pair_hidden_dim: int = 256,
        pair_bottleneck_dim: int = 64,
        aggregator_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.dropout = dropout
        self.pair_hidden_dim = pair_hidden_dim
        self.pair_bottleneck_dim = pair_bottleneck_dim
        self.aggregator_hidden_dim = aggregator_hidden_dim

        def projector() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.image_projector = projector()
        self.weight_projector = projector()
        self.mean_projector = projector()

        pair_input_dim = 4 * embedding_dim + 4
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, pair_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_bottleneck_dim, 1),
        )
        self.aggregator = nn.Sequential(
            nn.Linear(7, aggregator_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aggregator_hidden_dim, max(4, aggregator_hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(4, aggregator_hidden_dim // 2), 1),
        )

    def forward(
        self,
        features: Tensor,
        logits: Tensor,
        effective_weights: Tensor,
        mean_prototypes: Tensor,
    ) -> dict[str, Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape [B, {self.feature_dim}]")
        if logits.ndim != 2 or logits.shape[0] != features.shape[0]:
            raise ValueError("logits must have shape [B, C]")
        expected = (logits.shape[1], self.feature_dim)
        if tuple(effective_weights.shape) != expected or tuple(mean_prototypes.shape) != expected:
            raise ValueError(f"class representations must both have shape {expected}")
        if self.top_k > logits.shape[1]:
            raise ValueError("top_k exceeds number of classes")

        float_logits = logits.float()
        log_probs = F.log_softmax(float_logits, dim=-1)
        probabilities = log_probs.exp()
        topk_log_probs, topk_indices = log_probs.topk(self.top_k, dim=-1)
        topk_probs = probabilities.gather(1, topk_indices)
        topk_logits = float_logits.gather(1, topk_indices)

        image = self.image_projector(F.normalize(features.float(), dim=-1))
        selected_weights = F.normalize(effective_weights.float(), dim=-1)[topk_indices]
        selected_means = F.normalize(mean_prototypes.float(), dim=-1)[topk_indices]
        class_embedding = self.weight_projector(selected_weights) + self.mean_projector(selected_means)
        image_expanded = image.unsqueeze(1).expand(-1, self.top_k, -1)

        cosine = F.cosine_similarity(image_expanded, class_embedding, dim=-1).unsqueeze(-1)
        delta_logit = (topk_logits[:, :1] - topk_logits).unsqueeze(-1)
        if self.top_k == 1:
            rank = torch.zeros_like(topk_logits).unsqueeze(-1)
        else:
            rank_values = torch.arange(self.top_k, device=features.device, dtype=features.dtype)
            rank_values = rank_values / float(self.top_k - 1)
            rank = rank_values.view(1, self.top_k, 1).expand(features.shape[0], -1, -1)
        pair_input = torch.cat(
            (
                image_expanded,
                class_embedding,
                image_expanded * class_embedding,
                (image_expanded - class_embedding).abs(),
                cosine,
                topk_log_probs.unsqueeze(-1),
                delta_logit,
                rank,
            ),
            dim=-1,
        )
        pair_logits = self.pair_mlp(pair_input).squeeze(-1)

        predicted_pair = pair_logits[:, 0]
        if self.top_k == 1:
            pair_margin = torch.zeros_like(predicted_pair)
        else:
            pair_margin = predicted_pair - pair_logits[:, 1:].max(dim=-1).values
        pair_distribution = pair_logits.softmax(dim=-1)
        normalized_topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.top_k == 1:
            logit_margin = torch.zeros_like(predicted_pair)
        else:
            logit_margin = topk_logits[:, 0] - topk_logits[:, 1]

        aggregate_features = torch.stack(
            (
                predicted_pair,
                pair_margin,
                torch.logsumexp(pair_logits, dim=-1),
                _entropy(pair_distribution),
                logit_margin,
                topk_probs[:, 0],
                _entropy(normalized_topk_probs),
            ),
            dim=-1,
        )
        error_logit = self.aggregator(aggregate_features).squeeze(-1)
        return {
            "error_logit": error_logit,
            "pair_logits": pair_logits,
            "topk_indices": topk_indices,
            "topk_probabilities": topk_probs,
            "aggregate_features": aggregate_features,
        }


def probability_shape_features(logits: Tensor, top_k: int = 5) -> Tensor:
    """Seven stable probability-shape features used by the existing baseline."""
    if logits.shape[1] < 5:
        raise ValueError("Probability-shape features require at least five classes")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    top_log_probs, _ = log_probs.topk(5, dim=-1)
    top_probs = top_log_probs.exp()
    eps = 1e-8
    p1 = top_probs[:, 0].clamp(eps, 1 - eps)
    error_log_odds = torch.log1p(-p1) - torch.log(p1)
    log_ratios = top_log_probs[:, :3] - top_log_probs[:, 1:4]
    top5_sum = top_probs.sum(dim=-1)
    conditional = top_probs / top5_sum.unsqueeze(-1).clamp_min(eps)
    top5_entropy = _entropy(conditional)
    full_entropy = -(probs * log_probs).sum(dim=-1)
    return torch.cat(
        (
            error_log_odds.unsqueeze(-1),
            log_ratios,
            top5_sum.unsqueeze(-1),
            top5_entropy.unsqueeze(-1),
            full_entropy.unsqueeze(-1),
        ),
        dim=-1,
    )


class DirectCorrectnessDetector(nn.Module):
    """Low-capacity direct error head over global features and probability shape."""

    def __init__(
        self,
        feature_dim: int = 1024,
        embedding_dim: int = 64,
        top_k: int = 5,
        dropout: float = 0.3,
        **_: int,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.dropout = dropout
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + 7, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, features: Tensor, logits: Tensor, effective_weights: Tensor, mean_prototypes: Tensor):
        del effective_weights, mean_prototypes
        probability_features = probability_shape_features(logits)
        projected = self.feature_projector(F.normalize(features.float(), dim=-1))
        error_logit = self.mlp(torch.cat((projected, probability_features), dim=-1)).squeeze(-1)
        topk_indices = logits.float().topk(self.top_k, dim=-1).indices
        return {
            "error_logit": error_logit,
            "pair_logits": None,
            "topk_indices": topk_indices,
            "topk_probabilities": logits.float().softmax(-1).gather(1, topk_indices),
            "aggregate_features": probability_features,
        }


class ProbabilityOnlyDetector(nn.Module):
    """Controlled reimplementation of the established 7-D probability baseline."""

    def __init__(self, feature_dim: int = 1024, embedding_dim: int = 16, top_k: int = 5,
                 dropout: float = 0.15, **_: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(7, 16), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(16, 8), nn.GELU(), nn.Dropout(dropout), nn.Linear(8, 1),
        )

    def forward(self, features: Tensor, logits: Tensor, effective_weights: Tensor, mean_prototypes: Tensor):
        del features, effective_weights, mean_prototypes
        probability_features = probability_shape_features(logits)
        error_logit = self.mlp(probability_features).squeeze(-1)
        topk_indices = logits.float().topk(self.top_k, dim=-1).indices
        return {
            "error_logit": error_logit,
            "pair_logits": None,
            "topk_indices": topk_indices,
            "topk_probabilities": logits.float().softmax(-1).gather(1, topk_indices),
            "aggregate_features": probability_features,
        }


def build_detector(architecture: str, **config) -> nn.Module:
    if architecture == "matching":
        return CompetitionAwareFailureDetector(**config)
    if architecture == "direct":
        return DirectCorrectnessDetector(**config)
    if architecture == "probability_only":
        return ProbabilityOnlyDetector(**config)
    raise ValueError(f"Unknown detector architecture: {architecture}")


def effective_classifier_weights(vit: nn.Module) -> Tensor:
    """Return the eval-mode effective weight of BN(affine=False) -> Linear."""
    head = vit.head
    if isinstance(head, nn.Sequential) and len(head) == 2:
        batch_norm, linear = head[0], head[1]
        if not isinstance(batch_norm, nn.BatchNorm1d) or not isinstance(linear, nn.Linear):
            raise TypeError("Expected BatchNorm1d -> Linear classification head")
        inverse_std = torch.rsqrt(batch_norm.running_var.detach().float() + batch_norm.eps)
        scale = inverse_std
        if batch_norm.affine:
            scale = batch_norm.weight.detach().float() * inverse_std
        return linear.weight.detach().float() * scale.unsqueeze(0)
    if isinstance(head, nn.Linear):
        return head.weight.detach().float()
    raise TypeError("Unsupported classifier head")
