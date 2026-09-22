"""Models for the B1/B2/B3/M1/M2 Patch Cross-Attention ablation."""

from __future__ import annotations

import torch
from torch import nn


MODEL_NAMES = (
    "b1", "b2", "b3", "m1", "m2",
    "b1_global", "m1_effective", "m2_effective",
)


def effective_classifier_parameters(batch_norm: nn.BatchNorm1d, linear: nn.Linear):
    """Express eval-mode BN->Linear as one Linear operation in BN-input space."""
    if batch_norm.training:
        raise ValueError("effective classifier parameters require BatchNorm in eval mode")
    inverse_std = torch.rsqrt(batch_norm.running_var.detach() + batch_norm.eps)
    if batch_norm.affine:
        scale = batch_norm.weight.detach() * inverse_std
        shift = batch_norm.bias.detach() - batch_norm.running_mean.detach() * scale
    else:
        scale = inverse_std
        shift = -batch_norm.running_mean.detach() * scale
    weight = linear.weight.detach() * scale.unsqueeze(0)
    bias = linear.bias.detach() + linear.weight.detach().matmul(shift)
    return weight, bias


class ErrorHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class MeanPatchDetector(nn.Module):
    """B1: normalized-patch mean, without class information or attention."""

    def __init__(self, input_dim=1024, attention_dim=128, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.patch_projection = nn.Linear(input_dim, attention_dim)
        self.output_norm = nn.LayerNorm(attention_dim)
        self.error_head = ErrorHead(attention_dim, hidden_dim, dropout)

    def forward(self, patch_tokens, candidate_index=None, global_feature=None):
        feature = self.output_norm(self.patch_projection(patch_tokens.mean(1)))
        return {"error_logit": self.error_head(feature), "attended_feature": feature,
                "attention_weights": None}


class GlobalFeatureDetector(MeanPatchDetector):
    """B1-global: exact teacher pre-head global feature, without patch re-pooling."""

    def forward(self, patch_tokens, candidate_index=None, global_feature=None):
        if global_feature is None:
            raise ValueError("b1_global requires the teacher global_feature")
        feature = self.output_norm(self.patch_projection(global_feature))
        return {"error_logit": self.error_head(feature), "attended_feature": feature,
                "attention_weights": None}


class MeanPatchClassDetector(nn.Module):
    """B2: class-conditioned mean pooling without Cross-Attention."""

    def __init__(self, classifier_weights, input_dim=1024, attention_dim=128,
                 interaction_hidden_dim=320, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.register_buffer("classifier_weights", classifier_weights.detach().float().clone())
        self.patch_projection = nn.Linear(input_dim, attention_dim)
        self.class_projection = nn.Linear(input_dim, attention_dim)
        self.interaction = nn.Sequential(
            nn.Linear(4 * attention_dim, interaction_hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(interaction_hidden_dim, hidden_dim), nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens, candidate_index, global_feature=None):
        patch = self.patch_projection(patch_tokens.mean(1))
        query = self.class_projection(self.classifier_weights[candidate_index])
        features = torch.cat((patch, query, patch * query, (patch - query).abs()), dim=-1)
        feature = self.interaction(features)
        return {"error_logit": self.output(feature).squeeze(-1),
                "attended_feature": feature, "attention_weights": None}


def fixed_derangement(size: int, seed: int) -> torch.Tensor:
    """Create a deterministic permutation with no fixed points."""
    generator = torch.Generator().manual_seed(seed)
    identity = torch.arange(size)
    for _ in range(1000):
        permutation = torch.randperm(size, generator=generator)
        if not permutation.eq(identity).any():
            return permutation
    # Guaranteed fallback for extremely unlikely repeated failures.
    shift = int(torch.randint(1, size, (1,), generator=generator))
    return (identity + shift) % size


class PatchCrossAttentionDetector(nn.Module):
    """B3/M1/M2: one-query Cross-Attention without a Query residual bypass."""

    def __init__(self, classifier_weights, query_kind: str, input_dim=1024,
                 attention_dim=128, num_heads=4, hidden_dim=64, dropout=0.1,
                 permutation_seed=42001):
        super().__init__()
        if query_kind not in {"learnable", "shuffled_class", "top1_class"}:
            raise ValueError(f"unsupported query_kind={query_kind}")
        self.query_kind = query_kind
        weights = classifier_weights.detach().float().clone()
        self.register_buffer("classifier_weights", weights)
        self.query_projection = nn.Linear(input_dim, attention_dim)
        if query_kind == "learnable":
            self.learnable_query = nn.Parameter(torch.zeros(1, input_dim))
            nn.init.normal_(self.learnable_query, std=0.02)
        else:
            self.register_parameter("learnable_query", None)
        if query_kind == "shuffled_class":
            self.register_buffer("class_permutation", fixed_derangement(len(weights), permutation_seed))
        else:
            self.register_buffer("class_permutation", None)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout,
            kdim=input_dim, vdim=input_dim, batch_first=True,
        )
        self.output_norm = nn.LayerNorm(attention_dim)
        self.error_head = ErrorHead(attention_dim, hidden_dim, dropout)

    def query_source(self, candidate_index: torch.Tensor) -> torch.Tensor:
        if self.query_kind == "learnable":
            return self.learnable_query.expand(candidate_index.shape[0], -1)
        if self.query_kind == "shuffled_class":
            candidate_index = self.class_permutation[candidate_index]
        return self.classifier_weights[candidate_index]

    def forward(self, patch_tokens, candidate_index, global_feature=None):
        query = self.query_projection(self.query_source(candidate_index)).unsqueeze(1)
        attended, weights = self.cross_attention(
            query, patch_tokens, patch_tokens, need_weights=True, average_attn_weights=False
        )
        # Deliberately no `query + attended`: the detector cannot bypass Patch values.
        feature = self.output_norm(attended.squeeze(1))
        return {"error_logit": self.error_head(feature),
                "attended_feature": feature, "attention_weights": weights.squeeze(2)}


def build_detector(name: str, classifier_weights: torch.Tensor, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "b1":
        return MeanPatchDetector(**kwargs)
    if name == "b1_global":
        return GlobalFeatureDetector(**kwargs)
    if name == "b2":
        return MeanPatchClassDetector(classifier_weights, **kwargs)
    query_kinds = {
        "b3": "learnable", "m1": "shuffled_class", "m2": "top1_class",
        "m1_effective": "shuffled_class", "m2_effective": "top1_class",
    }
    if name in query_kinds:
        return PatchCrossAttentionDetector(classifier_weights, query_kinds[name], **kwargs)
    raise ValueError(f"model must be one of {MODEL_NAMES}, got {name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
