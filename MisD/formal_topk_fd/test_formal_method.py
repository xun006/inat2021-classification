from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "TopK-ProtoFD"))

import numpy as np
import torch
from torch import nn

from formal_topk_fd.engine import compute_loss
from formal_topk_fd.model import (
    CompetitionAwareFailureDetector, DirectCorrectnessDetector, ProbabilityOnlyDetector,
    effective_classifier_weights, probability_shape_features,
)
from topk_proto_fd.metrics import detection_metrics


def test_shapes_and_gradients() -> None:
    torch.manual_seed(0)
    batch, classes, dim, top_k = 4, 11, 16, 5
    model = CompetitionAwareFailureDetector(dim, 8, top_k, 0.0)
    features = torch.randn(batch, dim)
    logits = torch.randn(batch, classes)
    weights = torch.randn(classes, dim)
    means = torch.randn(classes, dim)
    outputs = model(features, logits, weights, means)
    assert outputs["error_logit"].shape == (batch,)
    assert outputs["pair_logits"].shape == (batch, top_k)
    assert outputs["aggregate_features"].shape == (batch, 7)
    targets = torch.tensor([0, 1, 2, 3])
    errors = logits.argmax(1).ne(targets)
    loss, parts = compute_loss(outputs, targets, errors, 2.0, 0.25)
    assert torch.isfinite(loss) and torch.isfinite(parts["pair_loss"])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_effective_weights() -> None:
    head = nn.Sequential(nn.BatchNorm1d(4, affine=False), nn.Linear(4, 3))
    head.eval()
    holder = nn.Module()
    holder.head = head
    expected = head[1].weight * torch.rsqrt(head[0].running_var + head[0].eps).unsqueeze(0)
    assert torch.allclose(effective_classifier_weights(holder), expected)


def test_metric_direction() -> None:
    labels = np.array([0, 0, 1, 1])
    perfect = detection_metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    reversed_scores = detection_metrics(labels, np.array([0.9, 0.8, 0.2, 0.1]))
    assert perfect["auroc"] == 1.0
    assert perfect["aupr_error"] == 1.0
    assert perfect["aurc"] < reversed_scores["aurc"]


def test_control_models() -> None:
    torch.manual_seed(1)
    features, logits = torch.randn(3, 16), torch.randn(3, 11)
    representations = torch.randn(11, 16)
    assert probability_shape_features(logits).shape == (3, 7)
    for model in (DirectCorrectnessDetector(16, 8, 5, 0.0), ProbabilityOnlyDetector(16, 16, 5, 0.0)):
        outputs = model(features, logits, representations, representations)
        assert outputs["error_logit"].shape == (3,)
        loss, parts = compute_loss(outputs, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0]), 2.43, 0.0)
        loss.backward()
        assert float(parts["pair_loss"]) == 0.0


if __name__ == "__main__":
    test_shapes_and_gradients()
    test_effective_weights()
    test_metric_direction()
    test_control_models()
    print("all formal-method CPU tests passed")
