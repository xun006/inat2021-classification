from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import detection_metrics


def weighted_bce(error_logits: torch.Tensor, error_targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    targets = error_targets.float()
    n_error = targets.sum()
    n_correct = targets.numel() - n_error
    weight = torch.log1p(n_correct / n_error.clamp_min(1.0))
    loss = F.binary_cross_entropy_with_logits(error_logits, targets, pos_weight=weight)
    return loss, float(weight.detach())


def run_epoch(
    model, loader, device, optimizer=None, amp: bool = True, max_grad_norm: float | None = None
) -> tuple[dict[str, float], float]:
    training = optimizer is not None
    model.train(training)
    labels_all, scores_all = [], []
    total_loss = total_count = 0
    for images, true_labels in loader:
        images = images.to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(training), amp_context:
            class_logits, error_logits, _ = model(images)
            error_targets = class_logits.argmax(dim=1).ne(true_labels)
            loss, _ = weighted_bce(error_logits, error_targets)
        if training:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.detector.parameters(), max_grad_norm)
            optimizer.step()
        count = images.shape[0]
        total_loss += float(loss.detach()) * count
        total_count += count
        labels_all.append(error_targets.detach().cpu().numpy())
        scores_all.append(error_logits.detach().float().sigmoid().cpu().numpy())
    metrics = detection_metrics(np.concatenate(labels_all), np.concatenate(scores_all))
    return metrics, total_loss / total_count
