from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from topk_proto_fd.metrics import detection_metrics
from topk_proto_fd.model import extract_classifier_features


def compute_loss(
    outputs: dict[str, torch.Tensor],
    true_labels: torch.Tensor,
    error_targets: torch.Tensor,
    error_pos_weight: float,
    pair_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    error_loss = F.binary_cross_entropy_with_logits(
        outputs["error_logit"],
        error_targets.float(),
        pos_weight=torch.as_tensor(error_pos_weight, device=error_targets.device),
    )
    if outputs["pair_logits"] is None:
        pair_loss = error_loss.new_zeros(())
    else:
        pair_targets = outputs["topk_indices"].eq(true_labels.unsqueeze(1)).float()
        pair_loss = F.binary_cross_entropy_with_logits(outputs["pair_logits"], pair_targets)
    total = error_loss + pair_loss_weight * pair_loss
    return total, {"error_loss": error_loss.detach(), "pair_loss": pair_loss.detach()}


def run_epoch(
    vit,
    detector,
    effective_weights,
    mean_prototypes,
    loader,
    device,
    error_pos_weight: float,
    pair_loss_weight: float,
    optimizer=None,
    amp: bool = True,
    max_grad_norm: float | None = None,
    collect_predictions: bool = False,
):
    training = optimizer is not None
    vit.eval()
    detector.train(training)
    labels_all, scores_all, predictions = [], [], []
    total_loss = total_error_loss = total_pair_loss = 0.0
    total_count = 0

    for images, true_labels in loader:
        images = images.to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if amp and device.type == "cuda"
            else nullcontext()
        )
        with torch.set_grad_enabled(training), context:
            with torch.no_grad():
                features = extract_classifier_features(vit, images)
                logits = vit.head(features)
            outputs = detector(features, logits, effective_weights, mean_prototypes)
            predicted_labels = logits.argmax(dim=-1)
            error_targets = predicted_labels.ne(true_labels)
            loss, parts = compute_loss(
                outputs, true_labels, error_targets, error_pos_weight, pair_loss_weight
            )
        if training:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(detector.parameters(), max_grad_norm)
            optimizer.step()

        count = images.shape[0]
        total_count += count
        total_loss += float(loss.detach()) * count
        total_error_loss += float(parts["error_loss"]) * count
        total_pair_loss += float(parts["pair_loss"]) * count
        labels_all.append(error_targets.detach().cpu().numpy())
        scores = outputs["error_logit"].detach().float().sigmoid().cpu().numpy()
        scores_all.append(scores)
        if collect_predictions:
            topk = outputs["topk_indices"].detach().cpu().numpy()
            for target, prediction, error, score, candidate_ids in zip(
                true_labels.detach().cpu().numpy(),
                predicted_labels.detach().cpu().numpy(),
                error_targets.detach().cpu().numpy(),
                scores,
                topk,
            ):
                predictions.append(
                    {
                        "true_label": int(target),
                        "pred_label": int(prediction),
                        "error_label": int(error),
                        "error_score": float(score),
                        "topk_indices": " ".join(map(str, candidate_ids.tolist())),
                    }
                )

    metrics = detection_metrics(np.concatenate(labels_all), np.concatenate(scores_all))
    losses = {
        "loss": total_loss / total_count,
        "error_loss": total_error_loss / total_count,
        "pair_loss": total_pair_loss / total_count,
    }
    return metrics, losses, predictions
