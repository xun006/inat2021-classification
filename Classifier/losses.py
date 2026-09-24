"""Loss functions for Stage-1 classification training.

L1: Independent Sigmoid BCE (multi-label BCE on one-hot targets).
L2: Indicator Margin Ranking Loss (hinge on z_y - max_{j!=y} z_j).
L3: Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1SigmoidBCE(nn.Module):
    """L1 = (1/C) * sum_i BCEWithLogits(z_i, y_i_onehot).

    Each of the C=4271 output nodes is treated as an independent binary
    classifier with sigmoid activation.  The target is a one-hot vector.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits:  [B, C] raw logits (pre-sigmoid).
            targets: [B]   integer class labels.
        """
        num_classes = logits.size(1)
        onehot = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, onehot, reduction=self.reduction)
        return loss


class L2IndicatorMarginRanking(nn.Module):
    """L2 = I(z_y - max_{j!=y} z_j < m) * [m - (z_y - max_{j!=y} z_j)].

    Implemented as a standard hinge:  max(0, m - (z_y - z_max_other)).
    When the correct-class logit does not lead the strongest competitor by at
    least margin *m*, a linear penalty is incurred; otherwise the loss is 0.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits:  [B, C] raw logits.
            targets: [B]   integer class labels.
        """
        batch_size = logits.size(0)
        arange = torch.arange(batch_size, device=logits.device)

        z_y = logits[arange, targets]                               # [B]
        masked = logits.clone()
        masked[arange, targets] = float("-inf")
        z_max_other = masked.max(dim=1).values                      # [B]

        diff = z_y - z_max_other                                    # [B]
        loss = torch.clamp(self.margin - diff, min=0.0)             # hinge
        return loss.mean()


class L3SupCon(nn.Module):
    """L3 = Supervised Contrastive Loss (Khosla et al., 2020).

    Operates on L2-normalised backbone features.  Requires a PK-sampler so
    that each anchor has at least K-1 positive samples in the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args:
            features: [B, D]  already L2-normalised.
            labels:   [B]     integer class labels.
        Returns:
            scalar loss (0 if no anchor has a positive pair).
        """
        batch_size = features.size(0)
        device = features.device
        if batch_size <= 1:
            return features.sum() * 0.0

        labels = labels.contiguous().view(-1, 1)

        # similarity matrix  [B, B]
        anchor_dot_contrast = torch.matmul(features, features.T)
        logits = anchor_dot_contrast / self.temperature

        # mask out self-similarity (diagonal)
        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        # positive mask: same label AND not self
        pos_mask = (labels == labels.T) & logits_mask

        # log-softmax over all non-self entries
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # per-anchor mean log-prob over positives
        pos_count = pos_mask.sum(dim=1).float()
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)

        # only anchors with at least one positive contribute
        valid = pos_count > 0
        if not valid.any():
            return features.sum() * 0.0
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class CombinedLoss(nn.Module):
    """Weighted combination:  L = lambda1*L1 + lambda2*L2 + lambda3*L3.

    For the three experiments:
        Exp-L1:   lambda1=1, lambda2=0, lambda3=0
        Exp-L1L3: lambda1=1, lambda2=0, lambda3=1
        Exp-L2L3: lambda1=0, lambda2=1, lambda3=1
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.0,
        lambda3: float = 0.0,
        margin: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.l1 = L1SigmoidBCE()
        self.l2 = L2IndicatorMarginRanking(margin=margin)
        self.l3 = L3SupCon(temperature=temperature)

    def forward(self, logits: torch.Tensor, features: torch.Tensor, targets: torch.Tensor):
        """Returns (total_loss, dict_of_components)."""
        components = {}
        total = logits.new_zeros(())

        if self.lambda1 > 0:
            l1_val = self.l1(logits, targets)
            components["l1"] = l1_val.detach()
            total = total + self.lambda1 * l1_val

        if self.lambda2 > 0:
            l2_val = self.l2(logits, targets)
            components["l2"] = l2_val.detach()
            total = total + self.lambda2 * l2_val

        if self.lambda3 > 0:
            l3_val = self.l3(features, targets)
            components["l3"] = l3_val.detach()
            total = total + self.lambda3 * l3_val

        components["total"] = total.detach()
        return total, components
