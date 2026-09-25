"""Loss definitions from To_do_list sections 39--44 (float32 arithmetic)."""
import torch
import torch.nn.functional as F


def true_and_wrong(logits, labels):
    true = logits.gather(1, labels[:, None]).squeeze(1)
    mask = F.one_hot(labels, logits.shape[1]).bool()
    wrong = logits.masked_fill(mask, -torch.inf).max(1).values
    return true, wrong


def classification_margin_loss(logits, labels, margin=1.0):
    true, wrong = true_and_wrong(logits.float(), labels)
    return F.relu(margin - true + wrong).mean()


def sigmoid_bce_loss(logits, labels, reduction="mean"):
    targets = F.one_hot(labels, logits.shape[1]).float()
    elements = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    if reduction == "mean":
        return elements.mean()
    if reduction == "class_sum":
        return elements.sum(1).mean()
    raise ValueError(reduction)


def confidence_ranking_loss(logits, labels, margin=0.1):
    logits = logits.float()
    maximum, prediction = logits.max(1)
    score = maximum.sigmoid()
    correct = prediction.eq(labels).detach()
    good, bad = score[correct], score[~correct]
    pairs = good.numel() * bad.numel()
    if not pairs:
        return score.sum() * 0, 0
    return F.relu(margin - good[:, None] + bad[None, :]).mean(), pairs


def components(logits, labels, cfg):
    l1 = classification_margin_loss(logits, labels, cfg["m_cls"])
    l2 = sigmoid_bce_loss(logits, labels, cfg["bce_reduction"])
    l3, pairs = confidence_ranking_loss(logits, labels, cfg["m_conf"])
    return l1, l2, l3, pairs


def ranking_weight(cfg, epoch):
    if not cfg["loss"].endswith("_l3") or epoch < cfg["l3_warmup"]:
        return 0.0
    return cfg["lambda3"] * min(1.0, (epoch - cfg["l3_warmup"] + 1) / cfg["l3_ramp"])
