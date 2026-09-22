#!/usr/bin/env python3
"""Train ViT feature + predicted-class embedding on natural data with weighted BCE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_misclassification_head import (
    fpr_at_95_tpr,
    load_split,
    risk_coverage,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("output/data"))
    p.add_argument("--class-map", type=Path, default=Path("data/inat2021/plants/class_to_idx.json"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/baselines/frozen_feature_class_mlp_natural_weighted"),
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-embedding-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class WeightedFeatureClassHead(nn.Module):
    def __init__(self, feature_dim, num_classes, embedding_dim, feature_mean, feature_std):
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.detector = nn.Sequential(
            nn.Linear(feature_dim + embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, visual_feature, predicted_class):
        visual_feature = (visual_feature - self.feature_mean) / self.feature_std
        class_feature = self.class_embedding(predicted_class)
        return self.detector(torch.cat([visual_feature, class_feature], dim=1)).squeeze(1)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)

    feature_store = np.load(args.data_dir / "teacher_val_features.npy", mmap_mode="r")
    class_to_idx = {k: int(v) for k, v in json.loads(args.class_map.read_text()).items()}
    split_dir = args.data_dir / "misdetection_splits"
    train_x, train_class, train_y, _ = load_split(
        split_dir / "train_natural.csv", feature_store, class_to_idx
    )
    test_x, test_class, test_y, test_frame = load_split(
        split_dir / "test.csv", feature_store, class_to_idx
    )
    n_errors = int(train_y.sum())
    n_correct = len(train_y) - n_errors
    if n_correct == 0 or n_errors == 0:
        raise ValueError("Training data must contain both labels")
    pos_weight_value = n_correct / n_errors

    model = WeightedFeatureClassHead(
        feature_dim=feature_store.shape[1],
        num_classes=len(class_to_idx),
        embedding_dim=args.class_embedding_dim,
        feature_mean=train_x.mean(0),
        feature_std=train_x.std(0, unbiased=False).clamp_min(1e-6),
    ).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_class, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_class, test_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    history = []
    print(
        f"natural train: {len(train_y)} samples, {n_correct} correct, "
        f"{n_errors} errors, pos_weight={pos_weight_value:.8f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = seen = 0
        for visual, predicted_class, target in train_loader:
            visual = visual.to(device)
            predicted_class = predicted_class.to(device)
            target_float = target.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            error_logit = model(visual, predicted_class)
            loss = criterion(error_logit, target_float)
            loss.backward()
            optimizer.step()
            n = target.size(0)
            loss_sum += loss.item() * n
            correct += (error_logit >= 0).eq(target_float.bool()).sum().item()
            seen += n
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs}: loss={row['train_loss']:.6f}, acc={row['train_accuracy']:.4%}")

    model.eval()
    batches = []
    with torch.inference_mode():
        for visual, predicted_class, _ in test_loader:
            logit = model(visual.to(device), predicted_class.to(device))
            batches.append(torch.sigmoid(logit).cpu())
    p_error = torch.cat(batches).numpy()
    p_correct = 1.0 - p_error
    probabilities = np.column_stack([p_correct, p_error])
    labels = test_y.numpy()
    decisions = probabilities.argmax(1)
    coverage, risk, aurc = risk_coverage(labels, p_error)
    metrics = {
        "method": "Frozen ViT feature + predicted-class embedding MLP (natural train, weighted BCE)",
        "auroc": float(roc_auc_score(labels, p_error)),
        "error_auprc": float(average_precision_score(labels, p_error)),
        "fpr_at_95_tpr": fpr_at_95_tpr(labels, p_error),
        "aurc": aurc,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(args.output_dir / "train_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame({"coverage": coverage, "risk": risk}).to_csv(
        args.output_dir / "risk_coverage.csv", index=False
    )
    pd.DataFrame(
        {
            "image_id": test_frame["image_id"],
            "feature_row": test_frame["feature_row"],
            "error_label": labels,
            "p_correct": p_correct,
            "p_error": p_error,
            "predicted_error_label": decisions,
            "predicted_label": np.where(decisions == 0, "correct", "error"),
            "max_probability": probabilities.max(1),
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(coverage, risk, linewidth=2)
    ax.set(
        xlabel="Coverage",
        ylabel="Selective risk (classification error rate)",
        title="Risk-Coverage: feature + predicted class, natural weighted",
        xlim=(0, 1),
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "risk_coverage.png", dpi=200)
    plt.close(fig)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        {
            "feature_dim": int(feature_store.shape[1]),
            "train_split": "train_natural.csv",
            "train_samples": len(train_y),
            "train_correct": n_correct,
            "train_errors": n_errors,
            "pos_weight": pos_weight_value,
            "loss": "BCEWithLogitsLoss",
            "probability_definition": "p_error=sigmoid(logit); p_correct=1-p_error",
            "test_samples": len(test_y),
            "model_selection": "none; final epoch evaluated",
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({"model": model.state_dict(), "config": config}, args.output_dir / "checkpoint_final.pth")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
