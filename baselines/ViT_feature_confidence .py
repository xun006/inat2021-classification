#!/usr/bin/env python3
"""Train a frozen ViT-feature + confidence-statistics detection head.

Input: 1024-D frozen ViT feature + Top-1 probability + margin + entropy
       + maximum logit.
Output: softmax probabilities [P(correct), P(error)] and argmax decision.
"""

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

from train_misclassification_head import fpr_at_95_tpr, risk_coverage, seed_everything


STAT_COLUMNS = ["teacher_top1_prob", "margin", "entropy", "max_logit"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("output/data"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/baselines/frozen_feature_confidence_mlp"),
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class VisualConfidenceHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        stat_mean: torch.Tensor,
        stat_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.register_buffer("stat_mean", stat_mean)
        self.register_buffer("stat_std", stat_std)
        self.detector = nn.Sequential(
            nn.Linear(feature_dim + len(STAT_COLUMNS), 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, visual_feature: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        visual_feature = (visual_feature - self.feature_mean) / self.feature_std
        stats = (stats - self.stat_mean) / self.stat_std
        return self.detector(torch.cat([visual_feature, stats], dim=1))


def load_split(csv_path: Path, feature_store: np.ndarray):
    frame = pd.read_csv(csv_path)
    required = {"feature_row", "error_label", *STAT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    rows = frame["feature_row"].to_numpy(dtype=np.int64)
    if rows.min() < 0 or rows.max() >= len(feature_store):
        raise ValueError(f"feature_row in {csv_path} is out of bounds")
    labels = frame["error_label"].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("error_label must contain only 0 and 1")
    visual = np.asarray(feature_store[rows], dtype=np.float32).copy()
    stats = frame[STAT_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(visual).all() or not np.isfinite(stats).all():
        raise ValueError(f"Non-finite input found in {csv_path}")
    return (
        torch.from_numpy(visual),
        torch.from_numpy(stats),
        torch.from_numpy(labels),
        frame,
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)

    feature_store = np.load(args.data_dir / "teacher_val_features.npy", mmap_mode="r")
    train_csv = args.data_dir / "misdetection_splits" / "train.csv"
    test_csv = args.data_dir / "misdetection_splits" / "test.csv"
    train_x, train_stats, train_y, _ = load_split(train_csv, feature_store)
    test_x, test_stats, test_y, test_frame = load_split(test_csv, feature_store)

    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    stat_mean = train_stats.mean(dim=0)
    stat_std = train_stats.std(dim=0, unbiased=False).clamp_min(1e-6)
    model = VisualConfidenceHead(
        feature_store.shape[1], feature_mean, feature_std, stat_mean, stat_std
    ).to(device)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_stats, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_stats, test_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = seen = 0
        for visual, stats, target in train_loader:
            visual, stats, target = visual.to(device), stats.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(visual, stats)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            n = target.size(0)
            loss_sum += loss.item() * n
            correct += logits.argmax(1).eq(target).sum().item()
            seen += n
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs}: loss={row['train_loss']:.6f}, acc={row['train_accuracy']:.4%}")

    model.eval()
    probability_batches = []
    with torch.inference_mode():
        for visual, stats, _ in test_loader:
            logits = model(visual.to(device), stats.to(device))
            probability_batches.append(torch.softmax(logits, dim=1).cpu())
    probabilities = torch.cat(probability_batches).numpy()
    labels = test_y.numpy()
    error_scores = probabilities[:, 1]
    decisions = probabilities.argmax(axis=1)
    coverage, risk, aurc = risk_coverage(labels, error_scores)
    metrics = {
        "method": "Frozen ViT feature + confidence statistics MLP",
        "auroc": float(roc_auc_score(labels, error_scores)),
        "error_auprc": float(average_precision_score(labels, error_scores)),
        "fpr_at_95_tpr": fpr_at_95_tpr(labels, error_scores),
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
            "p_correct": probabilities[:, 0],
            "p_error": probabilities[:, 1],
            "predicted_error_label": decisions,
            "predicted_label": np.where(decisions == 0, "correct", "error"),
            "max_softmax_probability": probabilities.max(axis=1),
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(coverage, risk, linewidth=2)
    ax.set(
        xlabel="Coverage",
        ylabel="Selective risk (classification error rate)",
        title="Risk-Coverage: frozen feature + confidence MLP",
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
            "confidence_columns": STAT_COLUMNS,
            "train_samples": len(train_y),
            "test_samples": len(test_y),
            "label_order": ["correct", "error"],
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
