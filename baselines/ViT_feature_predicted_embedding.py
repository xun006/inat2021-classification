#!/usr/bin/env python3
"""Train a frozen-feature misclassification head without a validation split.

Input: 1024-D frozen ViT feature + embedding of the teacher's predicted class.
Output: softmax probabilities [P(correct), P(error)] and their argmax decision.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("output/data"))
    p.add_argument("--output-dir", type=Path, default=Path("output/baselines/frozen_feature_class_mlp"))
    p.add_argument("--class-map", type=Path, default=Path("data/inat2021/plants/class_to_idx.json"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-embedding-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MisclassificationHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        embedding_dim: int,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
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
            nn.Linear(64, 2),
        )

    def forward(self, visual_feature: torch.Tensor, predicted_class: torch.Tensor) -> torch.Tensor:
        normalized = (visual_feature - self.feature_mean) / self.feature_std
        class_feature = self.class_embedding(predicted_class)
        return self.detector(torch.cat([normalized, class_feature], dim=1))


def load_split(
    csv_path: Path,
    feature_store: np.ndarray,
    class_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    required = {"feature_row", "candidate_class", "error_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    rows = frame["feature_row"].to_numpy(dtype=np.int64)
    if rows.min() < 0 or rows.max() >= len(feature_store):
        raise ValueError(f"feature_row in {csv_path} is out of bounds")
    try:
        class_ids = np.array([class_to_idx[x] for x in frame["candidate_class"]], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown candidate_class: {exc.args[0]}") from exc
    labels = frame["error_label"].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("error_label must contain only 0 and 1")
    features = np.asarray(feature_store[rows], dtype=np.float32).copy()
    return (
        torch.from_numpy(features),
        torch.from_numpy(class_ids),
        torch.from_numpy(labels),
        frame,
    )


def risk_coverage(labels: np.ndarray, error_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(error_scores, kind="stable")
    retained = np.arange(1, len(labels) + 1, dtype=np.float64)
    risk = np.cumsum(labels[order]) / retained
    return retained / len(labels), risk, float(risk.mean())


def fpr_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(0.95, tpr, fpr))


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)

    features_path = args.data_dir / "teacher_val_features.npy"
    train_csv = args.data_dir / "misdetection_splits" / "train.csv"
    test_csv = args.data_dir / "misdetection_splits" / "test.csv"
    feature_store = np.load(features_path, mmap_mode="r")
    if feature_store.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got {feature_store.shape}")
    class_to_idx = {k: int(v) for k, v in json.loads(args.class_map.read_text()).items()}

    train_x, train_class, train_y, _ = load_split(train_csv, feature_store, class_to_idx)
    test_x, test_class, test_y, test_frame = load_split(test_csv, feature_store, class_to_idx)
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_class, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_class, test_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    model = MisclassificationHead(
        feature_dim=feature_store.shape[1],
        num_classes=len(class_to_idx),
        embedding_dim=args.class_embedding_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for visual, predicted_class, target in train_loader:
            visual = visual.to(device)
            predicted_class = predicted_class.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(visual, predicted_class)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            batch = target.size(0)
            loss_sum += loss.item() * batch
            correct += logits.argmax(dim=1).eq(target).sum().item()
            seen += batch
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs}: loss={row['train_loss']:.6f}, acc={row['train_accuracy']:.4%}")

    model.eval()
    probability_batches = []
    with torch.inference_mode():
        for visual, predicted_class, _ in test_loader:
            logits = model(visual.to(device), predicted_class.to(device))
            probability_batches.append(torch.softmax(logits, dim=1).cpu())
    probabilities = torch.cat(probability_batches).numpy()
    labels = test_y.numpy()
    error_scores = probabilities[:, 1]
    decisions = probabilities.argmax(axis=1)
    max_probabilities = probabilities.max(axis=1)
    coverage, risk, aurc = risk_coverage(labels, error_scores)

    metrics = {
        "method": "Frozen ViT feature + predicted-class embedding MLP",
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

    predictions = pd.DataFrame(
        {
            "image_id": test_frame["image_id"] if "image_id" in test_frame else np.arange(len(labels)),
            "feature_row": test_frame["feature_row"],
            "error_label": labels,
            "p_correct": probabilities[:, 0],
            "p_error": probabilities[:, 1],
            "predicted_error_label": decisions,
            "predicted_label": np.where(decisions == 0, "correct", "error"),
            "max_softmax_probability": max_probabilities,
        }
    )
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(coverage, risk, linewidth=2.0)
    ax.set(xlabel="Coverage", ylabel="Selective risk (classification error rate)",
           title="Risk-Coverage: frozen feature + predicted-class MLP", xlim=(0, 1))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "risk_coverage.png", dpi=200)
    plt.close(fig)

    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update({
        "feature_dim": int(feature_store.shape[1]),
        "num_predicted_classes": len(class_to_idx),
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "label_order": ["correct", "error"],
        "model_selection": "none; final epoch evaluated",
    })
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({"model": model.state_dict(), "config": config}, args.output_dir / "checkpoint_final.pth")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
