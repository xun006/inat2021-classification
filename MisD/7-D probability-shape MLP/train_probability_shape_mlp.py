#!/usr/bin/env python3
"""Train an interpretable 7-feature MLP for teacher error detection.

The seven inputs are derived only from the teacher probability distribution:
  1. log((1-p1)/p1)
  2. log(p1/p2)
  3. log(p2/p3)
  4. log(p3/p4)
  5. sum(p1..p5)
  6. entropy of the probability distribution conditional on Top-5
  7. entropy of the full teacher probability distribution

Training, model selection, and final evaluation use the already-disjoint new
detector_train, detector_calibration, and official_val prediction CSVs.
"""

from __future__ import annotations

import argparse
import ast
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


FEATURE_NAMES = [
    "error_log_odds",
    "log_p1_over_p2",
    "log_p2_over_p3",
    "log_p3_over_p4",
    "top5_probability_sum",
    "top5_conditional_entropy",
    "full_entropy",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    predictions = root / "output/patch_cross_attention/teacher_predictions"
    parser.add_argument("--train-csv", type=Path, default=predictions / "detector_train.csv")
    parser.add_argument(
        "--calibration-csv", type=Path, default=predictions / "detector_calibration.csv"
    )
    parser.add_argument("--test-csv", type=Path, default=predictions / "official_val.csv")
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "output/patch_cross_attention/baselines/probability_shape_mlp",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_topk(value: object) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 5:
        raise ValueError(f"Expected at least five Top-k probabilities, got {value!r}")
    if not np.isfinite(probabilities[:5]).all():
        raise ValueError(f"Non-finite Top-k probabilities: {value!r}")
    return probabilities[:5]


def make_features(frame: pd.DataFrame) -> np.ndarray:
    required = {"teacher_topk_probs", "entropy", "error_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    top5 = np.stack([parse_topk(value) for value in frame["teacher_topk_probs"]])
    eps = 1e-8
    p = np.clip(top5, eps, 1.0)
    p1 = np.clip(p[:, 0], eps, 1.0 - eps)
    top5_sum = top5.sum(axis=1)
    conditional = top5 / np.clip(top5_sum[:, None], eps, None)
    conditional_entropy = -np.sum(
        conditional * np.log(np.clip(conditional, eps, 1.0)), axis=1
    )
    features = np.column_stack(
        [
            np.log((1.0 - p1) / p1),
            np.log(p[:, 0] / p[:, 1]),
            np.log(p[:, 1] / p[:, 2]),
            np.log(p[:, 2] / p[:, 3]),
            top5_sum,
            conditional_entropy,
            frame["entropy"].to_numpy(dtype=np.float64),
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("The constructed feature matrix contains non-finite values")
    return features


def load_frame(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    labels = frame["error_label"].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError(f"{path} must contain both binary error labels")
    return frame, make_features(frame), labels


class ProbabilityShapeMLP(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 16),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.network(normalized).squeeze(1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None, num_workers=0,
    )


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    with torch.inference_mode():
        for features, _ in loader:
            batches.append(torch.sigmoid(model(features.to(device))).cpu())
    return torch.cat(batches).numpy()


def fpr_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(0.95, tpr, fpr))


def risk_coverage(labels: np.ndarray, scores: np.ndarray):
    order = np.argsort(scores, kind="stable")
    retained = np.arange(1, len(labels) + 1, dtype=np.float64)
    risk = np.cumsum(labels[order]) / retained
    return retained / len(labels), risk, float(risk.mean())


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, train_x, train_y = load_frame(args.train_csv)
    _, val_x, val_y = load_frame(args.calibration_csv)
    test_frame, test_x, test_y = load_frame(args.test_csv)

    mean = torch.from_numpy(train_x.mean(axis=0))
    std = torch.from_numpy(train_x.std(axis=0)).clamp_min(1e-6)
    model = ProbabilityShapeMLP(mean, std).to(device)
    train_loader = make_loader(train_x, train_y, args.batch_size, True, args.seed)
    val_loader = make_loader(val_x, val_y, args.batch_size * 2, False, args.seed)
    test_loader = make_loader(test_x, test_y, args.batch_size * 2, False, args.seed)

    n_error = int(train_y.sum())
    n_correct = len(train_y) - n_error
    pos_weight = n_correct / n_error
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_auprc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * target.size(0)

        val_scores = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(val_y, val_scores))
        val_auprc = float(average_precision_score(val_y, val_scores))
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train_y),
            "validation_auroc": val_auc,
            "validation_error_auprc": val_auprc,
        })
        if val_auprc > best_auprc + 1e-6:
            best_auprc = val_auprc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d}: loss={history[-1]['train_loss']:.6f}, "
                f"val_auroc={val_auc:.6f}, val_error_auprc={val_auprc:.6f}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_scores = predict(model, test_loader, device)
    coverage, risk, aurc = risk_coverage(test_y, test_scores)
    metrics = {
        "method": "7-D probability-shape MLP",
        "auroc": float(roc_auc_score(test_y, test_scores)),
        "error_auprc": float(average_precision_score(test_y, test_scores)),
        "fpr_at_95_tpr": fpr_at_95_tpr(test_y, test_scores),
        "aurc": aurc,
        "best_validation_error_auprc": best_auprc,
        "best_epoch": best_epoch,
    }

    pd.DataFrame(history).to_csv(args.output_dir / "train_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame({"coverage": coverage, "risk": risk}).to_csv(
        args.output_dir / "risk_coverage.csv", index=False
    )
    predictions = pd.DataFrame({
        "sample_id": test_frame["sample_id"],
        "error_label": test_y,
        "p_error": test_scores,
        "p_correct": 1.0 - test_scores,
        "predicted_error_label": (test_scores >= 0.5).astype(np.int64),
    })
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(coverage, risk, linewidth=2)
    ax.set(xlabel="Coverage", ylabel="Selective risk", title="Risk-Coverage: 7-D probability-shape MLP", xlim=(0, 1))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "risk_coverage.png", dpi=200)
    plt.close(fig)

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "feature_names": FEATURE_NAMES,
        "architecture": "7-16-8-1, GELU, Dropout(0.15)",
        "loss": "BCEWithLogitsLoss",
        "pos_weight": pos_weight,
        "train_split": "detector_train",
        "selection_split": "detector_calibration",
        "test_split": "official_val",
        "selection_metric": "validation_error_auprc",
        "fit_samples": len(train_y),
        "validation_samples": len(val_y),
        "test_samples": len(test_y),
        "best_epoch": best_epoch,
    })
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({"model": best_state, "config": config}, args.output_dir / "checkpoint_best.pth")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
