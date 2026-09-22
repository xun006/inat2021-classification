#!/usr/bin/env python3
"""Train and evaluate the Classifier-Weight Compatibility baseline.

Compatibility residual head for a global feature and candidate weight.
The ViT features and classifier weights stay frozen.  Only two projections and
the residual error head are optimized.  A larger score always means that the
teacher's top-1 candidate is more likely to be wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_misclassification_head import (
    fpr_at_95_tpr,
    load_split,
    risk_coverage,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("output/data"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/checkpoint-24.pth")
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=Path("data/inat2021/plants/class_to_idx.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/baselines/classifier_weight_compatibility_softmax"),
    )
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Evaluate only the compatibility detector; do not load or fuse margin.",
    )
    return parser.parse_args()


def load_classifier_weights(checkpoint_path: Path) -> tuple[torch.Tensor, str]:
    """Load the frozen Linear classifier matrix from the teacher checkpoint."""
    # This project checkpoint includes optimizer/argparse objects and predates
    # PyTorch's weights-only format, so ordinary loading is required.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    candidates = ["head.1.weight", "head.weight", "classifier.weight", "fc.weight"]
    matches = [(key, state[key]) for key in candidates if key in state]
    if len(matches) != 1:
        raise ValueError(
            "Could not identify one classifier weight matrix; found "
            f"{[key for key, _ in matches]} among expected keys {candidates}"
        )
    key, weight = matches[0]
    if weight.ndim != 2 or not torch.is_floating_point(weight):
        raise ValueError(f"{key} must be a floating-point 2-D matrix, got {weight.shape}")
    return weight.detach().to(dtype=torch.float32).clone(), key


class ClassifierWeightCompatibility(nn.Module):
    """Compatibility residual head for a global feature and candidate weight."""

    def __init__(
        self,
        classifier_weights: torch.Tensor,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale: float,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        super().__init__()
        num_classes, feature_dim = classifier_weights.shape
        self.num_classes = num_classes
        self.residual_scale = residual_scale
        self.register_buffer("classifier_weights", classifier_weights)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.image_projection = nn.Linear(feature_dim, projection_dim)
        self.class_projection = nn.Linear(feature_dim, projection_dim)
        compatibility_dim = 4 * projection_dim + 1
        self.residual_head = nn.Sequential(
            nn.Linear(compatibility_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def residual_logits(
        self, visual_feature: torch.Tensor, candidate_class: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = (visual_feature - self.feature_mean) / self.feature_std
        wc = self.classifier_weights[candidate_class]
        u = self.image_projection(h)
        q = self.class_projection(wc)
        cosine = F.cosine_similarity(u, q, dim=1, eps=1e-8).unsqueeze(1)
        compatibility = torch.cat([u, q, u * q, (u - q).abs(), cosine], dim=1)
        residual_logits = self.residual_head(compatibility)
        return residual_logits, cosine.squeeze(1)

    def forward(
        self, visual_feature: torch.Tensor, candidate_class: torch.Tensor, margin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_logits, cosine = self.residual_logits(visual_feature, candidate_class)
        residual_probabilities = torch.softmax(residual_logits, dim=1)
        delta = residual_probabilities[:, 1]
        margin_error_score = -margin
        combined_error_logit = margin_error_score + self.residual_scale * delta
        return combined_error_logit, residual_logits, cosine


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_for(name: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, float | str]:
    _, _, aurc = risk_coverage(labels, scores)
    return {
        "method": name,
        "auroc": float(roc_auc_score(labels, scores)),
        "error_auprc": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": fpr_at_95_tpr(labels, scores),
        "aurc": aurc,
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.projection_dim <= 0 or args.hidden_dim < 2:
        raise ValueError("projection-dim must be positive and hidden-dim must be >= 2")
    if not 0 <= args.dropout < 1 or args.residual_scale <= 0:
        raise ValueError("dropout must be in [0,1) and residual-scale must be positive")
    if not 0 < args.validation_ratio < 1 or args.early_stopping_patience <= 0:
        raise ValueError("validation-ratio must be in (0,1) and patience must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)

    feature_path = args.data_dir / "teacher_val_features.npy"
    split_dir = args.data_dir / "misdetection_splits"
    train_csv, test_csv = split_dir / "train_natural.csv", split_dir / "test.csv"
    feature_store = np.load(feature_path, mmap_mode="r")
    class_to_idx = {k: int(v) for k, v in json.loads(args.class_map.read_text()).items()}
    classifier_weights, classifier_key = load_classifier_weights(args.checkpoint)
    if feature_store.ndim != 2 or feature_store.shape[1] != classifier_weights.shape[1]:
        raise ValueError(
            f"Feature shape {feature_store.shape} is incompatible with classifier "
            f"shape {tuple(classifier_weights.shape)}"
        )
    expected_ids = set(range(classifier_weights.shape[0]))
    if len(class_to_idx) != classifier_weights.shape[0] or set(class_to_idx.values()) != expected_ids:
        raise ValueError("class-map indices must be exactly 0..number_of_classifier_rows-1")

    all_train_x, all_train_class, all_train_y, all_train_frame = load_split(
        train_csv, feature_store, class_to_idx
    )
    test_x, test_class, test_y, test_frame = load_split(test_csv, feature_store, class_to_idx)
    test_margin = None
    if not args.standalone:
        test_margin = torch.from_numpy(test_frame["margin"].to_numpy(dtype=np.float32))
        if not torch.isfinite(test_margin).all():
            raise ValueError("margin contains non-finite values")

    all_indices = np.arange(len(all_train_y))
    fit_indices, validation_indices = train_test_split(
        all_indices,
        test_size=args.validation_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=all_train_y.numpy(),
    )
    fit_indices = torch.from_numpy(fit_indices)
    validation_indices = torch.from_numpy(validation_indices)
    train_x, train_class, train_y = (
        all_train_x[fit_indices], all_train_class[fit_indices], all_train_y[fit_indices]
    )
    validation_x, validation_class, validation_y = (
        all_train_x[validation_indices],
        all_train_class[validation_indices],
        all_train_y[validation_indices],
    )

    model = ClassifierWeightCompatibility(
        classifier_weights=classifier_weights,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        feature_mean=train_x.mean(0),
        feature_std=train_x.std(0, unbiased=False).clamp_min(1e-6),
    ).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_parameters = model.classifier_weights.numel()

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_class, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_x, validation_class, validation_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
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
    fit_errors = int(train_y.sum())
    fit_correct = len(train_y) - fit_errors
    pos_weight = fit_correct / fit_errors
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    )
    history: list[dict[str, float | int]] = []

    print(
        f"fit={len(train_y)} (errors={fit_errors}), validation={len(validation_y)} "
        f"(errors={int(validation_y.sum())}), test={len(test_y)} (errors={int(test_y.sum())}), "
        f"pos_weight={pos_weight:.6f}, "
        f"trainable={trainable_parameters:,}, "
        f"frozen_classifier={frozen_parameters:,}"
    )
    best_validation_auroc = -float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = seen = 0
        for visual, candidate, target in train_loader:
            visual, candidate = visual.to(device), candidate.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Residual is trained independently. Margin is intentionally absent
            # from both this input and this loss.
            residual_logits, _ = model.residual_logits(visual, candidate)
            loss = criterion(residual_logits, target)
            loss.backward()
            optimizer.step()
            batch_size = target.size(0)
            loss_sum += loss.item() * batch_size
            correct += residual_logits.argmax(1).eq(target).sum().item()
            seen += batch_size
        model.eval()
        validation_batches = []
        with torch.inference_mode():
            for visual, candidate, _ in validation_loader:
                residual_logits, _ = model.residual_logits(
                    visual.to(device), candidate.to(device)
                )
                validation_batches.append(torch.softmax(residual_logits, dim=1)[:, 1].cpu())
        validation_scores = torch.cat(validation_batches).numpy()
        validation_auroc = float(
            roc_auc_score(validation_y.numpy(), validation_scores)
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy_at_zero": correct / seen,
            "validation_residual_auroc": validation_auroc,
        }
        history.append(row)
        if validation_auroc > best_validation_auroc:
            best_validation_auroc = validation_auroc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs}: loss={row['train_loss']:.6f}, "
                f"acc@0={row['train_accuracy_at_zero']:.4%}, "
                f"val_residual_auroc={validation_auroc:.6f}"
            )
        if stale_epochs >= args.early_stopping_patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)

    model.eval()
    combined_batches, delta_batches, cosine_batches = [], [], []
    with torch.inference_mode():
        for visual, candidate, _ in test_loader:
            residual_logits, cosine = model.residual_logits(
                visual.to(device), candidate.to(device)
            )
            delta_batches.append(torch.softmax(residual_logits, dim=1)[:, 1].cpu())
            cosine_batches.append(cosine.cpu())
    residual_scores = torch.cat(delta_batches).numpy()
    cosines = torch.cat(cosine_batches).numpy()
    labels = test_y.numpy()

    score_sets = {"Classifier-weight compatibility detector": residual_scores}
    if not args.standalone:
        assert test_margin is not None
        margin_scores = -test_margin.numpy()
        combined_scores = margin_scores + args.residual_scale * residual_scores
        score_sets = {
            "Margin": margin_scores,
            "Compatibility residual only": residual_scores,
            f"Margin + {args.residual_scale:g} compatibility residual": combined_scores,
        }
    metrics = pd.DataFrame([metrics_for(name, labels, score) for name, score in score_sets.items()])
    coverage_frames = []
    for name, score in score_sets.items():
        coverage, risk, _ = risk_coverage(labels, score)
        coverage_frames.append(pd.DataFrame({"method": name, "coverage": coverage, "risk": risk}))
    curves = pd.concat(coverage_frames, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(args.output_dir / "train_history.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    curves.to_csv(args.output_dir / "risk_coverage.csv", index=False)
    prediction_columns = {
            "image_id": test_frame["image_id"],
            "feature_row": test_frame["feature_row"],
            "candidate_class": test_frame["candidate_class"],
            "error_label": labels,
            "compatibility_cosine": cosines,
            "residual_p_correct": 1.0 - residual_scores,
            "residual_p_error": residual_scores,
            "predicted_error_label": (residual_scores >= 0.5).astype(np.int64),
    }
    if not args.standalone:
        prediction_columns.update({
            "margin": test_margin.numpy(),
            "margin_error_score": margin_scores,
            "combined_error_score": combined_scores,
        })
    pd.DataFrame(prediction_columns).to_csv(args.output_dir / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name in score_sets:
        curve = curves[curves["method"] == name]
        ax.plot(curve["coverage"], curve["risk"], label=name, linewidth=1.8)
    ax.set(
        xlabel="Coverage",
        ylabel="Selective risk (classification error rate)",
        title="Risk-Coverage: classifier-weight compatibility",
        xlim=(0, 1),
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "risk_coverage.png", dpi=200)
    plt.close(fig)

    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "formula": (
                "score_error = softmax(residual_logits)[error]"
                if args.standalone
                else "score_error = -margin + residual_scale * softmax(residual_logits)[error]"
            ),
            "training_objective": "class-weighted CrossEntropyLoss(residual_logits, error_label); margin excluded",
            "positive_class": "error_label=1",
            "score_direction": "higher means more likely wrong",
            "classifier_weight_key": classifier_key,
            "classifier_weight_shape": list(classifier_weights.shape),
            "feature_shape": list(feature_store.shape),
            "compatibility_dim": 4 * args.projection_dim + 1,
            "trainable_parameters": trainable_parameters,
            "frozen_classifier_parameters": frozen_parameters,
            "train_split": str(train_csv.resolve()),
            "test_split": str(test_csv.resolve()),
            "natural_train_samples": len(all_train_y),
            "fit_samples": len(train_y),
            "validation_samples": len(validation_y),
            "test_samples": len(test_y),
            "pos_weight": pos_weight,
            "best_epoch": best_epoch,
            "best_validation_residual_auroc": best_validation_auroc,
            "model_selection": "best validation residual AUROC; test evaluated once afterward",
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "class_map_sha256": file_sha256(args.class_map),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({"model": model.state_dict(), "config": config}, args.output_dir / "checkpoint_final.pth")

    primary_row = metrics.iloc[-1]
    summary = f"""# Classifier-Weight Compatibility experiment

- Frozen teacher: `{args.checkpoint.resolve()}` (`{classifier_key}`, shape `{tuple(classifier_weights.shape)}`)
- Data: natural train `{len(all_train_y)}` split into fit `{len(train_y)}` and validation `{len(validation_y)}`; natural test `{len(test_y)}` with `{int(test_y.sum())}` errors
- Trainable parameters: `{trainable_parameters:,}`; projection dim: `{args.projection_dim}`; residual scale: `{args.residual_scale}`
- Training: residual-only class-weighted cross entropy with softmax inference, at most `{args.epochs}` epochs, best epoch `{best_epoch}`, validation AUROC `{best_validation_auroc:.6f}`, AdamW, lr `{args.learning_rate}`, weight decay `{args.weight_decay}`, seed `{args.seed}`
- Primary test result: AUROC `{primary_row.auroc:.6f}`, error-AUPRC `{primary_row.error_auprc:.6f}`, FPR@95TPR `{primary_row.fpr_at_95_tpr:.6f}`, AURC `{primary_row.aurc:.6f}`
- Test protocol: selected by validation residual AUROC; test evaluated once afterward

See `config.json` for hashes/environment, `train_history.csv` for optimization history,
`metrics.csv` for reported detector methods, and `test_predictions.csv`
for sample-level audit data.
"""
    (args.output_dir / "EXPERIMENT_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
