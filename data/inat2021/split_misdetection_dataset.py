"""Build train/test CSV splits for misclassification detection.

The full teacher-evaluated validation set is split first. Only the training
partition is balanced; the test partition keeps the deployment prevalence.
"""
from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {
    "image_path", "image_id", "ground_truth", "candidate_class",
    "is_correct", "teacher_top1_prob", "margin", "entropy", "max_logit",
    "energy", "feature_row",
}


def parse_binary(series: pd.Series, name: str) -> pd.Series:
    """Normalize a bool/0/1 column to int64 and reject ambiguous values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int64")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        bad = series[numeric.isna() | ~numeric.isin([0, 1])].unique()[:10]
        raise ValueError(f"{name} must contain only bool/0/1; bad values: {bad}")
    return numeric.astype("int64")


def describe(frame: pd.DataFrame) -> dict:
    errors = int(frame["error_label"].sum())
    total = len(frame)
    return {
        "samples": total,
        "correct": total - errors,
        "errors": errors,
        "error_rate": errors / total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="output/data/teacher_val_predictions.csv",
        help="CSV exported by the frozen teacher",
    )
    parser.add_argument(
        "--output-dir",
        default="output/data/misdetection_splits",
    )
    parser.add_argument(
        "--features",
        default="output/data/teacher_val_features.npy",
        help="Companion feature matrix indexed by feature_row",
    )
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be in (0, 1).")

    frame = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input CSV is empty.")
    if frame["image_id"].duplicated().any():
        examples = frame.loc[frame["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"image_id must be unique; duplicates include {examples}")

    frame = frame.copy()
    frame["is_correct"] = parse_binary(frame["is_correct"], "is_correct")
    frame["error_label"] = 1 - frame["is_correct"]
    if frame["error_label"].nunique() != 2:
        raise ValueError("The input must contain both correct and incorrect predictions.")

    feature_path = Path(args.features)
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature matrix does not exist: {feature_path}")
    features = np.load(feature_path, mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != len(frame):
        raise ValueError(
            "Feature matrix must be 2-D with one row per input CSV row; "
            f"got {features.shape} for {len(frame)} rows"
        )
    feature_rows = pd.to_numeric(frame["feature_row"], errors="coerce")
    if feature_rows.isna().any() or not np.array_equal(
        np.sort(feature_rows.astype("int64").to_numpy()), np.arange(len(frame))
    ):
        raise ValueError("feature_row must be a permutation of 0..N-1")

    # Split the natural-distribution data before any downsampling. This keeps
    # the final test set representative and prevents a sampled row appearing
    # in both partitions.
    natural_train, test = train_test_split(
        frame,
        test_size=args.test_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=frame["error_label"],
    )

    train_errors = natural_train[natural_train["error_label"] == 1]
    train_correct = natural_train[natural_train["error_label"] == 0]
    if len(train_correct) < len(train_errors):
        raise ValueError("Cannot form a 1:1 train set: fewer correct rows than errors.")
    sampled_correct = train_correct.sample(
        n=len(train_errors), replace=False, random_state=args.seed
    )
    train = pd.concat([train_errors, sampled_correct], ignore_index=True)
    train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    natural_train = natural_train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_ids = set(train["image_id"])
    test_ids = set(test["image_id"])
    if train_ids & test_ids:
        raise AssertionError("Train/test image_id overlap detected.")
    if train["error_label"].value_counts().to_dict().get(0) != len(train_errors):
        raise AssertionError("Balanced training split is not 1:1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    # Saved for audit/re-sampling; it is not an additional detector split.
    natural_train.to_csv(output_dir / "train_natural.csv", index=False)

    manifest = {
        "source_csv": str(Path(args.input).resolve()),
        "source_features": str(feature_path.resolve()),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "label_definition": "error_label=1 means teacher top-1 prediction is wrong",
        "feature_reference": (
            "feature_row indexes the companion teacher_val_features.npy file"
        ),
        "full": describe(frame),
        "train_natural_before_downsampling": describe(natural_train),
        "train_balanced": describe(train),
        "test_natural": describe(test),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
