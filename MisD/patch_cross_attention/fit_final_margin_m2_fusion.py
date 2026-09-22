#!/usr/bin/env python3
"""Fit and freeze the final Margin+M2 fusion on detector_calibration only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from teacher import MISD_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    metadata_path = MISD_ROOT / "output/patch_cross_attention/teacher_predictions/detector_calibration.csv"
    m2_path = MISD_ROOT / "output/patch_cross_attention/ablation/m2_seed0/calibration_predictions.csv"
    checkpoint_path = MISD_ROOT / "output/patch_cross_attention/ablation/m2_seed0/checkpoint_best.pth"
    output = MISD_ROOT / "output/patch_cross_attention/final_fusion/margin_m2_seed0/fusion_model.json"
    metadata = pd.read_csv(metadata_path)
    m2 = pd.read_csv(m2_path)[["sample_id", "error_label", "error_score"]]
    merged = metadata.merge(m2, on="sample_id", suffixes=("_metadata", "_m2"), validate="one_to_one")
    if len(merged) != len(metadata) or len(merged) != len(m2):
        raise ValueError("calibration metadata and M2 predictions are not identical sets")
    if not np.array_equal(merged.error_label_metadata, merged.error_label_m2):
        raise ValueError("calibration labels disagree")
    labels = merged.error_label_metadata.to_numpy(dtype=np.int64)
    features = np.column_stack((
        -merged.margin.to_numpy(dtype=np.float64),
        merged.error_score.to_numpy(dtype=np.float64),
    ))
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42
    ).fit(scaler.transform(features), labels)
    artifact = {
        "model": "standardized logistic regression",
        "feature_order": ["negative_margin", "m2_error_probability"],
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "seed": 42,
        "fit_split": "detector_calibration",
        "fit_samples": len(labels),
        "fit_errors": int(labels.sum()),
        "official_val_used": False,
        "m2_definition": "m2_seed0 original raw classifier-weight Query",
        "m2_checkpoint": str(checkpoint_path.resolve()),
        "m2_checkpoint_sha256": sha256(checkpoint_path),
        "calibration_metadata_sha256": sha256(metadata_path),
        "calibration_m2_predictions_sha256": sha256(m2_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    print(f"frozen fusion artifact: {output}")


if __name__ == "__main__":
    main()
