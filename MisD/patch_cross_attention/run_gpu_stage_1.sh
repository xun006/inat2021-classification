#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/export_teacher_predictions.py" \
  --splits detector_train detector_calibration official_val \
  --device cuda --batch-size 128 --num-workers 12

python "$SCRIPT_DIR/validate_online_teacher.py" \
  --device cuda --samples-per-split 100 --batch-size 16 --num-workers 4

python "$SCRIPT_DIR/evaluate_confidence_baselines.py"

python "$SCRIPT_DIR/../7-D probability-shape MLP/train_probability_shape_mlp.py" \
  --device cpu
