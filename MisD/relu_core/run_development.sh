#!/usr/bin/env bash
# Development-only REL-U-Core pipeline. This script cannot open official_val.
set -euo pipefail

ROOT=${ROOT:-/mnt/hdd8t/Mingle/xyyy}
CODE="$ROOT/MisD/relu_core"
CLASSIFIER_RUN=${CLASSIFIER_RUN:-$ROOT/Classifier/outputs/abc_seed42/C}
CLASSIFIER_CONFIG=${CLASSIFIER_CONFIG:-$CLASSIFIER_RUN/config.json}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$CLASSIFIER_RUN/best.pt}
CLASS_MAP=${CLASS_MAP:-$ROOT/MisD/data/class_to_idx.json}
OUT=${OUT:-$ROOT/MisD/output/relu_core_classifier_c_seed42}

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite or mix an existing REL-U run: $OUT" >&2
  exit 2
fi

python "$CODE/export_logits.py" \
  --split detector_train \
  --checkpoint "$TEACHER_CHECKPOINT" \
  --classifier-config "$CLASSIFIER_CONFIG" \
  --class-map "$CLASS_MAP" \
  --output-dir "$OUT/cache/detector_train" \
  --device cuda

python "$CODE/export_logits.py" \
  --split detector_calibration \
  --checkpoint "$TEACHER_CHECKPOINT" \
  --classifier-config "$CLASSIFIER_CONFIG" \
  --class-map "$CLASS_MAP" \
  --output-dir "$OUT/cache/detector_calibration" \
  --device cuda

python "$CODE/fit_select.py" \
  --train-cache "$OUT/cache/detector_train" \
  --calibration-cache "$OUT/cache/detector_calibration" \
  --checkpoint "$TEACHER_CHECKPOINT" \
  --classifier-config "$CLASSIFIER_CONFIG" \
  --class-map "$CLASS_MAP" \
  --output-dir "$OUT/selection" \
  --device cuda

echo "Development selection is locked at: $OUT/selection/selected_model.pt"
echo "Inspect calibration artifacts only; do not run final evaluation until the choice is frozen."
