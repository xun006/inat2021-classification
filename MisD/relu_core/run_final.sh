#!/usr/bin/env bash
# One-time held-out evaluation. Run only after development selection is frozen.
set -euo pipefail

ROOT=${ROOT:-/mnt/hdd8t/Mingle/xyyy}
CODE="$ROOT/MisD/relu_core"
CLASSIFIER_RUN=${CLASSIFIER_RUN:-$ROOT/Classifier/outputs/abc_seed42/C}
CLASSIFIER_CONFIG=${CLASSIFIER_CONFIG:-$CLASSIFIER_RUN/config.json}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$CLASSIFIER_RUN/best.pt}
CLASS_MAP=${CLASS_MAP:-$ROOT/MisD/data/class_to_idx.json}
OUT=${OUT:-$ROOT/MisD/output/relu_core_classifier_c_seed42}
SELECTED_MODEL=${SELECTED_MODEL:-$OUT/selection/selected_model.pt}
FINAL_OUTPUT=${FINAL_OUTPUT:-$OUT/final/selected}

if [[ ! -f "$SELECTED_MODEL" ]]; then
  echo "Locked model not found: $SELECTED_MODEL" >&2
  exit 2
fi
if [[ -e "$FINAL_OUTPUT" ]]; then
  echo "Refusing to overwrite an existing final evaluation: $FINAL_OUTPUT" >&2
  exit 2
fi

python "$CODE/evaluate.py" \
  --selected-model "$SELECTED_MODEL" \
  --checkpoint "$TEACHER_CHECKPOINT" \
  --classifier-config "$CLASSIFIER_CONFIG" \
  --class-map "$CLASS_MAP" \
  --output-dir "$FINAL_OUTPUT" \
  --device cuda \
  --allow-official-val
