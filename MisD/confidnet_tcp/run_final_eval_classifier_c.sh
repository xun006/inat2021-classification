#!/usr/bin/env bash
# Run only after selecting a ConfidNet seed using detector_calibration.
set -euo pipefail

ROOT=${ROOT:-/mnt/hdd8t/Mingle/xyyy}
CODE="$ROOT/MisD/confidnet_tcp"
CLASSIFIER_RUN=${CLASSIFIER_RUN:-$ROOT/Classifier/outputs/abc_seed42/C}
CLASSIFIER_CONFIG=${CLASSIFIER_CONFIG:-$CLASSIFIER_RUN/config.json}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$CLASSIFIER_RUN/best.pt}
OUT=${OUT:-$ROOT/MisD/output/confidnet_tcp_classifier_c_seed42}
: "${DETECTOR_CHECKPOINT:?Set DETECTOR_CHECKPOINT to the calibration-selected ConfidNet checkpoint}"
FINAL_OUTPUT=${FINAL_OUTPUT:-$OUT/final/selected}

if [[ -e "$FINAL_OUTPUT" ]]; then
  echo "Refusing to overwrite an existing final evaluation: $FINAL_OUTPUT" >&2
  exit 2
fi

python "$CODE/evaluate.py" \
  --detector-checkpoint "$DETECTOR_CHECKPOINT" \
  --teacher-kind classifier_c \
  --classifier-config "$CLASSIFIER_CONFIG" \
  --checkpoint "$TEACHER_CHECKPOINT" \
  --output-dir "$FINAL_OUTPUT" \
  --device cuda --allow-official-val
