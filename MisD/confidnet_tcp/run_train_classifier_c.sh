#!/usr/bin/env bash
# ConfidNet rerun using the controlled Classifier C teacher (LoRA + Head, CE).
# All artifacts use a dedicated namespace and cannot overwrite legacy runs.
set -euo pipefail

ROOT=${ROOT:-/mnt/hdd8t/Mingle/xyyy}
CODE="$ROOT/MisD/confidnet_tcp"
CLASSIFIER_RUN=${CLASSIFIER_RUN:-$ROOT/Classifier/outputs/abc_seed42/C}
CLASSIFIER_CONFIG=${CLASSIFIER_CONFIG:-$CLASSIFIER_RUN/config.json}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$CLASSIFIER_RUN/best.pt}
OUT=${OUT:-$ROOT/MisD/output/confidnet_tcp_classifier_c_seed42}
CACHE="$OUT/cache"

if [[ -e "$OUT" ]]; then
  echo "Refusing to mix or overwrite an existing run: $OUT" >&2
  echo "Set OUT to a new directory name for another rerun." >&2
  exit 2
fi

COMMON=(
  --teacher-kind classifier_c
  --classifier-config "$CLASSIFIER_CONFIG"
  --checkpoint "$TEACHER_CHECKPOINT"
  --cache-dir "$CACHE"
  --device cuda
)

python "$CODE/export_targets.py" \
  --split detector_train --output-dir "$CACHE" \
  --teacher-kind classifier_c --classifier-config "$CLASSIFIER_CONFIG" \
  --checkpoint "$TEACHER_CHECKPOINT" --device cuda
python "$CODE/export_targets.py" \
  --split detector_calibration --output-dir "$CACHE" \
  --teacher-kind classifier_c --classifier-config "$CLASSIFIER_CONFIG" \
  --checkpoint "$TEACHER_CHECKPOINT" --device cuda

for SEED in 0 1 2; do
  HEAD="$OUT/confidnet_seed_${SEED}/head"
  FINAL="$OUT/confidnet_seed_${SEED}/finetune"
  python "$CODE/train.py" --phase head --seed "$SEED" \
    --output-dir "$HEAD" "${COMMON[@]}"
  python "$CODE/train.py" --phase finetune --seed "$SEED" \
    --init-checkpoint "$HEAD/checkpoint_best.pth" \
    --output-dir "$FINAL" "${COMMON[@]}"
done
