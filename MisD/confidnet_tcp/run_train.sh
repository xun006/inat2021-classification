#!/usr/bin/env bash
# Run from /mnt/hdd8t/Mingle/xyyy. This script never opens official_val.
set -euo pipefail

ROOT=/mnt/hdd8t/Mingle/xyyy
CODE="$ROOT/MisD/confidnet_tcp"
OUT="$ROOT/MisD/output/confidnet_tcp"

python "$CODE/export_targets.py" --split detector_train --device cuda
python "$CODE/export_targets.py" --split detector_calibration --device cuda

for SEED in 0 1 2; do
  HEAD="$OUT/seed_${SEED}/head"
  FINAL="$OUT/seed_${SEED}/finetune"
  python "$CODE/train.py" --phase head --seed "$SEED" --output-dir "$HEAD" --device cuda
  python "$CODE/train.py" --phase finetune --seed "$SEED" --init-checkpoint "$HEAD/checkpoint_best.pth" \
    --output-dir "$FINAL" --device cuda
done
