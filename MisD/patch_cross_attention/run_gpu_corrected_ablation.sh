#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Coordinate-correction experiments only. They use detector_train and
# detector_calibration and never access official_val.
for MODEL_NAME in b1_global m1_effective m2_effective; do
  python "$SCRIPT_DIR/train_ablation.py" \
    --model "$MODEL_NAME" \
    --device cuda \
    --batch-size 32 \
    --num-workers 8 \
    --epochs 15 \
    --patience 3 \
    --seed 0
done
