#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for MODEL_NAME in b1 b2 b3 m1 m2; do
  python "$SCRIPT_DIR/train_ablation.py" \
    --model "$MODEL_NAME" \
    --device cuda \
    --batch-size 32 \
    --num-workers 8 \
    --epochs 15 \
    --patience 3 \
    --seed 0
done
