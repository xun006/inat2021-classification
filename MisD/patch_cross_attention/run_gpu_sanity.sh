#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Real-image end-to-end check. This does not access official_val.
python "$SCRIPT_DIR/train_ablation.py" \
  --model m2 \
  --device cuda \
  --batch-size 16 \
  --num-workers 4 \
  --epochs 5 \
  --patience 5 \
  --dropout 0.0 \
  --max-train-samples 256 \
  --balanced-debug-train \
  --max-calibration-samples 512 \
  --output-dir "$SCRIPT_DIR/../output/patch_cross_attention/sanity/m2_real_images"
