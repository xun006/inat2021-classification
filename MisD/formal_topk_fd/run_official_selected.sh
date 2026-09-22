#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd8t/Mingle/xyyy

CHECKPOINT="MisD/output/formal_topk_fd/screen/s4_direct/checkpoint_best.pth"
OUTPUT_DIR="MisD/output/formal_topk_fd/final/s4_direct_seed0_official_val"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing locked checkpoint: $CHECKPOINT" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python MisD/formal_topk_fd/evaluate.py \
  --detector-checkpoint "$CHECKPOINT" \
  --data-root MisD/data \
  --split official_val \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda

python MisD/formal_topk_fd/compare_official.py
