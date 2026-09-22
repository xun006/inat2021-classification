#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd8t/Mingle/xyyy

COMMON_ARGS=(
  --vit-checkpoint MisD/output/vit_large_linear_probe_4271/checkpoint_best.pth
  --prototype-path MisD/TopK-ProtoFD/prototypes/classifier_train_mean.pth
  --data-root MisD/data
  --top-k 5
  --batch-size 128
  --num-workers 8
  --epochs 20
  --patience 4
  --lr 1e-4
  --weight-decay 1e-3
  --device cuda
  --seed 0
)

# S1 changes only the positive-class weight relative to the original run.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python MisD/formal_topk_fd/train.py \
  "${COMMON_ARGS[@]}" \
  --architecture matching --embedding-dim 128 --pair-hidden-dim 256 \
  --pair-bottleneck-dim 64 --aggregator-hidden-dim 32 --dropout 0.3 \
  --error-pos-weight 2.43 --pair-loss-weight 0.25 \
  --output-dir MisD/output/formal_topk_fd/screen/s1_pos243

# S2 additionally removes the auxiliary pair objective.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python MisD/formal_topk_fd/train.py \
  "${COMMON_ARGS[@]}" \
  --architecture matching --embedding-dim 128 --pair-hidden-dim 256 \
  --pair-bottleneck-dim 64 --aggregator-hidden-dim 32 --dropout 0.3 \
  --error-pos-weight 2.43 --pair-loss-weight 0 \
  --output-dir MisD/output/formal_topk_fd/screen/s2_pos243_no_pair

# S3 is the reduced-capacity matching model (about half the original size).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python MisD/formal_topk_fd/train.py \
  "${COMMON_ARGS[@]}" \
  --architecture matching --embedding-dim 64 --pair-hidden-dim 128 \
  --pair-bottleneck-dim 32 --aggregator-hidden-dim 16 --dropout 0.3 \
  --error-pos-weight 2.43 --pair-loss-weight 0 \
  --output-dir MisD/output/formal_topk_fd/screen/s3_small_no_pair

# S4 directly predicts teacher correctness from h and the seven probability features.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python MisD/formal_topk_fd/train.py \
  "${COMMON_ARGS[@]}" \
  --architecture direct --embedding-dim 64 --dropout 0.3 \
  --error-pos-weight 2.43 --pair-loss-weight 0 \
  --output-dir MisD/output/formal_topk_fd/screen/s4_direct
