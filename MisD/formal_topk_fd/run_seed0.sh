#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd8t/Mingle/xyyy

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python MisD/formal_topk_fd/train.py \
  --vit-checkpoint MisD/output/vit_large_linear_probe_4271/checkpoint_best.pth \
  --prototype-path MisD/TopK-ProtoFD/prototypes/classifier_train_mean.pth \
  --data-root MisD/data \
  --output-dir MisD/output/formal_topk_fd/seed0 \
  --top-k 5 \
  --embedding-dim 128 \
  --dropout 0.3 \
  --pair-loss-weight 0.25 \
  --error-pos-weight 10.403524385902456 \
  --batch-size 128 \
  --num-workers 8 \
  --epochs 50 \
  --patience 6 \
  --lr 1e-4 \
  --weight-decay 1e-3 \
  --device cuda
