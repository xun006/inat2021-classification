#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Adjust CUDA_VISIBLE_DEVICES and --batch-size for your hardware.
# For multiple GPUs, replace `python` with e.g. `torchrun --nproc_per_node=4`.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python MisD/train_vit_classifier_head.py \
  --data-path MisD/data/classifier \
  --class-map MisD/data/class_to_idx.json \
  --pretrained models/PlantCLEF_Model/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth \
  --output-dir MisD/output/vit_large_linear_probe_4271 \
  --num-classes 4271 \
  --epochs 25 \
  --warmup-epochs 2 \
  --batch-size 128 \
  --num-workers 12 \
  --batch-size 128 \
  --global-pool
