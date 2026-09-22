#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd8t/Mingle/xyyy

# Adjust CUDA_VISIBLE_DEVICES and --batch-size for your hardware.
# For multiple GPUs, replace `python` with e.g. `torchrun --nproc_per_node=4`.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python /mnt/hdd8t/Mingle/xyyy/MisD/train_vit_classifier_head.py \
  --data-path /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier \
  --class-map /mnt/hdd8t/Mingle/xyyy/MisD/data/class_to_idx.json \
  --pretrained /mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth \
  --output-dir /mnt/hdd8t/Mingle/xyyy/MisD/output/vit_large_linear_probe_4271 \
  --num-classes 4271 \
  --epochs 25 \
  --warmup-epochs 2 \
  --batch-size 128 \
  --num-workers 12 \
  --batch-size 128 \
  --global-pool
