#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Exp-L1L3:  LoRA + Sigmoid + L1 (BCE) + L3 (SupCon)
# Uses PK-sampler (P=32, K=4 -> batch=128)
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/hdd8t/Mingle/xyyy}"
PYTHON="${PYTHON:-python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

DATA="${PROJECT_ROOT}/MisD/data/classifier"
CLASSMAP="${PROJECT_ROOT}/MisD/data/class_to_idx.json"
PRETRAINED="${PROJECT_ROOT}/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
OUT="${PROJECT_ROOT}/Classifier/output/exp_l1_l3"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${PROJECT_ROOT}/Classifier/train.py" \
  --data-path "${DATA}" \
  --class-map "${CLASSMAP}" \
  --pretrained "${PRETRAINED}" \
  --output-dir "${OUT}" \
  --loss-config l1_l3 \
  --num-classes 4271 \
  --global-pool \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
  --lora-targets qkv proj \
  --temperature 0.07 \
  --pk-P 8 --pk-K 4 \
  --epochs 20 --warmup-epochs 5 \
  --accum-iter 4 --lr 1e-4 --weight-decay 1e-3 \
  --num-workers 12 --seed 0

echo "Exp-L1L3 training complete. Exporting outputs..."

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${PROJECT_ROOT}/Classifier/export_outputs.py" \
  --data-path "${DATA}" \
  --class-map "${CLASSMAP}" \
  --pretrained "${PRETRAINED}" \
  --checkpoint "${OUT}/checkpoint_best.pth" \
  --output-dir "${OUT}" \
  --split val \
  --batch-size 64 --num-workers 8

echo "Exp-L1L3 done. Results in ${OUT}"
