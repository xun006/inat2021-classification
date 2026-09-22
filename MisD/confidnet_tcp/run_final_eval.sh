#!/usr/bin/env bash
# Execute only after selecting the checkpoint without inspecting official_val.
set -euo pipefail

ROOT=/mnt/hdd8t/Mingle/xyyy
python "$ROOT/MisD/confidnet_tcp/evaluate.py" \
  --detector-checkpoint "$ROOT/MisD/output/confidnet_tcp/seed_0/finetune/checkpoint_best.pth" \
  --output-dir "$ROOT/MisD/output/confidnet_tcp/final/seed_0" \
  --device cuda --allow-official-val
