#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: train_seed.sh SEED [OUTPUT_ROOT]}"
output_root="${2:-/mnt/hdd8t/Mingle/xyyy/Classifier/outputs}"

for group in A B C D G H I; do
  python -m Classifier.experiment train \
    --group "$group" --seed "$seed" --output-root "$output_root"
done
