#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: train_seed.sh SEED [OUTPUT_ROOT]}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
output_root="${2:-$repo_root/Classifier/outputs}"

cd "$repo_root"

for group in A B C D G H I; do
  python -m Classifier.experiment train \
    --group "$group" --seed "$seed" --output-root "$output_root"
done
