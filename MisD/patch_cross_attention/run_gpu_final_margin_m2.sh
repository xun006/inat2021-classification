#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/export_m2_official_val.py" \
  --device cuda --batch-size 32 --num-workers 8

python "$SCRIPT_DIR/evaluate_final_margin_m2.py" \
  --bootstrap-samples 10000 --seed 42
