#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run all three Stage-1 experiments sequentially.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========== Exp-L1 (BCE) =========="
bash "${SCRIPT_DIR}/run_exp_l1.sh"

echo "========== Exp-L1L3 (BCE + SupCon) =========="
bash "${SCRIPT_DIR}/run_exp_l1_l3.sh"

echo "========== Exp-L2L3 (Margin + SupCon) =========="
bash "${SCRIPT_DIR}/run_exp_l2_l3.sh"

echo "========== All Stage-1 experiments complete =========="
echo "Results:"
echo "  ${PROJECT_ROOT:-/mnt/hdd8t/Mingle/xyyy}/Classifier/output/exp_l1/"
echo "  ${PROJECT_ROOT:-/mnt/hdd8t/Mingle/xyyy}/Classifier/output/exp_l1_l3/"
echo "  ${PROJECT_ROOT:-/mnt/hdd8t/Mingle/xyyy}/Classifier/output/exp_l2_l3/"
