#!/bin/bash
# ============================================================================
# eval_only.sh — Evaluate existing prediction files (skip inference)
# ============================================================================
# Usage:
#   bash scripts/eval_only.sh                                   # all outputs
#   bash scripts/eval_only.sh outputs/casie_*_predictions.json  # specific file
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_DIR}"

if [ $# -gt 0 ]; then
    FILES=("$@")
else
    FILES=(outputs/*_predictions.json)
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[ERROR] No prediction files found in outputs/"
    exit 1
fi

for FILE in "${FILES[@]}"; do
    if [ ! -f "${FILE}" ]; then
        echo "[WARN] File not found: ${FILE}, skipping."
        continue
    fi
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Evaluating: ${FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python run_inference.py --eval_only "${FILE}"
done

echo ""
echo "[INFO] Evaluation complete."
