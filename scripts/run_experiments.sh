#!/bin/bash
# ============================================================================
# run_experiments.sh — Run AEC inference + evaluation on all available datasets
# ============================================================================
# Usage:
#   bash scripts/run_experiments.sh                                # all datasets
#   bash scripts/run_experiments.sh ace05-en                       # single dataset
#   bash scripts/run_experiments.sh ace05-en casie                 # multiple
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/run_experiments.sh
# ============================================================================
# Prerequisites:
#   1. vLLM server is running (see scripts/start_vllm.sh)
#   2. TextEE data is placed at data/raw/TextEE/<dataset>/split1/
# ============================================================================

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
K="${K:-3}"            # number of planning hypotheses
T="${T:-3}"            # max patch attempts per hypothesis
DELAY="${DELAY:-0.5}"  # seconds between API calls
SPLIT="${SPLIT:-test}"

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
TEXTEE_DIR="${PROJECT_DIR}/data/raw/TextEE"

# ── Determine which datasets to run ────────────────────────────────────────
ALL_DATASETS=("ace05-en" "casie" "fewevent" "genia2011" "speed")

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    # Auto-detect: only include datasets that have data
    DATASETS=()
    for ds in "${ALL_DATASETS[@]}"; do
        if [ -d "${TEXTEE_DIR}/${ds}" ]; then
            DATASETS+=("${ds}")
        fi
    done
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "[ERROR] No datasets found in ${TEXTEE_DIR}/"
    echo "        Available datasets: ${ALL_DATASETS[*]}"
    echo "        See README.md §2-3 for data preparation instructions."
    exit 1
fi

# ── Check vLLM server ──────────────────────────────────────────────────────
echo "[INFO] Checking vLLM server at ${BASE_URL} ..."
if ! curl -s --max-time 5 "${BASE_URL}/models" > /dev/null 2>&1; then
    echo "[ERROR] vLLM server not responding at ${BASE_URL}"
    echo "        Start it first: bash scripts/start_vllm.sh"
    exit 1
fi
echo "[INFO] vLLM server is up."

# ── Run inference ───────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  AEC Experiment Configuration"
echo "============================================"
echo "  Model:    ${MODEL}"
echo "  Server:   ${BASE_URL}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Split:    ${SPLIT}"
echo "  k=${K}, t=${T}, delay=${DELAY}s"
echo "============================================"
echo ""

cd "${PROJECT_DIR}"

TOTAL=${#DATASETS[@]}
CURRENT=0

for DATASET in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${CURRENT}/${TOTAL}] Dataset: ${DATASET}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check data exists
    SPLIT_FILE="${TEXTEE_DIR}/${DATASET}/split1/${SPLIT}.json"
    if [ ! -f "${SPLIT_FILE}" ]; then
        SPLIT_FILE="${TEXTEE_DIR}/${DATASET}/${SPLIT}.json"
    fi
    if [ ! -f "${SPLIT_FILE}" ]; then
        echo "[WARN] Split file not found for ${DATASET}, skipping."
        continue
    fi

    echo "[INFO] Running inference on ${DATASET} ..."
    OPENAI_API_KEY=EMPTY python run_inference.py \
        --dataset "${DATASET}" \
        --split "${SPLIT}" \
        --model "${MODEL}" \
        --base_url "${BASE_URL}" \
        --k "${K}" \
        --t "${T}" \
        --delay "${DELAY}" \
        --resume

    echo "[INFO] ${DATASET} done."
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"
echo "  Results saved in: ${PROJECT_DIR}/outputs/"
echo ""
echo "  Output files:"
for DATASET in "${DATASETS[@]}"; do
    MODEL_TAG=$(echo "${MODEL}" | sed 's/[^a-zA-Z0-9_-]/_/g')
    PRED_FILE="outputs/${DATASET}_${MODEL_TAG}_predictions.json"
    SCORE_FILE="outputs/${DATASET}_${MODEL_TAG}_predictions_scores.json"
    if [ -f "${PRED_FILE}" ]; then
        echo "    ${PRED_FILE}"
        if [ -f "${SCORE_FILE}" ]; then
            echo "    ${SCORE_FILE}"
            echo "    Scores: $(cat "${SCORE_FILE}")"
        fi
    fi
done
echo ""
