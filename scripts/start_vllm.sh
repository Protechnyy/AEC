#!/bin/bash
# ============================================================================
# start_vllm.sh — Deploy a local model with vLLM (OpenAI-compatible server)
# ============================================================================
# Usage:
#   bash scripts/start_vllm.sh                          # default: Qwen2.5-14B
#   bash scripts/start_vllm.sh meta-llama/Meta-Llama-3-8B-Instruct
#   MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8001 bash scripts/start_vllm.sh
# ============================================================================

set -euo pipefail

MODEL="${MODEL:-${1:-Qwen/Qwen2.5-14B-Instruct}}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

echo "============================================"
echo "  vLLM Server Configuration"
echo "============================================"
echo "  Model:   ${MODEL}"
echo "  Port:    ${PORT}"
echo "  Host:    ${HOST}"
echo "  TP size: ${TENSOR_PARALLEL}"
echo "  GPU mem: ${GPU_MEMORY_UTILIZATION}"
echo "  Max len: ${MAX_MODEL_LEN}"
echo "============================================"

# Install vllm if not present
if ! python -c "import vllm" 2>/dev/null; then
    echo "[INFO] vllm not found, installing..."
    pip install vllm
fi

# Launch vLLM server
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --dtype auto
