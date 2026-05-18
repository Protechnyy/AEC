#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

exec .venv/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --quantization awq \
  --dtype half \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs 1
