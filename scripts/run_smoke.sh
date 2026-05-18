#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" \
.venv/bin/python run_relation_inference.py \
  --data data/examples/smoke_re.jsonl \
  --schema data/examples/smoke_schema.json \
  --model "${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}" \
  --base_url "${BASE_URL:-http://127.0.0.1:8000/v1}" \
  --mode given_pairs \
  --k "${K:-1}" \
  --t "${T:-1}" \
  --output outputs/smoke_re_predictions.json
