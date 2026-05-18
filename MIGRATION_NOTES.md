# AEC Relation Extraction Migration Notes

This repository has been converted from event extraction to relation extraction.
The expected model output is now a JSON list of triples:

```json
[{"subject": "Alice", "object": "Acme Corp", "relation": "per:employee_of"}]
```

Do not use event-style outputs such as:

```text
[PerEmployeeOf(arg1="Alice", arg2="Acme Corp", evidence=["joined"])]
```

## Current Code State

Important modified files:

- `relation_agents.py`
- `relation_schema.py`
- `run_relation_inference.py`
- `scripts/prepare_semeval2010_task8.py`
- `README.md`
- `__init__.py`
- `data/examples/smoke_re.jsonl`
- `data/examples/smoke_schema.json`
- `data/schemas/semeval2010_task8.json`

The framework currently supports:

- sentence-level relation extraction
- given entity-pair relation classification
- end-to-end triple extraction over text spans
- simple micro-F1 evaluation over `(subject, object)` and `(subject, object, relation)`

It does not yet implement a full document-level RE benchmark protocol.

## Data Layout

Use this layout on the new server:

```text
AEC/
├── data/
│   ├── raw/
│   │   └── semeval2010_task8/
│   ├── processed/
│   │   └── semeval2010_task8/
│   │       ├── train.jsonl
│   │       └── test.jsonl
│   └── schemas/
│       └── semeval2010_task8.json
└── outputs/
```

The processed SemEval files can be regenerated from the raw files with:

```bash
.venv/bin/python scripts/prepare_semeval2010_task8.py prepare
```

## Model Layout

Keep large models outside the Git repository. Recommended layout:

```text
/home/users/yy/models/Llama3-8B/
```

On another server, use the equivalent local path and pass it to vLLM as the
model path.

## Environment Setup

Create and install the Python environment:

```bash
cd /path/to/AEC
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Verify the install:

```bash
.venv/bin/python -m py_compile __init__.py relation_agents.py relation_schema.py run_relation_inference.py scripts/prepare_semeval2010_task8.py
```

Check GPU visibility:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Serving Llama3-8B with vLLM

Start the OpenAI-compatible vLLM server. Adjust `CUDA_VISIBLE_DEVICES` for the
new machine.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/users/yy/models/Llama3-8B \
  --served-model-name Llama3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

If the model is a base Llama 3 model rather than an instruct/chat model, output
quality may be weak. Prefer an instruction-tuned 7B/8B model for real results
when possible, for example Qwen2.5-7B-Instruct-AWQ or Llama-3-8B-Instruct.

## First Real Experiment

Run a small SemEval subset first:

```bash
OPENAI_API_KEY=EMPTY \
.venv/bin/python run_relation_inference.py \
  --data data/processed/semeval2010_task8/test.jsonl \
  --schema data/schemas/semeval2010_task8.json \
  --model Llama3-8B \
  --base_url http://127.0.0.1:8000/v1 \
  --mode given_pairs \
  --max_samples 20 \
  --k 3 \
  --t 3 \
  --output outputs/semeval_llama3_8b_20.json
```

If that works, increase `--max_samples`, then run the full test set.

Evaluate an existing prediction file:

```bash
.venv/bin/python run_relation_inference.py \
  --eval_only outputs/semeval_llama3_8b_20.json \
  --schema data/schemas/semeval2010_task8.json
```

## Previously Verified

The triple-format code was checked with:

```bash
python3 -m py_compile __init__.py relation_agents.py relation_schema.py run_relation_inference.py scripts/prepare_semeval2010_task8.py
python3 scripts/prepare_semeval2010_task8.py prepare
python3 run_relation_inference.py --eval_only /tmp/aec_triple_eval.json --schema data/examples/smoke_schema.json
```

The old event-style output is rejected by the verifier because it is not strict
JSON triple output.
