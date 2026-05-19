# Agent-Relation-Coder

This repository is now specialized for relation extraction (RE).  It keeps the
multi-agent "extraction as code generation" idea from AEC, but all event
extraction code, schemas, prompts, and scorers have been removed.

The target output is a JSON list of relation triples:

```json
[{"subject": "Alice", "object": "Acme Corp", "relation": "per:employee_of"}]
```

## What It Supports

| Setting | Supported | Notes |
|---|---:|---|
| Sentence-level RE | Yes | The main intended setting. |
| Short paragraph / document input | Partial | The runner accepts long text, but there is no dedicated document graph, coreference, or cross-sentence candidate generator yet. |
| Given entity-pair relation classification | Yes | Use `--mode given_pairs` or provide `candidate_pairs`; the planner must choose whether each pair expresses each relation. |
| End-to-end relation triple extraction | Yes | Use `--mode end_to_end`, or omit candidate pairs in `--mode auto`; the planner proposes `(subject, object, relation)` triples directly. |
| Full DocRE benchmark protocol | Not yet | Needs entity clustering, mention-level aggregation, NA handling, and document-level metric adapters. |

## Framework

1. `RelationRetrievalAgent` generates synthetic examples for a relation schema.
2. `RelationPlanningAgent` proposes ranked subject/object hypotheses.
3. `RelationCodingAgent` emits JSON triples with `subject`, `object`, and `relation`.
4. `RelationVerificationAgent` parses the JSON and checks structural validity,
   schema labels, exact text grounding, and optional candidate pair constraints.

The outer loop traverses relation hypotheses.  The inner loop patches code using
verification diagnostics, matching the AEC dual-loop design but specialized to
relation extraction.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For this machine, use the 4090 as PCI bus GPU 1:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Start the recommended 4090 model server after the GPU has enough free memory:

```bash
scripts/start_qwen25_7b_awq_4090.sh
```

The script serves `Qwen/Qwen2.5-7B-Instruct-AWQ` at
`http://127.0.0.1:8000/v1` with vLLM.

## Relation Schema

Provide a JSON or JSONL schema for unlabeled inference:

```json
[
  {
    "relation_type": "per:employee_of",
    "description": "subject is a person who works for, is employed by, or holds a role at object.",
    "subject_role": "person",
    "object_role": "organization",
    "subject_type": "PERSON",
    "object_type": "ORG",
    "aliases": ["employee", "works for", "joined"]
  }
]
```

If `--schema` is omitted, relation types are inferred from gold labels in the
input data.  That is convenient for quick benchmark runs, but explicit schemas
are better for zero-shot experiments.

## Input Data

The runner accepts JSON arrays, JSON objects with `data` / `records` /
`examples` / `samples`, JSONL, and concatenated JSON objects.

### End-To-End Triple Extraction

```json
{
  "id": "ex1",
  "text": "Alice joined Acme Corp in 2024.",
  "relation_mentions": [
    {
      "relation": "per:employee_of",
      "subject": {"text": "Alice"},
      "object": {"text": "Acme Corp"}
    }
  ]
}
```

### TACRED-Style Given Pair

```json
{
  "token": ["Alice", "joined", "Acme", "Corp", "."],
  "subj_start": 0,
  "subj_end": 0,
  "obj_start": 2,
  "obj_end": 3,
  "relation": "per:employee_of"
}
```

### Explicit Candidate Pairs

```json
{
  "text": "Alice joined Acme Corp. Bob founded Beta Labs.",
  "candidate_pairs": [
    [{"text": "Alice"}, {"text": "Acme Corp"}],
    [{"text": "Bob"}, {"text": "Beta Labs"}]
  ]
}
```

For SemEval-2010 Task 8, `scripts/prepare_semeval2010_task8.py prepare`
keeps the original `e1/e2` candidate pair as input, but converts gold labels
such as `Product-Producer(e2,e1)` into semantic triples such as
`{"subject": "product mention", "object": "producer mention", "relation": "Product-Producer"}`.

## Running

Given-pair relation classification:

```bash
OPENAI_API_KEY=EMPTY \
.venv/bin/python run_relation_inference.py \
  --data data/re/test.jsonl \
  --schema data/re/schema.json \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --base_url http://127.0.0.1:8000/v1 \
  --mode given_pairs \
  --k 3 --t 3
```

End-to-end relation triple extraction:

```bash
OPENAI_API_KEY=EMPTY \
.venv/bin/python run_relation_inference.py \
  --data data/re/test.jsonl \
  --schema data/re/schema.json \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --base_url http://127.0.0.1:8000/v1 \
  --mode end_to_end \
  --k 3 --t 3
```

Smoke test after the server is up:

```bash
scripts/run_smoke.sh
```

Evaluate an existing prediction file:

```bash
.venv/bin/python run_relation_inference.py --eval_only outputs/my_predictions.json
```

## Metrics

The built-in scorer reports:

| Metric | Meaning |
|---|---|
| Entity Pair Identification | F1 over `(subject, object)` entity pairs. |
| Relation Classification | F1 over `(subject, object, relation)` triples. |

The scorer is intentionally simple and sentence-level oriented.  Public DocRE
datasets often require official evaluation scripts; use those scripts for final
numbers.

## Files

```text
AEC/
├── __init__.py
├── llm_utils.py
├── relation_agents.py
├── relation_schema.py
├── requirements.txt
├── run_relation_inference.py
├── scripts/
│   ├── run_smoke.sh
│   └── start_qwen25_7b_awq_4090.sh
└── data/examples/
    ├── smoke_re.jsonl
    └── smoke_schema.json
```
