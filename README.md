# Agent-Relation-Coder

This repository has been adapted from the original Agent-Event-Coder release into a zero-shot relation extraction project. The main path is now relation extraction only: the experiment entry point loads relation schemas, proposes entity-pair/relation hypotheses, generates executable Pydantic code for relation objects, verifies grounded triples, and evaluates exact relation triples.

## Method Adaptation

The original paper's transferable idea is schema-as-code multi-agent extraction.  In the relation version:

- Planning Agent proposes `(head, tail, relation_type)` hypotheses instead of event trigger hypotheses.
- Relation schemas are rendered as Pydantic class definitions, preserving the original schema-as-code idea.
- Coding Agent turns each hypothesis into executable Python code, assigning one Pydantic object to `result`.
- Verification Agent executes the generated code with only the schema classes in scope, validates it with Pydantic, then checks head/tail/evidence grounding in the source text.
- Evaluation reports micro triple precision/recall/F1 and relation-label classification F1.

## Chosen Datasets

Primary datasets for the zero-shot RE version:

- `fewrel`: best first benchmark for unseen-relation zero-shot splits.
- `wiki-zsl`: purpose-built zero-shot relation classification over Wikidata-style relations.
- `retacred`: cleaner TACRED-style sentence-level relation classification.
- `tacred`: broad news/web relation classification baseline with `no_relation`.
- `semeval2010`: compact sanity benchmark with interpretable relation names.
- `scierc`: domain-transfer benchmark for scientific relation extraction.

Schema starter files live in `utils/re_schema_generation/init_prompts/`. For full experiments, replace or extend those files with the exact relation inventory and definitions used by your split.

## Chosen Baselines

Use these as the corresponding RE baselines:

- `DirectRE`: one-step zero-shot JSON prediction.
- `CoT-RE`: DirectRE with brief evidence reasoning before final JSON.
- `DecomposeRE`: entity-pair proposal followed by relation classification.
- `GuidelineRE`: relation definitions and typed head/tail constraints in a code/schema prompt.
- `ChatIE-RE`: multi-turn QA-style relation extraction.
- `ZS-BERT/RelationPrompt`: historical specialist zero-shot RE baselines when non-LLM comparisons are needed.

## Data Format

The loader accepts JSONL or JSON records under:

```text
datasets/relation_splits/{dataset_name}/{split}.jsonl
datasets/relation_splits/{dataset_name}/{split}.json
```

Preferred normalized fields:

```json
{
  "id": "sample-1",
  "text": "Ada Lovelace was born in London.",
  "candidate_pairs": [{"head": "Ada Lovelace", "tail": "London"}],
  "relations": [{"head": "Ada Lovelace", "tail": "London", "relation_type": "place_of_birth"}]
}
```

The loader also handles common FewRel/TACRED-style fields such as `tokens`, `h`, `t`, `relation`, `subj_start`, `subj_end`, `obj_start`, and `obj_end`.

## Quick Start

A tiny demo split is included for smoke testing:

```bash
python main_experiment.py \
  --dataset_name fewrel \
  --split test \
  --max_samples 2 \
  --no_progress \
  --output_file ./results/fewrel_demo.json
```

LLM-backed run with an OpenAI-compatible endpoint:

```bash
export AEC_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export AEC_LLM_API_KEY="EMPTY"
export AEC_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"

python main_experiment.py \
  --dataset_name fewrel \
  --split test \
  --use_llm_plan \
  --use_llm_coding \
  --max_hypotheses 8 \
  --output_file ./results/fewrel_re.json
```

Print the selected datasets and baselines:

```bash
python main_experiment.py --list_recommendations
```
