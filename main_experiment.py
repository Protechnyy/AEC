from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable

from aec_pipeline import AECPipeline
from ontology import build_relation_definitions, load_relation_schemas
from utils.relation_data.loader import canonicalize_gold_relations, load_relation_samples
from utils.relation_evaluation.relation_scorer import evaluate_relation_predictions

ROOT_DIR = Path(__file__).resolve().parent

RECOMMENDED_DATASETS = {
    "fewrel": "FewRel 1.0/2.0 style entity-pair relation classification; useful for unseen-relation zero-shot splits.",
    "wiki-zsl": "Wiki-ZSL zero-shot relation classification; designed around relation labels unseen at training time.",
    "tacred": "TACRED sentence-level relation classification with no_relation; use as a broad news/web baseline.",
    "retacred": "Re-TACRED, a cleaner TACRED revision; preferred over TACRED when licensing/setup permits.",
    "semeval2010": "SemEval-2010 Task 8 relation classification; compact sanity benchmark with lexical relation names.",
    "scierc": "SciERC scientific IE relation extraction; useful for domain-transfer experiments.",
}

RECOMMENDED_BASELINES = {
    "DirectRE": "Single-step zero-shot prompt that outputs relation triples or the label for supplied entity pairs.",
    "CoT-RE": "DirectRE plus concise evidence reasoning before final code/JSON.",
    "DecomposeRE": "Entity-pair proposal followed by relation classification, matching the planning/coding ablation.",
    "GuidelineRE": "Schema/code-guided prompting with relation definitions and typed head/tail constraints.",
    "ChatIE-RE": "Multi-turn question-answering baseline adapted from ChatIE for relation extraction.",
    "ZS-BERT/RelationPrompt": "Historical specialist zero-shot RE baselines for non-LLM comparison.",
}


def merge_summary(total: dict[str, Any], current: dict[str, Any]) -> None:
    total["samples"] += 1
    for key in ("hypothesis_count", "validated_relation_count", "verified_pass_count", "verified_fail_count", "candidate_pair_count"):
        total[key] += int(current.get(key, 0))
    bucket = current.get("verifier_categories", {})
    if isinstance(bucket, dict):
        for name, value in bucket.items():
            total["verifier_categories"][name] = total["verifier_categories"].get(name, 0) + int(value)


def print_recommendations() -> None:
    print("Recommended zero-shot RE datasets:")
    for name, note in RECOMMENDED_DATASETS.items():
        print(f"- {name}: {note}")
    print("\nRecommended baselines:")
    for name, note in RECOMMENDED_BASELINES.items():
        print(f"- {name}: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent-Relation-Coder for zero-shot relation extraction.")
    parser.add_argument("--dataset_name", required=False, choices=sorted(RECOMMENDED_DATASETS))
    parser.add_argument("--split", default="test", choices=["train", "dev", "valid", "test"])
    parser.add_argument("--input_dir", default=str(ROOT_DIR / "datasets" / "relation_splits"))
    parser.add_argument("--schema_dir", default=str(ROOT_DIR / "utils" / "re_schema_generation" / "init_prompts"))
    parser.add_argument("--output_file", default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_hypotheses", type=int, default=8)
    parser.add_argument("--use_llm_plan", action="store_true")
    parser.add_argument("--use_llm_coding", action="store_true")
    parser.add_argument("--llm_model", default=os.getenv("AEC_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o")
    parser.add_argument("--llm_base_url", default=os.getenv("AEC_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "")
    parser.add_argument("--llm_api_key", default=os.getenv("AEC_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    parser.add_argument("--compact_output", action="store_true")
    parser.add_argument("--save_trace", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--list_recommendations", action="store_true", help="Print the chosen RE datasets and baselines, then exit.")
    args = parser.parse_args()

    if args.list_recommendations:
        print_recommendations()
        return
    if not args.dataset_name:
        parser.error("--dataset_name is required unless --list_recommendations is used")

    if args.llm_model:
        os.environ["AEC_LLM_MODEL"] = args.llm_model
    if args.llm_base_url:
        os.environ["AEC_LLM_BASE_URL"] = args.llm_base_url
    if args.llm_api_key:
        os.environ["AEC_LLM_API_KEY"] = args.llm_api_key

    schema_dir = Path(args.schema_dir)
    input_dir = Path(args.input_dir)
    schemas = load_relation_schemas(schema_dir, args.dataset_name)
    samples = load_relation_samples(input_dir, args.dataset_name, args.split, args.max_samples)
    for sample in samples:
        canonicalize_gold_relations(sample, schemas)

    pipeline = AECPipeline(
        max_hypotheses=args.max_hypotheses,
        use_llm_plan=args.use_llm_plan,
        use_llm_coding=args.use_llm_coding,
    )

    prediction_records: list[dict[str, Any]] = []
    aggregate_summary: dict[str, Any] = {
        "samples": 0,
        "hypothesis_count": 0,
        "validated_relation_count": 0,
        "verified_pass_count": 0,
        "verified_fail_count": 0,
        "candidate_pair_count": 0,
        "verifier_categories": {},
    }

    iterator = enumerate(samples, start=1)
    if not args.no_progress:
        iterator = tqdm(iterator, total=len(samples), desc=f"Running RE {args.dataset_name}/{args.split}", unit="sample")

    for sample_idx, sample in iterator:
        relations = pipeline.run_many(
            text=sample["text"],
            schemas=schemas,
            candidate_pairs=sample.get("candidate_pairs") or None,
            model=args.llm_model,
            base_url=args.llm_base_url or None,
            api_key=args.llm_api_key or None,
        )
        run_summary = dict(pipeline.last_run_summary)
        merge_summary(aggregate_summary, run_summary)
        record = {
            "Id": sample.get("id", sample_idx),
            "Input": sample["text"],
            "CandidatePairs": sample.get("candidate_pairs", []),
            "GoldRelations": sample.get("gold_relations", []),
            "PredictionRelations": [relation.dict() for relation in relations],
            "RunSummary": run_summary,
        }
        if args.save_trace or not args.compact_output:
            record["Trace"] = pipeline.last_run_trace
        prediction_records.append(record)

    metrics = evaluate_relation_predictions(prediction_records)
    output_file = Path(args.output_file) if args.output_file else ROOT_DIR / "results" / f"{args.dataset_name}_{args.split}_re_predictions.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "task": "zero_shot_relation_extraction",
        "dataset": args.dataset_name,
        "split": args.split,
        "num_samples": len(prediction_records),
        "schema_dir": str(schema_dir),
        "input_dir": str(input_dir),
        "relation_definitions": build_relation_definitions(schemas),
        "recommended_baselines": RECOMMENDED_BASELINES,
        "recommended_datasets": RECOMMENDED_DATASETS,
        "config": {
            "max_hypotheses": args.max_hypotheses,
            "use_llm_plan": args.use_llm_plan,
            "use_llm_coding": args.use_llm_coding,
            "compact_output": args.compact_output,
        },
        "metrics": metrics,
        "run_summary": aggregate_summary,
        "predictions": prediction_records,
    }
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {output_file}")
    print("Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Run summary:")
    print(json.dumps(aggregate_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
