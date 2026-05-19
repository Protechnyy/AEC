#!/usr/bin/env python3
"""Prepare SciERC processed JSON for the relation extraction runner."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]


RELATION_SCHEMAS = [
    {
        "relation_type": "USED-FOR",
        "description": (
            "The subject scientific entity is used for, applied to, or intended "
            "to solve the object task, process, application, or goal."
        ),
        "subject_role": "tool_or_entity",
        "object_role": "purpose_or_task",
        "aliases": ["used for", "applied to", "for"],
    },
    {
        "relation_type": "FEATURE-OF",
        "description": (
            "The subject is a feature, property, characteristic, constraint, or "
            "aspect of the object scientific entity."
        ),
        "subject_role": "feature",
        "object_role": "entity",
        "aliases": ["feature of", "property of", "aspect of"],
    },
    {
        "relation_type": "PART-OF",
        "description": (
            "The subject is a component, part, stage, element, or substructure "
            "of the object scientific entity."
        ),
        "subject_role": "part",
        "object_role": "whole",
        "aliases": ["part of", "component of", "element of"],
    },
    {
        "relation_type": "HYPONYM-OF",
        "description": (
            "The subject is a more specific kind, example, instance, subtype, "
            "or member of the broader object category."
        ),
        "subject_role": "specific_entity",
        "object_role": "general_category",
        "aliases": ["type of", "kind of", "such as", "instance of"],
    },
    {
        "relation_type": "EVALUATE-FOR",
        "description": (
            "The subject is a metric, criterion, evaluation aspect, quality, or "
            "measured result used to evaluate the object method, system, task, "
            "or entity."
        ),
        "subject_role": "metric_or_criterion",
        "object_role": "evaluated_entity",
        "aliases": ["evaluate for", "measured for", "performance on"],
    },
    {
        "relation_type": "CONJUNCTION",
        "description": (
            "The two entities are coordinated conjuncts or parallel items that "
            "play the same role in the sentence."
        ),
        "subject_role": "conjunct",
        "object_role": "conjunct",
        "aliases": ["and", "or", "as well as"],
        "symmetric": True,
    },
    {
        "relation_type": "COMPARE",
        "description": (
            "The two entities are explicitly compared, contrasted, or evaluated "
            "against one another."
        ),
        "subject_role": "compared_entity",
        "object_role": "comparison_target",
        "aliases": ["compare", "outperform", "better than", "versus"],
        "symmetric": True,
    },
]


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def span_text(flat_tokens: List[str], start: int, end: int) -> str:
    return " ".join(str(token) for token in flat_tokens[start : end + 1])


def entity_record(flat_tokens: List[str], item: List[Any]) -> dict:
    start, end, entity_type = item
    return {
        "text": span_text(flat_tokens, int(start), int(end)),
        "type": str(entity_type),
        "start": int(start),
        "end": int(end),
    }


def relation_record(flat_tokens: List[str], item: List[Any]) -> dict:
    subj_start, subj_end, obj_start, obj_end, relation = item
    return {
        "relation": str(relation),
        "subject": {
            "text": span_text(flat_tokens, int(subj_start), int(subj_end)),
            "start": int(subj_start),
            "end": int(subj_end),
        },
        "object": {
            "text": span_text(flat_tokens, int(obj_start), int(obj_end)),
            "start": int(obj_start),
            "end": int(obj_end),
        },
    }


def make_candidate_pairs(entities: List[dict], relations: List[dict]) -> List[List[dict]]:
    pairs: List[List[dict]] = []
    seen: set[tuple[str, str]] = set()

    for left, right in combinations(entities, 2):
        key = (left["text"], right["text"])
        if key not in seen:
            seen.add(key)
            pairs.append([left, right])

    # Relations should already be covered by the entity combinations, but keep
    # gold pairs if a processed file contains a relation span missing from NER.
    for rel in relations:
        left = rel["subject"]
        right = rel["object"]
        key = (left["text"], right["text"])
        reverse_key = (right["text"], left["text"])
        if key not in seen and reverse_key not in seen:
            seen.add(key)
            pairs.append([left, right])

    return pairs


def convert_split(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for doc in read_jsonl(in_path):
            doc_key = str(doc.get("doc_key", ""))
            sentences = doc.get("sentences", [])
            ner_by_sentence = doc.get("ner", [])
            relations_by_sentence = doc.get("relations", [])
            flat_tokens = [str(token) for sent in sentences for token in sent]

            for sent_idx, tokens in enumerate(sentences):
                entities = [
                    entity_record(flat_tokens, item)
                    for item in ner_by_sentence[sent_idx]
                ]
                relations = [
                    relation_record(flat_tokens, item)
                    for item in relations_by_sentence[sent_idx]
                ]
                record = {
                    "id": f"{doc_key}:{sent_idx}",
                    "doc_id": doc_key,
                    "sentence_id": sent_idx,
                    "text": " ".join(str(token) for token in tokens),
                    "entities": entities,
                    "candidate_pairs": make_candidate_pairs(entities, relations),
                    "relation_mentions": relations,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_schema(out_path: Path) -> None:
    schemas = []
    for item in RELATION_SCHEMAS:
        schema = {
            "subject_type": "SCIENTIFIC_ENTITY",
            "object_type": "SCIENTIFIC_ENTITY",
            **item,
        }
        schemas.append(schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schemas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    for split in ("train", "dev", "test"):
        convert_split(raw_dir / f"{split}.json", processed_dir / f"{split}.jsonl")
        print(f"Wrote {processed_dir / f'{split}.jsonl'}")
    write_schema(Path(args.schema))
    print(f"Wrote {args.schema}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_dir",
        default=str(ROOT / "data" / "raw" / "scierc" / "processed_data" / "json"),
    )
    parser.add_argument(
        "--processed_dir",
        default=str(ROOT / "data" / "processed" / "scierc"),
    )
    parser.add_argument(
        "--schema",
        default=str(ROOT / "data" / "schemas" / "scierc.json"),
    )
    parser.set_defaults(func=prepare)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
