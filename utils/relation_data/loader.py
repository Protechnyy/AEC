"""Dataset loading utilities for zero-shot relation extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from event_schema import RelationObject, RelationSchema
from ontology import normalize_relation_label

NO_RELATION_LABELS = {"", "no_relation", "none", "na", "n/a", "other"}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def label_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def canonical_relation_label(label: str, schemas: dict[str, RelationSchema]) -> str:
    key = label_key(label)
    for name, schema in schemas.items():
        candidates = [name, *schema.aliases]
        if key in {label_key(candidate) for candidate in candidates}:
            return name
    return label


def canonicalize_gold_relations(sample: dict[str, Any], schemas: dict[str, RelationSchema]) -> None:
    relations = sample.get("gold_relations", [])
    if not isinstance(relations, list):
        return
    for relation in relations:
        if isinstance(relation, dict) and isinstance(relation.get("relation_type"), str):
            relation["relation_type"] = canonical_relation_label(relation["relation_type"], schemas)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "samples", "instances", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        flattened: list[dict[str, Any]] = []
        for relation_label, value in data.items():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict):
                    copied = dict(item)
                    copied.setdefault("relation", relation_label)
                    flattened.append(copied)
        if flattened:
            return flattened
    raise ValueError(f"Unsupported dataset JSON structure in {path}")


def find_split_file(input_dir: Path, dataset_name: str, split: str) -> Path:
    candidates = [
        input_dir / dataset_name / f"{split}.jsonl",
        input_dir / dataset_name / f"{split}.json",
        input_dir / dataset_name.lower() / f"{split}.jsonl",
        input_dir / dataset_name.lower() / f"{split}.json",
        input_dir / f"{dataset_name}_{split}.jsonl",
        input_dir / f"{dataset_name}_{split}.json",
    ]
    split_path = next((path for path in candidates if path.exists()), None)
    if split_path is None:
        raise FileNotFoundError(
            f"No split file found for {dataset_name}/{split}. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )
    return split_path


def span_from_token_offsets(tokens: list[str], start: Any, end: Any) -> str:
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        return ""
    end_exclusive = end + 1
    if end_exclusive > len(tokens):
        end_exclusive = end
    return normalize_space(" ".join(tokens[start:end_exclusive]))


def extract_entity(value: Any, tokens: list[str] | None = None) -> str:
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, list) and value:
        if isinstance(value[0], str):
            return normalize_space(value[0])
        if tokens and all(isinstance(item, int) for item in value[:2]):
            return span_from_token_offsets(tokens, value[0], value[1])
    if isinstance(value, dict):
        for key in ("text", "name", "mention", "span"):
            if isinstance(value.get(key), str):
                return normalize_space(value[key])
        if tokens:
            start = value.get("start") if "start" in value else value.get("start_offset")
            end = value.get("end") if "end" in value else value.get("end_offset")
            return span_from_token_offsets(tokens, start, end)
    return ""


def fewrel_entity(raw: Any, tokens: list[str] | None) -> str:
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], str):
            return normalize_space(raw[0])
        if len(raw) >= 3 and tokens and isinstance(raw[2], list) and raw[2]:
            first = raw[2][0]
            if isinstance(first, list) and len(first) >= 2:
                return span_from_token_offsets(tokens, first[0], first[-1])
    return extract_entity(raw, tokens)


def relation_from_raw(raw: dict[str, Any], tokens: list[str] | None = None) -> RelationObject | None:
    label = raw.get("relation_type") or raw.get("relation") or raw.get("label") or raw.get("predicate")
    if not isinstance(label, str) or label.lower() in NO_RELATION_LABELS:
        return None
    head = extract_entity(raw.get("head"), tokens) or extract_entity(raw.get("subject"), tokens) or extract_entity(raw.get("subj"), tokens)
    tail = extract_entity(raw.get("tail"), tokens) or extract_entity(raw.get("object"), tokens) or extract_entity(raw.get("obj"), tokens)
    if not head or not tail:
        return None
    return RelationObject(relation_type=normalize_relation_label(label), head=head, tail=tail)


def normalize_sample(record: dict[str, Any], idx: int) -> dict[str, Any]:
    tokens = record.get("tokens") if isinstance(record.get("tokens"), list) else None
    text = record.get("text") or record.get("sentence")
    if not isinstance(text, str) and tokens:
        text = " ".join(str(token) for token in tokens)
    text = normalize_space(str(text or ""))

    gold: list[RelationObject] = []
    for key in ("relations", "relation_mentions", "triples", "labels"):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    relation = relation_from_raw(item, tokens)
                    if relation is not None:
                        gold.append(relation)

    pair_head = fewrel_entity(record.get("h"), tokens) or extract_entity(record.get("head"), tokens)
    pair_tail = fewrel_entity(record.get("t"), tokens) or extract_entity(record.get("tail"), tokens)
    label = record.get("relation") or record.get("label")
    if not pair_head and tokens and {"subj_start", "subj_end", "obj_start", "obj_end"} <= set(record):
        pair_head = span_from_token_offsets(tokens, record.get("subj_start"), record.get("subj_end"))
        pair_tail = span_from_token_offsets(tokens, record.get("obj_start"), record.get("obj_end"))
    if pair_head and pair_tail and isinstance(label, str) and label.lower() not in NO_RELATION_LABELS:
        candidate = RelationObject(relation_type=normalize_relation_label(label), head=pair_head, tail=pair_tail)
        if candidate.as_tuple() not in {item.as_tuple() for item in gold}:
            gold.append(candidate)

    candidate_pairs = []
    if pair_head and pair_tail:
        candidate_pairs.append({"head": pair_head, "tail": pair_tail})
    raw_pairs = record.get("candidate_pairs", [])
    if isinstance(raw_pairs, list):
        for item in raw_pairs:
            if isinstance(item, dict):
                head = extract_entity(item.get("head") or item.get("subject"), tokens)
                tail = extract_entity(item.get("tail") or item.get("object"), tokens)
                if head and tail:
                    candidate_pairs.append({"head": head, "tail": tail})

    return {
        "id": record.get("id", idx),
        "text": text,
        "candidate_pairs": candidate_pairs,
        "gold_relations": [item.dict() for item in gold],
        "raw": record,
    }


def load_relation_samples(input_dir: Path, dataset_name: str, split: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    split_path = find_split_file(input_dir, dataset_name, split)
    records = [normalize_sample(record, idx) for idx, record in enumerate(read_json_records(split_path))]
    return records if max_samples is None else records[:max_samples]
