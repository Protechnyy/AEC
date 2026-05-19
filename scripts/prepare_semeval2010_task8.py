#!/usr/bin/env python3
"""Prepare and export SemEval-2010 Task 8 data for Agent-Relation-Coder."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from AEC.relation_schema import (  # noqa: E402
    load_relation_schemas,
    schemas_by_relation_type,
)
from AEC.run_relation_inference import read_json_records, safe_parse_triples  # noqa: E402


RELATION_DESCRIPTIONS = {
    "Cause-Effect": "One nominal causes, produces, generates, or leads to the other.",
    "Component-Whole": "One nominal is a component, part, member, or constituent of the other whole.",
    "Content-Container": "One nominal is content located in or held by the other container.",
    "Entity-Destination": "One nominal moves, is sent, or is transferred to the other destination.",
    "Entity-Origin": "One nominal comes from, is derived from, or originates in the other.",
    "Instrument-Agency": "One nominal is an instrument, tool, or means used by the other agency.",
    "Member-Collection": "One nominal is a member, element, or item in the other collection.",
    "Message-Topic": "One nominal is a message, statement, or communication about the other topic.",
    "Product-Producer": "One nominal is a product created, manufactured, or produced by the other.",
}

RELATION_ROLES = {
    "Cause-Effect": ("cause", "effect"),
    "Component-Whole": ("component", "whole"),
    "Content-Container": ("content", "container"),
    "Entity-Destination": ("entity", "destination"),
    "Entity-Origin": ("entity", "origin"),
    "Instrument-Agency": ("instrument", "agency"),
    "Member-Collection": ("member", "collection"),
    "Message-Topic": ("message", "topic"),
    "Product-Producer": ("product", "producer"),
}


def base_relation(label: str) -> str:
    return label.split("(", 1)[0]


def parse_sentence_line(line: str) -> Tuple[str, str, str, str]:
    sent_id, tagged = line.rstrip("\n").split("\t", 1)
    e1_match = re.search(r"<e1>(.*?)</e1>", tagged)
    e2_match = re.search(r"<e2>(.*?)</e2>", tagged)
    if not e1_match or not e2_match:
        raise ValueError(f"Missing entity markers in sentence {sent_id}: {tagged}")
    e1 = e1_match.group(1)
    e2 = e2_match.group(1)
    text = (
        tagged.replace("<e1>", "")
        .replace("</e1>", "")
        .replace("<e2>", "")
        .replace("</e2>", "")
    )
    return sent_id, text, e1, e2


def read_labels(path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sent_id, label = line.split("\t", 1)
        labels[sent_id] = label.strip()
    return labels


def convert_split(sent_path: Path, label_path: Path, out_path: Path) -> None:
    labels = read_labels(label_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for line in sent_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sent_id, text, e1, e2 = parse_sentence_line(line)
            label = labels[sent_id]
            record: Dict[str, Any] = {
                "id": sent_id,
                "text": text,
                "candidate_pairs": [[{"text": e1}, {"text": e2}]],
                "relation_mentions": [],
                "semeval_label": label,
            }
            if label != "Other":
                relation = base_relation(label)
                source, target = relation_direction(label)
                mentions = {"e1": e1, "e2": e2}
                record["relation_mentions"].append(
                    {
                        "relation": relation,
                        "subject": {"text": mentions[source]},
                        "object": {"text": mentions[target]},
                        "semeval_label": label,
                    }
                )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def relation_direction(label: str) -> Tuple[str, str]:
    if label.endswith("(e1,e2)"):
        return "e1", "e2"
    if label.endswith("(e2,e1)"):
        return "e2", "e1"
    return "e1", "e2"


def create_schema(train_labels: Iterable[str], test_labels: Iterable[str], out_path: Path) -> None:
    relation_labels = sorted({base_relation(label) for label in list(train_labels) + list(test_labels) if label != "Other"})
    schemas: List[dict] = []
    for relation in relation_labels:
        subject_role, object_role = RELATION_ROLES.get(relation, ("subject", "object"))
        schemas.append(
            {
                "relation_type": relation,
                "description": (
                    f"{RELATION_DESCRIPTIONS.get(relation, 'A semantic relation between two nominals')} "
                    "SemEval labels use e1/e2 only to mark the two mentions; this "
                    f"schema uses semantic triples where subject is the {subject_role} "
                    f"and object is the {object_role}. If the official label is "
                    "Relation(e2,e1), the output triple must swap the two mentions."
                ),
                "subject_role": subject_role,
                "object_role": object_role,
                "subject_type": "NOMINAL",
                "object_type": "NOMINAL",
                "aliases": [relation.replace("-", " ")],
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schemas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    schema_path = Path(args.schema)

    train_labels = read_labels(raw_dir / "train" / "train_result_full.txt")
    test_labels = read_labels(raw_dir / "test" / "test_result_full.txt")

    convert_split(
        raw_dir / "train" / "train.txt",
        raw_dir / "train" / "train_result_full.txt",
        processed_dir / "train.jsonl",
    )
    convert_split(
        raw_dir / "test" / "test.txt",
        raw_dir / "test" / "test_result_full.txt",
        processed_dir / "test.jsonl",
    )
    create_schema(train_labels.values(), test_labels.values(), schema_path)

    print(f"Wrote {processed_dir / 'train.jsonl'}")
    print(f"Wrote {processed_dir / 'test.jsonl'}")
    print(f"Wrote {schema_path}")


def export_official(args: argparse.Namespace) -> None:
    schemas = load_relation_schemas(args.schema)
    schema_by_type = schemas_by_relation_type(schemas)
    records = read_json_records(args.predictions)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for record in records:
            triples, _ = safe_parse_triples(record.get("Prediction", "[]"))
            label = "Other"
            pair = None
            raw_pairs = record.get("candidate_pairs")
            if isinstance(raw_pairs, list) and raw_pairs:
                first = raw_pairs[0]
                if isinstance(first, list) and len(first) >= 2:
                    pair = (str(first[0]), str(first[1]))
            for triple in triples:
                relation = str(triple.get("relation", ""))
                if relation in schema_by_type:
                    subject = str(triple.get("subject", ""))
                    obj = str(triple.get("object", ""))
                    if pair and (subject, obj) == pair:
                        label = f"{relation}(e1,e2)"
                    elif pair and (subject, obj) == (pair[1], pair[0]):
                        label = f"{relation}(e2,e1)"
                    else:
                        label = relation
                    break
            out.write(f"{record.get('doc_id')}\t{label}\n")
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--raw_dir", default=str(ROOT / "data" / "raw" / "semeval2010_task8"))
    p_prepare.add_argument("--processed_dir", default=str(ROOT / "data" / "processed" / "semeval2010_task8"))
    p_prepare.add_argument("--schema", default=str(ROOT / "data" / "schemas" / "semeval2010_task8.json"))
    p_prepare.set_defaults(func=prepare)

    p_export = sub.add_parser("export-official")
    p_export.add_argument("--predictions", required=True)
    p_export.add_argument("--schema", default=str(ROOT / "data" / "schemas" / "semeval2010_task8.json"))
    p_export.add_argument("--output", default=str(ROOT / "outputs" / "semeval2010_task8_proposed.txt"))
    p_export.set_defaults(func=export_official)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
