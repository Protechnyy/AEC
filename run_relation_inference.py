"""
Run the multi-agent code-generation pipeline on relation extraction data.

The runner supports two common RE settings:

* end-to-end tuple extraction: find all ``(subject, object, relation)`` triples.
* given-pair relation classification: classify provided subject/object pairs.

Input can be JSON or JSONL.  The loader accepts common fields such as
``relation_mentions``, ``relations``, ``triples``, TACRED-style ``token`` /
``subj_start`` / ``obj_start`` records, and explicit ``candidate_pairs``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from prettytable import PrettyTable
except ImportError:  # pragma: no cover - exercised only in minimal envs
    class PrettyTable:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.field_names: List[str] = []
            self._rows: List[List[str]] = []

        def add_row(self, row: Sequence[str]) -> None:
            self._rows.append([str(item) for item in row])

        def __str__(self) -> str:
            rows = [self.field_names] + self._rows
            widths = [max(len(str(row[i])) for row in rows) for i in range(len(self.field_names))]

            def fmt(row: Sequence[str]) -> str:
                return " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row))

            sep = "-+-".join("-" * width for width in widths)
            return "\n".join([fmt(self.field_names), sep, *[fmt(row) for row in self._rows]])

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs
    def tqdm(iterable: Iterable[Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
        return iterable

_here = Path(__file__).resolve().parent
_parent = _here.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from AEC.relation_agents import (  # noqa: E402
    RelationCodingAgent,
    RelationPlanningAgent,
    RelationRetrievalAgent,
    RelationVerificationAgent,
    RelationVerificationError,
)
from AEC.relation_schema import (  # noqa: E402
    RelationSchema,
    build_relation_definitions,
    build_relation_definition,
    infer_relation_schemas,
    is_no_relation,
    load_relation_schemas,
    schemas_by_relation_type,
)


OUTPUT_DIR = _here / "outputs"


def compute_f1(pred_num: int, gold_num: int, match_num: int) -> Dict[str, float]:
    precision = match_num / pred_num if pred_num else 0.0
    recall = match_num / gold_num if gold_num else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def read_json_records(path: str | Path) -> List[dict]:
    """Read a JSON array, JSON object wrapper, JSONL, or concatenated JSON."""

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    try:
        raw = json.loads(stripped)
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
        if isinstance(raw, dict):
            for key in ("data", "records", "examples", "samples"):
                if isinstance(raw.get(key), list):
                    return [r for r in raw[key] if isinstance(r, dict)]
            return [raw]
    except json.JSONDecodeError:
        pass

    lines = [line for line in stripped.splitlines() if line.strip()]
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        records: List[dict] = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(stripped):
            obj, end = decoder.raw_decode(stripped, pos)
            if isinstance(obj, dict):
                records.append(obj)
            pos = end
            while pos < len(stripped) and stripped[pos] in " \t\r\n":
                pos += 1
        return records


def _tokens_to_text(tokens: Any) -> str:
    if isinstance(tokens, list):
        return " ".join(str(tok) for tok in tokens)
    return ""


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _span_from_token_indices(sample: Mapping[str, Any], start_key: str, end_key: str) -> str:
    tokens = sample.get("token") or sample.get("tokens")
    if not isinstance(tokens, list) or start_key not in sample or end_key not in sample:
        return ""
    try:
        start = int(sample[start_key])
        end = int(sample[end_key])
    except (TypeError, ValueError):
        return ""
    if start < 0 or end < start or end >= len(tokens):
        return ""
    return " ".join(str(tok) for tok in tokens[start : end + 1])


def span_text(value: Any, sample: Mapping[str, Any]) -> str:
    """Extract a text span from strings or common mention dictionaries."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return " ".join(value)
    if not isinstance(value, Mapping):
        return str(value)

    direct = _first_present(value, ("text", "span", "mention", "name", "value", "surface"))
    if isinstance(direct, str):
        return direct
    if isinstance(direct, list):
        return _tokens_to_text(direct)

    text = str(sample.get("text") or sample.get("sentence") or "")
    start = _first_present(value, ("start", "start_char", "char_start"))
    end = _first_present(value, ("end", "end_char", "char_end"))
    if text and start is not None and end is not None:
        try:
            return text[int(start) : int(end)]
        except (TypeError, ValueError):
            pass

    tokens = sample.get("token") or sample.get("tokens")
    token_start = _first_present(value, ("token_start", "start_token", "tok_start"))
    token_end = _first_present(value, ("token_end", "end_token", "tok_end"))
    if isinstance(tokens, list) and token_start is not None and token_end is not None:
        try:
            start_i = int(token_start)
            end_i = int(token_end)
            return " ".join(str(tok) for tok in tokens[start_i : end_i + 1])
        except (TypeError, ValueError):
            pass

    return ""


def _subject_object_pair(sample: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    subj = _first_present(sample, ("subj", "subject", "head", "h", "arg1", "e1"))
    obj = _first_present(sample, ("obj", "object", "tail", "t", "arg2", "e2"))
    arg1 = span_text(subj, sample)
    arg2 = span_text(obj, sample)

    if not arg1:
        arg1 = _span_from_token_indices(sample, "subj_start", "subj_end")
    if not arg2:
        arg2 = _span_from_token_indices(sample, "obj_start", "obj_end")

    if arg1 and arg2:
        return (arg1, arg2)
    return None


def _normalise_pair_item(item: Any, sample: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        arg1 = span_text(item[0], sample)
        arg2 = span_text(item[1], sample)
    elif isinstance(item, Mapping):
        arg1 = span_text(_first_present(item, ("arg1", "subject", "head", "subj", "e1")), sample)
        arg2 = span_text(_first_present(item, ("arg2", "object", "tail", "obj", "e2")), sample)
    else:
        return None
    return (arg1, arg2) if arg1 and arg2 else None


def _extract_candidate_pairs(sample: Mapping[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    raw_pairs = _first_present(sample, ("candidate_pairs", "pairs", "entity_pairs"))
    if isinstance(raw_pairs, list):
        for item in raw_pairs:
            pair = _normalise_pair_item(item, sample)
            if pair:
                pairs.append(pair)
    pair = _subject_object_pair(sample)
    if pair:
        pairs.append(pair)

    deduped: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return deduped


def normalise_relation_item(item: Mapping[str, Any], sample: Mapping[str, Any]) -> Optional[dict]:
    relation = _first_present(
        item,
        ("relation", "relation_type", "predicate", "type", "label", "class"),
    )
    if not isinstance(relation, str) or is_no_relation(relation):
        return None

    subject_value = _first_present(item, ("subject", "arg1", "head", "subj", "source", "e1"))
    object_value = _first_present(item, ("object", "arg2", "tail", "obj", "target", "e2"))

    if (subject_value is None or object_value is None) and isinstance(item.get("arguments"), list):
        arguments = item["arguments"]
        if len(arguments) >= 2:
            subject_value = subject_value if subject_value is not None else arguments[0]
            object_value = object_value if object_value is not None else arguments[1]

    subject = span_text(subject_value, sample)
    obj = span_text(object_value, sample)
    if not subject or not obj:
        return None

    return {
        "subject": subject,
        "object": obj,
        "relation": relation,
    }


def normalise_sample(sample: Mapping[str, Any]) -> dict:
    text = str(sample.get("text") or sample.get("sentence") or _tokens_to_text(sample.get("token") or sample.get("tokens")))
    relation_mentions: List[dict] = []
    has_relation_annotation = False

    relation_lists = ("relation_mentions", "relations", "triples", "relation_list")
    for key in relation_lists:
        if key in sample:
            has_relation_annotation = True
            raw_relations = sample.get(key)
            if isinstance(raw_relations, list):
                for item in raw_relations:
                    if isinstance(item, Mapping):
                        rel = normalise_relation_item(item, sample)
                        if rel:
                            relation_mentions.append(rel)

    if not relation_mentions and "relation" in sample:
        has_relation_annotation = True
        relation_type = sample.get("relation")
        pair = _subject_object_pair(sample)
        if isinstance(relation_type, str) and not is_no_relation(relation_type) and pair:
            relation_mentions.append(
                {
                    "subject": pair[0],
                    "object": pair[1],
                    "relation": relation_type,
                }
            )

    candidate_pairs = _extract_candidate_pairs(sample)
    return {
        "text": text,
        "relation_mentions": relation_mentions,
        "candidate_pairs": candidate_pairs,
        "has_relation_annotation": has_relation_annotation,
        "doc_id": sample.get("id") or sample.get("doc_id") or sample.get("guid") or sample.get("uid") or "",
        "raw": sample,
    }


_STRIP_PREFIXES = (
    "the ",
    "a ",
    "an ",
    "to ",
    "from ",
    "in ",
    "at ",
    "on ",
    "of ",
    "by ",
    "with ",
    "for ",
)


def _shrink_span(span: str, source_text: str) -> str:
    if not span or not source_text:
        return span
    span = span.strip()
    lower = span.lower()
    for prefix in sorted(_STRIP_PREFIXES, key=len, reverse=True):
        if lower.startswith(prefix):
            candidate = span[len(prefix) :]
            if candidate and candidate in source_text:
                return candidate
    return span


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json|python)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _extract_json_array(raw: str) -> str:
    raw = _strip_json_fence(raw)
    start = raw.find("[")
    if start < 0:
        return raw

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return raw[start:]


def _normalise_triple_item(item: Any) -> Tuple[Optional[dict], Optional[str]]:
    if not isinstance(item, Mapping):
        return None, "triple item is not a JSON object"

    subject = _first_present(item, ("subject", "arg1", "head", "subj", "source", "e1"))
    obj = _first_present(item, ("object", "arg2", "tail", "obj", "target", "e2"))
    relation = _first_present(item, ("relation", "relation_type", "predicate", "type", "label", "class"))

    if not isinstance(subject, str) or not subject:
        return None, "missing non-empty subject"
    if not isinstance(obj, str) or not obj:
        return None, "missing non-empty object"
    if not isinstance(relation, str) or not relation:
        return None, "missing non-empty relation"
    if is_no_relation(relation):
        return None, None

    return {"subject": subject, "object": obj, "relation": relation}, None


def safe_parse_triples(value: Any) -> Tuple[List[dict], List[str]]:
    """Parse prediction/label values as JSON triples.

    The canonical shape is ``[{"subject": "...", "object": "...",
    "relation": "..."}]``. ``arg1``/``arg2``/``relation_type`` are accepted
    only as input aliases so older converted data can be read, then normalized
    to the canonical keys.
    """

    if value is None:
        return [], []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped == "[]":
            return [], []
        try:
            data = json.loads(_extract_json_array(stripped))
        except json.JSONDecodeError as exc:
            return [], [f"JSONDecodeError: {exc}"]
    else:
        data = value

    if isinstance(data, Mapping):
        data = [data]
    if not isinstance(data, list):
        return [], [f"TypeError: expected JSON list, got {type(data).__name__}"]

    triples: List[dict] = []
    errors: List[str] = []
    for idx, item in enumerate(data):
        triple, error = _normalise_triple_item(item)
        if error:
            errors.append(f"item {idx}: {error}")
        if triple:
            triples.append(triple)
    return triples, errors


def serialize_triples(triples: Iterable[Mapping[str, Any]]) -> str:
    normalised: List[dict] = []
    for item in triples:
        triple, _error = _normalise_triple_item(item)
        if triple:
            normalised.append(triple)
    return json.dumps(normalised, ensure_ascii=False)


def pair_key(triple: Mapping[str, Any], schema_by_type: Mapping[str, RelationSchema]) -> Tuple[str, str]:
    subject = str(triple.get("subject", ""))
    obj = str(triple.get("object", ""))
    relation = str(triple.get("relation", ""))
    schema = schema_by_type.get(relation)
    if schema and schema.symmetric:
        subject, obj = sorted([subject, obj])
    return (subject, obj)


def triple_key(triple: Mapping[str, Any], schema_by_type: Mapping[str, RelationSchema]) -> Tuple[str, str, str]:
    subject, obj = pair_key(triple, schema_by_type)
    return (subject, obj, str(triple.get("relation", "")))


def postprocess_relation_prediction(code: str, source_text: str) -> str:
    triples, _ = safe_parse_triples(code)
    if not triples:
        return "[]"
    for triple in triples:
        triple["subject"] = _shrink_span(triple["subject"], source_text)
        triple["object"] = _shrink_span(triple["object"], source_text)
    return serialize_triples(triples)


def build_gold_label(
    relation_mentions: Sequence[Mapping[str, Any]],
    schema_by_type: Mapping[str, RelationSchema],
) -> str:
    triples: List[dict] = []
    for rel in relation_mentions:
        relation = str(rel.get("relation") or rel.get("relation_type") or "")
        schema = schema_by_type.get(relation)
        if not schema:
            continue
        subject = str(rel.get("subject") or rel.get("arg1") or "")
        obj = str(rel.get("object") or rel.get("arg2") or "")
        if not subject or not obj:
            continue
        triples.append(
            {
                "subject": subject,
                "object": obj,
                "relation": schema.relation_type,
            }
        )
    return serialize_triples(triples)


def merge_prediction_strings(
    predictions: Sequence[str],
    schema_by_type: Mapping[str, RelationSchema],
) -> str:
    merged: List[dict] = []
    seen: set[Tuple[str, str, str]] = set()
    for prediction in predictions:
        triples, _ = safe_parse_triples(prediction)
        for triple in triples:
            key = triple_key(triple, schema_by_type)
            if key not in seen:
                seen.add(key)
                merged.append(triple)
    return serialize_triples(merged)


def run_relation_pipeline(
    text: str,
    schema: RelationSchema,
    model: str,
    k: int = 3,
    t: int = 3,
    candidate_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    retriever: Optional[RelationRetrievalAgent] = None,
    planner: Optional[RelationPlanningAgent] = None,
    coder: Optional[RelationCodingAgent] = None,
    verifier: Optional[RelationVerificationAgent] = None,
) -> Tuple[str, bool]:
    retriever = retriever or RelationRetrievalAgent()
    planner = planner or RelationPlanningAgent()
    coder = coder or RelationCodingAgent()
    verifier = verifier or RelationVerificationAgent()

    schema_def = build_relation_definition(schema, include_base=True)
    exemplars = retriever.retrieve(schema_def, k=k, model=model)
    hypotheses = planner.generate_hypotheses(
        text=text,
        schema_definition=schema_def,
        relation_type=schema.relation_type,
        exemplars=exemplars,
        candidate_pairs=candidate_pairs,
        k=k,
        model=model,
    )

    for hyp in hypotheses:
        patch_feedback: Optional[str] = None
        for _attempt in range(1, t + 1):
            code_str = coder.generate_code(
                hypothesis=hyp,
                schema_definition=schema_def,
                text=text,
                exemplars=exemplars,
                patch_feedback=patch_feedback,
                model=model,
            )
            try:
                verifier.verify_code(
                    code_str,
                    text,
                    [schema],
                    expected_relation=schema.relation_type,
                    allowed_pairs=candidate_pairs,
                )
                return postprocess_relation_prediction(code_str, text), True
            except RelationVerificationError as exc:
                patch_feedback = str(exc)
    return "[]", False


def run_relation_classification_pipeline(
    text: str,
    schemas: Sequence[RelationSchema],
    candidate_pair: Tuple[str, str],
    model: str,
    t: int = 3,
    planner: Optional[RelationPlanningAgent] = None,
    coder: Optional[RelationCodingAgent] = None,
    verifier: Optional[RelationVerificationAgent] = None,
) -> Tuple[str, bool]:
    """Classify one candidate pair against all relation schemas in one call."""

    planner = planner or RelationPlanningAgent()
    coder = coder or RelationCodingAgent()
    verifier = verifier or RelationVerificationAgent()
    schema_by_type = schemas_by_relation_type(schemas)

    definitions = build_relation_definitions(schemas)
    hyp = planner.classify_pair(
        text=text,
        schema_definitions=definitions,
        candidate_pair=candidate_pair,
        model=model,
    )
    if hyp is None:
        return "[]", False
    if is_no_relation(hyp.relation):
        return "[]", True

    schema = schema_by_type.get(hyp.relation)
    if not schema:
        return "[]", False

    schema_def = build_relation_definition(schema, include_base=True)
    allowed_pairs = [candidate_pair, (candidate_pair[1], candidate_pair[0])]
    direct_code = serialize_triples(
        [
            {
                "subject": hyp.subject,
                "object": hyp.object,
                "relation": schema.relation_type,
            }
        ]
    )
    try:
        verifier.verify_code(
            direct_code,
            text,
            schemas,
            expected_relation=schema.relation_type,
            allowed_pairs=allowed_pairs,
        )
        return postprocess_relation_prediction(direct_code, text), True
    except RelationVerificationError as exc:
        patch_feedback: Optional[str] = str(exc)

    for _attempt in range(1, t + 1):
        code_str = coder.generate_code(
            hypothesis=hyp,
            schema_definition=schema_def,
            text=text,
            exemplars="",
            patch_feedback=patch_feedback,
            model=model,
        )
        try:
            verifier.verify_code(
                code_str,
                text,
                schemas,
                expected_relation=schema.relation_type,
                allowed_pairs=allowed_pairs,
            )
            return postprocess_relation_prediction(code_str, text), True
        except RelationVerificationError as exc:
            patch_feedback = str(exc)
    return "[]", False


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]],
    schemas: Sequence[RelationSchema],
) -> Dict[str, Any]:
    schema_by_type = schemas_by_relation_type(schemas)
    pair_pred_num = pair_gold_num = pair_match_num = 0
    rel_pred_num = rel_gold_num = rel_match_num = 0
    eval_errors: List[dict] = []

    for idx, record in enumerate(records):
        pred_triples, pred_errors = safe_parse_triples(record.get("Prediction", "[]"))
        gold_triples, gold_errors = safe_parse_triples(record.get("Label", "[]"))
        if pred_errors or gold_errors:
            eval_errors.append({"index": idx, "prediction_errors": pred_errors, "label_errors": gold_errors})

        pred_pairs = {pair_key(triple, schema_by_type) for triple in pred_triples}
        gold_pairs = {pair_key(triple, schema_by_type) for triple in gold_triples}
        pred_relations = {triple_key(triple, schema_by_type) for triple in pred_triples}
        gold_relations = {triple_key(triple, schema_by_type) for triple in gold_triples}

        pair_pred_num += len(pred_pairs)
        pair_gold_num += len(gold_pairs)
        pair_match_num += len(pred_pairs & gold_pairs)
        rel_pred_num += len(pred_relations)
        rel_gold_num += len(gold_relations)
        rel_match_num += len(pred_relations & gold_relations)

    pair_scores = compute_f1(pair_pred_num, pair_gold_num, pair_match_num)
    rel_scores = compute_f1(rel_pred_num, rel_gold_num, rel_match_num)

    table = PrettyTable()
    table.field_names = ["Metric", "Entity Pair Identification", "Relation Classification"]
    table.add_row(["Micro Precision", f"{pair_scores['precision'] * 100:.2f}", f"{rel_scores['precision'] * 100:.2f}"])
    table.add_row(["Micro Recall", f"{pair_scores['recall'] * 100:.2f}", f"{rel_scores['recall'] * 100:.2f}"])
    table.add_row(["Micro F1", f"{pair_scores['f1'] * 100:.2f}", f"{rel_scores['f1'] * 100:.2f}"])
    print(table)

    return {
        "entity_pair_precision": pair_scores["precision"],
        "entity_pair_recall": pair_scores["recall"],
        "entity_pair_f1": pair_scores["f1"],
        "relation_cls_precision": rel_scores["precision"],
        "relation_cls_recall": rel_scores["recall"],
        "relation_cls_f1": rel_scores["f1"],
        "eval_errors": eval_errors,
    }


def infer_schemas_from_output(records: Sequence[Mapping[str, Any]]) -> List[RelationSchema]:
    relation_labels: set[str] = set()
    for record in records:
        for field in ("Prediction", "Label"):
            triples, _ = safe_parse_triples(record.get(field, "[]"))
            relation_labels.update(str(triple["relation"]) for triple in triples if triple.get("relation"))
    return [
        RelationSchema(relation_type=label, class_name=label)
        for label in sorted(relation_labels)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-agent relation extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default=None, help="Input JSON/JSONL relation extraction data.")
    parser.add_argument("--schema", default=None, help="Optional JSON/JSONL relation schema file.")
    parser.add_argument("--model", default="gpt-4o", help="LLM model ID.")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible local server URL, e.g. vLLM.")
    parser.add_argument("--k", type=int, default=3, help="Planning hypotheses per relation type.")
    parser.add_argument("--t", type=int, default=3, help="Patch attempts per hypothesis.")
    parser.add_argument("--mode", choices=["auto", "end_to_end", "given_pairs"], default="auto")
    parser.add_argument("--relation_types", default=None, help="Comma-separated subset of relation labels to run.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--sample_seed", type=int, default=None, help="Randomly sample with this seed when limiting.")
    parser.add_argument("--output", default=None, help="Output predictions JSON path.")
    parser.add_argument("--delay", type=float, default=0.0, help="Sleep between relation-type calls.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file.")
    parser.add_argument("--no_eval", action="store_true", help="Skip evaluation after inference.")
    parser.add_argument("--eval_only", default=None, metavar="FILE", help="Only evaluate an existing predictions file.")
    return parser.parse_args()


def load_schemas_for_run(args: argparse.Namespace, samples: Optional[Sequence[Mapping[str, Any]]] = None) -> List[RelationSchema]:
    if args.schema:
        schemas = load_relation_schemas(args.schema)
    elif samples is not None:
        relation_types = [
            rel["relation"]
            for sample in samples
            for rel in sample.get("relation_mentions", [])
            if rel.get("relation")
        ]
        schemas = infer_relation_schemas(relation_types)
    else:
        schemas = []

    if args.relation_types:
        wanted = {item.strip() for item in args.relation_types.split(",") if item.strip()}
        schemas = [schema for schema in schemas if schema.relation_type in wanted or schema.class_name in wanted]
    return schemas


def main() -> None:
    args = parse_args()

    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    if args.eval_only:
        records = read_json_records(args.eval_only)
        schemas = load_schemas_for_run(args) if args.schema else infer_schemas_from_output(records)
        scores = evaluate_predictions(records, schemas)
        scores_path = str(args.eval_only).replace(".json", "_scores.json")
        with open(scores_path, "w", encoding="utf-8") as fh:
            json.dump({k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in scores.items()}, fh, indent=2)
        print(f"Scores saved -> {scores_path}")
        return

    if not args.data:
        print("ERROR: --data is required unless --eval_only is set.")
        sys.exit(1)

    raw_samples = [normalise_sample(sample) for sample in read_json_records(args.data)]
    if args.max_samples and args.max_samples < len(raw_samples):
        if args.sample_seed is None:
            raw_samples = raw_samples[: args.max_samples]
        else:
            rng = random.Random(args.sample_seed)
            raw_samples = rng.sample(raw_samples, args.max_samples)

    schemas = load_schemas_for_run(args, raw_samples)
    if not schemas:
        print(
            "ERROR: No relation schemas available. Provide --schema or use data "
            "with gold relation labels from which schemas can be inferred."
        )
        sys.exit(1)

    schema_by_type = schemas_by_relation_type(schemas)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_tag = re.sub(r"[^A-Za-z0-9_-]", "_", Path(args.data).stem)
    model_tag = re.sub(r"[^A-Za-z0-9_-]", "_", args.model)
    output_path = args.output or str(OUTPUT_DIR / f"{data_tag}_{model_tag}_re_predictions.json")

    retriever = RelationRetrievalAgent()
    planner = RelationPlanningAgent()
    coder = RelationCodingAgent()
    verifier = RelationVerificationAgent()

    predictions: List[Dict[str, Any]] = []
    if args.resume and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as fh:
            predictions = json.load(fh)
        print(f"Resuming from {len(predictions)} existing predictions.")

    skip = len(predictions)
    success_count = 0
    attempted_count = 0

    for idx, sample in enumerate(tqdm(raw_samples, desc="AEC-RE"), start=0):
        if idx < skip:
            continue

        text = sample["text"]
        sample_predictions: List[str] = []
        if args.mode == "given_pairs" or (args.mode == "auto" and sample["candidate_pairs"]):
            for pair in sample["candidate_pairs"]:
                attempted_count += 1
                pred_code, success = run_relation_classification_pipeline(
                    text=text,
                    schemas=schemas,
                    candidate_pair=pair,
                    model=args.model,
                    t=args.t,
                    planner=planner,
                    coder=coder,
                    verifier=verifier,
                )
                if success:
                    success_count += 1
                    sample_predictions.append(pred_code)
                if args.delay > 0:
                    time.sleep(args.delay)
        else:
            for schema in schemas:
                attempted_count += 1
                pred_code, success = run_relation_pipeline(
                    text=text,
                    schema=schema,
                    model=args.model,
                    k=args.k,
                    t=args.t,
                    candidate_pairs=None,
                    retriever=retriever,
                    planner=planner,
                    coder=coder,
                    verifier=verifier,
                )
                if success:
                    success_count += 1
                    sample_predictions.append(pred_code)
                if args.delay > 0:
                    time.sleep(args.delay)

        combined_prediction = merge_prediction_strings(sample_predictions, schema_by_type)
        gold_label = build_gold_label(sample["relation_mentions"], schema_by_type)
        predictions.append(
            {
                "Input": text,
                "Prediction": combined_prediction,
                "Label": gold_label,
                "task_type": "RE",
                "doc_id": sample["doc_id"],
                "candidate_pairs": sample["candidate_pairs"],
                "has_relation_annotation": sample["has_relation_annotation"],
            }
        )

        if len(predictions) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(predictions, fh, indent=2, ensure_ascii=False)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2, ensure_ascii=False)

    print(f"Done. {success_count} verified relation predictions from {attempted_count} schema attempts.")
    print(f"Predictions saved -> {output_path}")

    has_gold = any(record.get("has_relation_annotation") for record in predictions)
    if not args.no_eval and has_gold:
        scores = evaluate_predictions(predictions, schemas)
        scores_path = output_path.replace(".json", "_scores.json")
        with open(scores_path, "w", encoding="utf-8") as fh:
            json.dump({k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in scores.items()}, fh, indent=2)
        print(f"Scores saved -> {scores_path}")
    elif not has_gold:
        print("No gold relation annotations detected; skipping evaluation.")


if __name__ == "__main__":
    main()
