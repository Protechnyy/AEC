"""
Four-agent relation extraction framework.

The agents preserve the retrieval -> planning -> triple generation ->
verification loop, but the task output is standard relation extraction:

    [{"subject": "Alice", "object": "Acme Corp", "relation": "per:employee_of"}]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .llm_utils import call_llm


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _relation_labels(schemas_or_labels: Sequence[Any] | Mapping[str, Any]) -> set[str]:
    if isinstance(schemas_or_labels, Mapping):
        return {str(key) for key in schemas_or_labels}
    labels: set[str] = set()
    for item in schemas_or_labels:
        label = getattr(item, "relation_type", item)
        labels.add(str(label))
    return labels


def _symmetric_labels(schemas_or_labels: Sequence[Any] | Mapping[str, Any]) -> set[str]:
    if isinstance(schemas_or_labels, Mapping):
        values = schemas_or_labels.values()
    else:
        values = schemas_or_labels
    return {str(getattr(item, "relation_type", "")) for item in values if getattr(item, "symmetric", False)}


def parse_triple_json(code_string: str) -> List[dict]:
    """Parse a strict JSON list of relation triples."""

    payload = _extract_json_array(code_string)
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list")
    return data


@dataclass
class RelationHypothesis:
    """A ranked candidate relation triple proposed by the planning agent."""

    subject: str
    object: str
    relation: str
    confidence: float
    rationale: str = ""


@dataclass
class RelationRetrievalAgent:
    """Generate synthetic exemplars for a relation schema."""

    def retrieve(
        self,
        schema_definition: str,
        k: int = 3,
        model: str = "gpt-4o",
    ) -> str:
        system = "You generate concise examples for relation extraction."
        user = (
            f"Given this relation schema:\n\n{schema_definition}\n\n"
            f"Generate {k} fluent English sentences that clearly express this relation. "
            "Each sentence must contain both subject and object spans and a natural lexical cue. "
            "Output exactly one sentence per line, no numbering."
        )
        try:
            return call_llm(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model=model,
            )
        except Exception:
            return ""


class RelationPlanningAgent:
    """Propose candidate subject/object pairs for one target relation type."""

    def generate_hypotheses(
        self,
        text: str,
        schema_definition: str,
        relation_type: str,
        exemplars: str = "",
        candidate_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        k: int = 3,
        model: str = "gpt-4o",
    ) -> List[RelationHypothesis]:
        pair_block = ""
        if candidate_pairs:
            rendered = "\n".join(
                f"- subject={subject!r}, object={obj!r}" for subject, obj in candidate_pairs
            )
            pair_block = (
                "\nCandidate subject/object pairs are provided. Use only these exact spans, "
                "and return [] if none of the pairs expresses the target relation:\n"
                f"{rendered}\n"
            )

        exemplar_block = ""
        if exemplars:
            exemplar_block = f"\nExample sentences for context:\n{exemplars}\n"

        system = (
            "You are a planning agent for relation extraction. Given text and one "
            "target relation schema, produce a JSON array of candidate relation "
            "hypotheses. Each object must contain keys: subject, object, relation, "
            "confidence, rationale. subject and object must be exact spans copied "
            "from the text. relation must be the target relation label. Output [] "
            "when the relation is not expressed."
        )
        user = (
            f"Relation definition:\n{schema_definition}\n"
            f"{exemplar_block}"
            f"{pair_block}\n"
            f"Text:\n{text}\n\n"
            f"Target relation: {relation_type}\n"
            f"Return up to {k} hypotheses as a JSON array only."
        )
        try:
            raw = call_llm(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model=model,
            )
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            data = json.loads(match.group(0)) if match else []
        except Exception:
            data = []

        hypotheses: List[RelationHypothesis] = []
        if not isinstance(data, list):
            return hypotheses

        data.sort(key=lambda x: _safe_float(x.get("confidence", 0)) if isinstance(x, dict) else 0, reverse=True)
        for item in data[:k]:
            if not isinstance(item, dict):
                continue
            subject = item.get("subject") or item.get("arg1") or item.get("head") or ""
            obj = item.get("object") or item.get("arg2") or item.get("tail") or ""
            relation = item.get("relation") or item.get("relation_type") or relation_type
            if not isinstance(subject, str) or not isinstance(obj, str):
                continue
            hypotheses.append(
                RelationHypothesis(
                    subject=subject,
                    object=obj,
                    relation=str(relation),
                    confidence=_safe_float(item.get("confidence", 0.5)),
                    rationale=str(item.get("rationale", "")),
                )
            )
        return hypotheses

    def classify_pair(
        self,
        text: str,
        schema_definitions: str,
        candidate_pair: Tuple[str, str],
        model: str = "gpt-4o",
    ) -> Optional[RelationHypothesis]:
        """Choose one relation label or Other for a given subject/object pair."""

        entity_1, entity_2 = candidate_pair
        system = (
            "You are a planning agent for relation classification. Given text, "
            "relation schemas, and one candidate entity pair, choose exactly one "
            "relation label from the schemas, or output Other if no listed relation "
            "is expressed. For directed relations, also choose the correct subject "
            "and object orientation. The subject and object must be exactly the two "
            "provided entity mentions."
        )
        user = (
            f"Relation schemas:\n{schema_definitions}\n\n"
            f"Text:\n{text}\n\n"
            f"Candidate entity pair:\nentity_1 = {entity_1!r}\nentity_2 = {entity_2!r}\n\n"
            "Return only a JSON object with keys: subject, object, relation, "
            "confidence, rationale. If relation is not Other, subject and object "
            "must be entity_1/entity_2 in the semantically correct order. relation "
            "must be one schema relation label or exactly Other."
        )
        try:
            raw = call_llm(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model=model,
            )
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
        except Exception:
            return None

        if not isinstance(data, dict):
            return None
        relation = str(data.get("relation") or data.get("relation_type") or "Other")
        subject = data.get("subject") or data.get("arg1") or data.get("head") or entity_1
        obj = data.get("object") or data.get("arg2") or data.get("tail") or entity_2
        if subject not in candidate_pair or obj not in candidate_pair or subject == obj:
            subject, obj = entity_1, entity_2
        return RelationHypothesis(
            subject=str(subject),
            object=str(obj),
            relation=relation,
            confidence=_safe_float(data.get("confidence", 0.5)),
            rationale=str(data.get("rationale", "")),
        )


@dataclass
class RelationCodingAgent:
    """Generate JSON relation triples."""

    def generate_code(
        self,
        hypothesis: RelationHypothesis,
        schema_definition: str,
        text: str,
        exemplars: str = "",
        patch_feedback: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> str:
        exemplar_block = ""
        if exemplars:
            exemplar_block = (
                "Exemplar sentences:\n"
                + "\n".join(line for line in exemplars.splitlines() if line.strip())
                + "\n\n"
            )

        system = (
            "You are a relation extraction agent. Return a JSON array of triples.\n"
            "CRITICAL RULES:\n"
            "- Each triple must be an object with exactly these semantic fields: "
            "subject, object, relation.\n"
            "- subject and object MUST be exact minimal substrings copied from the text.\n"
            "- relation MUST be the relation label from the schema.\n"
            "- If the hinted pair does not express the relation, output [].\n"
            "- Do not output Python classes, arg1/arg2 fields, evidence, markdown, or explanation."
        )
        if patch_feedback:
            system += f"\n\nThe previous attempt failed. Fix this diagnostic:\n{patch_feedback}"

        user = (
            f"{exemplar_block}"
            f"Relation schema:\n{schema_definition}\n\n"
            f"Text to analyze:\n{text}\n\n"
            f"Hinted relation hypothesis:\n"
            f"relation = {hypothesis.relation!r}\n"
            f"subject = {hypothesis.subject!r}\n"
            f"object = {hypothesis.object!r}\n\n"
            'Return only JSON like [{"subject": "...", "object": "...", "relation": "..."}].'
        )
        raw = call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,
        )
        return _extract_json_array(raw)


class RelationVerificationError(Exception):
    """Raised when generated relation triples fail deterministic verification."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class RelationVerificationAgent:
    """Parse and validate generated relation triples."""

    check_args_in_text: bool = True
    check_relation_label: bool = True
    check_candidate_pairs: bool = True

    def verify_code(
        self,
        code_string: str,
        text: str,
        relation_schemas: Sequence[Any] | Mapping[str, Any],
        *,
        expected_relation: Optional[str] = None,
        allowed_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        errors: List[str] = []
        try:
            result = parse_triple_json(code_string)
        except json.JSONDecodeError as exc:
            raise RelationVerificationError([f"[T3-JSONError] {exc}"]) from exc
        except Exception as exc:
            raise RelationVerificationError([f"[T3-ParseError] {type(exc).__name__}: {exc}"]) from exc

        if not result:
            raise RelationVerificationError(["[T3-StructuralError] JSON list contains no relation triples."])

        valid_relations = _relation_labels(relation_schemas)
        symmetric_relations = _symmetric_labels(relation_schemas)
        allowed = set(allowed_pairs or [])

        for idx, triple in enumerate(result):
            if not isinstance(triple, dict):
                errors.append(f"[T3-StructuralError] Triple #{idx} must be a JSON object.")
                continue

            extra_fields = sorted(set(triple) - {"subject", "object", "relation"})
            if extra_fields:
                errors.append(
                    f"[T2-ExtraFields] Triple #{idx} has unsupported fields: {', '.join(extra_fields)}."
                )

            subject = triple.get("subject")
            obj = triple.get("object")
            relation = triple.get("relation")

            if not isinstance(subject, str) or not subject:
                errors.append(f"[T2-MissingSubject] Triple #{idx}.subject must be a non-empty string.")
            if not isinstance(obj, str) or not obj:
                errors.append(f"[T2-MissingObject] Triple #{idx}.object must be a non-empty string.")
            if not isinstance(relation, str) or not relation:
                errors.append(f"[T2-MissingRelation] Triple #{idx}.relation must be a non-empty string.")

            if isinstance(relation, str):
                if self.check_relation_label and relation not in valid_relations:
                    errors.append(
                        f"[T3-RelationLabel] Triple #{idx}.relation {relation!r} is not in the schema."
                    )
                if expected_relation and relation != expected_relation:
                    errors.append(
                        f"[T3-RelationMismatch] Expected relation {expected_relation!r}, got {relation!r}."
                    )

            if self.check_args_in_text:
                for attr, span in (("subject", subject), ("object", obj)):
                    if isinstance(span, str) and span and span not in text:
                        errors.append(
                            f"[T1-SpanGrounding] Triple #{idx}.{attr} span {span!r} "
                            "does not appear in the input text."
                        )

            if self.check_candidate_pairs and allowed_pairs and isinstance(subject, str) and isinstance(obj, str):
                symmetric_allowed = (
                    isinstance(relation, str)
                    and relation in symmetric_relations
                    and (obj, subject) in allowed
                )
                if (subject, obj) not in allowed and not symmetric_allowed:
                    errors.append(
                        f"[T1-PairConstraint] Pair ({subject!r}, {obj!r}) is not one "
                        "of the provided candidate subject/object pairs."
                    )

        if errors:
            raise RelationVerificationError(errors)
