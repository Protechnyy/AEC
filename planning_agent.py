"""Planning agent for zero-shot relation extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from event_schema import RelationSchema
from llm_utils import _load_json_reply, call_llm


def _norm(value: str) -> str:
    return " ".join(value.split()).strip()


def _label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


@dataclass
class RelationHypothesis:
    """A candidate relation triple before verification."""

    head: str
    tail: str
    relation_type: str
    confidence: float = 0.0
    rationale: str = ""


class PlanningAgent:
    """Propose relation triples or classify supplied entity pairs."""

    entity_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9._/-]*(?:\s+[A-Z][A-Za-z0-9._/-]*)*|"
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+|https?://\S+|\d+(?:\.\d+)?%?)\b"
    )

    def __init__(self) -> None:
        self.last_planner_debug: dict[str, Any] = {}

    def generate_hypotheses(
        self,
        text: str,
        schemas: dict[str, RelationSchema],
        *,
        candidate_pairs: list[dict[str, str]] | None = None,
        max_candidates: int = 8,
    ) -> list[RelationHypothesis]:
        if candidate_pairs:
            return self._heuristic_for_candidate_pairs(text, schemas, candidate_pairs, max_candidates)
        return self._heuristic_open_extraction(text, schemas, max_candidates)

    def generate_hypotheses_with_llm(
        self,
        text: str,
        relation_definitions: str,
        *,
        schemas: dict[str, RelationSchema],
        candidate_pairs: list[dict[str, str]] | None = None,
        max_candidates: int = 8,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> list[RelationHypothesis]:
        allowed = ", ".join(schemas)
        pair_block = json.dumps(candidate_pairs or [], ensure_ascii=False, indent=2)
        task = (
            "Classify only the supplied entity pairs. For each pair, choose one allowed relation type "
            "when the sentence explicitly supports it; otherwise omit the pair."
            if candidate_pairs
            else "Extract all explicit relation triples from the text."
        )
        messages = [
            {
                "role": "system",
                "content": "You are the Planning Agent in a zero-shot relation extraction system. Return strict JSON only.",
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    "Rules:\n"
                    "1. Use only relation types from the ontology.\n"
                    "2. Copy head and tail spans verbatim from the text.\n"
                    "3. Prefer explicit, text-supported relations over world knowledge.\n"
                    "4. Do not output no_relation/Other; omit unsupported pairs.\n"
                    "5. For asymmetric relations, preserve head/tail direction.\n\n"
                    "Return a JSON array. Each item must contain head, tail, relation_type, confidence, and rationale.\n\n"
                    f"Allowed relation types: {allowed}\n\n"
                    f"Relation definitions:\n{relation_definitions}\n\n"
                    f"Candidate pairs:\n{pair_block if candidate_pairs else 'None'}\n\n"
                    f"Text:\n{text}"
                ),
            },
        ]
        reply = call_llm(
            messages,
            model=model,
            base_url=base_url,
            api_key=api_key,
            request_tag="re_planning",
            max_tokens=768,
        )
        data = _load_json_reply(reply)
        if not isinstance(data, list):
            return []

        hypotheses: list[RelationHypothesis] = []
        for item in data[: max_candidates * 2]:
            if not isinstance(item, dict):
                continue
            head = item.get("head")
            tail = item.get("tail")
            relation_type = item.get("relation_type") or item.get("relation")
            if not all(isinstance(x, str) and x.strip() for x in (head, tail, relation_type)):
                continue
            matched = self.match_schema_name(str(relation_type), schemas)
            if matched is None:
                continue
            confidence = item.get("confidence", 0.0)
            if isinstance(confidence, int):
                confidence = float(confidence)
            if not isinstance(confidence, float):
                confidence = 0.0
            hypotheses.append(
                RelationHypothesis(
                    head=_norm(str(head)),
                    tail=_norm(str(tail)),
                    relation_type=matched,
                    confidence=max(0.0, min(1.0, confidence)),
                    rationale=str(item.get("rationale") or ""),
                )
            )
        return self._dedupe_hypotheses(hypotheses, max_candidates)

    def match_schema_name(self, label: str, schemas: dict[str, RelationSchema]) -> str | None:
        normalized = _label_key(label)
        for name, schema in schemas.items():
            candidates = {_label_key(name), *(_label_key(alias) for alias in schema.aliases)}
            if normalized in candidates:
                return name
        return None

    def _heuristic_for_candidate_pairs(
        self,
        text: str,
        schemas: dict[str, RelationSchema],
        candidate_pairs: list[dict[str, str]],
        max_candidates: int,
    ) -> list[RelationHypothesis]:
        text_lower = text.lower()
        hyps: list[RelationHypothesis] = []
        for pair in candidate_pairs:
            head = _norm(str(pair.get("head", "")))
            tail = _norm(str(pair.get("tail", "")))
            if not head or not tail:
                continue
            best: RelationHypothesis | None = None
            for schema in schemas.values():
                score = self._schema_surface_score(text_lower, schema)
                if score <= 0:
                    continue
                candidate = RelationHypothesis(
                    head=head,
                    tail=tail,
                    relation_type=schema.relation_type,
                    confidence=score,
                    rationale="A relation name or alias appears in the sentence near the supplied entity pair.",
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
            if best is not None:
                hyps.append(best)
        return self._dedupe_hypotheses(sorted(hyps, key=lambda h: h.confidence, reverse=True), max_candidates)

    def _heuristic_open_extraction(
        self,
        text: str,
        schemas: dict[str, RelationSchema],
        max_candidates: int,
    ) -> list[RelationHypothesis]:
        entities = self._find_entity_like_spans(text)[:12]
        if len(entities) < 2:
            return []
        scored_schemas = sorted(
            ((self._schema_surface_score(text.lower(), schema), schema) for schema in schemas.values()),
            key=lambda item: item[0],
            reverse=True,
        )
        hyps: list[RelationHypothesis] = []
        for score, schema in scored_schemas:
            if score <= 0:
                continue
            for left_idx, head in enumerate(entities):
                for tail in entities[left_idx + 1 : left_idx + 4]:
                    if head.lower() == tail.lower():
                        continue
                    hyps.append(
                        RelationHypothesis(
                            head=head,
                            tail=tail,
                            relation_type=schema.relation_type,
                            confidence=score,
                            rationale="Nearby entity-like spans co-occur with a relation surface cue.",
                        )
                    )
                    if len(hyps) >= max_candidates:
                        return hyps
        return hyps

    def _find_entity_like_spans(self, text: str) -> list[str]:
        seen: set[str] = set()
        entities: list[str] = []
        for match in self.entity_pattern.finditer(text):
            span = _norm(match.group(0).strip(".,;:()[]{}\"'"))
            if len(span) < 2 or span.lower() in seen:
                continue
            seen.add(span.lower())
            entities.append(span)
        return entities

    def _schema_surface_score(self, text_lower: str, schema: RelationSchema) -> float:
        cues = [schema.relation_type.replace("_", " "), *schema.aliases]
        score = 0.0
        for cue in cues:
            normalized = cue.lower().replace("/", " ")
            if normalized and normalized in text_lower:
                score = max(score, 0.72 if normalized == schema.relation_type.lower().replace("_", " ") else 0.65)
        return score

    def _dedupe_hypotheses(
        self,
        hypotheses: list[RelationHypothesis],
        max_candidates: int,
    ) -> list[RelationHypothesis]:
        deduped: list[RelationHypothesis] = []
        seen: set[tuple[str, str, str]] = set()
        for hyp in hypotheses:
            key = (hyp.relation_type.lower(), hyp.head.lower(), hyp.tail.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hyp)
            if len(deduped) >= max_candidates:
                break
        return deduped
