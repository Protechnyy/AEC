"""
Four-agent relation extraction framework.

The agents preserve the retrieval -> planning -> coding -> verification loop
and make the relation tuple the unit of work:

    RelationClass(arg1="entity span", arg2="entity span", evidence=[...])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from .llm_utils import call_llm


@dataclass
class RelationHypothesis:
    """A ranked candidate relation tuple proposed by the planning agent."""

    arg1: str
    arg2: str
    relation_type: str
    confidence: float
    rationale: str = ""
    evidence: List[str] = field(default_factory=list)


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
            f"Given this executable relation definition:\n\n{schema_definition}\n\n"
            f"Generate {k} fluent English sentences that clearly express this relation. "
            "Each sentence must contain both argument spans and a natural lexical cue. "
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
    """Propose candidate argument pairs for one target relation type."""

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
                f"- arg1={arg1!r}, arg2={arg2!r}" for arg1, arg2 in candidate_pairs
            )
            pair_block = (
                "\nCandidate argument pairs are provided. Use only these exact spans, "
                "and return [] if none of the pairs expresses the target relation:\n"
                f"{rendered}\n"
            )

        exemplar_block = ""
        if exemplars:
            exemplar_block = f"\nExample sentences for context:\n{exemplars}\n"

        system = (
            "You are a planning agent for relation extraction. Given text and one "
            "target relation schema, produce a JSON array of candidate relation "
            "hypotheses. Each object must contain keys: arg1, arg2, relation_type, "
            "confidence, rationale, evidence. arg1 and arg2 must be exact spans "
            "copied from the text. evidence is a list of exact text spans that "
            "support the relation. Output [] when the relation is not expressed."
        )
        user = (
            f"Relation definition:\n{schema_definition}\n"
            f"{exemplar_block}"
            f"{pair_block}\n"
            f"Text:\n{text}\n\n"
            f"Target relation type: {relation_type}\n"
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

        data.sort(key=lambda x: float(x.get("confidence", 0)) if isinstance(x, dict) else 0, reverse=True)
        for item in data[:k]:
            if not isinstance(item, dict):
                continue
            arg1 = item.get("arg1") or item.get("subject") or item.get("head") or ""
            arg2 = item.get("arg2") or item.get("object") or item.get("tail") or ""
            if not isinstance(arg1, str) or not isinstance(arg2, str):
                continue
            evidence = item.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            elif not isinstance(evidence, list):
                evidence = []
            hypotheses.append(
                RelationHypothesis(
                    arg1=arg1,
                    arg2=arg2,
                    relation_type=str(item.get("relation_type") or relation_type),
                    confidence=float(item.get("confidence", 0.5)),
                    rationale=str(item.get("rationale", "")),
                    evidence=[str(e) for e in evidence if e],
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
        """Choose one relation type or Other for a given argument pair."""

        arg1, arg2 = candidate_pair
        system = (
            "You are a planning agent for relation classification. Given text, "
            "relation schemas, and one candidate argument pair, choose exactly "
            "one relation_type from the schemas, or output Other if no listed "
            "relation is expressed. The argument spans must be interpreted in "
            "the given order."
        )
        user = (
            f"Relation schemas:\n{schema_definitions}\n\n"
            f"Text:\n{text}\n\n"
            f"Candidate pair:\narg1 = {arg1!r}\narg2 = {arg2!r}\n\n"
            "Return only a JSON object with keys: relation_type, confidence, "
            "rationale, evidence. relation_type must be one schema relation_type "
            "or exactly Other. evidence must be a list of exact text spans."
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
        relation_type = str(data.get("relation_type", "Other"))
        evidence = data.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        elif not isinstance(evidence, list):
            evidence = []
        return RelationHypothesis(
            arg1=arg1,
            arg2=arg2,
            relation_type=relation_type,
            confidence=float(data.get("confidence", 0.5)),
            rationale=str(data.get("rationale", "")),
            evidence=[str(e) for e in evidence if e],
        )


@dataclass
class RelationCodingAgent:
    """Generate executable relation class instantiations."""

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
                "# Exemplar sentences:\n"
                + "\n".join(f"# {line}" for line in exemplars.splitlines() if line.strip())
                + "\n\n"
            )

        system = (
            "You are a coding agent for relation extraction. Complete the Python "
            "result with a list of relation class instances.\n"
            "CRITICAL RULES:\n"
            "- Use the class defined in the schema, not a generic class.\n"
            "- arg1 and arg2 MUST be exact minimal substrings copied from the text.\n"
            "- evidence must be a list of exact text spans supporting the relation; "
            "use [] if no short evidence phrase is appropriate.\n"
            "- If the hinted pair does not express the relation, output [].\n"
            "Output ONLY the Python list expression, with no markdown or explanation."
        )
        if patch_feedback:
            system += f"\n\nThe previous attempt failed. Fix this diagnostic:\n{patch_feedback}"

        evidence_hint = ", ".join(repr(e) for e in hypothesis.evidence) or "[]"
        user = (
            f"{exemplar_block}"
            f"# Relation schema\n{schema_definition}\n\n"
            f"# Text to analyze\ntext = {text!r}\n\n"
            f"# Hinted relation hypothesis\n"
            f"# relation_type = {hypothesis.relation_type!r}\n"
            f"# arg1 = {hypothesis.arg1!r}\n"
            f"# arg2 = {hypothesis.arg2!r}\n"
            f"# evidence = {evidence_hint}\n\n"
            "# Complete this assignment:\nresult = "
        )
        raw = call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,
        )
        raw = re.sub(r"^```(?:python)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = raw.strip()

        bracket_depth = 0
        cut = len(raw)
        for i, ch in enumerate(raw):
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    cut = i + 1
                    break
        if cut < len(raw):
            raw = raw[:cut]
        return raw.strip()


class RelationVerificationError(Exception):
    """Raised when relation code fails deterministic verification."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class RelationVerificationAgent:
    """Execute and validate generated relation code."""

    check_args_in_text: bool = True
    check_evidence_in_text: bool = True
    check_relation_class: bool = True
    check_candidate_pairs: bool = True

    def verify_code(
        self,
        code_string: str,
        text: str,
        class_namespace: dict[str, Any],
        *,
        expected_class: Optional[str] = None,
        allowed_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        errors: List[str] = []
        try:
            result = eval(code_string, {"__builtins__": {}}, class_namespace)  # noqa: S307
        except SyntaxError as exc:
            raise RelationVerificationError([f"[T3-SyntaxError] {exc}"]) from exc
        except TypeError as exc:
            raise RelationVerificationError([f"[T2/T3-TypeError] {exc}"]) from exc
        except NameError as exc:
            raise RelationVerificationError([f"[T3-NameError] {exc}"]) from exc
        except Exception as exc:
            raise RelationVerificationError([f"[T3-Error] {type(exc).__name__}: {exc}"]) from exc

        if not isinstance(result, (list, tuple)):
            result = [result]
        instances = [item for item in result if not isinstance(item, type) and hasattr(item, "__dict__")]
        if not instances:
            raise RelationVerificationError(["[T3-StructuralError] eval() produced no relation instances."])

        allowed = set(allowed_pairs or [])
        for inst in instances:
            class_name = type(inst).__name__
            if self.check_relation_class and expected_class and class_name != expected_class:
                errors.append(
                    f"[T3-ClassMismatch] Expected class {expected_class}, got {class_name}."
                )

            arg1 = getattr(inst, "arg1", None)
            arg2 = getattr(inst, "arg2", None)
            if not isinstance(arg1, str) or not arg1:
                errors.append(f"[T2-MissingArg1] {class_name}.arg1 must be a non-empty string.")
            if not isinstance(arg2, str) or not arg2:
                errors.append(f"[T2-MissingArg2] {class_name}.arg2 must be a non-empty string.")

            if self.check_args_in_text:
                for attr, span in (("arg1", arg1), ("arg2", arg2)):
                    if isinstance(span, str) and span and span not in text:
                        errors.append(
                            f"[T1-SpanGrounding] {class_name}.{attr} span {span!r} "
                            "does not appear in the input text."
                        )

            if self.check_candidate_pairs and allowed_pairs and isinstance(arg1, str) and isinstance(arg2, str):
                symmetric_allowed = bool(getattr(inst, "symmetric", False)) and (arg2, arg1) in allowed
                if (arg1, arg2) not in allowed and not symmetric_allowed:
                    errors.append(
                        f"[T1-PairConstraint] Pair ({arg1!r}, {arg2!r}) is not one "
                        "of the provided candidate pairs."
                    )

            evidence = getattr(inst, "evidence", [])
            if evidence is None:
                evidence = []
            if not isinstance(evidence, list):
                errors.append(f"[T2-EvidenceType] {class_name}.evidence must be a list.")
            elif self.check_evidence_in_text:
                for span in evidence:
                    if isinstance(span, str) and span and span not in text:
                        errors.append(
                            f"[T1-EvidenceGrounding] Evidence span {span!r} "
                            "does not appear in the input text."
                        )

        if errors:
            raise RelationVerificationError(errors)
