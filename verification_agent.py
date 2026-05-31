"""Verification agent for zero-shot relation extraction."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from event_schema import BaseModel, RelationObject, RelationSchema


def _norm(value: str) -> str:
    return " ".join(value.split()).strip()


@dataclass
class VerificationError(Exception):
    message: str
    category: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "category": self.category, "details": self.details}


class VerificationAgent:
    """Execute generated relation code and verify Pydantic/schema constraints."""

    def verify_code(
        self,
        code: str,
        schemas: dict[str, RelationSchema],
        text: str,
        *,
        confidence: float = 0.0,
        rationale: str = "",
    ) -> RelationObject:
        self._check_code_shape(code)
        model_classes = {schema.class_name: schema.generate_pydantic_model() for schema in schemas.values()}
        namespace: dict[str, Any] = dict(model_classes)
        try:
            exec(compile(code, "<relation-coder>", "exec"), {"__builtins__": {}}, namespace)
        except Exception as exc:
            raise VerificationError(
                f"Generated relation code failed to execute: {exc}",
                "code_execution_error",
                {"error": f"{exc.__class__.__name__}: {exc}", "code": code},
            ) from exc

        result = namespace.get("result")
        if result is None:
            raise VerificationError(
                "Generated code must assign exactly one relation object to variable `result`.",
                "missing_result",
                {"code": code},
            )
        if not isinstance(result, BaseModel):
            raise VerificationError(
                "`result` must be an instance of one of the generated Pydantic relation classes.",
                "result_not_pydantic_model",
                {"actual_type": type(result).__name__},
            )

        payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        relation = RelationObject(
            relation_type=str(payload.get("relation_type", "")),
            head=str(payload.get("head", "")),
            tail=str(payload.get("tail", "")),
            evidence=str(payload.get("evidence", "")),
            confidence=confidence,
            rationale=rationale,
        )
        self.verify(relation, schemas, text)
        return relation

    def _check_code_shape(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise VerificationError(
                f"Generated code has invalid Python syntax: {exc}",
                "syntax_error",
                {"error": str(exc), "code": code},
            ) from exc
        allowed_nodes = (
            ast.Module,
            ast.Assign,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Call,
            ast.keyword,
            ast.Constant,
            ast.Expr,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise VerificationError(
                    f"Generated code contains unsupported syntax: {node.__class__.__name__}.",
                    "unsafe_code_shape",
                    {"node": node.__class__.__name__, "code": code},
                )
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
        if len(assignments) != 1:
            raise VerificationError(
                "Generated code must contain exactly one assignment to `result`.",
                "unsafe_code_shape",
                {"assignment_count": len(assignments), "code": code},
            )
        target = assignments[0].targets[0] if assignments[0].targets else None
        if not isinstance(target, ast.Name) or target.id != "result":
            raise VerificationError(
                "Generated code must assign the relation object to `result`.",
                "missing_result",
                {"code": code},
            )

    def verify(self, relation: RelationObject, schemas: dict[str, RelationSchema], text: str) -> bool:
        if relation.relation_type not in schemas:
            raise VerificationError(
                f"Unknown relation type {relation.relation_type!r}.",
                "relation_type_not_in_schema",
                {"relation_type": relation.relation_type},
            )

        normalized_text = _norm(text).lower()
        for field_name in ("head", "tail"):
            value = _norm(getattr(relation, field_name))
            if not value or value.lower() not in normalized_text:
                raise VerificationError(
                    f"{field_name} span {value!r} is not grounded in the source text.",
                    "entity_not_in_text",
                    {"field": field_name, "value": value},
                )

        if relation.head.lower() == relation.tail.lower():
            raise VerificationError(
                "Head and tail entity spans must be distinct.",
                "self_relation",
                {"head": relation.head, "tail": relation.tail},
            )

        if relation.evidence and relation.evidence.lower() not in normalized_text:
            raise VerificationError(
                f"Evidence span {relation.evidence!r} is not grounded in the source text.",
                "evidence_not_in_text",
                {"evidence": relation.evidence},
            )

        return True
