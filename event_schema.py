"""Relation schema definitions for zero-shot relation extraction.

This repository was adapted from Agent-Event-Coder.  The original schema file
is intentionally reused, but it now defines relation extraction primitives only:
a relation ontology entry and a grounded relation triple.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

try:
    from pydantic import BaseModel, Field, create_model
except ImportError:  # pragma: no cover - exercised only before dependencies are installed.
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    create_model = None  # type: ignore[assignment]


def python_identifier(value: str) -> str:
    """Convert a relation label into a safe Python class name."""

    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not cleaned:
        cleaned = "Relation"
    if cleaned[0].isdigit():
        cleaned = f"R_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_relation"
    return cleaned


@dataclass(frozen=True)
class RelationSchema:
    """A single relation type in a zero-shot relation ontology."""

    relation_type: str
    definition: str = ""
    head_type: str = "ENTITY"
    tail_type: str = "ENTITY"
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RelationSchema":
        relation_type = raw.get("relation_type") or raw.get("name") or raw.get("label")
        if not isinstance(relation_type, str) or not relation_type.strip():
            raise ValueError("Relation schema must define a non-empty relation_type/name/label")
        definition = raw.get("definition") or raw.get("description") or ""
        aliases = raw.get("aliases", ())
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []
        return cls(
            relation_type=relation_type.strip(),
            definition=str(definition).strip(),
            head_type=str(raw.get("head_type") or raw.get("subject_type") or "ENTITY").strip(),
            tail_type=str(raw.get("tail_type") or raw.get("object_type") or "ENTITY").strip(),
            aliases=tuple(str(alias).strip() for alias in aliases if str(alias).strip()),
        )

    @property
    def class_name(self) -> str:
        return python_identifier(self.relation_type)

    def generate_pydantic_model(self) -> type[BaseModel]:
        """Generate a Pydantic model that encodes this relation schema."""

        if create_model is None or Field is None:
            raise RuntimeError("pydantic is required for schema-as-code verification. Run `pip install -r requirements.txt`.")
        return create_model(
            self.class_name,
            __base__=BaseModel,
            __module__=__name__,
            relation_type=(Literal[self.relation_type], Field(default=self.relation_type)),  # type: ignore[valid-type]
            head=(str, Field(..., min_length=1, description=f"Head entity span; expected type {self.head_type}.")),
            tail=(str, Field(..., min_length=1, description=f"Tail entity span; expected type {self.tail_type}.")),
            evidence=(str, Field(default="", description="Optional evidence span copied from the source text.")),
        )

    def to_code_definition(self) -> str:
        """Render this schema as Python/Pydantic code for LLM prompts."""

        alias_text = f" aliases={list(self.aliases)!r}" if self.aliases else ""
        return "\n".join(
            [
                f"class {self.class_name}(BaseModel):",
                f"    \"\"\"{self.definition or self.relation_type}{alias_text}\"\"\"",
                f"    relation_type: Literal[{self.relation_type!r}] = {self.relation_type!r}",
                f"    head: str = Field(..., description='Head entity span; expected type {self.head_type}.')",
                f"    tail: str = Field(..., description='Tail entity span; expected type {self.tail_type}.')",
                "    evidence: str = ''",
            ]
        )


@dataclass
class RelationObject:
    """Grounded relation prediction: head, relation type, tail."""

    relation_type: str
    head: str
    tail: str
    evidence: str = ""
    confidence: float = 0.0
    rationale: str = ""

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.relation_type, self.head, self.tail)

    def dict(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "head": self.head,
            "tail": self.tail,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }
