"""Schema-aware retrieval helpers for zero-shot relation extraction."""

from __future__ import annotations

from dataclasses import dataclass

from event_schema import RelationSchema


@dataclass
class RetrievalAgent:
    """Return compact relation-definition reminders for the planner."""

    example_db: dict[str, list[str]] | None = None

    def retrieve(self, schema: RelationSchema, k: int = 3) -> list[str]:
        if self.example_db and schema.relation_type in self.example_db:
            return self.example_db[schema.relation_type][:k]
        examples = [
            f"Relation '{schema.relation_type}': {schema.definition or 'no definition provided'}",
            f"Head type: {schema.head_type}; tail type: {schema.tail_type}.",
        ]
        if schema.aliases:
            examples.append("Aliases: " + ", ".join(schema.aliases))
        return examples[:k]
