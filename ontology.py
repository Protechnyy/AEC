"""Relation ontology loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

from event_schema import RelationSchema, python_identifier


def normalize_relation_label(value: str) -> str:
    return python_identifier(value)


def load_relation_schemas(schema_dir: Path, dataset_name: str) -> dict[str, RelationSchema]:
    candidates = [
        schema_dir / f"{dataset_name}.json",
        schema_dir / f"{dataset_name.lower()}.json",
        schema_dir / "relations.json",
    ]
    schema_path = next((path for path in candidates if path.exists()), None)
    if schema_path is None:
        raise FileNotFoundError(
            f"No relation schema found for {dataset_name!r}. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )

    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("relations") or data.get("schemas") or []
    if not isinstance(data, list):
        raise ValueError(f"Relation schema file {schema_path} must contain a list or a dict with 'relations'.")
    schemas = [RelationSchema.from_dict(item) for item in data if isinstance(item, dict)]
    return {schema.relation_type: schema for schema in schemas}


def build_relation_definitions(schemas: dict[str, RelationSchema]) -> str:
    blocks = [
        "from typing import Literal",
        "from pydantic import BaseModel, Field",
        "",
    ]
    for schema in schemas.values():
        blocks.extend([schema.to_code_definition(), ""])
    return "\n".join(blocks)
