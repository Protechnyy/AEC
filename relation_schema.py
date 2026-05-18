"""
Relation schema utilities.

Each relation label is mapped to a valid Python class that can be instantiated
by the coding agent and checked deterministically by the verification agent.
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


NO_RELATION_LABELS = {
    "",
    "no_relation",
    "no relation",
    "none",
    "na",
    "n/a",
    "other",
    "false",
}


@dataclass
class RelationSchema:
    """Executable schema metadata for one relation type."""

    relation_type: str
    class_name: str
    arg1_role: str = "arg1"
    arg2_role: str = "arg2"
    arg1_type: str = "Entity"
    arg2_type: str = "Entity"
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    symmetric: bool = False


class Relation:
    """Base class used by dynamically generated relation classes."""

    relation_type = "Relation"
    arg1_role = "arg1"
    arg2_role = "arg2"
    arg1_type = "Entity"
    arg2_type = "Entity"
    description = ""
    aliases: List[str] = []
    symmetric = False

    def __init__(
        self,
        arg1: Optional[str] = None,
        arg2: Optional[str] = None,
        evidence: Optional[List[str]] = None,
    ) -> None:
        self.arg1 = arg1 if arg1 is not None else ""
        self.arg2 = arg2 if arg2 is not None else ""
        self.evidence = evidence if evidence is not None else []

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(arg1={self.arg1!r}, "
            f"arg2={self.arg2!r}, evidence={self.evidence!r})"
        )


def is_no_relation(label: Optional[str]) -> bool:
    """Return True if *label* is a conventional negative relation label."""

    if label is None:
        return True
    return str(label).strip().lower() in NO_RELATION_LABELS


def relation_label_to_class_name(label: str) -> str:
    """Convert a relation label such as ``per:employee_of`` to a class name."""

    words = re.findall(r"[A-Za-z0-9]+", label)
    if not words:
        words = ["Relation"]
    name = "".join(w[:1].upper() + w[1:] for w in words)
    if not name or name[0].isdigit() or keyword.iskeyword(name):
        name = f"Relation{name}"
    return name


def _unique_class_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}{i}" in used:
        i += 1
    name = f"{base}{i}"
    used.add(name)
    return name


def _coerce_aliases(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def schema_from_mapping(
    item: Mapping[str, Any],
    *,
    used_class_names: Optional[set[str]] = None,
) -> RelationSchema:
    """Build a :class:`RelationSchema` from a JSON-compatible mapping."""

    relation_type = (
        item.get("relation_type")
        or item.get("type")
        or item.get("label")
        or item.get("name")
    )
    if not isinstance(relation_type, str) or not relation_type.strip():
        raise ValueError(f"Relation schema is missing a relation_type: {item!r}")

    base_class_name = str(item.get("class_name") or relation_label_to_class_name(relation_type))
    if used_class_names is not None:
        class_name = _unique_class_name(base_class_name, used_class_names)
    else:
        class_name = base_class_name

    description = item.get("description") or item.get("definition") or item.get("doc") or ""
    arg1_role = item.get("arg1_role") or item.get("subject_role") or item.get("head_role") or "arg1"
    arg2_role = item.get("arg2_role") or item.get("object_role") or item.get("tail_role") or "arg2"

    return RelationSchema(
        relation_type=relation_type,
        class_name=class_name,
        arg1_role=str(arg1_role),
        arg2_role=str(arg2_role),
        arg1_type=str(item.get("arg1_type") or item.get("subject_type") or item.get("head_type") or "Entity"),
        arg2_type=str(item.get("arg2_type") or item.get("object_type") or item.get("tail_type") or "Entity"),
        description=str(description),
        aliases=_coerce_aliases(item.get("aliases") or item.get("cues") or item.get("keywords")),
        symmetric=bool(item.get("symmetric", False)),
    )


def _load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_relation_schemas(path: str | Path) -> List[RelationSchema]:
    """Load relation schemas from JSON or JSONL.

    Supported shapes:

    * ``[{"relation_type": "per:employee_of", ...}, ...]``
    * ``{"relations": [...]}``
    * ``{"per:employee_of": {"description": "...", ...}, ...}``
    * ``["per:employee_of", "org:founded_by"]``
    """

    raw = _load_json_or_jsonl(Path(path))
    if isinstance(raw, Mapping) and "relations" in raw:
        raw_items = raw["relations"]
    elif isinstance(raw, Mapping):
        raw_items = []
        for label, value in raw.items():
            if isinstance(value, Mapping):
                raw_items.append({"relation_type": label, **dict(value)})
            else:
                raw_items.append({"relation_type": label, "description": str(value)})
    else:
        raw_items = raw

    if not isinstance(raw_items, list):
        raise ValueError(f"Unsupported relation schema file format: {path}")

    used: set[str] = set()
    schemas: List[RelationSchema] = []
    for item in raw_items:
        if isinstance(item, str):
            item = {"relation_type": item}
        if not isinstance(item, Mapping):
            raise ValueError(f"Unsupported relation schema entry: {item!r}")
        if is_no_relation(str(item.get("relation_type") or item.get("label") or item.get("type") or "")):
            continue
        schemas.append(schema_from_mapping(item, used_class_names=used))
    return schemas


def infer_relation_schemas(relation_types: Iterable[str]) -> List[RelationSchema]:
    """Infer minimal schemas from observed relation type labels."""

    used: set[str] = set()
    schemas: List[RelationSchema] = []
    for relation_type in sorted(set(str(r) for r in relation_types if not is_no_relation(str(r)))):
        schemas.append(schema_from_mapping({"relation_type": relation_type}, used_class_names=used))
    return schemas


def build_relation_definition(schema: RelationSchema, *, include_base: bool = True) -> str:
    """Return an executable-looking class definition for prompt context."""

    lines: List[str] = []
    if include_base:
        lines.extend(
            [
                "from typing import List, Optional",
                "",
                "class Relation:",
                "    arg1: str",
                "    arg2: str",
                "    evidence: List[str]",
                "",
            ]
        )

    lines.append(f"# Relation type: {schema.relation_type}")
    if schema.description:
        for desc_line in schema.description.splitlines():
            lines.append(f"# Description: {desc_line}")
    if schema.aliases:
        lines.append(f"# Common cues: {', '.join(schema.aliases)}")
    lines.extend(
        [
            f"class {schema.class_name}(Relation):",
            f"    relation_type = {schema.relation_type!r}",
            f"    arg1_role = {schema.arg1_role!r}",
            f"    arg2_role = {schema.arg2_role!r}",
            f"    arg1_type = {schema.arg1_type!r}",
            f"    arg2_type = {schema.arg2_type!r}",
            f"    symmetric = {schema.symmetric!r}",
            "    arg1: str",
            "    arg2: str",
            "    evidence: List[str]",
        ]
    )
    return "\n".join(lines)


def build_relation_definitions(schemas: Sequence[RelationSchema]) -> str:
    """Return prompt definitions for all relation schemas."""

    chunks: List[str] = []
    for i, schema in enumerate(schemas):
        chunks.append(build_relation_definition(schema, include_base=(i == 0)))
    return "\n\n".join(chunks)


def make_relation_namespace(schemas: Sequence[RelationSchema]) -> Dict[str, Any]:
    """Create a namespace containing executable relation classes for ``eval``."""

    namespace: Dict[str, Any] = {
        "Relation": Relation,
        "List": List,
        "Optional": Optional,
    }
    for schema in schemas:
        attrs = {
            "__module__": __name__,
            "relation_type": schema.relation_type,
            "arg1_role": schema.arg1_role,
            "arg2_role": schema.arg2_role,
            "arg1_type": schema.arg1_type,
            "arg2_type": schema.arg2_type,
            "description": schema.description,
            "aliases": list(schema.aliases),
            "symmetric": schema.symmetric,
        }
        namespace[schema.class_name] = type(schema.class_name, (Relation,), attrs)
    return namespace


def schemas_by_relation_type(schemas: Sequence[RelationSchema]) -> Dict[str, RelationSchema]:
    return {schema.relation_type: schema for schema in schemas}


def schemas_by_class_name(schemas: Sequence[RelationSchema]) -> Dict[str, RelationSchema]:
    return {schema.class_name: schema for schema in schemas}


def relation_instance_key(instance: Any, schema_by_class: Mapping[str, RelationSchema]) -> Tuple[str, str, str]:
    """Return a canonical ``(relation_type, arg1, arg2)`` key."""

    class_name = type(instance).__name__
    schema = schema_by_class.get(class_name)
    relation_type = getattr(instance, "relation_type", class_name)
    arg1 = getattr(instance, "arg1", "")
    arg2 = getattr(instance, "arg2", "")
    if schema and schema.symmetric:
        arg1, arg2 = sorted([arg1, arg2])
    return (relation_type, arg1, arg2)
