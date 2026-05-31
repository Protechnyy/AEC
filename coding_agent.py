"""Coding agent for zero-shot relation extraction."""

from __future__ import annotations

from dataclasses import dataclass

from event_schema import RelationSchema
from llm_utils import _load_json_reply, call_llm
from planning_agent import RelationHypothesis


def _norm(value: str) -> str:
    return " ".join(value.split()).strip()


@dataclass
class CodingAgent:
    """Convert relation hypotheses into executable schema-as-code snippets."""

    def generate_relation_code(
        self,
        hypothesis: RelationHypothesis,
        text: str,
        *,
        schemas: dict[str, RelationSchema],
        use_llm_coding: bool = False,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> str:
        schema = schemas[hypothesis.relation_type]
        if not use_llm_coding:
            return "\n".join(
                [
                    f"result = {schema.class_name}(",
                    f"    relation_type={schema.relation_type!r},",
                    f"    head={hypothesis.head!r},",
                    f"    tail={hypothesis.tail!r},",
                    "    evidence='',",
                    ")",
                ]
            )

        schema_definitions = "\n\n".join(item.to_code_definition() for item in schemas.values())
        allowed_classes = ", ".join(schema.class_name for schema in schemas.values())
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Coding Agent in a schema-as-code relation extraction system. "
                    "Return only Python code, no Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write executable Python code that instantiates exactly one Pydantic relation object.\n\n"
                    "Rules:\n"
                    "1. Use one of the provided Pydantic classes and assign it to a variable named result.\n"
                    "2. Keep the relation type fixed to the planning hypothesis unless the class would be invalid.\n"
                    "3. head, tail, and evidence must be exact spans copied from the text.\n"
                    "4. Do not define new classes, functions, imports, or helper variables.\n"
                    "5. Return code only. The verifier will execute it with the schema classes already in scope.\n\n"
                    f"Allowed classes: {allowed_classes}\n\n"
                    f"Schema code:\n{schema_definitions}\n\n"
                    f"Planning hypothesis:\n"
                    f"- relation_type: {hypothesis.relation_type}\n"
                    f"- class: {schema.class_name}\n"
                    f"- head: {hypothesis.head}\n"
                    f"- tail: {hypothesis.tail}\n"
                    f"- rationale: {hypothesis.rationale}\n\n"
                    f"Text:\n{text}\n\n"
                    "Return code like:\n"
                    f"result = {schema.class_name}(relation_type={schema.relation_type!r}, head='...', tail='...', evidence='...')"
                ),
            },
        ]
        reply = call_llm(
            messages,
            model=model,
            base_url=base_url,
            api_key=api_key,
            request_tag="re_coding",
            max_tokens=384,
        )
        return self._extract_code(reply)

    def _extract_code(self, reply: str) -> str:
        fenced = _load_json_reply(reply)
        if isinstance(fenced, dict) and isinstance(fenced.get("code"), str):
            return fenced["code"].strip()
        if isinstance(fenced, str):
            return fenced.strip()
        import re

        match = re.search(r"```(?:python)?\s*([\s\S]*?)```", reply)
        if match:
            return match.group(1).strip()
        return reply.strip()

    def generate_relation_objects(self, *args, **kwargs):
        """Deprecated compatibility shim: code execution now happens in the verifier."""

        raise RuntimeError(
            "CodingAgent now emits executable Pydantic code via generate_relation_code(); "
            "use VerificationAgent.verify_code() to execute and validate it."
        )
