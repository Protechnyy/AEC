"""
Retrieval agent for the AEC pipeline.

The retrieval agent generates exemplar sentences that illustrate how the
target event type is realised in natural language.  These exemplars are
passed to the Planning Agent to ground trigger hypotheses.

Paper §3.1 (Retrieval Agent)
------------------------------
The LLM is prompted with the Python dataclass definition of the event schema
and asked to produce fluent sentences containing a clear trigger word and as
many populated argument roles as possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm_utils import call_llm


def _extract_event_type_and_roles(schema_definition: str) -> tuple[str, list[str]]:
    event_type = ""
    roles: list[str] = []

    for line in schema_definition.splitlines():
        stripped = line.strip()
        if stripped.startswith("class ") and "(" in stripped:
            event_type = stripped.split()[1].split("(")[0].strip(":")
            continue
        if ":" in stripped and not stripped.startswith("@") and not stripped.startswith("class "):
            field = stripped.split(":", 1)[0].strip()
            if field and field != "mention":
                roles.append(field)

    return event_type, roles


@dataclass
class RetrievalAgent:
    """Retrieval agent that generates exemplar sentences via an LLM."""

    def retrieve(
        self,
        schema_definition: str,
        k: int = 3,
        model: str = "gpt-4o",
    ) -> str:
        """Generate *k* exemplar sentences using an LLM.

        Parameters
        ----------
        schema_definition : str
            The Python dataclass definition of the target event type.
        k : int, optional
            Number of exemplar sentences to generate.  Paper default: 3.
        model : str, optional
            LLM to use.  Defaults to ``"gpt-4o"``.

        Returns
        -------
        str
            A newline-separated string of *k* exemplar sentences.
        """
        event_type, roles = _extract_event_type_and_roles(schema_definition)
        roles_text = ", ".join(roles)

        system = "You are a helpful example generator for event extraction."
        user = (
            f"Event type: {event_type or 'UnknownEvent'}\n"
            f"Roles: {roles_text}\n\n"
            f"Write {k} English sentence"
            f"{'' if k == 1 else 's'} that contain"
            f"{'' if k == 1 else ''} a clear mention of the {event_type or 'event'} trigger "
            f"and populate{'s' if k == 1 else ''} all roles.\n"
            f"Output exactly one sentence per line."
        )
        try:
            return call_llm(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                model=model,
            )
        except Exception:
            return ""
