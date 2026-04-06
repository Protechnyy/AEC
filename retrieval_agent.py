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
        system = "You are a helpful example generator for event extraction."
        user = (
            f"Given the following event type definition:\n\n{schema_definition}\n\n"
            f"Generate {k} fluent English sentences. Each sentence must:\n"
            f"1. Contain a clear trigger word or phrase for this event type.\n"
            f"2. Mention entities filling as many argument roles as possible.\n"
            f"3. Be realistic and varied.\n\n"
            f"Output exactly one sentence per line, no numbering."
        )
        try:
            return call_llm(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                model=model,
            )
        except Exception:
            return ""
