"""
Planning agent for the AEC pipeline.

The planning agent analyses the input text conditioned on event type
definitions and proposes a ranked list of trigger-type hypotheses.

Paper §3.2 (Planning Agent)
----------------------------
The agent is prompted with the event type definition (as a Python dataclass)
and the input sentence.  It returns a JSON array of candidate (trigger,
event_type) pairs ranked by confidence.  These are converted into
:class:`Hypothesis` objects and passed to the Coding Agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

from .llm_utils import call_llm


@dataclass
class Hypothesis:
    """Data structure representing a trigger-type hypothesis."""

    trigger: str
    event_type: str
    confidence: float
    rationale: str


class PlanningAgent:
    """Planning agent that produces ranked trigger hypotheses via an LLM."""

    def generate_hypotheses(
        self,
        text: str,
        schema_definition: str,
        exemplars: str = "",
        k: int = 3,
        model: str = "gpt-4o",
    ) -> List[Hypothesis]:
        """Generate a ranked list of up to *k* trigger hypotheses.

        Parameters
        ----------
        text : str
            The input sentence or paragraph to analyse.
        schema_definition : str
            The Python ``@dataclass`` definition for the target event type.
        exemplars : str, optional
            Newline-separated exemplar sentences from the Retrieval Agent.
        k : int, optional
            Maximum number of hypotheses to return.  Defaults to 3.
        model : str, optional
            LLM identifier.  Defaults to ``"gpt-4o"``.

        Returns
        -------
        List[Hypothesis]
            A list of hypotheses sorted by descending confidence.
        """
        system = (
            "You are an assistant for event extraction. "
            "Given a piece of text and an event type definition (as a Python dataclass), "
            "produce a JSON array of trigger-hypothesis objects. "
            "Each object must have keys: 'trigger' (exact span from text), "
            "'event_type' (class name), 'confidence' (0-1 float), 'rationale' (string).\n"
            "IMPORTANT: The 'trigger' MUST be the EXACT minimal word(s) copied verbatim "
            "from the text that best indicate the event is occurring. Typically this is "
            "a single verb or noun. Do NOT paraphrase, do NOT include surrounding context, "
            "and do NOT use multi-word phrases when a single word suffices. "
            "If multiple events of this type exist in the text, include a separate "
            "hypothesis for each distinct trigger."
        )
        exemplar_block = ""
        if exemplars:
            exemplar_block = f"\nExample sentences for context:\n{exemplars}\n"

        user = (
            f"Event definition:\n{schema_definition}\n"
            f"{exemplar_block}\n"
            f"Text:\n{text}\n\n"
            f"Identify all events of this type in the text (up to {k} candidates). "
            f"Each trigger must be the exact minimal word(s) from the text. "
            f"Output only a JSON array, no explanation."
        )
        try:
            raw = call_llm(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                model=model,
            )
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else []
            hypotheses: List[Hypothesis] = []
            if isinstance(data, list):
                has_confidence = any("confidence" in item for item in data if isinstance(item, dict))
                if has_confidence:
                    data.sort(
                        key=lambda x: float(x.get("confidence", 0)), reverse=True
                    )
                for rank, item in enumerate(data[:k]):
                    default_conf = max(0.0, 1.0 - (rank / max(1, len(data))))
                    hypotheses.append(Hypothesis(
                        trigger=item.get("trigger", ""),
                        event_type=item.get("event_type", ""),
                        confidence=float(item.get("confidence", default_conf)),
                        rationale=item.get("rationale", ""),
                    ))
            if hypotheses:
                return hypotheses
        except Exception:
            pass
        # Fallback: use the first token as a heuristic trigger
        return [Hypothesis(
            trigger=text.split()[0] if text.split() else "",
            event_type="",
            confidence=0.5,
            rationale="fallback",
        )]
