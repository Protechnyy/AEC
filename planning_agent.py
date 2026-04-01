"""
Planning agent for the AEC pipeline.

The planning agent analyses the input text conditioned on event type
definitions and proposes a ranked list of trigger–type hypotheses.

Paper §3.2 (Planning Agent)
----------------------------
The agent is prompted with the full set of event type definitions (as Python
dataclasses) and the input sentence.  It returns a JSON array of candidate
(trigger, event_type) pairs ranked by confidence.  These are converted into
:class:`Hypothesis` objects and passed to the Coding Agent.

Note: ``run_inference.py`` implements this agent's logic directly as
``run_planning_agent()``.  The method below mirrors that logic and is
provided as a callable class interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Hypothesis:
    """Data structure representing a trigger–type hypothesis."""

    trigger: str
    event_type: str
    confidence: float
    rationale: str


class PlanningAgent:
    """Planning agent that produces ranked trigger hypotheses via an LLM."""

    def generate_hypotheses_with_llm(
        self,
        text: str,
        event_definitions: str,
        max_candidates: int = 6,
    ) -> List[Hypothesis]:
        """Generate trigger hypotheses by querying a large language model.

        Parameters
        ----------
        text : str
            The input sentence or paragraph to analyse.
        event_definitions : str
            A string containing Python dataclass definitions for all relevant
            event types.
        max_candidates : int, optional
            Maximum number of hypotheses to return.  Defaults to 6.

        Returns
        -------
        List[Hypothesis]
            A list of hypotheses sorted by descending confidence.
        """
        from .llm_utils import extract_trigger_event_pairs

        pairs = extract_trigger_event_pairs(text, event_definitions)
        hypotheses: List[Hypothesis] = []
        n = len(pairs[:max_candidates])
        for rank, (trigger, event_type) in enumerate(pairs[:max_candidates]):
            confidence = 1.0 - rank * (0.9 / max(n, 1))
            confidence = max(0.1, confidence)
            rationale = (
                f"LLM identified '{trigger}' as a trigger for event type "
                f"'{event_type}' (rank {rank + 1}/{n})."
            )
            hypotheses.append(
                Hypothesis(
                    trigger=trigger,
                    event_type=event_type,
                    confidence=confidence,
                    rationale=rationale,
                )
            )
        return hypotheses
