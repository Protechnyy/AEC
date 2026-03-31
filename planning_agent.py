"""
Planning agent for the AEC pipeline.

The planning agent analyses the input text (possibly conditioned on
retrieval exemplars) and proposes a ranked list of trigger–type
hypotheses.  In the original AEC paper the planning agent leverages
large language models to generate and explain multiple candidate
event triggers along with their associated confidence scores and
rationales.  This implementation uses a very simple heuristic
baseline: it scans the input text for occurrences of the event type
name and, if none are found, it falls back to suggesting the most
frequent verb or noun tokens as candidate triggers.

The returned hypotheses are sorted in descending order of the
confidence score.  Each hypothesis includes a short rationale
explaining why the trigger is plausible.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from .event_schema import EventSchema


@dataclass
class Hypothesis:
    """Data structure representing a trigger–type hypothesis."""

    trigger: str
    event_type: str
    confidence: float
    rationale: str


class PlanningAgent:
    """A lightweight planning agent that proposes trigger hypotheses.

    This agent uses simple string matching and frequency heuristics to
    generate candidate triggers.  It is intended as a placeholder for
    more sophisticated language model reasoning in the full AEC
    framework.
    """

    def __init__(self) -> None:
        pass

    _tokeniser = re.compile(r"\b\w+\b", re.UNICODE)

    def generate_hypotheses(
        self,
        text: str,
        schema: EventSchema,
        examples: List[str],
        k: int = 3,
    ) -> List[Hypothesis]:
        """Generate a ranked list of up to ``k`` trigger hypotheses.

        Parameters
        ----------
        text : str
            The input text from which to extract events.
        schema : EventSchema
            The target event schema.
        examples : List[str]
            Unused in this heuristic implementation but accepted for API
            compatibility.  More sophisticated planners could use the
            exemplars to guide disambiguation.
        k : int, optional
            Maximum number of hypotheses to return.  Defaults to 3.
        """
        text_lower = text.lower()
        tokens = self._tokeniser.findall(text_lower)
        # If the event type name appears verbatim in the text, treat it as a strong cue
        candidates: List[Hypothesis] = []
        event_token = schema.event_type.lower()
        # positions where the event type occurs
        for match in re.finditer(re.escape(event_token), text_lower):
            trig = text[match.start() : match.end()]
            rationale = (
                f"The substring '{trig}' exactly matches the event type name '{schema.event_type}' in the text."
            )
            candidates.append(
                Hypothesis(trigger=trig, event_type=schema.event_type, confidence=1.0, rationale=rationale)
            )
        if not candidates:
            # As a fallback, choose the most frequent tokens as proxy triggers
            # Exclude very common stop words to avoid trivial suggestions
            stop_words = {
                "the",
                "a",
                "an",
                "of",
                "in",
                "and",
                "to",
                "for",
                "on",
                "with",
                "as",
                "by",
                "is",
                "was",
                "were",
                "be",
                "been",
                "are",
            }
            counter = Counter(tok for tok in tokens if tok not in stop_words)
            for word, freq in counter.most_common(k):
                confidence = 0.5 + 0.5 * (freq / counter.most_common(1)[0][1])  # scale between 0.5 and 1.0
                rationale = (
                    f"The token '{word}' appears {freq} times in the text and is considered a potential trigger."
                )
                # Use the capitalisation from the original text if possible
                orig_match = re.search(rf"\b{re.escape(word)}\b", text)
                trig = orig_match.group(0) if orig_match else word
                candidates.append(
                    Hypothesis(trigger=trig, event_type=schema.event_type, confidence=confidence, rationale=rationale)
                )
        # sort by confidence descending
        candidates.sort(key=lambda h: h.confidence, reverse=True)
        return candidates[:k]

    def generate_hypotheses_with_llm(
        self,
        text: str,
        event_definitions: str,
        max_candidates: int = 6,
    ) -> List[Hypothesis]:
        """Generate trigger hypotheses by querying a large language model.

        This method calls :func:`~aec.llm_utils.extract_trigger_event_pairs`
        which prompts an LLM (default: GPT-4o) with the input *text* and a
        string of Python dataclass event definitions.  The LLM returns a list
        of ``(trigger, event_type)`` pairs which are converted into ranked
        :class:`Hypothesis` objects.  Hypotheses are assigned decreasing
        confidence scores in the order returned by the model, reflecting the
        assumption that the model lists its most confident predictions first.

        Parameters
        ----------
        text : str
            The input sentence or paragraph to analyse.
        event_definitions : str
            A string containing Python dataclass definitions for all relevant
            event types (as returned by
            :meth:`~aec.ontology.OntologyManager.build_definitions`).
        max_candidates : int, optional
            Maximum number of hypotheses to return.  Defaults to 6.

        Returns
        -------
        List[Hypothesis]
            A list of hypotheses sorted by descending confidence.  An empty
            list is returned if the LLM call fails or produces no parseable
            output.
        """
        from .llm_utils import extract_trigger_event_pairs  # local import to avoid hard dep

        pairs = extract_trigger_event_pairs(text, event_definitions)
        hypotheses: List[Hypothesis] = []
        n = len(pairs[:max_candidates])
        for rank, (trigger, event_type) in enumerate(pairs[:max_candidates]):
            # Assign confidence scores that decrease linearly from 1.0
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