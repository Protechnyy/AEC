"""
Coding agent for the AEC pipeline.

The coding agent takes the highest-ranked trigger hypothesis produced by the
planning agent and generates executable Python code that instantiates the
target event schema class with argument values extracted directly from the
input text.

Paper §3.3 (Coding Agent)
--------------------------
The agent receives:

* The event schema expressed as a Python ``@dataclass`` definition.
* The trigger–type hypothesis (trigger span + event type name).
* The original input sentence.
* Optionally: retrieved exemplar sentences and patch feedback from a
  previous failed verification attempt.

It produces a Python instantiation string such as::

    [Databreach(mention="hacked", tool=["phishing"], victim=["HealthCo"],
                number_of_data=[], time=["March 2024"], place=[])]

This string can be passed directly to ``eval()`` after the event class
definitions have been imported.  The scorer in
``utils/code_evaluation/events_scorer.py`` operates on exactly this format.

Note: ``run_inference.py`` implements this agent's logic directly as
``run_coding_agent()``, which uses the task-type-aware prompt headers from
``TASK_HEADERS``/``TASK_FOOTERS``.  The method below mirrors that logic and
is provided as a callable class interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .planning_agent import Hypothesis


@dataclass
class CodingAgent:
    """Coding agent that produces event instantiation code via an LLM."""

    def generate_code_with_llm(
        self,
        hypothesis: Hypothesis,
        schema_definition: str,
        text: str,
        exemplars: str = "",
        patch_feedback: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> str:
        """Generate a Python instantiation code string using an LLM.

        Parameters
        ----------
        hypothesis : Hypothesis
            The trigger-type hypothesis to use for this coding attempt.
        schema_definition : str
            The Python ``@dataclass`` definition for the target event type.
        text : str
            The original input sentence.
        exemplars : str, optional
            Newline-separated exemplar sentences from the Retrieval Agent.
        patch_feedback : str, optional
            Diagnostic error message from a previous failed verification
            attempt (inner-loop patching).
        model : str, optional
            LLM identifier.  Defaults to ``"gpt-4o"``.

        Returns
        -------
        str
            A Python expression such as::

                [EventClass(mention="trigger", role=["span"], ...)]
        """
        from .llm_utils import call_llm

        exemplar_block = ""
        if exemplars:
            lines = [f"# {line}" for line in exemplars.strip().splitlines() if line.strip()]
            exemplar_block = "# Example sentences for this event type:\n" + "\n".join(lines) + "\n\n"

        task_header = (
            "# This is an event extraction task. Extract structured events from the text.\n"
            "# Output the extracted information as a Python list of event class instances.\n"
            "# For each role, extract the exact text span; use [] if absent.\n"
            "# The following lines describe the task definition"
        )
        text_block = f'# This is the text to analyze\ntext = "{text}"'
        footer = (
            "# The list called result should contain the instances for the following "
            f"events according to the guidelines above:\n"
            f"# Trigger word: \"{hypothesis.trigger}\"  Event type: {hypothesis.event_type}\n"
            "result = "
        )

        user_prompt = (
            f"{exemplar_block}"
            f"{task_header}\n\n"
            f"{schema_definition}\n\n"
            f"{text_block}\n\n"
            f"{footer}"
        )

        system_prompt = (
            "You are a coding agent for event extraction.  "
            "Complete the Python assignment `result = ` with a list containing exactly one "
            "instantiation of the event class.  "
            "Extract spans verbatim from the text; use [] for absent roles.  "
            "Output only the Python list expression, nothing else."
        )

        if patch_feedback:
            system_prompt += (
                "\n\nThe previous attempt had the following error — fix it:\n"
                f"{patch_feedback}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = call_llm(messages, model=model)
        raw = re.sub(r"^```(?:python)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
        return raw.strip()
