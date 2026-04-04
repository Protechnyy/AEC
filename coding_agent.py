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
* The trigger-type hypothesis (trigger span + event type name).
* The original input sentence.
* Optionally: retrieved exemplar sentences and patch feedback from a
  previous failed verification attempt.

It produces a Python instantiation string such as::

    [Databreach(mention="hacked", tool=["phishing"], victim=["HealthCo"],
                number_of_data=[], time=["March 2024"], place=[])]

This string can be passed directly to ``eval()`` after the event class
definitions have been imported.  The scorer in
``utils/code_evaluation/events_scorer.py`` operates on exactly this format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .planning_agent import Hypothesis
from .llm_utils import call_llm


# ── Task-type-aware prompt templates (paper §3.3) ───────────────────────────

TASK_HEADERS = {
    "e2e": (
        "# This is an event extraction task where the goal is to extract structured events "
        "from the text. A structured event contains an event trigger word, an event type, "
        "the arguments participating in the event, and their roles in the event. For each "
        "different event type, please output the extracted information from the text into "
        "python-style dictionaries where the first key will be 'mention' with the value of "
        "the event trigger. Next, please output the arguments and their roles following the "
        "same format. The event type definitions and their argument roles are defined next."
    ),
    "ed": (
        "# This is an event detection task where the goal is to identify event triggers and "
        "their types in the text. For each event, please output the extracted information "
        "into python-style dictionaries where the key is 'mention' with the value of the "
        "event trigger. The event type definitions are defined next."
    ),
    "eae": (
        "# This is an event argument extraction task where the goal is to extract the "
        "arguments of a given event trigger in the text. The event trigger and its type "
        "are provided. Please output the extracted arguments and their roles into python-"
        "style dictionaries. The event type definitions and their argument roles are defined next."
    ),
}

TASK_FOOTERS = {
    "e2e": "# The list called result should contain the instances for the following events according to the guidelines above:\nresult = \n",
    "ed":  "# The list called result should contain the instances for the following events according to the guidelines above:\nresult = \n",
    "eae": "# The list called result contains the instances for the following events according to the guidelines above\n# 1. \"{trigger}\" triggers a {event_name} event.\n\nresult = \n",
}


@dataclass
class CodingAgent:
    """Coding agent that produces event instantiation code via an LLM."""

    def generate_code(
        self,
        hypothesis: Hypothesis,
        schema_definition: str,
        text: str,
        task_type: str = "e2e",
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
        task_type : str, optional
            One of ``"e2e"``, ``"ed"``, ``"eae"``.  Defaults to ``"e2e"``.
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
        header = TASK_HEADERS[task_type]
        footer = TASK_FOOTERS[task_type]

        exemplar_block = ""
        if exemplars:
            exemplar_block = (
                "# Exemplar sentences:\n"
                + "\n".join(f"# {l}" for l in exemplars.splitlines() if l.strip())
                + "\n\n"
            )

        trigger = hypothesis.trigger
        event_type = hypothesis.event_type

        footer_filled = (
            footer.format(trigger=trigger, event_name=event_type)
            if "{trigger}" in footer
            else footer
        )

        user_prompt = (
            f"{exemplar_block}"
            f"{header}\n\n"
            f"{schema_definition}\n\n"
            f"# This is the text to analyze\n"
            f'text = "{text}"\n\n'
            f'# Hint: trigger word is "{trigger}"\n'
            f"{footer_filled}"
        )

        system_prompt = (
            "You are a coding agent for event extraction. "
            "Complete the Python assignment `result = ` with a list containing "
            "the event class instantiation.\n"
            "CRITICAL RULES:\n"
            "- The 'mention' field MUST be the EXACT minimal trigger word(s) copied "
            "verbatim from the text. Use the shortest word that indicates the event "
            "(typically one verb or noun). Do NOT paraphrase or extend.\n"
            "- Each argument span MUST be the SHORTEST exact substring from the text "
            "that fills the role. Do NOT add articles (the/a/an), prepositions "
            "(to/from/in/at/of/by/with), or surrounding context. "
            "For example, use 'United States' not 'the United States', "
            "use 'Red Sea' not 'to the Red Sea'.\n"
            "- Only include arguments that are EXPLICITLY stated in the text for that role. "
            "Do NOT infer missing participants from world knowledge, discourse context, or "
            "weak clues. If a role is uncertain, output [].\n"
            "- Prefer the head noun / named entity itself, not possessors or descriptive "
            "modifiers. For example, prefer 'mother' over \"baby's mother\" when both are "
            "supported by the text, and prefer 'Republican Guards' over "
            "\"Saddam's Iraqi Republican Guards\" when the shorter span is present.\n"
            "- Do not use a long clause as an argument span. Copy only the minimal noun "
            "phrase that fills the role.\n"
            "- Use [] for absent roles.\n"
            "Output ONLY the Python list expression (e.g. [ClassName(mention=..., role=[...])]), "
            "nothing else - no markdown, no explanation."
        )
        if patch_feedback:
            system_prompt += (
                f"\n\nThe previous attempt failed. Fix this error:\n{patch_feedback}"
            )

        raw = call_llm(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            model=model,
        )
        # Strip code fences
        raw = re.sub(r"^```(?:python)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
        # Strip trailing commentary (e.g. "# Note: ...") after the closing ']'
        # that LLMs sometimes append despite instructions.
        raw = raw.strip()
        bracket_depth = 0
        cut = len(raw)
        for i, ch in enumerate(raw):
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    cut = i + 1
                    break
        if cut < len(raw):
            raw = raw[:cut]
        return raw.strip()
