"""
Coding agent for the AEC pipeline.

The coding agent is the core of the AEC framework.  It takes the highest-
ranked trigger hypothesis produced by the planning agent and generates
executable Python code that instantiates the target event schema class with
argument values extracted directly from the input text.

Paper §3.3 (Coding Agent)
--------------------------
The agent receives three inputs:

* The event schema expressed as a Python ``@dataclass`` definition (with typed
  fields for ``mention`` and each argument role).
* The trigger–type hypothesis (trigger span + event type name).
* The original input sentence.

It produces a Python *instantiation string* such as::

    [Databreach(mention="hacked", tool=["phishing"], victim=["HealthCo"],
                number_of_data=[], time=["March 2024"], place=[])]

This string can be passed directly to the Python evaluator (``eval()``) after
the event class definitions have been imported.  The scorer in
``utils/code_evaluation/events_scorer.py`` operates on exactly this format
via its ``safe_eval`` function.

Two interfaces are provided:

* :meth:`generate_code_with_llm` – the paper's approach.  Prompts an LLM
  to complete the ``result = `` line in a code-completion prompt that mirrors
  the input format produced by ``utils/code_prompts/prepare_dataset.py``.
* :meth:`generate_event_object` – a heuristic baseline (no LLM) that returns
  an :class:`~AEC.event_schema.EventObject` with the trigger filled in and
  all argument lists empty.  Useful for offline testing when no API key is
  available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .event_schema import EventSchema, EventObject
from .planning_agent import Hypothesis


@dataclass
class CodingAgent:
    """Coding agent that produces event instantiation code or objects."""

    # ------------------------------------------------------------------ #
    # Paper-aligned LLM method                                            #
    # ------------------------------------------------------------------ #

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

        This is the method described in the paper.  The prompt presents the
        event schema as a Python dataclass, the source text, and optionally
        retrieved exemplars.  The model is asked to complete the line
        ``result = `` with a Python list containing one instantiation of the
        event class.

        Parameters
        ----------
        hypothesis : Hypothesis
            The trigger-type hypothesis to use for this coding attempt.
        schema_definition : str
            The Python ``@dataclass`` definition for the target event type as
            a string (e.g. as loaded from the ontology ``init_prompts`` files
            or produced by :meth:`~AEC.ontology.OntologyManager.build_definitions`).
        text : str
            The original input sentence.
        exemplars : str, optional
            Newline-separated exemplar sentences produced by the Retrieval
            Agent.  When non-empty they are added as inline comments before
            the task prompt to help the model understand the event type.
        patch_feedback : str, optional
            Diagnostic error message from a previous failed verification
            attempt (inner-loop patching).  When provided the model is asked
            to fix the specific issue rather than generating from scratch.
        model : str, optional
            LLM identifier.  Defaults to ``"gpt-4o"``.

        Returns
        -------
        str
            A Python expression such as::

                [EventClass(mention="trigger", role=["span"], ...)]

            that can be ``eval()``-ed after the event class definitions have
            been imported.
        """
        from .llm_utils import call_llm  # local import

        # Build exemplar comment block
        exemplar_block = ""
        if exemplars:
            lines = [f"# {line}" for line in exemplars.strip().splitlines() if line.strip()]
            exemplar_block = "# Example sentences for this event type:\n" + "\n".join(lines) + "\n\n"

        # Build the code-completion prompt (mirrors prepare_dataset.py format)
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
        # Strip markdown code fences if the model adds them
        raw = re.sub(r"^```(?:python)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
        return raw.strip()

    # ------------------------------------------------------------------ #
    # Heuristic baseline (no LLM needed)                                  #
    # ------------------------------------------------------------------ #

    def generate_event_object(
        self, hypothesis: Hypothesis, schema: EventSchema, text: str
    ) -> EventObject:
        """Instantiate an :class:`EventObject` from a trigger hypothesis.

        This is the heuristic baseline used when no LLM is available.  It
        records the trigger from the hypothesis and initialises every argument
        role to an empty list.  Subclasses can override this method to add
        proper argument extraction logic.
        """
        args: Dict[str, List[str]] = {role: [] for role in schema.roles}
        return EventObject(event_type=schema.event_type, trigger=hypothesis.trigger, arguments=args)

    def generate_code(
        self, hypothesis: Hypothesis, schema: EventSchema, text: str
    ) -> str:
        """Return a heuristic Python code snippet (no LLM).

        The snippet imports ``EventObject`` from ``AEC.event_schema`` and
        constructs an instance using the trigger and empty arguments.  This
        is a development/debug utility; use :meth:`generate_code_with_llm`
        for paper-aligned output.
        """
        arg_dict_items = ", ".join([f"'{role}': []" for role in schema.roles])
        code = (
            "from AEC.event_schema import EventObject\n"
            f"event = EventObject(event_type='{schema.event_type}', "
            f"trigger='{hypothesis.trigger}', "
            f"arguments={{ {arg_dict_items} }})\n"
            "event"
        )
        return code
