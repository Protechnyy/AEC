"""
Verification agent for the AEC pipeline.

The verification agent checks that a candidate :class:`EventObject` produced
by the coding agent is consistent with two sources of truth:

1. **Schema consistency** – the event type matches the schema, and every
   expected argument role is present with no unexpected extras.
2. **Text grounding** – the trigger span and every non-empty argument span
   can be found verbatim as a substring of the original input sentence.

If any check fails the agent raises a :class:`VerificationError` that
contains a list of human-readable error messages.  The pipeline catches
``VerificationError`` and either retries with a different hypothesis or
abandons the current event type.

Both checks can be toggled individually via constructor parameters, which
makes it easy to run ablation experiments where only schema validation or
only text grounding is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .event_schema import EventSchema, EventObject


class VerificationError(Exception):
    """Raised when an event object fails one or more verification checks.

    Attributes
    ----------
    errors : List[str]
        A list of human-readable error messages describing each failed check.
    """

    def __init__(self, errors: List[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class VerificationAgent:
    """Validate event objects against the schema and the input text.

    Parameters
    ----------
    check_trigger_in_text : bool
        When ``True`` (default) the agent verifies that the trigger span
        appears verbatim as a substring of the input text.  Disable this
        to allow triggers that are paraphrases of the text.
    check_args_in_text : bool
        When ``True`` (default) the agent verifies that every non-empty
        argument span appears verbatim as a substring of the input text.
        Disable this to allow abstractive argument values.
    check_schema_roles : bool
        When ``True`` (default) the agent verifies that the set of argument
        roles in the event object exactly matches the roles declared in the
        schema (no missing roles, no extra roles).
    """

    check_trigger_in_text: bool = True
    check_args_in_text: bool = True
    check_schema_roles: bool = True

    def verify(self, event_obj: EventObject, schema: EventSchema, text: str) -> None:
        """Verify *event_obj* against *schema* and *text*.

        Parameters
        ----------
        event_obj : EventObject
            The candidate event object to verify.
        schema : EventSchema
            The target event schema.
        text : str
            The original input sentence or paragraph.

        Raises
        ------
        VerificationError
            If one or more verification checks fail.  All failing checks are
            collected and reported together so that a caller can diagnose
            multiple issues in a single pass.
        """
        errors: List[str] = []

        # --- Check 1: event type must match the schema ----------------------
        if event_obj.event_type != schema.event_type:
            errors.append(
                f"Event type mismatch: schema expects '{schema.event_type}' "
                f"but event object has '{event_obj.event_type}'."
            )

        # --- Check 2: trigger span must appear in the input text ------------
        if self.check_trigger_in_text:
            if not event_obj.trigger:
                errors.append("Trigger span is empty.")
            elif event_obj.trigger not in text:
                errors.append(
                    f"Trigger span '{event_obj.trigger}' does not appear "
                    f"in the input text."
                )

        # --- Check 3: argument roles must match the schema ------------------
        if self.check_schema_roles:
            expected_roles = set(schema.roles.keys())
            actual_roles = set(event_obj.arguments.keys())
            missing_roles = expected_roles - actual_roles
            extra_roles = actual_roles - expected_roles
            if missing_roles:
                errors.append(
                    f"Missing argument roles (declared in schema but absent "
                    f"from event object): {sorted(missing_roles)}."
                )
            if extra_roles:
                errors.append(
                    f"Unexpected argument roles (present in event object but "
                    f"not declared in schema): {sorted(extra_roles)}."
                )

        # --- Check 4: argument spans must appear in the input text ----------
        if self.check_args_in_text:
            for role, spans in event_obj.arguments.items():
                for span in spans:
                    if span and span not in text:
                        errors.append(
                            f"Argument span '{span}' for role '{role}' "
                            f"does not appear in the input text."
                        )

        if errors:
            raise VerificationError(errors)

    # ------------------------------------------------------------------ #
    # Paper-aligned code-execution verification (T1 / T2 / T3)           #
    # ------------------------------------------------------------------ #

    def verify_code(
        self,
        code_string: str,
        text: str,
        class_globals: dict,
    ) -> None:
        """Verify a raw Python instantiation string by executing it.

        This implements the three-stage verification described in the paper
        (§3.4 Verification Agent):

        * **T3 — Structural check:** The code compiles and runs without
          ``SyntaxError``, ``TypeError`` (missing/unexpected arguments), or
          ``NameError`` (hallucinated class names).  These errors are caught
          and reported verbatim so they can be fed back to the Coding Agent
          as patching feedback.
        * **T2 — Type check:** Argument values conform to their declared
          types.  This is enforced implicitly when the dataclass/Pydantic
          constructor raises a ``TypeError`` or ``ValueError`` on invalid
          inputs.
        * **T1 — Semantic check:** The ``mention`` (trigger) field of every
          extracted event object appears verbatim in the input text.

        Parameters
        ----------
        code_string : str
            The raw LLM output, e.g.
            ``[Databreach(mention="hacked", victim=["HealthCo"], ...)]``.
        text : str
            The original input sentence used to ground the trigger and
            argument spans.
        class_globals : dict
            A dictionary of event class definitions that will be available
            when ``eval()`` is called.  Typically obtained by calling
            ``vars(module)`` on the imported ``*_definitions_new`` module,
            or by using ``import_star`` from ``utils/code_prompts/utils.py``.

        Raises
        ------
        VerificationError
            If any of T1, T2, or T3 fails.  The error message is the
            diagnostic ``ε`` fed back to the Coding Agent for patching.
        """
        errors: List[str] = []

        # T3 + T2: Try to execute the code string ----------------------------
        result = None
        try:
            result = eval(code_string, class_globals)  # noqa: S307
        except SyntaxError as exc:
            errors.append(f"[T3-SyntaxError] {exc}")
        except TypeError as exc:
            errors.append(f"[T2/T3-TypeError] {exc}")
        except NameError as exc:
            errors.append(f"[T3-NameError] {exc}")
        except Exception as exc:
            errors.append(f"[T3-Error] {type(exc).__name__}: {exc}")

        if errors:
            raise VerificationError(errors)

        # Unwrap result: accept a list or a bare instance
        if not isinstance(result, (list, tuple)):
            result = [result]
        # Filter out raw class objects (model occasionally outputs the class itself)
        instances = [r for r in result if not isinstance(r, type)]
        if not instances:
            errors.append("[T3-StructuralError] eval() produced no event instances.")
            raise VerificationError(errors)

        # T1: trigger (mention) must appear in the input text ----------------
        if self.check_trigger_in_text:
            for inst in instances:
                mention = getattr(inst, "mention", None)
                if mention is None:
                    errors.append(
                        f"[T1-MissingMention] Event instance of type "
                        f"'{type(inst).__name__}' has no 'mention' field."
                    )
                elif mention and mention not in text:
                    errors.append(
                        f"[T1-SemanticError] Trigger span '{mention}' does not "
                        f"appear in the input text."
                    )

        # T1 (args): argument spans must appear in text ----------------------
        if self.check_args_in_text:
            for inst in instances:
                for attr, val in vars(inst).items():
                    if attr == "mention":
                        continue
                    if isinstance(val, list):
                        for span in val:
                            if isinstance(span, str) and span and span not in text:
                                errors.append(
                                    f"[T1-ArgHallucination] Argument span '{span}' "
                                    f"(role '{attr}') not found in input text."
                                )

        if errors:
            raise VerificationError(errors)
