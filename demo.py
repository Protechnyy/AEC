"""
AEC Pipeline Demo
=================

Run the AEC pipeline on example sentences using the heuristic planner
(no API key required) or the LLM-based planner (requires OPENAI_API_KEY).

Usage
-----
# Heuristic mode (no API key needed):
    cd /home/users/yy/code
    python -m AEC.demo

# LLM mode (requires OpenAI key):
    OPENAI_API_KEY=sk-... python -m AEC.demo --llm

# Run the evaluation scorer on a prediction file:
    python -m AEC.demo --eval demo_result/beam_predictions-70b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import textwrap

# Allow running as `python demo.py` from inside the AEC directory
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from AEC.event_schema import EventSchema, EventObject
from AEC.aec_pipeline import AECPipeline
from AEC.ontology import OntologyManager
from AEC.verification_agent import VerificationAgent


# ---------------------------------------------------------------------------
# Inline schemas for the demo (no data files needed)
# ---------------------------------------------------------------------------

DEMO_SCHEMAS = {
    "Attack": EventSchema(
        "Attack",
        {"attacker": str, "victim": str, "weapon": str, "place": str},
    ),
    "Die": EventSchema(
        "Die",
        {"agent": str, "victim": str, "instrument": str, "place": str},
    ),
    "Transfer-Money": EventSchema(
        "Transfer-Money",
        {"giver": str, "recipient": str, "beneficiary": str, "money": str},
    ),
    "Arrest-Jail": EventSchema(
        "Arrest-Jail",
        {"agent": str, "person": str, "crime": str, "place": str},
    ),
    "End-Position": EventSchema(
        "End-Position",
        {"person": str, "entity": str, "place": str},
    ),
}

DEMO_SENTENCES = [
    ("The soldiers attacked the village with mortars, killing dozens of civilians.", "Attack"),
    ("A suicide bomber killed himself and three bystanders in the crowded market.", "Die"),
    ("The bank transferred $5 million to the offshore account.", "Transfer-Money"),
    ("Police arrested the suspect at his apartment in downtown Chicago.", "Arrest-Jail"),
    ("The CEO resigned from his position at the company after the scandal.", "End-Position"),
]


def run_heuristic_demo() -> None:
    """Run the heuristic pipeline (no OpenAI key needed)."""
    print("=" * 70)
    print("AEC Pipeline — Heuristic Mode (no API key required)")
    print("=" * 70)
    print()

    pipeline = AECPipeline(
        max_hypotheses=3,
        max_patches=2,
        use_llm_plan=False,
    )

    for text, event_type in DEMO_SENTENCES:
        schema = DEMO_SCHEMAS[event_type]
        print(f"Text  : {textwrap.shorten(text, width=70)}")
        print(f"Schema: {event_type}  roles={list(schema.roles.keys())}")

        result: EventObject | None = pipeline.run(text=text, schema=schema)

        if result is None:
            print("Result: [pipeline returned None — no hypothesis passed verification]")
        else:
            print(f"Result: trigger='{result.trigger}'  event_type='{result.event_type}'")
            for role, spans in result.arguments.items():
                if spans:
                    print(f"        {role}: {spans}")
        print()


def run_llm_demo() -> None:
    """Run the LLM-based pipeline (requires OPENAI_API_KEY)."""
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.  Export it and re-run with --llm.")
        sys.exit(1)

    print("=" * 70)
    print("AEC Pipeline — LLM Mode (gpt-4o)")
    print("=" * 70)
    print()

    # Build a temporary ontology from the inline schemas
    # OntologyManager.from_directory needs files; here we build it manually.
    from AEC.ontology import OntologyManager
    from dataclasses import dataclass

    # Write schemas to a temp file so OntologyManager can load them
    import tempfile, pathlib

    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    schema_lines = ["from dataclasses import dataclass", "from typing import List", ""]
    for name, schema in DEMO_SCHEMAS.items():
        schema_lines.append("@dataclass")
        schema_lines.append(f"class {name}:")
        schema_lines.append("    mention: str")
        for role in schema.roles:
            schema_lines.append(f"    {role}: List")
        schema_lines.append("")
    (tmp_dir / "demo.py").write_text("\n".join(schema_lines))

    ontology_manager = OntologyManager.from_directory(str(tmp_dir))
    pipeline = AECPipeline(
        ontology_manager=ontology_manager,
        max_hypotheses=3,
        max_patches=2,
        use_llm_plan=True,
    )

    for text, event_type in DEMO_SENTENCES[:3]:  # limit to 3 to reduce API cost
        print(f"Text      : {textwrap.shorten(text, width=66)}")
        print(f"Event type: {event_type}")
        try:
            result = pipeline.run(
                text=text,
                dataset="demo",
                event_type=event_type,
                use_llm_plan=True,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            print()
            continue

        if result is None:
            print("Result    : [no event extracted]")
        else:
            print(f"Result    : trigger='{result.trigger}'  event_type='{result.event_type}'")
            for role, spans in result.arguments.items():
                if spans:
                    print(f"            {role}: {spans}")
        print()


def run_eval(input_file: str) -> None:
    """Run the evaluation scorer on a JSON predictions file.

    The file must contain a list of objects with at least:
      - "Input"      : the input prompt text
      - "Prediction" : the model's prediction string
      - "Label"      : the gold label string
    """
    print(f"Evaluating: {input_file}")
    print()

    # The scorer lives in utils/code_evaluation and does sys-level imports
    scorer_dir = os.path.join(_here, "utils", "code_evaluation")
    if scorer_dir not in sys.path:
        sys.path.insert(0, scorer_dir)

    # Make event class definitions importable inside safe_eval
    import importlib
    try:
        import all_ee_definitions  # noqa: F401
    except ImportError:
        print(
            "WARNING: Could not import all_ee_definitions from "
            f"{scorer_dir}.  Make sure utils/code_evaluation/ is present."
        )
        sys.exit(1)

    from events_scorer import micro_e2e_scores, micro_ed_scores, micro_eae_scores

    data = json.load(open(input_file))
    if not isinstance(data, list) or not data:
        print("ERROR: input file must contain a non-empty JSON list.")
        sys.exit(1)

    # Check the file is in scorer format (needs Input/Prediction/Label keys)
    first = data[0]
    if not all(k in first for k in ("Input", "Prediction", "Label")):
        missing = [k for k in ("Input", "Prediction", "Label") if k not in first]
        print(
            f"ERROR: The file '{input_file}' is not in scorer format.\n"
            f"  Missing keys: {missing}\n"
            f"  Found keys  : {list(first.keys())}\n\n"
            "The scorer expects a JSON list of objects with:\n"
            '  "Input"      : the full prompt fed to the model\n'
            '  "Prediction" : the model\'s raw output string (may include "assistant" prefix)\n'
            '  "Label"      : the gold Python instantiation string, e.g. [Die(mention="killed", ...)]\n'
            '  "task_type"  : (optional) "E2E", "ED", or "EAE"  (default: "E2E")\n\n'
            "The file you provided appears to be a beam-search prediction file with a\n"
            "different schema (doc_id / text / trigger / gold_trigger / ...).\n"
            "Run utils/code_prompts/prepare_dataset.py first to generate scorer-compatible files."
        )
        sys.exit(1)

    # Detect task type from first item (default to E2E)
    task_type = first.get("task_type", "E2E").upper()
    if task_type == "ED":
        micro_ed_scores(data)
    elif task_type == "EAE":
        micro_eae_scores(data)
    else:
        micro_e2e_scores(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AEC pipeline demo and evaluation runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # heuristic demo (no API key):
              python -m AEC.demo

              # LLM-based demo (needs OPENAI_API_KEY):
              OPENAI_API_KEY=sk-... python -m AEC.demo --llm

              # evaluate a predictions file:
              python -m AEC.demo --eval demo_result/beam_predictions-70b.json
        """),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use the LLM-based planner (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--eval",
        metavar="FILE",
        default=None,
        help="Path to a JSON predictions file to evaluate.",
    )
    args = parser.parse_args()

    if args.eval:
        run_eval(args.eval)
    elif args.llm:
        run_llm_demo()
    else:
        run_heuristic_demo()


if __name__ == "__main__":
    main()
