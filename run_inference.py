"""
AEC Inference Runner — Reproducing Paper Results
=================================================

This script runs the full AEC multi-agent pipeline over any TextEE test split
and saves predictions in the format expected by the evaluation scorer.

Paper reference: "Extracting Events Like Code: A Multi-Agent Programming
Framework for Zero-Shot Event Extraction", AAAI 2026 (arXiv 2511.13118)

Quick start
-----------
1.  Install dependencies::

        conda activate AEC
        pip install -r requirements.txt

2.  Make sure the TextEE processed data is available at ``data/raw/TextEE``
    (the standard TextEE split format: ``<dataset>/split1/{train,dev,test}.json``).

3.  Run inference::

        # with GPT-4o (best results in paper)
        OPENAI_API_KEY=sk-... python run_inference.py \\
            --dataset ace05-en \\
            --model gpt-4o \\
            --output outputs/ace05-en_gpt4o_predictions.json

        # with a local Llama3-70B served via vLLM on localhost:8000
        python run_inference.py \\
            --dataset casie \\
            --model llama3-70b \\
            --base_url http://localhost:8000/v1 \\
            --output outputs/casie_llama70b_predictions.json

4.  Evaluate::

        cd utils/code_evaluation
        python events_scorer.py --input_file ../../outputs/ace05-en_gpt4o_predictions.json

Datasets supported (paper Table 1)
------------------------------------
    ace05-en   (News, 33 event types, E2E)
    fewevent   (General, 100 event types, ED)
    genia2011  (Biomedical, 9 event types, E2E)
    speed      (Epidemiology, 7 event types, ED)
    casie      (Cybersecurity, 5 event types, E2E)

Pipeline (Algorithm 1)
-----------------------
For each test sample:
  1. Retrieval Agent → k exemplar sentences (LLM)
  2. Planning Agent  → ranked hypotheses (LLM), k=3
  3. Outer loop over hypotheses (confidence descending):
       Inner loop, up to t=3 patch attempts:
         Coding Agent  → Python instantiation code (LLM)
         Verification  → T1 semantic + T2 type + T3 structural checks
         If pass  → record prediction and break
         If fail  → feed diagnostic ε back to Coding Agent (patch)
     If all inner attempts fail → advance to next hypothesis
  4. If all hypotheses fail → record empty prediction []
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ── make AEC importable when running as `python run_inference.py` from AEC/ ──
_here = Path(__file__).resolve().parent
_parent = _here.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from AEC.llm_utils import call_llm
from AEC.verification_agent import VerificationAgent, VerificationError

# ── paths ──────────────────────────────────────────────────────────────────
TEXTEE_DIR = _here / "data" / "raw" / "TextEE"
SCHEMA_DEF_DIR = _here / "utils" / "code_schema_generation" / "python_event_defs"
INIT_PROMPTS_DIR = _here / "utils" / "code_schema_generation" / "init_prompts"
OUTPUT_DIR = _here / "outputs"

# ── dataset → task type mapping (from utils/code_prompts/utils.py) ─────────
DATASET_TASK = {
    "ace05-en":   "e2e",
    "fewevent":   "ed",
    "genia2011":  "e2e",
    "genia2013":  "e2e",
    "speed":      "ed",
    "casie":      "e2e",
    "wikievents": "eae",
    "rams":       "eae",
    "maven":      "ed",
    "mlee":       "e2e",
    "phee":       "e2e",
    "richere-en": "e2e",
    "muc4":       "eae",
    "mee":        "ed",
    "m2e2":       "e2e",
    "geneva":     "eae",
}

# Event-type name cleaning (mirroring clean_event_name in utils/code_prompts/utils.py)
SCHEMA_SPLITTERS = {
    "ace05-en":   ":",
    "speed":      "None",
    "fewevent":   ".",
    "casie":      ":",
    "genia2011":  "None",
    "genia2013":  "None",
    "wikievents": "None",
    "rams":       ".",
    "maven":      "None",
    "mlee":       "None",
    "phee":       "None",
    "richere-en": ":",
    "muc4":       "None",
    "mee":        "_",
    "m2e2":       ":",
    "geneva":     "None",
}

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
    "e2e":  "# The list called result should contain the instances for the following events according to the guidelines above:\nresult = \n",
    "ed":   "# The list called result should contain the instances for the following events according to the guidelines above:\nresult = \n",
    "eae":  "# The list called result contains the instances for the following events according to the guidelines above\n# 1. \"{trigger}\" triggers a {event_name} event.\n\nresult = \n",
}


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def clean_event_name(event_type: str, dataset_name: str) -> str:
    """Convert TextEE event type string to Python class name."""
    splitter = SCHEMA_SPLITTERS.get(dataset_name, "None")
    event_type = event_type.replace("n/a", "Na")
    if splitter == "None":
        parts = [event_type]
    else:
        parts = event_type.split(splitter)

    if len(parts) > 1:
        parent = parts[0].replace("-", "").replace(".", "_")
        if dataset_name == "casie":
            parent = parent.title()
        child = "_".join(parts[1:]).replace("-", "").replace(".", "_")
        if dataset_name == "rams":
            return f"{parent}_{child}(Event)"
        return f"{child}({parent}Event)"
    else:
        name = event_type.replace("-", "").replace(".", "_")
        if dataset_name == "speed":
            name = name.title()
        return f"{name}(Event)"


def load_schema_definitions(dataset_name: str) -> Dict[str, str]:
    """Load per-event-type schema strings from the init_prompts directory.

    Returns a dict mapping class name (e.g. ``"Die"``) → dataclass string.
    """
    init_file = INIT_PROMPTS_DIR / f"{dataset_name}.txt"
    if not init_file.exists():
        raise FileNotFoundError(
            f"Schema init_prompts file not found: {init_file}\n"
            f"Run utils/code_schema_generation/generate_schema.py first."
        )
    text = init_file.read_text(encoding="utf-8")
    schemas: Dict[str, str] = {}
    # Split on @dataclass blocks; use the same line-scanning approach as
    # utils/code_prompts/utils.py::load_init_prompts to avoid matching
    # "class" inside "@dataclass".
    blocks = re.split(r"(?=@dataclass)", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("class "):
                # e.g. "class Attack(ConflictEvent):" → key = "Attack"
                cls_name = line.split("(")[0].replace("class", "").strip().rstrip(":")
                schemas[cls_name] = block
                break
    return schemas


def import_event_classes(dataset_name: str) -> dict:
    """Import the event class definitions for a dataset and return their namespace.

    The classes are defined in ``utils/code_schema_generation/python_event_defs/``.
    """
    defs_dir = str(SCHEMA_DEF_DIR)
    if defs_dir not in sys.path:
        sys.path.insert(0, defs_dir)
    mod_name = f"{dataset_name}_definitions_new"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        raise FileNotFoundError(
            f"Event class definitions not found: {SCHEMA_DEF_DIR / mod_name}.py\n"
            f"Run utils/code_schema_generation/generate_schema.py first."
        )
    return vars(mod)


def load_schema_roles(dataset_name: str) -> Dict[str, List[str]]:
    """Return ordered argument role names per class (excluding 'mention').

    Parses the init_prompts text file which is the authoritative source for
    role ordering.  Used by ``build_gold_label`` to emit arguments in the
    same order as the class constructor, making the gold label eval()-safe.
    """
    path = INIT_PROMPTS_DIR / f"{dataset_name}.txt"
    if not path.exists():
        return {}
    result: Dict[str, List[str]] = {}
    current: Optional[str] = None
    roles: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("class "):
            if current:
                result[current] = roles
            current = s.split("(")[0].replace("class", "").strip()
            roles = []
        elif s.startswith("mention"):
            pass  # skip trigger field; handled separately
        elif ":" in s and current and not s.startswith(("#", "@", "def")):
            role = s.split(":")[0].strip()
            if role:
                roles.append(role)
        elif s == "" and current:
            result[current] = roles
            current = None
            roles = []
    if current:
        result[current] = roles
    return result


def build_gold_label(
    event_mentions: List[dict],
    dataset_name: str,
    schema_roles: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Convert TextEE gold event_mentions to a Python instantiation string.

    Mirrors ``utils/code_prompts/prepare_dataset.py::prepare_train_output``.
    The returned string is the ``Label`` field in the predictions JSON and
    is evaluated by the scorer's ``safe_eval`` against the Prediction field.
    """
    if not event_mentions:
        return "[]"

    parts: List[str] = []
    for em in event_mentions:
        class_name = clean_event_name(em["event_type"], dataset_name).split("(")[0].strip()
        trigger = em.get("trigger", {})
        mention = trigger.get("text", "") if isinstance(trigger, dict) else str(trigger)

        # Collect gold argument spans grouped by (lowercased) role name
        arg_gold: Dict[str, List[str]] = {}
        for arg in em.get("arguments", []):
            role = arg.get("role", "").lower()
            span = arg.get("text", "")
            if role and span:
                arg_gold.setdefault(role, []).append(span)

        # Emit roles in schema order (guarantees the constructor call is valid)
        ordered_roles = (schema_roles or {}).get(class_name, list(arg_gold.keys()))
        arg_strs: List[str] = [f"mention={mention!r}"]
        for role in ordered_roles:
            spans = arg_gold.get(role, [])
            arg_strs.append(f"{role}={spans!r}")

        parts.append(f"{class_name}({', '.join(arg_strs)})")

    return "[" + ", ".join(parts) + "]"


# ══════════════════════════════════════════════════════════════════════════════
# AEC Four-Agent Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_retrieval_agent(schema_def: str, k: int, model: str) -> str:
    """Agent 1 — Generate k exemplar sentences for the schema."""
    system = "You are a helpful example generator for event extraction."
    user = (
        f"Given the following event type definition:\n\n{schema_def}\n\n"
        f"Generate {k} fluent English sentences. Each sentence must:\n"
        f"1. Contain a clear trigger word or phrase for this event type.\n"
        f"2. Mention entities filling as many argument roles as possible.\n"
        f"3. Be realistic and varied.\n\n"
        f"Output exactly one sentence per line, no numbering."
    )
    try:
        return call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
        )
    except Exception:
        return ""


def run_planning_agent(
    text: str,
    schema_def: str,
    exemplars: str,
    task_type: str,
    k: int,
    model: str,
) -> List[Dict[str, Any]]:
    """Agent 2 — Produce ranked trigger-type hypotheses.

    Returns a list of dicts sorted by confidence (descending), each with
    keys: ``trigger``, ``event_type``, ``confidence``, ``rationale``.
    """
    system = (
        "You are an assistant for event extraction. "
        "Given a piece of text and an event type definition (as a Python dataclass), "
        "produce a JSON array of trigger-hypothesis objects.  "
        "Each object must have keys: 'trigger' (exact span from text), "
        "'event_type' (class name), 'confidence' (0-1 float), 'rationale' (string)."
    )
    exemplar_block = ""
    if exemplars:
        exemplar_block = f"\nExample sentences for context:\n{exemplars}\n"

    user = (
        f"Event definition:\n{schema_def}\n"
        f"{exemplar_block}\n"
        f"Text:\n{text}\n\n"
        f"Identify up to {k} candidate trigger spans. "
        f"Output only a JSON array, no explanation."
    )
    try:
        raw = call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
        )
        # Extract JSON array from the response
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else []
        if isinstance(data, list):
            data.sort(key=lambda x: float(x.get("confidence", 0)), reverse=True)
            return data[:k]
    except Exception:
        pass
    # Fallback: use the event type name as a heuristic trigger
    return [{"trigger": text.split()[0], "event_type": "", "confidence": 0.5, "rationale": "fallback"}]


def run_coding_agent(
    hypothesis: Dict[str, Any],
    schema_def: str,
    text: str,
    task_type: str,
    exemplars: str,
    patch_feedback: Optional[str],
    model: str,
) -> str:
    """Agent 3 — Generate Python instantiation code for the event."""
    header = TASK_HEADERS[task_type]
    footer = TASK_FOOTERS[task_type]

    exemplar_block = ""
    if exemplars:
        exemplar_block = (
            "# Exemplar sentences:\n"
            + "\n".join(f"# {l}" for l in exemplars.splitlines() if l.strip())
            + "\n\n"
        )

    trigger = hypothesis.get("trigger", "")
    event_type = hypothesis.get("event_type", "")

    # For EAE, fill in the trigger/event_name placeholders
    footer_filled = footer.format(trigger=trigger, event_name=event_type) if "{trigger}" in footer else footer

    user_prompt = (
        f"{exemplar_block}"
        f"{header}\n\n"
        f"{schema_def}\n\n"
        f"# This is the text to analyze\n"
        f'text = "{text}"\n\n'
        f'# Hint: trigger word is "{trigger}"\n'
        f"{footer_filled}"
    )

    system_prompt = (
        "You are a coding agent for event extraction. "
        "Complete the Python assignment `result = ` with a list containing "
        "exactly one instantiation of the event class defined above. "
        "Extract argument spans verbatim from the text; use [] for absent roles. "
        "Output ONLY the Python list expression (e.g. [ClassName(mention=..., role=[...])]), "
        "nothing else — no markdown, no explanation."
    )
    if patch_feedback:
        system_prompt += (
            f"\n\nThe previous attempt failed. Fix this error:\n{patch_feedback}"
        )

    raw = call_llm(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
    )
    # Strip code fences
    raw = re.sub(r"^```(?:python)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())
    return raw.strip()


def run_aec_pipeline(
    text: str,
    schema_def: str,
    class_name: str,
    task_type: str,
    class_globals: dict,
    model: str,
    k: int = 3,
    t: int = 3,
    verifier: Optional[VerificationAgent] = None,
) -> Tuple[str, bool]:
    """Run the full four-agent AEC pipeline (Algorithm 1).

    Returns
    -------
    (prediction_str, success)
        ``prediction_str`` is the final code string (may be ``"[]"`` if all
        hypotheses fail).  ``success`` is True if verification passed.
    """
    if verifier is None:
        verifier = VerificationAgent()

    # Step 1 — Retrieval
    exemplars = run_retrieval_agent(schema_def, k=k, model=model)

    # Step 2 — Planning
    hypotheses = run_planning_agent(
        text=text,
        schema_def=schema_def,
        exemplars=exemplars,
        task_type=task_type,
        k=k,
        model=model,
    )

    # Step 3 — Dual-loop refinement (Algorithm 1)
    for hyp in hypotheses:
        patch_feedback: Optional[str] = None
        for attempt in range(1, t + 1):
            # Coding agent
            code_str = run_coding_agent(
                hypothesis=hyp,
                schema_def=schema_def,
                text=text,
                task_type=task_type,
                exemplars=exemplars,
                patch_feedback=patch_feedback,
                model=model,
            )
            # Verification agent
            try:
                verifier.verify_code(code_str, text, class_globals)
                return code_str, True          # success
            except VerificationError as ve:
                patch_feedback = str(ve)       # feed diagnostic ε to next attempt
        # All t attempts failed for this hypothesis → try next

    return "[]", False


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(predictions_path: str) -> None:
    """Evaluate a predictions file using events_scorer.py."""
    eval_dir = str(_here / "utils" / "code_evaluation")
    defs_dir = str(SCHEMA_DEF_DIR)
    for d in (eval_dir, defs_dir):
        if d not in sys.path:
            sys.path.insert(0, d)

    with open(predictions_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not data:
        print("Empty predictions file — nothing to evaluate.")
        return

    # Import event class definitions so safe_eval can instantiate them
    try:
        import all_ee_definitions  # noqa: F401
    except ImportError:
        print(
            "WARNING: Could not import all_ee_definitions.\n"
            f"  Expected at: {eval_dir}/all_ee_definitions.py\n"
            "  Skipping automatic evaluation."
        )
        return

    from events_scorer import micro_e2e_scores, micro_ed_scores, micro_eae_scores

    task_type = data[0].get("task_type", "E2E").upper()
    print(f"\n{'='*60}")
    print(f"Evaluating {predictions_path}  (task={task_type})")
    print("=" * 60)

    if task_type == "ED":
        scores = micro_ed_scores(data)
    elif task_type == "EAE":
        scores = micro_eae_scores(data)
    else:
        scores = micro_e2e_scores(data)

    scores_path = predictions_path.replace(".json", "_scores.json")
    scalar_scores = {k: round(v * 100, 2) for k, v in scores.items() if not isinstance(v, list)}
    with open(scores_path, "w", encoding="utf-8") as fh:
        json.dump(scalar_scores, fh, indent=2)
    print(f"Scores (×100) saved → {scores_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AEC inference on a TextEE dataset split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", default=None,
                   choices=list(DATASET_TASK.keys()),
                   help="Dataset name (e.g. ace05-en, casie, fewevent). Required unless --eval_only.")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"],
                   help="Which split to evaluate on (default: test).")
    p.add_argument("--textee_dir", default=str(TEXTEE_DIR),
                   help="Root directory of TextEE processed data.")
    p.add_argument("--model", default="gpt-4o",
                   help="LLM to use (gpt-4o, gpt-3.5-turbo, or any OpenAI-compatible model).")
    p.add_argument("--base_url", default=None,
                   help="Base URL for OpenAI-compatible local servers (e.g. vLLM). "
                        "Leave unset for the official OpenAI API.")
    p.add_argument("--k", type=int, default=3,
                   help="Number of planning hypotheses (paper default: 3).")
    p.add_argument("--t", type=int, default=3,
                   help="Max inner-loop patch attempts per hypothesis (paper default: 3).")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit evaluation to the first N samples (for debugging).")
    p.add_argument("--output", default=None,
                   help="Output JSON file path. Defaults to outputs/<dataset>_<model>_predictions.json")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds to sleep between API calls (avoids rate limits).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from an existing output file (skip already-processed samples).")
    p.add_argument("--no_eval", action="store_true",
                   help="Skip automatic evaluation after inference.")
    p.add_argument("--eval_only", default=None, metavar="FILE",
                   help="Skip inference; only evaluate an existing predictions JSON file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── eval-only mode ───────────────────────────────────────────────────────
    if args.eval_only:
        if not Path(args.eval_only).exists():
            print(f"ERROR: file not found: {args.eval_only}")
            sys.exit(1)
        run_evaluation(args.eval_only)
        return

    if not args.dataset:
        print("ERROR: --dataset is required unless --eval_only is set.")
        sys.exit(1)

    # ── resolve output path ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    output_path = args.output or str(
        OUTPUT_DIR / f"{args.dataset}_{model_tag}_predictions.json"
    )

    # ── configure custom base_url for local models ──────────────────────────
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    # ── load test data ───────────────────────────────────────────────────────
    split_file = Path(args.textee_dir) / args.dataset / "split1" / f"{args.split}.json"
    if not split_file.exists():
        split_file = Path(args.textee_dir) / args.dataset / f"{args.split}.json"
    if not split_file.exists():
        print(
            f"ERROR: Could not find split file at:\n"
            f"  {split_file}\n\n"
            f"Download TextEE data to data/raw/TextEE/{args.dataset}/split1/{args.split}.json\n"
            f"See https://github.com/ej0cl6/TextEE for instructions."
        )
        sys.exit(1)

    raw_text = split_file.read_text(encoding="utf-8")
    lines = [l for l in raw_text.splitlines() if l.strip()]
    try:
        raw_data = [json.loads(l) for l in lines]
    except json.JSONDecodeError:
        raw_data = json.loads(raw_text)  # fallback: JSON array format

    if args.max_samples:
        raw_data = raw_data[: args.max_samples]
    print(f"Loaded {len(raw_data)} samples from {split_file}")

    # ── load schemas and event class definitions ─────────────────────────────
    task_type = DATASET_TASK[args.dataset]
    try:
        schema_defs  = load_schema_definitions(args.dataset)
        schema_roles = load_schema_roles(args.dataset)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    try:
        class_globals = import_event_classes(args.dataset)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    verifier = VerificationAgent(
        check_trigger_in_text=True,
        check_args_in_text=True,
        check_schema_roles=False,  # role-set check done at eval time by scorer
    )

    # ── resume: skip already-processed samples ───────────────────────────────
    predictions: List[Dict[str, Any]] = []
    if args.resume and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as fh:
            predictions = json.load(fh)
        print(f"Resuming from {len(predictions)} existing predictions.")
    skip = len(predictions)

    # ── run inference ────────────────────────────────────────────────────────
    n_success = 0

    for idx, sample in enumerate(
        tqdm(raw_data, desc=f"AEC [{args.dataset}]"), start=0
    ):
        if idx < skip:
            continue  # already done

        text: str = sample.get("text", "")
        event_mentions: List[dict] = sample.get("event_mentions", [])
        gold_label = build_gold_label(event_mentions, args.dataset, schema_roles)

        # Group mentions by event type; run one pipeline call per type
        seen_types: set = set()
        sample_predictions: List[str] = []

        for em in event_mentions:
            raw_type = em.get("event_type", "")
            class_name = clean_event_name(raw_type, args.dataset).split("(")[0].strip()

            if class_name in seen_types:
                continue
            seen_types.add(class_name)

            # Case-insensitive schema lookup
            schema_def = schema_defs.get(class_name) or next(
                (v for k, v in schema_defs.items() if k.lower() == class_name.lower()),
                None,
            )
            if not schema_def:
                continue

            pred_code, success = run_aec_pipeline(
                text=text,
                schema_def=schema_def,
                class_name=class_name,
                task_type=task_type,
                class_globals=class_globals,
                model=args.model,
                k=args.k,
                t=args.t,
                verifier=verifier,
            )
            if success:
                n_success += 1
            sample_predictions.append(pred_code)

            if args.delay > 0:
                time.sleep(args.delay)

        # Merge per-type predictions into one list string
        merged = ", ".join(p.strip("[]") for p in sample_predictions if p and p != "[]")
        combined_pred = f"[{merged}]" if merged else "[]"

        predictions.append({
            "Input":      text,
            "Prediction": combined_pred,
            "Label":      gold_label,
            "task_type":  task_type.upper(),
            "doc_id":     sample.get("wnd_id", sample.get("doc_id", "")),
        })

        # Checkpoint every 10 samples
        if len(predictions) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(predictions, fh, indent=2, ensure_ascii=False)

    # ── final save ───────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2, ensure_ascii=False)

    print(f"\nDone. {n_success} verified predictions out of {len(predictions)} samples.")
    print(f"Predictions saved → {output_path}")

    # ── automatic evaluation ─────────────────────────────────────────────────
    if not args.no_eval:
        run_evaluation(output_path)
    else:
        print(f"\nTo evaluate manually:")
        print(f"  cd utils/code_evaluation")
        print(f"  python events_scorer.py --input_file ../../{output_path}")


if __name__ == "__main__":
    main()
