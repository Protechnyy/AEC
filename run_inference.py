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
import random
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

from AEC.retrieval_agent import RetrievalAgent
from AEC.planning_agent import PlanningAgent
from AEC.coding_agent import CodingAgent
from AEC.verification_agent import VerificationAgent, VerificationError

# ── paths ──────────────────────────────────────────────────────────────────
TEXTEE_DIR = _here / "data" / "raw" / "TextEE"
SCHEMA_DEF_DIR = _here / "utils" / "code_schema_generation" / "python_event_defs"
INIT_PROMPTS_DIR = _here / "utils" / "code_schema_generation" / "init_prompts"
OUTPUT_DIR = _here / "outputs"

# ── dataset → default sample size (paper §4.1) ──────────────────────────────
# "For CASIE, due to its smaller size, we sample 50 test instances."
DATASET_SAMPLE_SIZE: Dict[str, int] = {
    "casie": 50,
}

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
        # Normalize hyphenated field names to underscores so the LLM sees
        # valid Python identifiers (e.g. "issues-addressed" → "issues_addressed").
        normalized_lines = []
        for line in block.splitlines():
            if line.startswith("class "):
                cls_name = line.split("(")[0].replace("class", "").strip().rstrip(":")
                normalized_lines.append(line)
            elif ":" in line and not line.strip().startswith(("#", "@", "def")):
                # Field definition line — replace hyphens with underscores
                # in the field name portion (before the colon).
                field_part, rest = line.split(":", 1)
                field_part = field_part.replace("-", "_")
                normalized_lines.append(f"{field_part}:{rest}")
            else:
                normalized_lines.append(line)
        block = "\n".join(normalized_lines)
        if cls_name:
            schemas[cls_name] = block
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
            role = s.split(":")[0].strip().replace("-", "_")
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

        # Collect gold argument spans grouped by (lowercased) role name.
        # Replace hyphens with underscores so the label is valid Python
        # (e.g. "issues-addressed" → "issues_addressed").
        arg_gold: Dict[str, List[str]] = {}
        for arg in em.get("arguments", []):
            role = arg.get("role", "").lower().replace("-", "_")
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
# AEC Four-Agent Pipeline (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════

def run_aec_pipeline(
    text: str,
    schema_def: str,
    class_name: str,
    task_type: str,
    class_globals: dict,
    model: str,
    k: int = 3,
    t: int = 3,
    retriever: Optional[RetrievalAgent] = None,
    planner: Optional[PlanningAgent] = None,
    coder: Optional[CodingAgent] = None,
    verifier: Optional[VerificationAgent] = None,
) -> Tuple[str, bool]:
    """Run the full four-agent AEC pipeline (Algorithm 1).

    Returns
    -------
    (prediction_str, success)
        ``prediction_str`` is the final code string (may be ``"[]"`` if all
        hypotheses fail).  ``success`` is True if verification passed.
    """
    retriever = retriever or RetrievalAgent()
    planner = planner or PlanningAgent()
    coder = coder or CodingAgent()
    verifier = verifier or VerificationAgent()

    # Step 1 — Retrieval Agent
    exemplars = retriever.retrieve(schema_def, k=k, model=model)

    # Step 2 — Planning Agent
    hypotheses = planner.generate_hypotheses(
        text=text,
        schema_definition=schema_def,
        exemplars=exemplars,
        k=k,
        model=model,
    )

    # Step 3 — Dual-loop refinement (Algorithm 1)
    for hyp in hypotheses:
        patch_feedback: Optional[str] = None
        for attempt in range(1, t + 1):
            # Coding Agent
            code_str = coder.generate_code(
                hypothesis=hyp,
                schema_definition=schema_def,
                text=text,
                task_type=task_type,
                exemplars=exemplars,
                patch_feedback=patch_feedback,
                model=model,
            )
            # Verification Agent
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
        # Try JSONL (one JSON object per line)
        raw_data = [json.loads(l) for l in lines]
    except json.JSONDecodeError:
        try:
            # Try single JSON array
            raw_data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Concatenated pretty-printed JSON objects (TextEE default format):
            # parse by streaming through the text with JSONDecoder.
            raw_data = []
            decoder = json.JSONDecoder()
            text_stripped = raw_text.strip()
            pos = 0
            while pos < len(text_stripped):
                obj, end = decoder.raw_decode(text_stripped, pos)
                raw_data.append(obj)
                # Skip whitespace between objects
                pos = end
                while pos < len(text_stripped) and text_stripped[pos] in ' \t\r\n':
                    pos += 1

    # Apply dataset-specific sample size (paper §4.1) or user override
    sample_size = args.max_samples or DATASET_SAMPLE_SIZE.get(args.dataset)
    if sample_size and sample_size < len(raw_data):
        rng = random.Random(42)
        raw_data = rng.sample(raw_data, sample_size)
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

    # ── instantiate agents ────────────────────────────────────────────────────
    retriever = RetrievalAgent()
    planner = PlanningAgent()
    coder = CodingAgent()
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
                retriever=retriever,
                planner=planner,
                coder=coder,
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
