"""
AEC – Agent-Event-Coder
=======================

A multi-agent framework for zero-shot event extraction that treats event
extraction as a code-generation problem.  Four LLM-based agents collaborate
in a dual-loop refinement algorithm:

1. Retrieval Agent  — generates exemplar sentences for the target event type
2. Planning Agent   — proposes ranked (trigger, event_type) hypotheses
3. Coding Agent     — generates a Python instantiation string for the event
4. Verification Agent — checks T1/T2/T3 constraints; feeds errors back for patching

Paper: "Extracting Events Like Code: A Multi-Agent Programming Framework
for Zero-Shot Event Extraction", AAAI 2026 (arXiv 2511.13118)

To reproduce paper results run ``run_inference.py`` directly::

    cd /path/to/AEC
    OPENAI_API_KEY=EMPTY python run_inference.py \\
        --dataset ace05-en \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --base_url http://localhost:8000/v1 \\
        --k 3 --t 3
"""

from .planning_agent import PlanningAgent, Hypothesis
from .retrieval_agent import RetrievalAgent
from .coding_agent import CodingAgent
from .verification_agent import VerificationAgent, VerificationError
from .ontology import OntologyManager

__all__ = [
    "PlanningAgent",
    "Hypothesis",
    "RetrievalAgent",
    "CodingAgent",
    "VerificationAgent",
    "VerificationError",
    "OntologyManager",
]
