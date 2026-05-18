"""
Agent-Relation-Coder
====================

AEC-style multi-agent framework specialized for relation extraction.

The framework treats relation extraction as code generation over executable
relation schemas.  Four LLM agents collaborate in a dual-loop workflow:

1. RelationRetrievalAgent generates synthetic relation exemplars.
2. RelationPlanningAgent proposes ranked argument-pair hypotheses.
3. RelationCodingAgent emits Python relation-class instantiations.
4. RelationVerificationAgent executes and checks structural/text grounding
   constraints, then feeds diagnostics back for patching.
"""

from .relation_agents import (
    RelationCodingAgent,
    RelationHypothesis,
    RelationPlanningAgent,
    RelationRetrievalAgent,
    RelationVerificationAgent,
    RelationVerificationError,
)
from .relation_schema import Relation, RelationSchema

__all__ = [
    "Relation",
    "RelationSchema",
    "RelationHypothesis",
    "RelationRetrievalAgent",
    "RelationPlanningAgent",
    "RelationCodingAgent",
    "RelationVerificationAgent",
    "RelationVerificationError",
]
