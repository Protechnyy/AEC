"""
Agent-Relation-Coder
====================

AEC-style multi-agent framework specialized for relation extraction.

The framework treats relation extraction as JSON triple generation over
relation schemas. Four LLM agents collaborate in a dual-loop workflow:

1. RelationRetrievalAgent generates synthetic relation exemplars.
2. RelationPlanningAgent proposes ranked subject/object hypotheses.
3. RelationCodingAgent emits JSON triples with subject, object, and relation.
4. RelationVerificationAgent parses and checks structural/text grounding
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
from .relation_schema import RelationSchema

__all__ = [
    "RelationSchema",
    "RelationHypothesis",
    "RelationRetrievalAgent",
    "RelationPlanningAgent",
    "RelationCodingAgent",
    "RelationVerificationAgent",
    "RelationVerificationError",
]
