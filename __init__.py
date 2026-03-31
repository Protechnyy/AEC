"""
AEC – Agent-Event-Coder
=======================

A multi-agent framework for zero-shot event extraction that treats event
extraction as a code-generation problem.  Given an input sentence the pipeline
produces an :class:`~aec.event_schema.EventObject` whose trigger and argument
spans are drawn directly from the text.

Quickstart
----------
>>> from AEC.event_schema import EventSchema
>>> from AEC.aec_pipeline import AECPipeline
>>> schema = EventSchema("Attack", {"attacker": str, "victim": str, "weapon": str})
>>> pipeline = AECPipeline()
>>> result = pipeline.run("The soldiers attacked the village with mortars.", schema=schema)
>>> print(result.trigger)
"""

from .event_schema import EventSchema, EventObject
from .planning_agent import PlanningAgent, Hypothesis
from .retrieval_agent import RetrievalAgent
from .coding_agent import CodingAgent
from .verification_agent import VerificationAgent, VerificationError
from .ontology import OntologyManager
from .aec_pipeline import AECPipeline

__all__ = [
    "EventSchema",
    "EventObject",
    "PlanningAgent",
    "Hypothesis",
    "RetrievalAgent",
    "CodingAgent",
    "VerificationAgent",
    "VerificationError",
    "OntologyManager",
    "AECPipeline",
]
