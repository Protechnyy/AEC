"""Multi-agent pipeline for zero-shot relation extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coding_agent import CodingAgent
from event_schema import RelationObject, RelationSchema
from ontology import build_relation_definitions
from planning_agent import PlanningAgent, RelationHypothesis
from verification_agent import VerificationAgent, VerificationError


@dataclass
class AECPipeline:
    """Relation extraction pipeline using the original AEC orchestration slot."""

    planning_agent: PlanningAgent = field(default_factory=PlanningAgent)
    coding_agent: CodingAgent = field(default_factory=CodingAgent)
    verification_agent: VerificationAgent = field(default_factory=VerificationAgent)
    max_hypotheses: int = 8
    use_llm_plan: bool = False
    use_llm_coding: bool = False
    last_run_trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    last_run_summary: dict[str, Any] = field(default_factory=dict, init=False)

    def run_many(
        self,
        text: str,
        schemas: dict[str, RelationSchema],
        *,
        candidate_pairs: list[dict[str, str]] | None = None,
        use_llm_plan: bool | None = None,
        use_llm_coding: bool | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> list[RelationObject]:
        self.last_run_trace = []
        effective_llm_plan = self.use_llm_plan if use_llm_plan is None else use_llm_plan
        effective_llm_coding = self.use_llm_coding if use_llm_coding is None else use_llm_coding
        definitions = build_relation_definitions(schemas)

        if effective_llm_plan:
            hypotheses = self.planning_agent.generate_hypotheses_with_llm(
                text=text,
                relation_definitions=definitions,
                schemas=schemas,
                candidate_pairs=candidate_pairs,
                max_candidates=self.max_hypotheses,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        else:
            hypotheses = self.planning_agent.generate_hypotheses(
                text=text,
                schemas=schemas,
                candidate_pairs=candidate_pairs,
                max_candidates=self.max_hypotheses,
            )

        validated: list[RelationObject] = []
        seen: set[tuple[str, str, str]] = set()
        verify_pass = 0
        verify_fail = 0
        verifier_categories: dict[str, int] = {}

        for hypothesis in hypotheses:
            hyp_trace = self._hypothesis_trace(hypothesis)
            generated_code = self.coding_agent.generate_relation_code(
                hypothesis,
                text,
                schemas=schemas,
                use_llm_coding=effective_llm_coding,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            hyp_trace["generated_code"] = generated_code
            try:
                relation = self.verification_agent.verify_code(
                    generated_code,
                    schemas,
                    text,
                    confidence=hypothesis.confidence,
                    rationale=hypothesis.rationale,
                )
                verify_pass += 1
                hyp_trace.setdefault("verification", []).append(
                    {"status": "passed", "relation": relation.dict()}
                )
                key = (
                    relation.relation_type.lower(),
                    relation.head.lower(),
                    relation.tail.lower(),
                )
                if key not in seen:
                    seen.add(key)
                    validated.append(relation)
            except VerificationError as exc:
                verify_fail += 1
                payload = exc.to_dict()
                category = str(payload.get("category", "unknown"))
                verifier_categories[category] = verifier_categories.get(category, 0) + 1
                hyp_trace.setdefault("verification", []).append(
                    {"status": "failed", "error": str(exc), "error_info": payload}
                )
            self.last_run_trace.append(hyp_trace)

        self.last_run_summary = {
            "hypothesis_count": len(hypotheses),
            "validated_relation_count": len(validated),
            "verified_pass_count": verify_pass,
            "verified_fail_count": verify_fail,
            "verifier_categories": verifier_categories,
            "candidate_pair_count": len(candidate_pairs or []),
        }
        return validated

    def _hypothesis_trace(self, hypothesis: RelationHypothesis) -> dict[str, Any]:
        return {
            "head": hypothesis.head,
            "tail": hypothesis.tail,
            "relation_type": hypothesis.relation_type,
            "confidence": hypothesis.confidence,
            "rationale": hypothesis.rationale,
        }
