"""DOR-inspired evidence review: retain alternatives, gaps and bounded conclusions."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class HypothesisAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fault_id: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    missing_required_definition_ids: list[str] = Field(default_factory=list)
    verdict: Literal["SUPPORTED_CANDIDATE", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]


class DiagnosticAssessment(BaseModel):
    """An auditable reasoning summary, not an unrestricted chain of thought."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-diagnostic-assessment.v1"] = "astrograph-diagnostic-assessment.v1"
    hypotheses: list[HypothesisAssessment]
    conclusion_status: Literal["SUPPORTED_CANDIDATE", "MODEL_HYPOTHESIS", "ALTERNATIVES", "UNKNOWN"]
    selected_fault_id: str | None = None
    additional_observation_definition_ids: list[str] = Field(default_factory=list)
    causal_identification: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    control_actions_authorized: Literal[False] = False
    bounded_candidate_fault_ids: list[str] = Field(default_factory=list)
    complementary_status: str | None = None


def assess_summary(summary: dict) -> DiagnosticAssessment:
    """Build the canonical evidence matrix before model interpretation."""
    hypotheses = []
    for c in summary["candidates"]:
        verdict = ("CONTRADICTED" if c["contradicting_evidence_ids"] else
                   "SUPPORTED_CANDIDATE" if c["eligible_for_explanation"] else "INSUFFICIENT_EVIDENCE")
        hypotheses.append(HypothesisAssessment(
            fault_id=c["fault_id"], verdict=verdict,
            **{k: c[k] for k in ("supporting_evidence_ids", "contradicting_evidence_ids",
                                "missing_required_definition_ids")}))
    selected = summary["suspected_fault_id"]
    decision = summary.get('diagnostic_decision', {})
    bounded = decision.get('candidate_fault_ids', [])
    status = "SUPPORTED_CANDIDATE" if selected else "ALTERNATIVES" if bounded or summary["status"] == "AMBIGUOUS" else "UNKNOWN"
    if selected and decision.get('status') == 'MODEL_ONLY':
        status = "MODEL_HYPOTHESIS"
    return DiagnosticAssessment(
        hypotheses=hypotheses, conclusion_status=status, selected_fault_id=selected,
        bounded_candidate_fault_ids=bounded, complementary_status=decision.get('status'),
        additional_observation_definition_ids=sorted({d for h in hypotheses for d in h.missing_required_definition_ids}))


def validate_assessment(actual: DiagnosticAssessment, expected: DiagnosticAssessment) -> None:
    """Model review must preserve the entire evidence matrix and uncertainty."""
    if actual.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError("assessment changed canonical evidence, alternatives or authority")
