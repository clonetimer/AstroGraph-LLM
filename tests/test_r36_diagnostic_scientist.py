"""Reconstructed R36.6 contract tests from the source handoff."""
from astrograph.agents.controlled_tools import ControlledToolRegistry, RegisteredTool
from astrograph.agents.mission.diagnostic_scientist import (
    DiagnosticHypothesisProposal,
    DiagnosticScientistAgent,
    EvidenceRequest,
    ExperimentEvidence,
    ExperimentPlan,
    ScientistStatus,
)
from astrograph.evidence.quality import EvidenceSourceCategory, QualityFactor, assess_evidence_quality


def complete_quality(evidence_id: str):
    f = QualityFactor(value=1.0, basis="test")
    return assess_evidence_quality(
        evidence_id=evidence_id, source_category=EvidenceSourceCategory.SIMULATION,
        source_reliability=f, measurement_quality=f, temporal_consistency=f, model_agreement=f,
    )


def test_agent_strips_computed_quality_fields_on_tool_reingress():
    registry = ControlledToolRegistry()
    evidence = ExperimentEvidence(
        evidence_id="E1", hypothesis_id="H1", likelihood_ratio=4.0,
        edge_confidence=1.0, source_ref="tool:test", quality=complete_quality("E1"),
    )
    registry.register(
        RegisteredTool(tool_id="experiment", description="test experiment"),
        lambda args: evidence.model_dump(mode="json"),
    )
    result = DiagnosticScientistAgent(registry).run(
        hypotheses=[DiagnosticHypothesisProposal(
            hypothesis_id="H1", fault_id="F1", label="fault", prior_probability=0.2, rationale="test"
        )],
        requests=[EvidenceRequest(
            request_id="R1", hypothesis_id="H1", question="q", required_observation="o"
        )],
        plans=[ExperimentPlan(plan_id="P1", request_id="R1", hypothesis_id="H1", tool_id="experiment")],
    )
    assert result.status == ScientistStatus.COMPLETE
    assert result.accepted_evidence_ids == ("E1",)
    assert result.posteriors[0].posterior_probability > result.posteriors[0].prior_probability


def test_agent_without_evidence_is_incomplete():
    result = DiagnosticScientistAgent(ControlledToolRegistry()).run(
        hypotheses=[DiagnosticHypothesisProposal(
            hypothesis_id="H1", fault_id="F1", label="fault", prior_probability=0.2, rationale="test"
        )],
        requests=[], plans=[],
    )
    assert result.status == ScientistStatus.INCOMPLETE
    assert "NO_QUALIFIED_EXPERIMENT_EVIDENCE" in result.blockers
