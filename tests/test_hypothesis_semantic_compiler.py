from __future__ import annotations

from pathlib import Path

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.investigation import HypothesisSet, HypothesisStatus, HypothesisType
from astrograph.services import (
    DeterministicReferenceReasoner,
    HypothesisSemanticCompiler,
    SafetyValidationService,
    StructuredHypothesisProposalService,
)


def _bundle() -> EpisodeEvidenceBundle:
    return EpisodeEvidenceBundle.model_validate_json(
        Path("tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json")
        .read_text(encoding="utf-8")
    )


def _fault(hypotheses: HypothesisSet, fault_id: str):
    return next(
        item for item in hypotheses.hypotheses
        if item.hypothesis_type == HypothesisType.REGISTERED_FAULT
        and item.target_fault_id == fault_id
    )


class _SemanticDeviationBackend:
    model_id = "semantic-deviation"
    model_version = "1"

    def generate_structured(self, *, agent_id, output_model, payload):
        candidate = HypothesisSet.model_validate(payload["reference_hypothesis_set"])
        bad_fact = next(
            item["evidence_fact_id"]
            for item in payload["evidence_facts"]
            if item["status"] == "NOT_SATISFIED"
        )
        updated = []
        for item in candidate.hypotheses:
            if item.target_fault_id == "THERMAL_HEATER_STUCK_ON":
                item = item.model_copy(update={
                    "status": HypothesisStatus.CONTRADICTED,
                    "supporting_evidence_fact_ids": [bad_fact],
                    "contradicting_evidence_fact_ids": [],
                })
            updated.append(item)
        return candidate.model_copy(update={"hypotheses": updated})


def test_semantic_compiler_restores_canonical_status_and_evidence_membership():
    bundle = _bundle()
    reference = DeterministicReferenceReasoner.build(bundle)
    proposal = _SemanticDeviationBackend().generate_structured(
        agent_id="test", output_model=HypothesisSet,
        payload={
            "reference_hypothesis_set": reference.model_dump(mode="json"),
            "evidence_facts": [item.model_dump(mode="json") for item in bundle.evidence_facts],
        },
    )

    compiled = HypothesisSemanticCompiler().compile(proposal, bundle, reference=reference)
    expected = _fault(reference, "THERMAL_HEATER_STUCK_ON")
    actual = _fault(compiled, "THERMAL_HEATER_STUCK_ON")

    assert actual.status == expected.status == HypothesisStatus.SUPPORTED
    assert actual.supporting_evidence_fact_ids == expected.supporting_evidence_fact_ids
    assert actual.contradicting_evidence_fact_ids == expected.contradicting_evidence_fact_ids
    meta = compiled.validation_metadata["semantic_compiler"]
    assert meta["semantic_override_count"] == 1
    assert meta["formal_status_owned_by_llm"] is False


def test_proposal_service_compiles_semantically_wrong_but_allowlisted_output():
    bundle = _bundle()
    result = StructuredHypothesisProposalService().reason(
        bundle, backend=_SemanticDeviationBackend()
    )
    heater = _fault(result, "THERMAL_HEATER_STUCK_ON")
    assert heater.status == HypothesisStatus.SUPPORTED
    assert all(
        fact_id.startswith("EF-")
        for fact_id in heater.supporting_evidence_fact_ids
    )
    assert result.validation_metadata["semantic_compiler"]["semantic_override_count"] == 1


def test_safety_critic_blocks_uncompiled_semantic_mismatch():
    bundle = _bundle()
    reference = DeterministicReferenceReasoner.build(bundle)
    proposal = _SemanticDeviationBackend().generate_structured(
        agent_id="test", output_model=HypothesisSet,
        payload={
            "reference_hypothesis_set": reference.model_dump(mode="json"),
            "evidence_facts": [item.model_dump(mode="json") for item in bundle.evidence_facts],
        },
    )
    critique = SafetyValidationService().critique(proposal, bundle)
    codes = {item.code for item in critique.findings}
    assert critique.blocking is True
    assert "HYPOTHESIS_SEMANTICS_MISMATCH" in codes
    assert "SUPPORTING_EVIDENCE_NOT_SATISFIED" in codes
