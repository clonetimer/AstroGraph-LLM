from __future__ import annotations

import pytest

from astrograph.agents.mission.models import AgentNarrative, CitedParagraph
from astrograph.evidence.executable import (
    ComparisonOperator,
    ConjunctiveFeatureCondition,
    EvidenceDefinition,
    EvidenceFactStatus,
    EvidenceRole,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
    FeatureResultValidity,
    classify_feature_result_validity,
    evaluate_evidence,
)
from astrograph.integrations.platform_agent import _validate_narrative
from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeClaim, PlatformKnowledgeExplanation, _validate_explanation,
)


def _feature(*, fid: str = "F1", value: float | None = 2.0, flags: list[str] | None = None, sample_count: int = 4) -> FeatureResult:
    return FeatureResult(
        feature_result_id=f"FR-{fid}",
        feature_definition_id=fid,
        run_id="RUN",
        paired_nominal_run_id="NOM",
        episode_id="EP",
        resolved_field="sensor.current_a",
        component_instance="RW-0",
        value=value,
        unit="A",
        window_start_s=0.0,
        window_end_s=1.0,
        sample_count=sample_count,
        algorithm_version="test.v1",
        input_data_sha256="0" * 64,
        quality_flags=flags or [],
    )


def _definition(*, conjunct: bool = False) -> EvidenceDefinition:
    return EvidenceDefinition(
        evidence_definition_id="EDEF-1",
        target_fault_id="F-RW",
        name="current high",
        feature_definition_id="F1",
        role=EvidenceRole.REQUIRED,
        comparison=ComparisonOperator.GREATER_THAN,
        threshold=1.0,
        threshold_unit="A",
        and_feature_conditions=(
            [ConjunctiveFeatureCondition(
                feature_definition_id="F2",
                comparison=ComparisonOperator.GREATER_THAN,
                threshold=1.0,
                threshold_unit="A",
            )]
            if conjunct else []
        ),
    )


def test_nonfinite_feature_fails_closed_as_invalid_data_before_threshold_comparison() -> None:
    feature = _feature(value=float("nan"), flags=["NONFINITE_INPUT"])
    validity, _ = classify_feature_result_validity(feature)
    assert validity in {FeatureResultValidity.INVALID_QUALITY_FLAG, FeatureResultValidity.INVALID_NONFINITE}
    fact = evaluate_evidence(_definition(), feature)
    assert fact.status == EvidenceFactStatus.INVALID_DATA
    assert fact.observed_value is None
    assert "threshold comparison not performed" in fact.explanation.lower()


def test_zero_sample_feature_cannot_vote_formal_evidence() -> None:
    feature = _feature(value=2.0, sample_count=0)
    fact = evaluate_evidence(_definition(), feature)
    assert fact.status == EvidenceFactStatus.INVALID_DATA
    assert fact.observed_value is None


def test_missing_quality_flag_maps_to_missing_data_not_negative_evidence() -> None:
    feature = _feature(value=2.0, flags=["MISSING_WINDOW_DATA"])
    fact = evaluate_evidence(_definition(), feature)
    assert fact.status == EvidenceFactStatus.MISSING_DATA
    assert fact.observed_value is None


def test_invalid_conjunct_fails_closed_instead_of_becoming_not_satisfied() -> None:
    primary = _feature(fid="F1", value=2.0)
    bad_conjunct = _feature(fid="F2", value=float("nan"), flags=["NONFINITE_INPUT"])
    fact = evaluate_evidence(_definition(conjunct=True), primary, all_feature_results=[primary, bad_conjunct])
    assert fact.status == EvidenceFactStatus.INVALID_DATA
    assert "conjunct F2" in fact.explanation


def test_mixed_evidence_and_knowledge_citations_do_not_disable_numeric_grounding() -> None:
    snapshot = {
        "candidate_fault_id": "ADCS_RW_JAM",
        "evidence": [{
            "evidence_id": "E1",
            "field": "sensor.adcs.rw.motor_current_a_0",
            "observed_value": 0.2,
            "unit": "A",
        }],
    }
    narrative = AgentNarrative(
        candidate_fault_id="ADCS_RW_JAM",
        paragraphs=[CitedParagraph(
            text="Motor current is 900 A; the maintenance note also discusses bearing wear.",
            citation_ids=["E1", "K1"],
        )],
    )
    with pytest.raises(ValueError, match="numerically inconsistent"):
        _validate_narrative(narrative, snapshot=snapshot, visible={"E1", "K1"})


def test_mixed_citation_allows_measurement_when_value_matches_evidence() -> None:
    snapshot = {
        "candidate_fault_id": "ADCS_RW_JAM",
        "evidence": [{"evidence_id": "E1", "observed_value": 0.2, "unit": "A"}],
    }
    narrative = AgentNarrative(
        candidate_fault_id="ADCS_RW_JAM",
        paragraphs=[CitedParagraph(text="Motor current is 0.2 A.", citation_ids=["E1", "K1"])],
    )
    _validate_narrative(narrative, snapshot=snapshot, visible={"E1", "K1"})


def test_mixed_hypothesis_citations_cannot_bypass_evidence_measurement_grounding() -> None:
    snapshot = {
        "candidate_fault_id": "ADCS_RW_JAM",
        "evidence": [{"evidence_id": "E1", "observed_value": 0.2, "unit": "A"}],
    }
    explanation = PlatformKnowledgeExplanation(
        candidate_fault_id="ADCS_RW_JAM",
        abstained=False,
        summary="Frozen evidence remains 0.2 A.",
        claims=[PlatformKnowledgeClaim(
            kind="HYPOTHESIS",
            text="The measured current is 900 A and may indicate wear.",
            citation_ids=["E1", "K1"],
        )],
    )
    with pytest.raises(ValueError, match="numerically inconsistent"):
        _validate_explanation(
            explanation, snapshot,
            [{"knowledge_id": "K1"}], [], [],
        )
