"""Reconstructed R36.2 contract tests from the source handoff."""
import pytest
from pydantic import ValidationError

from astrograph.evidence.quality import (
    EvidenceQualityStatus,
    EvidenceSourceCategory,
    QualityFactor,
    assess_evidence_quality,
)


def q(value: float) -> QualityFactor:
    return QualityFactor(value=value, basis="explicit test basis", source_refs=("test",))


def test_complete_quality_has_multiplicative_confidence():
    result = assess_evidence_quality(
        evidence_id="E1",
        source_category=EvidenceSourceCategory.SIMULATION,
        source_reliability=q(0.5),
        measurement_quality=q(0.8),
        temporal_consistency=q(0.9),
        model_agreement=q(1.0),
    )
    assert result.status == EvidenceQualityStatus.COMPLETE
    assert result.completeness == 1.0
    assert result.overall_confidence == pytest.approx(0.36)


def test_partial_quality_never_fabricates_overall_confidence():
    result = assess_evidence_quality(
        evidence_id="E2",
        source_category=EvidenceSourceCategory.FLIGHT_TELEMETRY,
        source_reliability=q(1.0),
    )
    assert result.status == EvidenceQualityStatus.PARTIAL
    assert result.overall_confidence is None


def test_source_category_does_not_generate_scores():
    result = assess_evidence_quality(
        evidence_id="E3",
        source_category=EvidenceSourceCategory.FLIGHT_TELEMETRY,
    )
    assert result.status == EvidenceQualityStatus.UNASSESSED
    assert result.completeness == 0.0
    assert result.overall_confidence is None


def test_factor_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        q(1.01)
