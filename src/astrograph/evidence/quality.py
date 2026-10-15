"""Evidence-quality assessment independent of evidence existence/provenance.

R36 intentionally does not infer reliability from a source name.
Every scored dimension needs an explicit numeric assessment and an
auditable basis. Overall confidence exists only when all four required
dimensions are present.
"""
from __future__ import annotations

from enum import Enum
from math import prod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class EvidenceSourceCategory(str, Enum):
    FLIGHT_TELEMETRY = "FLIGHT_TELEMETRY"
    GROUND_TEST = "GROUND_TEST"
    SIMULATION = "SIMULATION"
    ENGINEERING_DOCUMENT = "ENGINEERING_DOCUMENT"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    OPERATOR_OBSERVATION = "OPERATOR_OBSERVATION"
    OTHER = "OTHER"


class EvidenceQualityStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNASSESSED = "UNASSESSED"


class QualityFactor(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    value: float = Field(ge=0.0, le=1.0)
    basis: str = Field(min_length=1)
    source_refs: tuple[str, ...] = ()


class EvidenceQualityAssessment(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["astrograph-evidence-quality.v1"] = (
        "astrograph-evidence-quality.v1"
    )

    evidence_id: str = Field(min_length=1)
    source_category: EvidenceSourceCategory

    source_reliability: QualityFactor | None = None
    measurement_quality: QualityFactor | None = None
    temporal_consistency: QualityFactor | None = None
    model_agreement: QualityFactor | None = None

    claim_boundary: str = (
        "Evidence quality is an auditable screening score, not probability "
        "of truth. Source category alone never determines a numeric score."
    )

    @computed_field
    @property
    def completeness(self) -> float:
        factors = (
            self.source_reliability,
            self.measurement_quality,
            self.temporal_consistency,
            self.model_agreement,
        )
        return sum(factor is not None for factor in factors) / 4.0

    @computed_field
    @property
    def status(self) -> EvidenceQualityStatus:
        if self.completeness == 1.0:
            return EvidenceQualityStatus.COMPLETE
        if self.completeness == 0.0:
            return EvidenceQualityStatus.UNASSESSED
        return EvidenceQualityStatus.PARTIAL

    @computed_field
    @property
    def screening_score(self) -> float | None:
        """Heuristic conjunctive score; it has no probability semantics."""
        if self.status != EvidenceQualityStatus.COMPLETE:
            return None

        values = [
            self.source_reliability.value,
            self.measurement_quality.value,
            self.temporal_consistency.value,
            self.model_agreement.value,
        ]
        return float(prod(values))

    @computed_field
    @property
    def overall_confidence(self) -> float | None:
        """Compatibility alias for :attr:`screening_score`.

        New research text and inference receipts must call this a screening
        score.  The alias remains so stored v1 payload consumers do not break.
        """
        return self.screening_score


def assess_evidence_quality(
    *,
    evidence_id: str,
    source_category: EvidenceSourceCategory,
    source_reliability: QualityFactor | None = None,
    measurement_quality: QualityFactor | None = None,
    temporal_consistency: QualityFactor | None = None,
    model_agreement: QualityFactor | None = None,
) -> EvidenceQualityAssessment:
    return EvidenceQualityAssessment(
        evidence_id=evidence_id,
        source_category=source_category,
        source_reliability=source_reliability,
        measurement_quality=measurement_quality,
        temporal_consistency=temporal_consistency,
        model_agreement=model_agreement,
    )


__all__ = [
    "EvidenceSourceCategory",
    "EvidenceQualityStatus",
    "QualityFactor",
    "EvidenceQualityAssessment",
    "assess_evidence_quality",
]
