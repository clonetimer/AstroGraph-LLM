"""Application-driven contract for calibration/reference trust.

This module deliberately does not select thresholds or mutate runtime policy.  It
makes the trust state of a reference window explicit so downstream calibration
cannot silently assume that an arbitrary prefix is nominal.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceSelectionMode(str, Enum):
    VERIFIED_NOMINAL = "VERIFIED_NOMINAL"
    PREFIX_ASSUMED_NOMINAL = "PREFIX_ASSUMED_NOMINAL"
    LABEL_FILTERED_RESEARCH = "LABEL_FILTERED_RESEARCH"
    ADAPTIVE_CAUSAL = "ADAPTIVE_CAUSAL"


class ReferenceTrustState(str, Enum):
    VERIFIED_NOMINAL = "VERIFIED_NOMINAL"
    ASSUMED_UNVERIFIED = "ASSUMED_UNVERIFIED"
    CONTAMINATED = "CONTAMINATED"
    INSUFFICIENT = "INSUFFICIENT"


class CalibrationTransferState(str, Enum):
    NOT_ASSESSED = "NOT_ASSESSED"
    PASSED = "PASSED"
    FAILED = "FAILED"


class ReferenceUseDecision(str, Enum):
    QUALIFIED = "QUALIFIED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    REJECTED = "REJECTED"


class TrustedReferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-trusted-reference.v1"
    reference_id: str = Field(min_length=1)
    domain_id: str = Field(min_length=1)
    selection_mode: ReferenceSelectionMode
    trust_state: ReferenceTrustState
    sample_count: int = Field(ge=1)
    start_time: str
    end_time: str
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_overlap_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    generated_with_outcome_labels: bool = False
    transfer_state: CalibrationTransferState = CalibrationTransferState.NOT_ASSESSED
    claim_boundary: str = "Reference trust and deployment calibration transfer are separate claims."

    @model_validator(mode="after")
    def validate_state(self) -> "TrustedReferenceRecord":
        if self.trust_state is ReferenceTrustState.CONTAMINATED:
            if self.label_overlap_fraction is None or self.label_overlap_fraction <= 0.0:
                raise ValueError("CONTAMINATED reference requires positive label_overlap_fraction evidence")
        if self.trust_state is ReferenceTrustState.VERIFIED_NOMINAL:
            if self.label_overlap_fraction is not None and self.label_overlap_fraction > 0.0:
                raise ValueError("VERIFIED_NOMINAL cannot have positive known label overlap")
        if self.selection_mode is ReferenceSelectionMode.LABEL_FILTERED_RESEARCH and not self.generated_with_outcome_labels:
            raise ValueError("LABEL_FILTERED_RESEARCH must declare generated_with_outcome_labels=true")
        return self


class ReferenceQualification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-trusted-reference-qualification.v1"
    reference_id: str
    decision: ReferenceUseDecision
    reasons: tuple[str, ...]
    formal_calibration_eligible: bool


def qualify_reference(
    record: TrustedReferenceRecord,
    *,
    minimum_samples: int = 1000,
    max_known_overlap_fraction: float = 0.0,
    require_transfer_pass: bool = False,
) -> ReferenceQualification:
    reasons: list[str] = []
    rejected = False
    research_only = False

    if record.sample_count < minimum_samples or record.trust_state is ReferenceTrustState.INSUFFICIENT:
        rejected = True
        reasons.append("reference sample support is insufficient")
    if record.trust_state is ReferenceTrustState.CONTAMINATED:
        rejected = True
        reasons.append("reference contains known labeled event contamination")
    if record.label_overlap_fraction is not None and record.label_overlap_fraction > max_known_overlap_fraction:
        rejected = True
        reasons.append("known label overlap exceeds allowed reference contamination")
    if record.trust_state is ReferenceTrustState.ASSUMED_UNVERIFIED or record.label_overlap_fraction is None:
        research_only = True
        reasons.append("nominal status is assumed rather than verified")
    if record.generated_with_outcome_labels:
        research_only = True
        reasons.append("reference construction used outcome labels and is research-only")
    if require_transfer_pass and record.transfer_state is not CalibrationTransferState.PASSED:
        research_only = True
        reasons.append("deployment calibration transfer is not independently passed")

    if rejected:
        decision = ReferenceUseDecision.REJECTED
    elif research_only:
        decision = ReferenceUseDecision.RESEARCH_ONLY
    else:
        decision = ReferenceUseDecision.QUALIFIED
    if not reasons:
        reasons.append("reference meets declared nominal, provenance, support, and transfer requirements")
    return ReferenceQualification(
        reference_id=record.reference_id,
        decision=decision,
        reasons=tuple(reasons),
        formal_calibration_eligible=decision is ReferenceUseDecision.QUALIFIED,
    )
