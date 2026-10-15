"""Fail-closed governance contracts for design, test, and closure documents.

Access decisions in this module are deliberately independent from retrieval
scores and caller-provided prompt text.  A quarantined current-episode truth
document can never be made visible to a runtime diagnosis by adding an access
tag.
"""
from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentType(str, Enum):
    DESIGN_PROPOSAL = "DESIGN_PROPOSAL"
    DESIGN_MANUAL = "DESIGN_MANUAL"
    FMEA = "FMEA"
    TEST_PROCEDURE = "TEST_PROCEDURE"
    TEST_LOG = "TEST_LOG"
    ANOMALY_REPORT = "ANOMALY_REPORT"
    CLOSURE_REPORT = "CLOSURE_REPORT"
    CAMPAIGN_REPORT = "CAMPAIGN_REPORT"


class DocumentVisibilityClass(str, Enum):
    DESIGN_PUBLIC = "DESIGN_PUBLIC"
    OBSERVATION_CAUSAL = "OBSERVATION_CAUSAL"
    RESOLVED_CASE_TRAIN_ONLY = "RESOLVED_CASE_TRAIN_ONLY"
    TRUTH_QUARANTINED = "TRUTH_QUARANTINED"


class DocumentReviewState(str, Enum):
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    MACHINE_VALIDATED = "MACHINE_VALIDATED"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    REJECTED = "REJECTED"


class DocumentAuthority(str, Enum):
    NON_NORMATIVE = "NON_NORMATIVE"
    ENGINEERING_REFERENCE = "ENGINEERING_REFERENCE"
    APPROVED_SOURCE = "APPROVED_SOURCE"


class DocumentAccessMode(str, Enum):
    RUNTIME_DIAGNOSIS = "RUNTIME_DIAGNOSIS"
    OFFLINE_EVALUATION = "OFFLINE_EVALUATION"
    CORPUS_BUILD = "CORPUS_BUILD"


class ClaimEpistemicStatus(str, Enum):
    SOURCE_FACT = "SOURCE_FACT"
    DESIGN_PROPOSAL = "DESIGN_PROPOSAL"
    ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    ONTOLOGY_ASSOCIATION = "ONTOLOGY_ASSOCIATION"
    HISTORICAL_RESOLUTION = "HISTORICAL_RESOLUTION"
    ILLUSTRATIVE = "ILLUSTRATIVE"


class ClaimTemporalScope(str, Enum):
    TIME_INDEPENDENT = "TIME_INDEPENDENT"
    TIME_BOUNDED = "TIME_BOUNDED"


class DocumentEvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_locator: str = Field(min_length=1)
    source_chunk_id: str | None = None
    source_fragment_id: str | None = None
    source_visibility_class: DocumentVisibilityClass | None = None
    source_provenance_group_id: str | None = None
    epistemic_status: ClaimEpistemicStatus
    source_review_state: DocumentReviewState
    source_authority: DocumentAuthority
    source_generator_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    synthetic_from_simulation: bool = False
    temporal_scope: ClaimTemporalScope
    valid_time_start_s: float | None = None
    valid_time_end_s: float | None = None
    observable_cutoff_time_s: float | None = None
    unit: str | None = None
    applicability: list[str] = Field(default_factory=list)
    source_span_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_eligible: bool = False
    source_runtime_vote_allowed: bool = False
    runtime_evidence_eligible: bool = False

    @field_validator("applicability")
    @classmethod
    def applicability_is_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("applicability must contain unique nonempty values")
        return value

    @model_validator(mode="after")
    def provenance_and_authority_are_coherent(self) -> "DocumentEvidenceClaim":
        if self.synthetic_from_simulation and self.source_generator_sha256 is None:
            raise ValueError("synthetic claims require source_generator_sha256")
        if (
            self.valid_time_start_s is not None
            and self.valid_time_end_s is not None
            and self.valid_time_start_s > self.valid_time_end_s
        ):
            raise ValueError("claim valid_time_start_s must not exceed valid_time_end_s")
        if self.temporal_scope == ClaimTemporalScope.TIME_BOUNDED:
            if self.valid_time_end_s is None or self.observable_cutoff_time_s is None:
                raise ValueError("time-bounded claims require end time and observable cutoff")
            if self.valid_time_end_s > self.observable_cutoff_time_s:
                raise ValueError("claim contains information after its observable cutoff")
        elif any(
            value is not None
            for value in (
                self.valid_time_start_s,
                self.valid_time_end_s,
                self.observable_cutoff_time_s,
            )
        ):
            raise ValueError("time-independent claims cannot carry an observation time range")

        non_publishable = {
            ClaimEpistemicStatus.DESIGN_PROPOSAL,
            ClaimEpistemicStatus.ENGINEERING_ESTIMATE,
            ClaimEpistemicStatus.MODEL_INFERENCE,
            ClaimEpistemicStatus.ILLUSTRATIVE,
        }
        if self.publication_eligible and (
            self.source_review_state != DocumentReviewState.HUMAN_REVIEWED
            or self.epistemic_status in non_publishable
        ):
            raise ValueError("publishable claims require human review and an admissible epistemic status")
        if self.epistemic_status in {
            *non_publishable,
            ClaimEpistemicStatus.ONTOLOGY_ASSOCIATION,
        } and self.runtime_evidence_eligible:
            raise ValueError(
                f"{self.epistemic_status.value} claims cannot be runtime evidence"
            )
        if self.runtime_evidence_eligible and (
            self.source_review_state != DocumentReviewState.HUMAN_REVIEWED
            or not self.source_runtime_vote_allowed
            or not self.publication_eligible
            or self.epistemic_status != ClaimEpistemicStatus.SOURCE_FACT
        ):
            raise ValueError(
                "runtime evidence claims require reviewed, published source facts with voting authority"
            )
        return self


class GovernedDocumentDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-governed-document-descriptor.v1"
    document_id: str = Field(min_length=1)
    document_type: DocumentType
    visibility_class: DocumentVisibilityClass
    source_uri: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_state: DocumentReviewState = DocumentReviewState.REVIEW_REQUIRED
    authority: DocumentAuthority = DocumentAuthority.NON_NORMATIVE
    provenance_group_id: str | None = None
    episode_id: str | None = None
    valid_time_start_s: float | None = None
    valid_time_end_s: float | None = None
    observable_cutoff_time_s: float | None = None
    template_version: str | None = None
    license_id: str | None = None
    usage_boundary: str | None = None
    generator_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    contains_root_cause: bool = False
    contains_episode_truth: bool = False
    synthetic_from_simulation: bool = False
    runtime_vote_allowed: bool = False

    @model_validator(mode="after")
    def visibility_contract_is_coherent(self) -> "GovernedDocumentDescriptor":
        if (
            self.valid_time_start_s is not None
            and self.valid_time_end_s is not None
            and self.valid_time_start_s > self.valid_time_end_s
        ):
            raise ValueError("valid_time_start_s must not exceed valid_time_end_s")
        if self.visibility_class == DocumentVisibilityClass.OBSERVATION_CAUSAL:
            if self.episode_id is None or self.observable_cutoff_time_s is None:
                raise ValueError(
                    "causal observation documents require episode_id and observable_cutoff_time_s"
                )
            if (
                self.valid_time_end_s is not None
                and self.valid_time_end_s > self.observable_cutoff_time_s
            ):
                raise ValueError("document contains observations after its causal cutoff")
        if self.visibility_class == DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY:
            if self.provenance_group_id is None or not self.contains_root_cause:
                raise ValueError(
                    "resolved historical cases require provenance_group_id and root cause content"
                )
        if self.contains_episode_truth and self.visibility_class not in {
            DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
            DocumentVisibilityClass.TRUTH_QUARANTINED,
        }:
            raise ValueError(
                "episode truth must be a governed historical resolution or quarantined"
            )
        if self.visibility_class == DocumentVisibilityClass.TRUTH_QUARANTINED and self.runtime_vote_allowed:
            raise ValueError("quarantined truth can never vote at runtime")
        if self.review_state != DocumentReviewState.HUMAN_REVIEWED and self.runtime_vote_allowed:
            raise ValueError("only human-reviewed documents may allow runtime evidence")
        return self


def bind_document_evidence_claim(
    *,
    claim_id: str,
    subject: str,
    predicate: str,
    object: str,
    source_span: str,
    source_locator: str,
    source_chunk_id: str | None = None,
    source_fragment_id: str | None = None,
    descriptor: GovernedDocumentDescriptor,
    epistemic_status: ClaimEpistemicStatus,
    unit: str | None = None,
    applicability: list[str] | None = None,
    publication_eligible: bool = False,
    runtime_evidence_eligible: bool = False,
) -> DocumentEvidenceClaim:
    """Bind a claim to a governed source without dropping provenance or time."""

    if not source_span.strip():
        raise ValueError("document claim source_span must be nonempty")
    time_bounded = descriptor.visibility_class == DocumentVisibilityClass.OBSERVATION_CAUSAL
    return DocumentEvidenceClaim(
        claim_id=claim_id,
        subject=subject,
        predicate=predicate,
        object=object,
        source_document_id=descriptor.document_id,
        source_document_sha256=descriptor.source_sha256,
        source_locator=source_locator,
        source_chunk_id=source_chunk_id,
        source_fragment_id=source_fragment_id,
        source_visibility_class=descriptor.visibility_class,
        source_provenance_group_id=descriptor.provenance_group_id,
        epistemic_status=epistemic_status,
        source_review_state=descriptor.review_state,
        source_authority=descriptor.authority,
        source_generator_sha256=descriptor.generator_sha256,
        synthetic_from_simulation=descriptor.synthetic_from_simulation,
        temporal_scope=(
            ClaimTemporalScope.TIME_BOUNDED
            if time_bounded
            else ClaimTemporalScope.TIME_INDEPENDENT
        ),
        valid_time_start_s=descriptor.valid_time_start_s if time_bounded else None,
        valid_time_end_s=descriptor.valid_time_end_s if time_bounded else None,
        observable_cutoff_time_s=(
            descriptor.observable_cutoff_time_s if time_bounded else None
        ),
        unit=unit,
        applicability=applicability or [],
        source_span_sha256=hashlib.sha256(source_span.encode("utf-8")).hexdigest(),
        publication_eligible=publication_eligible,
        source_runtime_vote_allowed=descriptor.runtime_vote_allowed,
        runtime_evidence_eligible=runtime_evidence_eligible,
    )


class DocumentAccessContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: DocumentAccessMode = DocumentAccessMode.RUNTIME_DIAGNOSIS
    episode_id: str | None = None
    current_provenance_group_id: str | None = None
    observation_time_s: float | None = None
    allowed_historical_provenance_group_ids: list[str] = Field(default_factory=list)

    @field_validator("allowed_historical_provenance_group_ids")
    @classmethod
    def groups_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("historical provenance groups must be unique and nonempty")
        return value


class DocumentAccessDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    reason_code: str = Field(min_length=1)


def evaluate_document_access(
    descriptor: GovernedDocumentDescriptor,
    context: DocumentAccessContext,
) -> DocumentAccessDecision:
    """Apply non-bypassable document visibility and causal-time gates."""

    if context.mode == DocumentAccessMode.CORPUS_BUILD:
        return DocumentAccessDecision(allowed=True, reason_code="CORPUS_BUILD_ALLOWED")

    if descriptor.review_state in {
        DocumentReviewState.REVIEW_REQUIRED,
        DocumentReviewState.REJECTED,
    }:
        return DocumentAccessDecision(allowed=False, reason_code="DOCUMENT_NOT_REVIEWED")

    visibility = descriptor.visibility_class
    if visibility == DocumentVisibilityClass.TRUTH_QUARANTINED:
        if context.mode == DocumentAccessMode.OFFLINE_EVALUATION:
            return DocumentAccessDecision(allowed=True, reason_code="EVALUATOR_TRUTH_ACCESS")
        return DocumentAccessDecision(allowed=False, reason_code="TRUTH_QUARANTINED")

    if visibility == DocumentVisibilityClass.DESIGN_PUBLIC:
        if descriptor.review_state != DocumentReviewState.HUMAN_REVIEWED:
            return DocumentAccessDecision(
                allowed=False,
                reason_code="DESIGN_HUMAN_REVIEW_REQUIRED",
            )
        return DocumentAccessDecision(allowed=True, reason_code="DESIGN_PUBLIC_ALLOWED")

    if visibility == DocumentVisibilityClass.OBSERVATION_CAUSAL:
        if context.episode_id is None or descriptor.episode_id != context.episode_id:
            return DocumentAccessDecision(allowed=False, reason_code="EPISODE_MISMATCH")
        if context.observation_time_s is None:
            return DocumentAccessDecision(allowed=False, reason_code="OBSERVATION_TIME_REQUIRED")
        cutoff = descriptor.observable_cutoff_time_s
        if cutoff is None or cutoff > context.observation_time_s:
            return DocumentAccessDecision(allowed=False, reason_code="FUTURE_OBSERVATION_BLOCKED")
        return DocumentAccessDecision(allowed=True, reason_code="CAUSAL_OBSERVATION_ALLOWED")

    if visibility == DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY:
        if descriptor.review_state != DocumentReviewState.HUMAN_REVIEWED:
            return DocumentAccessDecision(
                allowed=False,
                reason_code="HISTORICAL_HUMAN_REVIEW_REQUIRED",
            )
        group_id = descriptor.provenance_group_id
        if group_id is None:
            return DocumentAccessDecision(allowed=False, reason_code="HISTORICAL_GROUP_MISSING")
        if group_id == context.current_provenance_group_id:
            return DocumentAccessDecision(allowed=False, reason_code="CURRENT_GROUP_CLOSURE_BLOCKED")
        if group_id not in set(context.allowed_historical_provenance_group_ids):
            return DocumentAccessDecision(allowed=False, reason_code="HISTORICAL_GROUP_NOT_ALLOWED")
        return DocumentAccessDecision(allowed=True, reason_code="HISTORICAL_CASE_ALLOWED")

    return DocumentAccessDecision(allowed=False, reason_code="UNKNOWN_VISIBILITY_CLASS")


__all__ = [
    "ClaimEpistemicStatus",
    "ClaimTemporalScope",
    "DocumentAccessContext",
    "DocumentAccessDecision",
    "DocumentAccessMode",
    "DocumentAuthority",
    "DocumentEvidenceClaim",
    "DocumentReviewState",
    "DocumentType",
    "DocumentVisibilityClass",
    "GovernedDocumentDescriptor",
    "bind_document_evidence_claim",
    "evaluate_document_access",
]
