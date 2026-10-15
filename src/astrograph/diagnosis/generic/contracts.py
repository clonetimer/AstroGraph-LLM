from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def stable_id(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20].upper()}"


class DataQualityStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


class HeadCalibrationStatus(str, Enum):
    CALIBRATED = "CALIBRATED"
    DETERMINISTIC_CONTRACT = "DETERMINISTIC_CONTRACT"
    NOT_CALIBRATED = "NOT_CALIBRATED"


class HeadExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    NOT_EVALUATED = "NOT_EVALUATED"


class AnomalyDecision(str, Enum):
    NORMAL = "NORMAL"
    ANOMALOUS = "ANOMALOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DATA_INVALID = "DATA_INVALID"


class OpenSetDecision(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    IN_DISTRIBUTION = "IN_DISTRIBUTION"
    OOD = "OOD"


class EffectPattern(str, Enum):
    CALIBRATED_NO_DEVIATION = "CALIBRATED_NO_DEVIATION"
    PERSISTENT_BALANCE_DEVIATION = "PERSISTENT_BALANCE_DEVIATION"
    RAW_POINTWISE_TOLERANCE_EXCEEDANCE = "RAW_POINTWISE_TOLERANCE_EXCEEDANCE"


class FailureMechanism(str, Enum):
    BIAS = "BIAS"
    DRIFT = "DRIFT"
    GAIN_LOSS = "GAIN_LOSS"
    STUCK = "STUCK"
    SATURATION = "SATURATION"
    FRICTION_INCREASE = "FRICTION_INCREASE"
    CAPACITY_LOSS = "CAPACITY_LOSS"
    IMPEDANCE_INCREASE = "IMPEDANCE_INCREASE"
    LEAKAGE = "LEAKAGE"
    INTERMITTENT = "INTERMITTENT"
    UNMODELED_BALANCE_DISTURBANCE = "UNMODELED_BALANCE_DISTURBANCE"
    UNKNOWN = "UNKNOWN"


class MechanismDecision(str, Enum):
    NO_ANOMALY = "NO_ANOMALY"
    CANDIDATES = "CANDIDATES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_EVALUATED = "NOT_EVALUATED"


class LocalizationRole(str, Enum):
    AFFECTED_COMPONENT = "AFFECTED_COMPONENT"
    ROOT_CAUSE = "ROOT_CAUSE"


class LocalizationDecision(str, Enum):
    NO_ANOMALY = "NO_ANOMALY"
    AFFECTED_COMPONENT_IDENTIFIED = "AFFECTED_COMPONENT_IDENTIFIED"
    ROOT_CAUSE_IDENTIFIED = "ROOT_CAUSE_IDENTIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_EVALUATED = "NOT_EVALUATED"


class HypothesisStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReasoningDecision(str, Enum):
    NO_ANOMALY = "NO_ANOMALY"
    HYPOTHESES = "HYPOTHESES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_EVALUATED = "NOT_EVALUATED"


class GenericDisposition(str, Enum):
    NORMAL = "NORMAL"
    DIAGNOSED = "DIAGNOSED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DATA_INVALID = "DATA_INVALID"
    OOD = "OOD"


class DataQualityContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DataQualityStatus
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def invalid_requires_reason(self) -> "DataQualityContext":
        if self.status == DataQualityStatus.INVALID and not self.reason_codes:
            raise ValueError("invalid data quality requires at least one reason code")
        return self


class AssetTopologyContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(min_length=1)
    asset_instance_id: str = Field(min_length=1)
    component_instance_id: str = Field(min_length=1)
    component_archetype_id: str = Field(min_length=1)
    asset_path: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def path_matches_declared_instances(self) -> "AssetTopologyContext":
        if self.asset_path[0] != self.asset_instance_id:
            raise ValueError("asset path must start at asset_instance_id")
        if self.asset_path[-1] != self.component_instance_id:
            raise ValueError("asset path must end at component_instance_id")
        if len(self.asset_path) != len(set(self.asset_path)):
            raise ValueError("asset path cannot contain a cycle")
        return self


class RootCauseEvidenceContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    boundary_inputs_observed: bool
    state_channel_integrity_verified: bool
    input_channel_integrity_verified: bool
    time_alignment_verified: bool
    independent_repetitions: int = Field(ge=0)
    evidence_ids: list[str]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False

    @property
    def supports_root_cause(self) -> bool:
        return (
            self.boundary_inputs_observed
            and self.state_channel_integrity_verified
            and self.input_channel_integrity_verified
            and self.time_alignment_verified
            and self.independent_repetitions >= 2
            and bool(self.evidence_ids)
        )


class HeadReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    head_id: str = Field(min_length=1)
    head_version: str = Field(min_length=1)
    calibration_status: HeadCalibrationStatus
    calibration_id: str = "NONE"
    calibration_version: str = "NONE"
    calibration_sha256: str = Field(
        default="NONE", pattern=r"^(?:NONE|[0-9a-f]{64})$"
    )
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False


class HeadEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class AnomalyHeadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-generic-anomaly-head.v1"] = (
        "astrograph-generic-anomaly-head.v1"
    )
    anomaly_result_id: str
    episode_id: str
    decision: AnomalyDecision
    anomaly_score: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    raw_violation_count: int = Field(ge=0)
    raw_violation_fraction: float = Field(ge=0.0, le=1.0)
    evidence: list[HeadEvidence]
    receipt: HeadReceipt

    @model_validator(mode="after")
    def calibrated_decision_matches_score(self) -> "AnomalyHeadResult":
        if self.decision == AnomalyDecision.ANOMALOUS and self.anomaly_score < self.decision_threshold:
            raise ValueError("anomalous decision must meet the declared threshold")
        if self.decision == AnomalyDecision.NORMAL and self.anomaly_score >= self.decision_threshold:
            raise ValueError("normal decision cannot meet the declared anomaly threshold")
        return self


class MechanismCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: FailureMechanism
    screening_score: float = Field(ge=0.0, le=1.0)
    calibrated: bool
    evidence_ids: list[str]


class MechanismHeadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-generic-mechanism-head.v1"] = (
        "astrograph-generic-mechanism-head.v1"
    )
    mechanism_result_id: str
    episode_id: str
    execution_status: HeadExecutionStatus
    decision: MechanismDecision
    open_set_decision: OpenSetDecision = OpenSetDecision.NOT_EVALUATED
    effect_patterns: list[EffectPattern]
    candidates: list[MechanismCandidate]
    evidence: list[HeadEvidence]
    receipt: HeadReceipt


class LocalizationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_path: list[str]
    component_instance_id: str
    role: LocalizationRole
    score: float = Field(ge=0.0, le=1.0)
    root_cause_supported: bool
    evidence_ids: list[str]

    @model_validator(mode="after")
    def role_matches_claim(self) -> "LocalizationCandidate":
        if self.role == LocalizationRole.ROOT_CAUSE and not self.root_cause_supported:
            raise ValueError("root-cause localization requires root-cause support")
        if self.role == LocalizationRole.AFFECTED_COMPONENT and self.root_cause_supported:
            raise ValueError("affected-component evidence cannot claim root-cause support")
        return self


class LocalizationHeadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-generic-localization-head.v1"] = (
        "astrograph-generic-localization-head.v1"
    )
    localization_result_id: str
    episode_id: str
    execution_status: HeadExecutionStatus
    decision: LocalizationDecision
    candidates: list[LocalizationCandidate]
    evidence: list[HeadEvidence]
    receipt: HeadReceipt


class DiagnosticHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    mechanism_id: FailureMechanism
    localization_path: list[str]
    status: HypothesisStatus
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    missing_requirements: list[str]
    falsification_checks: list[str]

    @model_validator(mode="after")
    def supported_requires_complete_evidence(self) -> "DiagnosticHypothesis":
        if self.status == HypothesisStatus.SUPPORTED and self.missing_requirements:
            raise ValueError("supported hypothesis cannot retain missing requirements")
        return self


class ReasoningHeadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-generic-reasoning-head.v1"] = (
        "astrograph-generic-reasoning-head.v1"
    )
    reasoning_result_id: str
    episode_id: str
    execution_status: HeadExecutionStatus
    decision: ReasoningDecision
    hypotheses: list[DiagnosticHypothesis]
    receipt: HeadReceipt


class GenericDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-generic-decision.v1"] = (
        "astrograph-generic-decision.v1"
    )
    decision_id: str
    episode_id: str
    disposition: GenericDisposition
    accepted_hypothesis_id: str | None = None
    reason_codes: list[str]
    unresolved_requirements: list[str]
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False
    formal_release_allowed: Literal[False] = False
    claim_boundary: str = (
        "Generic deterministic technical output only. Affected-component evidence is not root-cause "
        "localization, screening scores are not calibrated probabilities, and G3/G4 are not implied."
    )

    @model_validator(mode="after")
    def accepted_hypothesis_matches_disposition(self) -> "GenericDecisionRecord":
        requires_hypothesis = self.disposition == GenericDisposition.DIAGNOSED
        if requires_hypothesis and self.accepted_hypothesis_id is None:
            raise ValueError("accepted diagnosis requires a hypothesis ID")
        if not requires_hypothesis and self.accepted_hypothesis_id is not None:
            raise ValueError("non-accepting disposition cannot retain a hypothesis ID")
        return self


class GenericDiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-generic-diagnosis.v1"] = (
        "astrograph-generic-diagnosis.v1"
    )
    episode_id: str
    anomaly: AnomalyHeadResult
    mechanism: MechanismHeadResult
    localization: LocalizationHeadResult
    reasoning: ReasoningHeadResult
    decision: GenericDecisionRecord


class GenericHeadBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    head_id: str = Field(min_length=1)
    head_version: str = Field(min_length=1)
    calibration_status: HeadCalibrationStatus
    formal_output_allowed: bool


class GenericHeadManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-generic-head-manifest.v1"]
    pipeline_id: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)
    bindings: list[GenericHeadBinding] = Field(min_length=4)
    open_set_ood_calibrated: bool
    known_mechanism_classification_calibrated: bool
    root_cause_localization_calibrated: bool
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ordered_bindings(self) -> "GenericHeadManifest":
        sequences = [item.sequence for item in self.bindings]
        head_ids = [item.head_id for item in self.bindings]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("generic head sequence must be contiguous and ordered")
        if len(head_ids) != len(set(head_ids)):
            raise ValueError("generic head IDs must be unique")
        return self


__all__ = [
    "AnomalyDecision",
    "AnomalyHeadResult",
    "AssetTopologyContext",
    "DataQualityContext",
    "DataQualityStatus",
    "DiagnosticHypothesis",
    "EffectPattern",
    "FailureMechanism",
    "GenericDecisionRecord",
    "GenericDiagnosisResult",
    "GenericDisposition",
    "GenericHeadBinding",
    "GenericHeadManifest",
    "HeadCalibrationStatus",
    "HeadEvidence",
    "HeadExecutionStatus",
    "HeadReceipt",
    "HypothesisStatus",
    "LocalizationCandidate",
    "LocalizationDecision",
    "LocalizationHeadResult",
    "LocalizationRole",
    "MechanismCandidate",
    "MechanismDecision",
    "MechanismHeadResult",
    "OpenSetDecision",
    "ReasoningDecision",
    "ReasoningHeadResult",
    "RootCauseEvidenceContext",
    "stable_id",
]
