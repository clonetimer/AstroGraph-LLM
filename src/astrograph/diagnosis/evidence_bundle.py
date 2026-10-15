from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SHA256_PATTERN = r"^[0-9a-fA-F]{64}$"


class DiagnosisReleaseStatus(str, Enum):
    DETERMINATE_DIAGNOSIS = "DETERMINATE_DIAGNOSIS"
    CONDITIONAL_DIAGNOSIS = "CONDITIONAL_DIAGNOSIS"
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"


class ModelEvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    prediction_id: str = Field(min_length=1)
    prediction_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    probability_ranking_id: str = Field(min_length=1)
    top_fault_id: str | None = None
    calibrated: bool = False
    claim_boundary: str = (
        "Model evidence is a ranking input. It cannot publish a determinate diagnosis without KG evidence, "
        "supported capability axes, and a passing policy gate."
    )


class KGEvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_id: str = Field(min_length=1)
    knowledge_version: str = Field(min_length=1)
    nodes_sha256: str = Field(pattern=_SHA256_PATTERN)
    edges_sha256: str = Field(pattern=_SHA256_PATTERN)
    query_ids: list[str] = Field(min_length=1)
    query_sha256_values: list[str] = Field(min_length=1)
    read_user: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    graph_backend: str = "neo4j"
    claim_boundary: str = (
        "KG evidence is scoped to the imported release and read-only query templates. It is not flight evidence."
    )

    @field_validator("query_sha256_values")
    @classmethod
    def validate_query_hashes(cls, value: list[str]) -> list[str]:
        invalid = [item for item in value if not re.fullmatch(_SHA256_PATTERN, item)]
        if invalid:
            raise ValueError("query_sha256_values must contain SHA256 hex strings")
        return value


class SafetyReplayReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_id: str = Field(min_length=1)
    replay_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    status: str = Field(min_length=1)
    wrong_automatic_accept_rate: float | None = Field(default=None, ge=0.0)
    claim_boundary: str = (
        "SafetyReplayReference represents an independently executed replay corpus or safety campaign. "
        "A same-request deterministic-core comparison is not a safety replay."
    )


class CoreComparisonReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    reference_core_version: str = Field(min_length=1)
    core_diagnosis_status: str = Field(min_length=1)
    core_predicted_case_id: str | None = None
    accepted_fault_id: str | None = None
    independence_level: Literal[
        "LLM_INDEPENDENT_SHARED_REFERENCE_IMPLEMENTATION",
        "IMPLEMENTATION_INDEPENDENT",
    ] = "LLM_INDEPENDENT_SHARED_REFERENCE_IMPLEMENTATION"
    claim_boundary: str = (
        "Core comparison is a same-request cross-check. Unless independence_level is IMPLEMENTATION_INDEPENDENT, "
        "it must not be represented as an independent safety replay."
    )


class ReleaseQualificationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime_mode: Literal["SHADOW", "QUALIFICATION", "RELEASE"] = "QUALIFICATION"
    formal_release_authorization: Literal["NOT_GRANTED", "GRANTED"] = "NOT_GRANTED"
    authority_id: str | None = None
    claim_boundary: str = (
        "Release context declares runtime mode and authorization state; it does not itself grant authority."
    )


class CapabilityAxesEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported_axes: list[str] = Field(default_factory=list)
    unsupported_axes: list[str] = Field(default_factory=list)
    requested_axes: list[str] = Field(default_factory=list)
    capability_manifest_id: str = Field(min_length=1)
    capability_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_axis_sets(self) -> "CapabilityAxesEvidence":
        supported = set(self.supported_axes)
        unsupported = set(self.unsupported_axes)
        overlap = supported & unsupported
        if overlap:
            raise ValueError(f"capability axes cannot be both supported and unsupported: {sorted(overlap)}")
        return self

    @property
    def requested_unsupported_axes(self) -> list[str]:
        unsupported = set(self.unsupported_axes)
        return sorted(axis for axis in self.requested_axes if axis in unsupported)


class DiagnosisEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-diagnosis-evidence-bundle.v1"
    bundle_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    telemetry_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    model_evidence: ModelEvidenceReference | None = None
    kg_evidence: KGEvidenceReference | None = None
    safety_replay: SafetyReplayReference | None = None
    core_comparison: CoreComparisonReference | None = None
    capability_axes: CapabilityAxesEvidence
    release_context: ReleaseQualificationContext | None = None
    upstream_formal_release_allowed: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "The bundle carries structured evidence for a policy gate. It does not by itself authorize diagnosis release."
    )

    @field_validator("split")
    @classmethod
    def normalize_split(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"TRAIN", "VALIDATION", "TEST", "ONLINE", "UNKNOWN"}:
            raise ValueError("split must be TRAIN, VALIDATION, TEST, ONLINE, or UNKNOWN")
        return normalized


class DiagnosisPolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-diagnosis-policy-decision.v1"
    bundle_id: str
    status: DiagnosisReleaseStatus
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "Policy decision only evaluates evidence completeness and capability boundaries. It is not a clinical, "
        "flight, or operational authority."
    )


class ReleaseQualificationGate:
    gate_id = "release-qualification-gate-v1"
    role = "RELEASE_QUALIFICATION"
    def decide(
        self,
        bundle: DiagnosisEvidenceBundle,
        conflict_flags: list[str] | None = None,
        *,
        blocking_conflicts: list[str] | None = None,
        quality_warnings: list[str] | None = None,
        open_set_actions: list[str] | None = None,
        semantic_gate_blockers: list[str] | None = None,
        calibration_gate_blockers: list[str] | None = None,
        mechanism_gate_blockers: list[str] | None = None,
    ) -> DiagnosisPolicyDecision:
        """Evaluate formal release evidence without conflating diagnostics.

        ``conflict_flags`` is retained for backward compatibility. New callers must
        pass the typed channels so ranking disagreement and KG redundancy remain
        warnings rather than formal evidence contradictions.
        """

        blocking: list[str] = []
        warnings: list[str] = []

        effective_blocking = list(blocking_conflicts or [])
        if blocking_conflicts is None and conflict_flags:
            # Legacy callers historically treated every flag as blocking. Preserve
            # that behavior at the API boundary while migrated internal callers use
            # explicit typed channels.
            effective_blocking.extend(conflict_flags)
        if effective_blocking:
            blocking.append("EVIDENCE_CONFLICT_DETECTED")
            warnings.extend(effective_blocking)

        if quality_warnings:
            warnings.extend(quality_warnings)

        if open_set_actions:
            blocking.append("OPEN_SET_ACTION_REQUIRED")
            warnings.extend(open_set_actions)

        if semantic_gate_blockers:
            blocking.append("SEMANTIC_QUALIFICATION_NOT_SUPPORTED")
            warnings.extend(semantic_gate_blockers)

        if calibration_gate_blockers:
            blocking.append("MODEL_EVIDENCE_CALIBRATION_NOT_SUPPORTED")
            warnings.extend(calibration_gate_blockers)

        if mechanism_gate_blockers:
            blocking.append("MECHANISM_SCOPE_QUALIFICATION_NOT_SUPPORTED")
            warnings.extend(mechanism_gate_blockers)

        if bundle.model_evidence is None:
            blocking.append("MISSING_MODEL_EVIDENCE")
        if bundle.kg_evidence is None:
            blocking.append("MISSING_KG_EVIDENCE")
        if bundle.safety_replay is None:
            blocking.append("MISSING_SAFETY_REPLAY")
            if bundle.core_comparison is not None:
                warnings.append("CORE_COMPARISON_PRESENT_BUT_NOT_SAFETY_REPLAY")
        elif bundle.safety_replay.status.upper() != "PASS":
            blocking.append("SAFETY_REPLAY_NOT_PASS")

        release_context_blocks = bool(
            bundle.release_context is not None
            and (
                bundle.release_context.runtime_mode == "SHADOW"
                or bundle.release_context.formal_release_authorization != "GRANTED"
            )
        )
        if release_context_blocks:
            blocking.append("UPSTREAM_FORMAL_RELEASE_NOT_ALLOWED")
            warnings.append(
                "release_context="
                f"{bundle.release_context.runtime_mode}/{bundle.release_context.formal_release_authorization}"
            )
        elif bundle.upstream_formal_release_allowed is False:
            blocking.append("UPSTREAM_FORMAL_RELEASE_NOT_ALLOWED")

        unsupported_requested = bundle.capability_axes.requested_unsupported_axes
        if unsupported_requested:
            blocking.append("REQUESTED_UNSUPPORTED_CAPABILITY_AXIS")
            warnings.append(f"unsupported_axes={','.join(unsupported_requested)}")

        if bundle.split == "TEST":
            warnings.append("TEST split evidence must not be used for threshold tuning or knowledge edits")

        blocking = list(dict.fromkeys(blocking))
        warnings = list(dict.fromkeys(warnings))

        if not blocking:
            return DiagnosisPolicyDecision(
                bundle_id=bundle.bundle_id,
                status=DiagnosisReleaseStatus.DETERMINATE_DIAGNOSIS,
                warnings=warnings,
            )
        if (
            "REQUESTED_UNSUPPORTED_CAPABILITY_AXIS" in blocking
            or "MISSING_MODEL_EVIDENCE" in blocking
            or "OPEN_SET_ACTION_REQUIRED" in blocking
            or "SEMANTIC_QUALIFICATION_NOT_SUPPORTED" in blocking
            or "MODEL_EVIDENCE_CALIBRATION_NOT_SUPPORTED" in blocking
            or "MECHANISM_SCOPE_QUALIFICATION_NOT_SUPPORTED" in blocking
        ):
            status = DiagnosisReleaseStatus.UNKNOWN
        else:
            status = DiagnosisReleaseStatus.CONDITIONAL_DIAGNOSIS
        return DiagnosisPolicyDecision(
            bundle_id=bundle.bundle_id,
            status=status,
            blocking_reasons=blocking,
            warnings=warnings,
        )


# Backward-compatible public name. New code should use ReleaseQualificationGate.
DiagnosisPolicyGate = ReleaseQualificationGate
