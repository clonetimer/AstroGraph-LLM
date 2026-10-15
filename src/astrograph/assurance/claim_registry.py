"""Fail-closed scientific claim registry for AstroGraph.

The registry binds every reported claim to a dataset evidence profile, project
claim policy, evaluation use, independent statistical unit, immutable artifact
hashes, and an explicit label-access boundary.  It is intentionally independent
of model code so that claim validation can run before metrics are published.

This module does not promote B006 beyond WIP and contains no performance
thresholds.  In particular, ADCS groups 4/9 are permanently post-hoc under the
B006 policy supplied in ``configs/assurance``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .label_ontology_v3 import LabelOntologyRecord

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ZERO_HASH = "0" * 64


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Return a canonical JSON SHA-256 for a JSON-compatible value."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_keys(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}")


def _enum_tuple(enum_cls: type[Enum], values: Iterable[Any]) -> tuple[Enum, ...]:
    return tuple(enum_cls(value) for value in values)


def _normalise_group(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("group identifier must not be blank")
    return text


def _validate_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


class ProjectStatus(str, Enum):
    VALIDATED_STABLE_BASELINE = "VALIDATED_STABLE_BASELINE"
    WIP_NOT_RELEASED = "WIP_NOT_RELEASED"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"
    RELEASED = "RELEASED"


class EvaluationUse(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    POST_HOC_AUDIT = "POST_HOC_AUDIT"
    CONFIRMATORY = "CONFIRMATORY"


class TaskType(str, Enum):
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    TEMPORAL_LOCALIZATION = "TEMPORAL_LOCALIZATION"
    FAULT_IDENTIFICATION = "FAULT_IDENTIFICATION"
    ROOT_CAUSE_ATTRIBUTION = "ROOT_CAUSE_ATTRIBUTION"
    CALIBRATION = "CALIBRATION"
    SELECTIVE_PREDICTION = "SELECTIVE_PREDICTION"
    EXPLANATION_FIDELITY = "EXPLANATION_FIDELITY"


class EvidenceTier(str, Enum):
    S0_INTERNAL_SYNTHETIC = "S0_INTERNAL_SYNTHETIC"
    S1_EXTERNAL_SIMULATION = "S1_EXTERNAL_SIMULATION"
    S2_REAL_TELEMETRY_ANOMALY = "S2_REAL_TELEMETRY_ANOMALY"
    S3_PHYSICAL_TESTBED = "S3_PHYSICAL_TESTBED"


class ClaimScope(str, Enum):
    INTERNAL_METHOD_VALIDITY = "INTERNAL_METHOD_VALIDITY"
    POST_HOC_EVIDENCE = "POST_HOC_EVIDENCE"
    TASK_LEVEL_EXTERNAL_VALIDITY = "TASK_LEVEL_EXTERNAL_VALIDITY"
    SUBSYSTEM_PHYSICAL_VALIDITY = "SUBSYSTEM_PHYSICAL_VALIDITY"
    DOMAIN_EXTERNAL_VALIDITY = "DOMAIN_EXTERNAL_VALIDITY"
    UNIVERSAL_KG_GAIN = "UNIVERSAL_KG_GAIN"
    HIL_VALIDITY = "HIL_VALIDITY"
    FLIGHT_VALIDITY = "FLIGHT_VALIDITY"
    FLIGHT_QUALIFICATION = "FLIGHT_QUALIFICATION"
    SAFETY_QUALIFICATION = "SAFETY_QUALIFICATION"
    G0_G4_STATUS = "G0_G4_STATUS"


class StatisticalUnit(str, Enum):
    TRAJECTORY_GROUP = "TRAJECTORY_GROUP"
    MONTE_CARLO_GROUP = "MONTE_CARLO_GROUP"
    EXPERIMENT_ID = "EXPERIMENT_ID"
    EVENT_CLUSTER = "EVENT_CLUSTER"
    CHANNEL_EVENT_CLUSTER = "CHANNEL_EVENT_CLUSTER"
    SEALED_CAMPAIGN_BLOCK = "SEALED_CAMPAIGN_BLOCK"


class RecordState(str, Enum):
    DRAFT = "DRAFT"
    EVIDENCE_BOUND = "EVIDENCE_BOUND"
    FROZEN = "FROZEN"
    WITHDRAWN = "WITHDRAWN"


class LabelAccess(str, Enum):
    LABEL_FREE_RUNTIME = "LABEL_FREE_RUNTIME"
    EVALUATOR_ONLY = "EVALUATOR_ONLY"
    METADATA_ONLY = "METADATA_ONLY"


class ArtifactRole(str, Enum):
    TELEMETRY = "TELEMETRY"
    MODEL_PACKAGE = "MODEL_PACKAGE"
    TEMPORAL_POLICY = "TEMPORAL_POLICY"
    EVIDENCE_PACKAGE = "EVIDENCE_PACKAGE"
    FUSION_POLICY = "FUSION_POLICY"
    REJECTION_POLICY = "REJECTION_POLICY"
    SCORE_TRACE = "SCORE_TRACE"
    PREDICTIONS = "PREDICTIONS"
    GROUND_TRUTH = "GROUND_TRUTH"
    METRICS = "METRICS"
    STATISTICS_PLAN = "STATISTICS_PLAN"
    RUN_MANIFEST = "RUN_MANIFEST"
    EXPLANATION_TRACE = "EXPLANATION_TRACE"


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: ValidationSeverity
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity.value, "message": self.message}


class ClaimValidationError(ValueError):
    """Raised when a claim violates one or more scientific-integrity rules."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        joined = "; ".join(f"{item.code}: {item.message}" for item in self.issues)
        super().__init__(joined)


@dataclass(frozen=True)
class ProjectClaimPolicy:
    project_version: str
    project_status: ProjectStatus
    permanently_post_hoc_groups: tuple[str, ...]
    globally_frozen_claim_scopes: tuple[ClaimScope, ...]
    qualification_change_allowed: bool = False
    confirmatory_requires_unseen_data: bool = True
    confirmatory_requires_policy_freeze: bool = True
    confirmatory_requires_truth_join_after_prediction_freeze: bool = True
    schema_version: str = "astrograph-project-claim-policy.v1"

    def __post_init__(self) -> None:
        if not self.project_version.strip():
            raise ValueError("project_version must not be blank")
        object.__setattr__(
            self,
            "permanently_post_hoc_groups",
            tuple(_normalise_group(x) for x in self.permanently_post_hoc_groups),
        )
        if len(set(self.permanently_post_hoc_groups)) != len(self.permanently_post_hoc_groups):
            raise ValueError("permanently_post_hoc_groups contains duplicates")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectClaimPolicy":
        allowed = {
            "schema_version", "project_version", "project_status",
            "permanently_post_hoc_groups", "globally_frozen_claim_scopes",
            "qualification_change_allowed", "confirmatory_requires_unseen_data",
            "confirmatory_requires_policy_freeze",
            "confirmatory_requires_truth_join_after_prediction_freeze",
        }
        _strict_keys(data, allowed, "ProjectClaimPolicy")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-project-claim-policy.v1")),
            project_version=str(data["project_version"]),
            project_status=ProjectStatus(data["project_status"]),
            permanently_post_hoc_groups=tuple(data.get("permanently_post_hoc_groups", ())),
            globally_frozen_claim_scopes=tuple(
                ClaimScope(x) for x in data.get("globally_frozen_claim_scopes", ())
            ),
            qualification_change_allowed=bool(data.get("qualification_change_allowed", False)),
            confirmatory_requires_unseen_data=bool(data.get("confirmatory_requires_unseen_data", True)),
            confirmatory_requires_policy_freeze=bool(data.get("confirmatory_requires_policy_freeze", True)),
            confirmatory_requires_truth_join_after_prediction_freeze=bool(
                data.get("confirmatory_requires_truth_join_after_prediction_freeze", True)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_version": self.project_version,
            "project_status": self.project_status.value,
            "permanently_post_hoc_groups": list(self.permanently_post_hoc_groups),
            "globally_frozen_claim_scopes": [x.value for x in self.globally_frozen_claim_scopes],
            "qualification_change_allowed": self.qualification_change_allowed,
            "confirmatory_requires_unseen_data": self.confirmatory_requires_unseen_data,
            "confirmatory_requires_policy_freeze": self.confirmatory_requires_policy_freeze,
            "confirmatory_requires_truth_join_after_prediction_freeze": (
                self.confirmatory_requires_truth_join_after_prediction_freeze
            ),
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class DatasetEvidenceProfile:
    dataset_id: str
    evidence_tier: EvidenceTier
    domain_scope: str
    allowed_tasks: tuple[TaskType, ...]
    allowed_evaluation_uses: tuple[EvaluationUse, ...]
    allowed_claim_scopes: tuple[ClaimScope, ...]
    forbidden_claim_scopes: tuple[ClaimScope, ...]
    allowed_statistical_units: tuple[StatisticalUnit, ...]
    confirmatory_eligible: bool
    previously_accessed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = "astrograph-dataset-evidence-profile.v1"

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be blank")
        if not self.domain_scope.strip():
            raise ValueError("domain_scope must not be blank")
        overlap = set(self.allowed_claim_scopes) & set(self.forbidden_claim_scopes)
        if overlap:
            raise ValueError(f"claim scopes cannot be both allowed and forbidden: {sorted(x.value for x in overlap)}")
        if not self.allowed_tasks:
            raise ValueError("allowed_tasks must not be empty")
        if not self.allowed_statistical_units:
            raise ValueError("allowed_statistical_units must not be empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetEvidenceProfile":
        allowed = {
            "schema_version", "dataset_id", "evidence_tier", "domain_scope",
            "allowed_tasks", "allowed_evaluation_uses", "allowed_claim_scopes",
            "forbidden_claim_scopes", "allowed_statistical_units",
            "confirmatory_eligible", "previously_accessed", "notes",
        }
        _strict_keys(data, allowed, "DatasetEvidenceProfile")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-dataset-evidence-profile.v1")),
            dataset_id=str(data["dataset_id"]),
            evidence_tier=EvidenceTier(data["evidence_tier"]),
            domain_scope=str(data["domain_scope"]),
            allowed_tasks=tuple(TaskType(x) for x in data["allowed_tasks"]),
            allowed_evaluation_uses=tuple(EvaluationUse(x) for x in data["allowed_evaluation_uses"]),
            allowed_claim_scopes=tuple(ClaimScope(x) for x in data["allowed_claim_scopes"]),
            forbidden_claim_scopes=tuple(ClaimScope(x) for x in data.get("forbidden_claim_scopes", ())),
            allowed_statistical_units=tuple(StatisticalUnit(x) for x in data["allowed_statistical_units"]),
            confirmatory_eligible=bool(data["confirmatory_eligible"]),
            previously_accessed=bool(data["previously_accessed"]),
            notes=tuple(str(x) for x in data.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "evidence_tier": self.evidence_tier.value,
            "domain_scope": self.domain_scope,
            "allowed_tasks": [x.value for x in self.allowed_tasks],
            "allowed_evaluation_uses": [x.value for x in self.allowed_evaluation_uses],
            "allowed_claim_scopes": [x.value for x in self.allowed_claim_scopes],
            "forbidden_claim_scopes": [x.value for x in self.forbidden_claim_scopes],
            "allowed_statistical_units": [x.value for x in self.allowed_statistical_units],
            "confirmatory_eligible": self.confirmatory_eligible,
            "previously_accessed": self.previously_accessed,
            "notes": list(self.notes),
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    role: ArtifactRole
    sha256: str
    label_access: LabelAccess
    frozen_before_truth_join: bool
    path_hint: Optional[str] = None
    schema_version: str = "astrograph-claim-artifact-ref.v1"

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be blank")
        _validate_sha256(self.sha256, "ArtifactRef.sha256")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        allowed = {
            "schema_version", "artifact_id", "role", "sha256", "label_access",
            "frozen_before_truth_join", "path_hint",
        }
        _strict_keys(data, allowed, "ArtifactRef")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-claim-artifact-ref.v1")),
            artifact_id=str(data["artifact_id"]),
            role=ArtifactRole(data["role"]),
            sha256=str(data["sha256"]),
            label_access=LabelAccess(data["label_access"]),
            frozen_before_truth_join=bool(data["frozen_before_truth_join"]),
            path_hint=None if data.get("path_hint") is None else str(data["path_hint"]),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "role": self.role.value,
            "sha256": self.sha256,
            "label_access": self.label_access.value,
            "frozen_before_truth_join": self.frozen_before_truth_join,
        }
        if self.path_hint is not None:
            result["path_hint"] = self.path_hint
        return result


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    claim_text: str
    project_version: str
    project_status: ProjectStatus
    project_policy_sha256: str
    dataset_id: str
    dataset_profile_sha256: str
    task_type: TaskType
    evaluation_use: EvaluationUse
    claim_scope: ClaimScope
    statistical_unit: StatisticalUnit
    groups_used: tuple[str, ...]
    parameter_selection_groups: tuple[str, ...]
    policy_frozen_before_evaluation: bool
    truth_join_after_prediction_freeze: bool
    data_previously_used_for_development: bool
    qualification_change_requested: bool
    record_state: RecordState
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    label_ontology_record_sha256: Optional[str] = None
    primary_endpoint: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = "astrograph-claim-record.v1"

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError("claim_id must not be blank")
        if not self.claim_text.strip():
            raise ValueError("claim_text must not be blank")
        _validate_sha256(self.project_policy_sha256, "ClaimRecord.project_policy_sha256")
        _validate_sha256(self.dataset_profile_sha256, "ClaimRecord.dataset_profile_sha256")
        if self.label_ontology_record_sha256 is not None:
            _validate_sha256(self.label_ontology_record_sha256, "ClaimRecord.label_ontology_record_sha256")
        object.__setattr__(self, "groups_used", tuple(_normalise_group(x) for x in self.groups_used))
        object.__setattr__(
            self,
            "parameter_selection_groups",
            tuple(_normalise_group(x) for x in self.parameter_selection_groups),
        )
        if len(set(self.groups_used)) != len(self.groups_used):
            raise ValueError("groups_used contains duplicates")
        if len(set(self.parameter_selection_groups)) != len(self.parameter_selection_groups):
            raise ValueError("parameter_selection_groups contains duplicates")
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact_id values must be unique within a claim")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimRecord":
        allowed = {
            "schema_version", "claim_id", "claim_text", "project_version",
            "project_status", "project_policy_sha256", "dataset_id",
            "dataset_profile_sha256", "task_type", "evaluation_use",
            "claim_scope", "statistical_unit", "groups_used",
            "parameter_selection_groups", "policy_frozen_before_evaluation",
            "truth_join_after_prediction_freeze", "data_previously_used_for_development",
            "qualification_change_requested", "record_state", "artifacts",
            "label_ontology_record_sha256", "primary_endpoint", "notes",
        }
        _strict_keys(data, allowed, "ClaimRecord")
        return cls(
            schema_version=str(data.get("schema_version", "astrograph-claim-record.v1")),
            claim_id=str(data["claim_id"]),
            claim_text=str(data["claim_text"]),
            project_version=str(data["project_version"]),
            project_status=ProjectStatus(data["project_status"]),
            project_policy_sha256=str(data["project_policy_sha256"]),
            dataset_id=str(data["dataset_id"]),
            dataset_profile_sha256=str(data["dataset_profile_sha256"]),
            task_type=TaskType(data["task_type"]),
            evaluation_use=EvaluationUse(data["evaluation_use"]),
            claim_scope=ClaimScope(data["claim_scope"]),
            statistical_unit=StatisticalUnit(data["statistical_unit"]),
            groups_used=tuple(data.get("groups_used", ())),
            parameter_selection_groups=tuple(data.get("parameter_selection_groups", ())),
            policy_frozen_before_evaluation=bool(data["policy_frozen_before_evaluation"]),
            truth_join_after_prediction_freeze=bool(data["truth_join_after_prediction_freeze"]),
            data_previously_used_for_development=bool(data["data_previously_used_for_development"]),
            qualification_change_requested=bool(data.get("qualification_change_requested", False)),
            record_state=RecordState(data["record_state"]),
            artifacts=tuple(ArtifactRef.from_dict(x) for x in data.get("artifacts", ())),
            label_ontology_record_sha256=(
                None if data.get("label_ontology_record_sha256") is None
                else str(data["label_ontology_record_sha256"])
            ),
            primary_endpoint=None if data.get("primary_endpoint") is None else str(data["primary_endpoint"]),
            notes=tuple(str(x) for x in data.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "project_version": self.project_version,
            "project_status": self.project_status.value,
            "project_policy_sha256": self.project_policy_sha256,
            "dataset_id": self.dataset_id,
            "dataset_profile_sha256": self.dataset_profile_sha256,
            "task_type": self.task_type.value,
            "evaluation_use": self.evaluation_use.value,
            "claim_scope": self.claim_scope.value,
            "statistical_unit": self.statistical_unit.value,
            "groups_used": list(self.groups_used),
            "parameter_selection_groups": list(self.parameter_selection_groups),
            "policy_frozen_before_evaluation": self.policy_frozen_before_evaluation,
            "truth_join_after_prediction_freeze": self.truth_join_after_prediction_freeze,
            "data_previously_used_for_development": self.data_previously_used_for_development,
            "qualification_change_requested": self.qualification_change_requested,
            "record_state": self.record_state.value,
            "artifacts": [x.to_dict() for x in self.artifacts],
            "notes": list(self.notes),
        }
        if self.label_ontology_record_sha256 is not None:
            result["label_ontology_record_sha256"] = self.label_ontology_record_sha256
        if self.primary_endpoint is not None:
            result["primary_endpoint"] = self.primary_endpoint
        return result

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


_RUNTIME_ROLES = {
    ArtifactRole.MODEL_PACKAGE,
    ArtifactRole.TEMPORAL_POLICY,
    ArtifactRole.EVIDENCE_PACKAGE,
    ArtifactRole.FUSION_POLICY,
    ArtifactRole.REJECTION_POLICY,
    ArtifactRole.SCORE_TRACE,
    ArtifactRole.PREDICTIONS,
    ArtifactRole.EXPLANATION_TRACE,
}
_EVALUATOR_ROLES = {ArtifactRole.GROUND_TRUTH, ArtifactRole.METRICS}
_PREDICTION_FREEZE_ROLES = {ArtifactRole.SCORE_TRACE, ArtifactRole.PREDICTIONS, ArtifactRole.EXPLANATION_TRACE}


def validate_claim(
    record: ClaimRecord,
    profile: DatasetEvidenceProfile,
    policy: ProjectClaimPolicy,
    label_record: Optional["LabelOntologyRecord"] = None,
) -> tuple[ValidationIssue, ...]:
    """Validate a claim and return all fail-closed issues.

    The function never reads model outcomes.  It validates only frozen metadata,
    hashes, task/evidence boundaries, group provenance, and label-access roles.
    """

    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue(code, ValidationSeverity.ERROR, message))

    def warning(code: str, message: str) -> None:
        issues.append(ValidationIssue(code, ValidationSeverity.WARNING, message))

    if record.project_version != policy.project_version:
        error("PROJECT_VERSION_MISMATCH", "claim project_version does not match the frozen project policy")
    if record.project_status != policy.project_status:
        error("PROJECT_STATUS_MISMATCH", "claim project_status does not match the frozen project policy")
    if record.project_policy_sha256 != policy.sha256:
        error("PROJECT_POLICY_HASH_MISMATCH", "claim does not bind the supplied project policy")
    if record.dataset_id != profile.dataset_id:
        error("DATASET_ID_MISMATCH", "claim dataset_id does not match the supplied dataset profile")
    if record.dataset_profile_sha256 != profile.sha256:
        error("DATASET_PROFILE_HASH_MISMATCH", "claim does not bind the supplied dataset profile")

    if record.task_type not in profile.allowed_tasks:
        error("TASK_OUTSIDE_DATASET_SCOPE", f"{record.task_type.value} is not allowed for {profile.dataset_id}")
    if record.evaluation_use not in profile.allowed_evaluation_uses:
        error("EVALUATION_USE_OUTSIDE_DATASET_SCOPE", f"{record.evaluation_use.value} is not allowed for {profile.dataset_id}")
    if record.claim_scope not in profile.allowed_claim_scopes:
        error("CLAIM_SCOPE_NOT_ALLOWED", f"{record.claim_scope.value} is not allowed for {profile.dataset_id}")
    if record.claim_scope in profile.forbidden_claim_scopes:
        error("CLAIM_SCOPE_EXPLICITLY_FORBIDDEN", f"{record.claim_scope.value} is explicitly forbidden for {profile.dataset_id}")
    if record.claim_scope in policy.globally_frozen_claim_scopes:
        error("CLAIM_SCOPE_GLOBALLY_FROZEN", f"{record.claim_scope.value} is frozen by the B006 project policy")
    if record.statistical_unit not in profile.allowed_statistical_units:
        error("INVALID_INDEPENDENT_UNIT", f"{record.statistical_unit.value} is not an allowed independent unit for {profile.dataset_id}")

    exposed = set(policy.permanently_post_hoc_groups)
    selected = set(record.parameter_selection_groups)
    used = set(record.groups_used)
    if selected & exposed:
        error("POST_HOC_GROUP_USED_FOR_SELECTION", "permanently exposed groups cannot be used for policy or parameter selection")
    if used & exposed and record.evaluation_use is not EvaluationUse.POST_HOC_AUDIT:
        error("EXPOSED_GROUP_MISCLASSIFIED", "a claim using permanently exposed groups must be POST_HOC_AUDIT")
    if record.evaluation_use is EvaluationUse.POST_HOC_AUDIT and record.claim_scope is not ClaimScope.POST_HOC_EVIDENCE:
        error("POST_HOC_SCOPE_REQUIRED", "post-hoc results must use the POST_HOC_EVIDENCE claim scope")

    if record.qualification_change_requested and not policy.qualification_change_allowed:
        error("QUALIFICATION_CHANGE_FORBIDDEN", "the current project policy forbids qualification-state changes")

    if record.evaluation_use is EvaluationUse.CONFIRMATORY:
        if record.record_state is not RecordState.FROZEN:
            error("CONFIRMATORY_RECORD_NOT_FROZEN", "confirmatory claims require a frozen claim record")
        if not profile.confirmatory_eligible:
            error("DATASET_NOT_CONFIRMATORY_ELIGIBLE", "dataset profile is not currently eligible for confirmatory use")
        if policy.confirmatory_requires_unseen_data and (
            record.data_previously_used_for_development or profile.previously_accessed
        ):
            error("CONFIRMATORY_DATA_PREVIOUSLY_ACCESSED", "confirmatory evidence must be untouched before policy freeze")
        if record.parameter_selection_groups:
            error("CONFIRMATORY_SELECTION_GROUPS_PRESENT", "confirmatory records cannot contain parameter-selection groups")
        if policy.confirmatory_requires_policy_freeze and not record.policy_frozen_before_evaluation:
            error("CONFIRMATORY_POLICY_NOT_FROZEN", "policy must be frozen before confirmatory evaluation")
        if (
            policy.confirmatory_requires_truth_join_after_prediction_freeze
            and not record.truth_join_after_prediction_freeze
        ):
            error("CONFIRMATORY_LABEL_ISOLATION_UNPROVEN", "truth must be joined only after prediction artifacts are frozen")
        if used & exposed:
            error("EXPOSED_GROUP_CANNOT_BE_CONFIRMATORY", "permanently post-hoc groups cannot support confirmatory claims")

    role_counts: dict[ArtifactRole, int] = {}
    for artifact in record.artifacts:
        role_counts[artifact.role] = role_counts.get(artifact.role, 0) + 1
        if artifact.role in _RUNTIME_ROLES and artifact.label_access is not LabelAccess.LABEL_FREE_RUNTIME:
            error("RUNTIME_ARTIFACT_LABEL_ACCESS", f"{artifact.artifact_id} is runtime-facing but not label-free")
        if artifact.role in _EVALUATOR_ROLES and artifact.label_access is not LabelAccess.EVALUATOR_ONLY:
            error("EVALUATOR_ARTIFACT_ACCESS", f"{artifact.artifact_id} must be evaluator-only")
        if artifact.role in _PREDICTION_FREEZE_ROLES and not artifact.frozen_before_truth_join:
            error("PREDICTION_NOT_FROZEN_BEFORE_TRUTH", f"{artifact.artifact_id} must be frozen before truth join")

    if record.record_state in {RecordState.EVIDENCE_BOUND, RecordState.FROZEN}:
        required = {ArtifactRole.PREDICTIONS, ArtifactRole.GROUND_TRUTH, ArtifactRole.METRICS, ArtifactRole.RUN_MANIFEST}
        if record.task_type is TaskType.TEMPORAL_LOCALIZATION:
            required.add(ArtifactRole.SCORE_TRACE)
        missing = sorted(role.value for role in required if role_counts.get(role, 0) == 0)
        if missing:
            error("MISSING_EVIDENCE_ARTIFACTS", f"evidence-bound claim is missing required roles: {missing}")
        if not record.truth_join_after_prediction_freeze:
            error("TRUTH_JOIN_ORDER_NOT_ENFORCED", "evidence-bound claims require truth join after prediction freeze")

    label_tasks = {
        TaskType.TEMPORAL_LOCALIZATION,
        TaskType.FAULT_IDENTIFICATION,
        TaskType.ROOT_CAUSE_ATTRIBUTION,
    }
    if record.task_type in label_tasks:
        if record.label_ontology_record_sha256 is None:
            error("LABEL_ONTOLOGY_BINDING_REQUIRED", f"{record.task_type.value} requires a label ontology record")
        elif label_record is None:
            error("LABEL_ONTOLOGY_RECORD_NOT_SUPPLIED", "bound label ontology record was not supplied to the validator")
        else:
            if record.label_ontology_record_sha256 != label_record.sha256:
                error("LABEL_ONTOLOGY_HASH_MISMATCH", "claim label ontology hash does not match the supplied record")
            support = label_record.support_for_task(record.task_type.value)
            for message in support.errors:
                error("LABEL_ONTOLOGY_TASK_UNSUPPORTED", message)
            for message in support.warnings:
                warning("LABEL_ONTOLOGY_WARNING", message)

    if not record.groups_used and record.record_state is not RecordState.DRAFT:
        error("INDEPENDENT_GROUPS_REQUIRED", "evidence-bound and frozen claims must enumerate independent groups")
    if record.evaluation_use is EvaluationUse.DEVELOPMENT and not record.parameter_selection_groups:
        warning("NO_SELECTION_GROUPS_RECORDED", "development record has no explicit parameter-selection groups")

    return tuple(issues)


def require_valid_claim(
    record: ClaimRecord,
    profile: DatasetEvidenceProfile,
    policy: ProjectClaimPolicy,
    label_record: Optional["LabelOntologyRecord"] = None,
) -> tuple[ValidationIssue, ...]:
    issues = validate_claim(record, profile, policy, label_record)
    errors = [item for item in issues if item.severity is ValidationSeverity.ERROR]
    if errors:
        raise ClaimValidationError(errors)
    return issues


@dataclass(frozen=True)
class LedgerVerification:
    valid: bool
    entry_count: int
    final_entry_sha256: str
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "entry_count": self.entry_count,
            "final_entry_sha256": self.final_entry_sha256,
            "issues": list(self.issues),
        }


def verify_claim_ledger(path: Path | str) -> LedgerVerification:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return LedgerVerification(True, 0, _ZERO_HASH, ())
    issues: list[str] = []
    previous = _ZERO_HASH
    count = 0
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                issues.append(f"line {line_number}: blank lines are not permitted")
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                issues.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            expected_fields = {
                "schema_version", "sequence", "previous_entry_sha256",
                "record", "record_sha256", "entry_sha256",
            }
            if set(entry) != expected_fields:
                issues.append(f"line {line_number}: unexpected ledger fields")
                continue
            count += 1
            if entry["sequence"] != count:
                issues.append(f"line {line_number}: non-contiguous sequence")
            if entry["previous_entry_sha256"] != previous:
                issues.append(f"line {line_number}: previous-entry hash mismatch")
            computed_record = content_sha256(entry["record"])
            if entry["record_sha256"] != computed_record:
                issues.append(f"line {line_number}: record hash mismatch")
            payload = {key: value for key, value in entry.items() if key != "entry_sha256"}
            computed_entry = content_sha256(payload)
            if entry["entry_sha256"] != computed_entry:
                issues.append(f"line {line_number}: entry hash mismatch")
            previous = entry["entry_sha256"]
    return LedgerVerification(not issues, count, previous, tuple(issues))


def append_claim_record(
    path: Path | str,
    record: ClaimRecord,
    profile: DatasetEvidenceProfile,
    policy: ProjectClaimPolicy,
    label_record: Optional["LabelOntologyRecord"] = None,
) -> str:
    """Validate and append a claim to a hash-chained JSONL ledger.

    Existing bytes are never rewritten.  A corrupted ledger fails closed.
    """

    require_valid_claim(record, profile, policy, label_record)
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    verification = verify_claim_ledger(ledger_path)
    if not verification.valid:
        raise ClaimValidationError(
            [ValidationIssue("LEDGER_INTEGRITY_FAILURE", ValidationSeverity.ERROR, x) for x in verification.issues]
        )
    payload = {
        "schema_version": "astrograph-claim-registry-ledger-entry.v1",
        "sequence": verification.entry_count + 1,
        "previous_entry_sha256": verification.final_entry_sha256,
        "record": record.to_dict(),
        "record_sha256": record.sha256,
    }
    payload["entry_sha256"] = content_sha256(payload)
    encoded = _canonical_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(ledger_path, flags, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    post = verify_claim_ledger(ledger_path)
    if not post.valid or post.final_entry_sha256 != payload["entry_sha256"]:
        raise RuntimeError("claim ledger failed post-append verification")
    return payload["entry_sha256"]


def load_project_policy(path: Path | str) -> ProjectClaimPolicy:
    return ProjectClaimPolicy.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_dataset_profiles(path: Path | str) -> dict[str, DatasetEvidenceProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != {"schema_version", "profiles"}:
        raise ValueError("dataset profile document must contain only schema_version and profiles")
    profiles = [DatasetEvidenceProfile.from_dict(item) for item in raw["profiles"]]
    result = {profile.dataset_id: profile for profile in profiles}
    if len(result) != len(profiles):
        raise ValueError("dataset_id values must be unique")
    return result
