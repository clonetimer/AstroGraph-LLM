from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContractStatus(str, Enum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"
    SUPERSEDED = "SUPERSEDED"


class GateSeverity(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class ReleaseDecisionStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    BLOCKED = "BLOCKED"


class TaskStepStatus(str, Enum):
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class DataAcceptanceLevel(str, Enum):
    ACCEPTANCE_VALID = "ACCEPTANCE_VALID"
    EFFECT_VALID_RESEARCH_ONLY = "EFFECT_VALID_RESEARCH_ONLY"
    INVALID = "INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class HiddenTruthAccessStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ENFORCED_NO_ACCESS = "ENFORCED_NO_ACCESS"
    VIOLATION = "VIOLATION"
    UNVERIFIED = "UNVERIFIED"


class FrozenArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: str = Field(min_length=1)
    immutable_during_candidate: bool = True


class AcceptanceCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: GateSeverity = GateSeverity.HARD
    expected_status: CheckStatus = CheckStatus.PASS
    evaluator: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


class ResourceBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_candidates: int = Field(default=5, ge=1)
    max_formal_gate_queries: int = Field(default=1, ge=1)
    max_runtime_seconds: int = Field(default=300, ge=1)
    max_retries_per_error: int = Field(default=2, ge=0)
    max_storage_growth_bytes: int = Field(default=268_435_456, ge=0)
    max_sat_agent_files_changed: int = Field(default=0, ge=0)


class ImprovementContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-improvement-contract.v2"
    contract_id: str = Field(min_length=1)
    status: ContractStatus = ContractStatus.FROZEN
    baseline_version: str = Field(min_length=1)
    target_version: str = Field(min_length=1)
    baseline_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_manifest_path: str = Field(min_length=1)
    baseline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_test_count: int = Field(ge=1)
    baseline_test_inventory_path: str = ""
    baseline_test_inventory_sha256: str = ""
    baseline_tree_size_bytes: int = Field(default=0, ge=0)
    objective: str = Field(min_length=1)
    in_scope: list[str] = Field(min_length=1)
    out_of_scope: list[str] = Field(min_length=1)
    sat_agent_allowed_changes: list[str] = Field(default_factory=list)
    required_deliverables: list[str] = Field(min_length=1)
    prohibited_actions: list[str] = Field(min_length=1)
    protected_paths: list[str] = Field(default_factory=list)
    frozen_artifacts: list[FrozenArtifact] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1)
    resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)
    stop_conditions: list[str] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def inventory_hash_pair(self) -> "ImprovementContract":
        if bool(self.baseline_test_inventory_path) != bool(self.baseline_test_inventory_sha256):
            raise ValueError("baseline test inventory path and sha256 must be provided together")
        if self.baseline_test_inventory_sha256 and (
            len(self.baseline_test_inventory_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.baseline_test_inventory_sha256)
        ):
            raise ValueError("invalid baseline test inventory sha256")
        return self

    @field_validator(
        "in_scope", "out_of_scope", "sat_agent_allowed_changes", "required_deliverables",
        "prohibited_actions", "protected_paths", "stop_conditions",
    )
    @classmethod
    def unique_strings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("contract lists must not contain duplicates")
        return value

    @field_validator("frozen_artifacts")
    @classmethod
    def unique_frozen_paths(cls, value: list[FrozenArtifact]) -> list[FrozenArtifact]:
        paths = [item.path for item in value]
        if len(paths) != len(set(paths)):
            raise ValueError("frozen artifact paths must be unique")
        return value

    @field_validator("acceptance_criteria")
    @classmethod
    def unique_criteria(cls, value: list[AcceptanceCriterion]) -> list[AcceptanceCriterion]:
        ids = [item.criterion_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("acceptance criterion ids must be unique")
        return value


class ChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-change-request.v2"
    request_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    changed_paths: list[str] = Field(min_length=1)
    expected_metric_impacts: dict[str, str] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    rollback_plan: str = Field(min_length=1)
    requested_at: str = Field(default_factory=utc_now)

    @field_validator("changed_paths", "risks")
    @classmethod
    def unique_paths(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("change request lists must not contain duplicates")
        return value


class TaskStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: TaskStepStatus = TaskStepStatus.PLANNED
    evidence: list[str] = Field(default_factory=list)
    error_code: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def completed_requires_evidence(self) -> "TaskStep":
        if self.status == TaskStepStatus.COMPLETED and not self.evidence:
            raise ValueError("completed task steps require evidence")
        if self.status in {TaskStepStatus.FAILED, TaskStepStatus.BLOCKED} and not self.error_code:
            raise ValueError("failed or blocked task steps require an error_code")
        return self


class TaskBoard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-task-board.v1"
    board_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    goal: str = Field(min_length=1)
    planned_steps: list[TaskStep] = Field(min_length=1)
    missing_inputs: list[str] = Field(default_factory=list)
    scope_changes: list[str] = Field(default_factory=list)
    deliverability: str = Field(min_length=1)
    updated_at: str = Field(default_factory=utc_now)

    @field_validator("planned_steps")
    @classmethod
    def unique_step_ids(cls, value: list[TaskStep]) -> list[TaskStep]:
        ids = [item.step_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("task step ids must be unique")
        return value


class MaterialPassport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-material-passport.v1"
    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    logical_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    generated_by: str = Field(min_length=1)
    source_artifact_hashes: list[str] = Field(default_factory=list)
    data_acceptance_level: DataAcceptanceLevel = DataAcceptanceLevel.NOT_APPLICABLE
    allowed_uses: list[str] = Field(min_length=1)
    claims: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)

    @field_validator("source_artifact_hashes")
    @classmethod
    def valid_source_hashes(cls, value: list[str]) -> list[str]:
        for item in value:
            if len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item):
                raise ValueError("source artifact hashes must be lowercase sha256 values")
        return value


class GateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str = Field(min_length=1)
    status: CheckStatus
    severity: GateSeverity = GateSeverity.HARD
    observed: str = ""
    expected: str = "PASS"
    evidence_paths: list[str] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    error_code: str = ""


class GateQueryReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-gate-query-receipt.v1"
    receipt_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    query_index: int = Field(ge=1)
    state_id: str = Field(min_length=1)
    reserved_at: str = Field(default_factory=utc_now)


class GateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-gate-evidence.v2"
    gate_run_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    candidate_tree_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_tree_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_id: str = Field(min_length=1)
    evaluator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_query_index: int = Field(ge=1)
    gate_query_receipt_path: str = Field(min_length=1)
    gate_query_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: str
    finished_at: str
    checks: list[GateCheck] = Field(min_length=1)
    evidence_artifacts: dict[str, str] = Field(default_factory=dict)
    tests_collected: int = Field(default=0, ge=0)
    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    tests_skipped: int = Field(default=0, ge=0)
    tests_errors: int = Field(default=0, ge=0)
    tests_timed_out: bool = False
    baseline_test_nodeids_total: int = Field(default=0, ge=0)
    baseline_test_nodeids_missing: list[str] = Field(default_factory=list)
    hidden_truth_access_status: HiddenTruthAccessStatus = HiddenTruthAccessStatus.UNVERIFIED
    hidden_truth_accessed: bool | None = None
    modified_evaluator_paths: list[str] = Field(default_factory=list)
    source_tree_clean: bool
    candidate_storage_growth_bytes: int = 0
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("checks")
    @classmethod
    def unique_check_ids(cls, value: list[GateCheck]) -> list[GateCheck]:
        ids = [item.criterion_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("gate check ids must be unique")
        return value

    @model_validator(mode="after")
    def count_consistency(self) -> "GateEvidence":
        if self.tests_passed + self.tests_failed + self.tests_errors + self.tests_skipped != self.tests_collected:
            raise ValueError("pytest result counts do not add up to tests_collected")
        if self.hidden_truth_accessed is True and self.hidden_truth_access_status != HiddenTruthAccessStatus.VIOLATION:
            raise ValueError("hidden truth access boolean and status disagree")
        return self


class ReleaseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-release-decision.v2"
    decision_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    gate_run_id: str = Field(min_length=1)
    gate_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ReleaseDecisionStatus
    reasons: list[str] = Field(min_length=1)
    blockers: list[str] = Field(default_factory=list)
    requires_human_review: bool
    claim_boundary: str = Field(min_length=1)
    decided_at: str = Field(default_factory=utc_now)


class FailureMemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-failure-memory-entry.v1"
    failure_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    signature: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    failed_approach: str = Field(min_length=1)
    affected_scope: list[str] = Field(min_length=1)
    evidence_paths: list[str] = Field(default_factory=list)
    do_not_repeat: list[str] = Field(min_length=1)
    recoverable: bool
    recommended_next_action: str = Field(min_length=1)
    recorded_at: str = Field(default_factory=utc_now)


class RollbackManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-rollback-manifest.v1"
    rollback_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    baseline_version: str = Field(min_length=1)
    baseline_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_paths: list[str] = Field(min_length=1)
    rollback_command: str = Field(min_length=1)
    verification_command: str = Field(min_length=1)
    verified: bool = False
    created_at: str = Field(default_factory=utc_now)


class ContractValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-contract-validation-report.v2"
    status: CheckStatus
    contract_id: str
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_root: str
    candidate_version: str = ""
    changed_paths: list[str] = Field(default_factory=list)
    out_of_scope_changes: list[str] = Field(default_factory=list)
    missing_deliverables: list[str] = Field(default_factory=list)
    frozen_artifact_errors: list[str] = Field(default_factory=list)
    actual_tree_size_bytes: int = Field(default=0, ge=0)
    storage_growth_bytes: int = 0
    blocking_reasons: list[str] = Field(default_factory=list)
