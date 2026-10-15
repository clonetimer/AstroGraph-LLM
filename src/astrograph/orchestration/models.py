from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PermissionScope(str, Enum):
    READ_DATA = "READ_DATA"
    WRITE_DATA = "WRITE_DATA"
    READ_KNOWLEDGE = "READ_KNOWLEDGE"
    GOVERN_KNOWLEDGE = "GOVERN_KNOWLEDGE"
    RUN_DIAGNOSIS = "RUN_DIAGNOSIS"
    WRITE_AUDIT = "WRITE_AUDIT"
    READ_STATE = "READ_STATE"
    WRITE_STATE = "WRITE_STATE"
    READ_ARTIFACTS = "READ_ARTIFACTS"
    WRITE_ARTIFACTS = "WRITE_ARTIFACTS"
    RUN_EXPERIMENT = "RUN_EXPERIMENT"
    RENDER_REPORT = "RENDER_REPORT"
    READ_MEMORY = "READ_MEMORY"
    WRITE_MEMORY = "WRITE_MEMORY"
    REVIEW_MEMORY = "REVIEW_MEMORY"
    MANAGE_PLUGINS = "MANAGE_PLUGINS"


class SideEffectLevel(str, Enum):
    NONE = "NONE"
    READ_ONLY = "READ_ONLY"
    WRITE_ARTIFACT = "WRITE_ARTIFACT"
    MUTATE_STATE = "MUTATE_STATE"
    EXTERNAL_IO = "EXTERNAL_IO"


class VersionPins(BaseModel):
    """Version pins carried by orchestration envelopes.

    New payloads should use protocol_* field names. Historical contract_* keys are
    accepted as aliases only so archived API/data payloads remain readable.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    astrograph_version: str = "0.5.0"
    policy_version: str = ""
    knowledge_version: str = ""
    data_protocol_version: str = Field(
        default="",
        validation_alias=AliasChoices("data_protocol_version", "dataset_contract_version"),
    )
    fault_protocol_version: str = Field(
        default="",
        validation_alias=AliasChoices("fault_protocol_version", "fault_contract_version"),
    )
    interface_protocol_version: str = Field(
        default="",
        validation_alias=AliasChoices("interface_protocol_version", "contract_version"),
    )
    tool_versions: dict[str, str] = Field(default_factory=dict)


class ExecutionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic_only: bool = True
    allow_llm: bool = False
    allow_unregistered_tools: bool = False
    timeout_s: float = Field(default=60.0, gt=0.0)
    retry_count: int = Field(default=0, ge=0)
    fail_closed: bool = True

    @field_validator("allow_unregistered_tools")
    @classmethod
    def forbid_unregistered_tools(cls, value: bool) -> bool:
        if value:
            raise ValueError("unregistered tools are not allowed in AstroGraph V0.5.0 orchestration")
        return value


class WorkflowContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    actor: str = "system"
    created_at: str = Field(default_factory=_now)
    permissions: list[PermissionScope] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_permissions(self, required: list[PermissionScope]) -> bool:
        owned = set(self.permissions)
        return set(required).issubset(owned)


class ProtocolReference(BaseModel):
    """Reference to a versioned protocol, with read-only compatibility aliases."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    protocol_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("protocol_id", "contract_id"),
    )
    protocol_version: str = Field(
        default="",
        validation_alias=AliasChoices("protocol_version", "contract_version"),
    )
    protocol_sha256: str = Field(
        default="",
        validation_alias=AliasChoices("protocol_sha256", "contract_sha256"),
    )


class TaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = "astrograph-task-envelope.v1"
    task_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    input_refs: dict[str, str] = Field(default_factory=dict)
    protocol: ProtocolReference | None = Field(
        default=None,
        validation_alias=AliasChoices("protocol", "contract"),
    )
    version_pins: VersionPins = Field(default_factory=VersionPins)
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    permission_scope: list[PermissionScope] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-execution-result.v1"
    task_id: str
    workflow_id: str
    status: ExecutionStatus
    output_payload: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    input_hash: str = ""
    output_hash: str = ""
    tool_versions: dict[str, str] = Field(default_factory=dict)
    started_at: str = Field(default_factory=_now)
    finished_at: str | None = None
    error: ExecutionError | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def succeeded(
        cls,
        *,
        task_id: str,
        workflow_id: str,
        output_payload: dict[str, Any],
        tool_versions: dict[str, str] | None = None,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ExecutionResult":
        return cls(
            task_id=task_id,
            workflow_id=workflow_id,
            status=ExecutionStatus.SUCCEEDED,
            output_payload=output_payload,
            tool_versions=tool_versions or {},
            finished_at=_now(),
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        *,
        task_id: str,
        workflow_id: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> "ExecutionResult":
        return cls(
            task_id=task_id,
            workflow_id=workflow_id,
            status=ExecutionStatus.FAILED,
            error=ExecutionError(code=code, message=message, details=details or {}),
            finished_at=_now(),
            warnings=warnings or [],
        )
