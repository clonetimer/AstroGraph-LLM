from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from astrograph.contracts import STABLE_CONTRACTS, StableContractBundle
from .artifacts import RuntimeArtifactStore, RuntimeStageArtifact
from .execution_control import CancellationToken, RuntimeExecutionPolicy, StageExecutionProfile


class RuntimeStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class StageStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RuntimeRequest(BaseModel):
    """The only public execution envelope understood by AstroGraphRuntime."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-runtime-request.v1"
    request_id: str = Field(min_length=1)
    pipeline_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stage_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: RuntimeErrorRecord | None = None
    artifact: RuntimeStageArtifact | None = None

    @classmethod
    def succeeded(
        cls,
        payload: dict[str, Any],
        *,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        artifact: RuntimeStageArtifact | None = None,
    ) -> "StageResult":
        return cls(
            status=StageStatus.SUCCEEDED,
            payload=payload,
            warnings=warnings or [],
            metadata=metadata or {},
            artifact=artifact,
        )

    @classmethod
    def failed(
        cls,
        *,
        code: str,
        message: str,
        stage_id: str,
        details: dict[str, Any] | None = None,
    ) -> "StageResult":
        return cls(
            status=StageStatus.FAILED,
            error=RuntimeErrorRecord(
                code=code,
                message=message,
                stage_id=stage_id,
                details=details or {},
            ),
        )


class RuntimeStageReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str
    stage_version: str
    status: StageStatus
    elapsed_ms: float = Field(ge=0.0)
    warning_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: RuntimeErrorRecord | None = None
    artifact_type: str | None = None
    artifact_sha256: str | None = None
    artifact_semantic_sha256: str | None = None


class RuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-runtime-result.v1"
    status: RuntimeStatus
    request_id: str
    pipeline_id: str
    output: dict[str, Any] = Field(default_factory=dict)
    stages: list[RuntimeStageReceipt] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: RuntimeErrorRecord | None = None
    contracts: StableContractBundle = Field(default_factory=lambda: STABLE_CONTRACTS)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class RuntimeStageContext:
    """Non-serializable execution context injected by the runtime factory.

    Domain dependencies live here so the stable request/result protocols never expose
    repository clients, model objects, LLM clients, or extension implementations.
    """

    project_root: Path | None = None
    dependencies: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)
    artifacts: RuntimeArtifactStore = field(default_factory=lambda: RuntimeArtifactStore(execution_scope="unscoped"))
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    execution_policy: RuntimeExecutionPolicy = field(default_factory=RuntimeExecutionPolicy)
    stage_profile: StageExecutionProfile | None = None


@runtime_checkable
class RuntimeStage(Protocol):
    stage_id: str
    stage_version: str

    def execute(self, payload: dict[str, Any], context: RuntimeStageContext) -> StageResult:
        ...
