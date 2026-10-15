from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchedulerMode(str, Enum):
    SERIAL = "SERIAL"
    SAFE_PARALLEL = "SAFE_PARALLEL"


class ExecutionBoundary(str, Enum):
    IN_PROCESS = "IN_PROCESS"
    ISOLATED_PROCESS = "ISOLATED_PROCESS"


class CancellationMode(str, Enum):
    NONE = "NONE"
    COOPERATIVE = "COOPERATIVE"
    PROVIDER_ENFORCED = "PROVIDER_ENFORCED"
    HARD_ISOLATED = "HARD_ISOLATED"


class StageSideEffect(str, Enum):
    NONE = "NONE"
    READ_ONLY = "READ_ONLY"
    WRITE_ARTIFACT = "WRITE_ARTIFACT"
    MUTATE_STATE = "MUTATE_STATE"
    EXTERNAL_IO = "EXTERNAL_IO"


class StageExecutionProfile(BaseModel):
    """Machine-checkable execution/scheduling contract for one RuntimeStage.

    `parallel_safe` alone is deliberately insufficient. A stage is eligible for the
    bounded parallel scheduler only when replay equivalence is certified, payload is
    pass-through, side effects are non-mutating, and its cancellation contract satisfies
    the request policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-stage-execution-profile.v1"
    stage_id: str = Field(min_length=1)
    parallel_safe: bool = False
    replay_equivalence_certified: bool = False
    payload_passthrough: bool = True
    cancellation_mode: CancellationMode = CancellationMode.COOPERATIVE
    execution_boundary: ExecutionBoundary = ExecutionBoundary.IN_PROCESS
    parallel_execution_boundary: ExecutionBoundary = ExecutionBoundary.IN_PROCESS
    isolation_adapter_id: str | None = None
    cancellation_qualification_id: str | None = None
    provider_qualification_ids: tuple[str, ...] = ()
    side_effect: StageSideEffect = StageSideEffect.READ_ONLY
    resource_units: int = Field(default=1, ge=1, le=1024)
    timeout_s: float | None = Field(default=None, gt=0.0)
    notes: str = ""

    @model_validator(mode="after")
    def validate_parallel_claim(self) -> "StageExecutionProfile":
        if self.parallel_safe and not self.replay_equivalence_certified:
            raise ValueError("parallel_safe requires replay_equivalence_certified")
        if self.parallel_safe and not self.payload_passthrough:
            raise ValueError("parallel_safe requires payload_passthrough")
        if self.parallel_safe and self.side_effect in {
            StageSideEffect.WRITE_ARTIFACT,
            StageSideEffect.MUTATE_STATE,
        }:
            raise ValueError("parallel_safe stages cannot mutate state or write artifacts")
        if self.cancellation_mode == CancellationMode.HARD_ISOLATED:
            if self.execution_boundary != ExecutionBoundary.ISOLATED_PROCESS:
                raise ValueError("HARD_ISOLATED requires ISOLATED_PROCESS execution_boundary")
            if not self.isolation_adapter_id:
                raise ValueError("HARD_ISOLATED requires isolation_adapter_id")
            if not self.cancellation_qualification_id:
                raise ValueError("HARD_ISOLATED requires cancellation_qualification_id")
        if self.execution_boundary == ExecutionBoundary.ISOLATED_PROCESS and not self.isolation_adapter_id:
            raise ValueError("ISOLATED_PROCESS requires isolation_adapter_id")
        if self.parallel_execution_boundary == ExecutionBoundary.ISOLATED_PROCESS:
            if not self.isolation_adapter_id:
                raise ValueError("parallel ISOLATED_PROCESS requires isolation_adapter_id")
            if self.parallel_safe and not self.cancellation_qualification_id:
                raise ValueError("parallel ISOLATED_PROCESS requires cancellation_qualification_id")
        if self.cancellation_mode == CancellationMode.PROVIDER_ENFORCED and not self.cancellation_qualification_id:
            raise ValueError("PROVIDER_ENFORCED requires cancellation_qualification_id")
        return self

    @property
    def hard_cancel_capable(self) -> bool:
        return (
            (
                self.cancellation_mode in {
                    CancellationMode.PROVIDER_ENFORCED,
                    CancellationMode.HARD_ISOLATED,
                }
                and bool(self.cancellation_qualification_id)
            )
            or (
                self.parallel_execution_boundary == ExecutionBoundary.ISOLATED_PROCESS
                and bool(self.isolation_adapter_id)
                and bool(self.cancellation_qualification_id)
            )
        )


class RuntimeExecutionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-runtime-execution-policy.v1"
    scheduler_mode: SchedulerMode = SchedulerMode.SERIAL
    max_parallel_stages: int = Field(default=1, ge=1, le=64)
    max_resource_units: int = Field(default=1, ge=1, le=4096)
    request_timeout_s: float | None = Field(default=None, gt=0.0)
    require_hard_cancel_for_parallel: bool = True
    allow_cooperative_parallel: bool = False
    isolated_termination_grace_s: float = Field(default=0.25, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_policy(self) -> "RuntimeExecutionPolicy":
        if self.scheduler_mode == SchedulerMode.SERIAL:
            return self
        if self.max_parallel_stages < 2:
            raise ValueError("SAFE_PARALLEL requires max_parallel_stages >= 2")
        if not self.require_hard_cancel_for_parallel and not self.allow_cooperative_parallel:
            raise ValueError(
                "disabling hard-cancel requirement requires explicit allow_cooperative_parallel"
            )
        return self


class RuntimeCancelledError(RuntimeError):
    pass


class RuntimeDeadlineExceeded(RuntimeCancelledError):
    pass


class CancellationToken:
    """Thread-safe request/stage cancellation signal with monotonic deadline support."""

    def __init__(self, *, deadline_monotonic: float | None = None, parent: "CancellationToken | None" = None) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._deadline = deadline_monotonic
        self._parent = parent

    @property
    def deadline_monotonic(self) -> float | None:
        return self._deadline

    @property
    def reason(self) -> str:
        if self._parent is not None and self._parent.is_cancelled():
            return self._parent.reason
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
                self._event.set()

    def is_cancelled(self) -> bool:
        if self._parent is not None and self._parent.is_cancelled():
            return True
        if self._event.is_set():
            return True
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return True
        return False

    def remaining_s(self) -> float | None:
        deadlines = [d for d in (self._deadline, self._parent.deadline_monotonic if self._parent else None) if d is not None]
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - time.monotonic())

    def checkpoint(self) -> None:
        if self._parent is not None:
            self._parent.checkpoint()
        if self._event.is_set():
            raise RuntimeCancelledError(self.reason or "cancelled")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise RuntimeDeadlineExceeded("runtime execution deadline exceeded")

    def child(self, *, timeout_s: float | None = None) -> "CancellationToken":
        deadline = self._deadline
        if timeout_s is not None:
            local = time.monotonic() + timeout_s
            deadline = local if deadline is None else min(deadline, local)
        return CancellationToken(deadline_monotonic=deadline, parent=self)


def execution_policy_from_request(payload: dict[str, Any], metadata: dict[str, Any]) -> RuntimeExecutionPolicy:
    """Resolve execution policy without adding another public request envelope.

    Runtime-specific overrides live under `metadata.execution_control`. For the online
    pipeline, the existing TaskEnvelope.execution_policy.timeout_s remains authoritative
    when no explicit Runtime timeout is supplied.
    """

    raw = metadata.get("execution_control") or {}
    if not isinstance(raw, dict):
        raise ValueError("metadata.execution_control must be an object")
    raw = dict(raw)
    if "request_timeout_s" not in raw:
        envelope = payload.get("envelope")
        if isinstance(envelope, dict):
            envelope_policy = envelope.get("execution_policy")
            if isinstance(envelope_policy, dict) and envelope_policy.get("timeout_s") is not None:
                raw["request_timeout_s"] = float(envelope_policy["timeout_s"])
    return RuntimeExecutionPolicy.model_validate(raw)
