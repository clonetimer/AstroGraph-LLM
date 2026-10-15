from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter
import time
from typing import Any, Iterable, Mapping

from astrograph.contracts import STABLE_CONTRACTS
from astrograph.provider_lifecycle import BUILTIN_PROVIDER_QUALIFICATIONS
from .artifacts import RuntimeArtifactStore, RuntimeStageArtifact
from .contracts import (
    RuntimeErrorRecord,
    RuntimeRequest,
    RuntimeResult,
    RuntimeStage,
    RuntimeStageContext,
    RuntimeStageReceipt,
    RuntimeStatus,
    StageResult,
    StageStatus,
)
from .execution_control import (
    CancellationMode,
    CancellationToken,
    ExecutionBoundary,
    RuntimeCancelledError,
    RuntimeDeadlineExceeded,
    RuntimeExecutionPolicy,
    SchedulerMode,
    StageExecutionProfile,
    execution_policy_from_request,
)
from .isolation import IsolatedExecutionError, IsolatedStageInvocation, execute_isolated_stage


@dataclass(frozen=True, slots=True)
class RuntimePipeline:
    pipeline_id: str
    stages: tuple[RuntimeStage, ...]
    description: str = ""
    dependencies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    execution_profiles: Mapping[str, StageExecutionProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("pipeline_id must be nonempty")
        if not self.stages:
            raise ValueError("runtime pipeline must contain at least one stage")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"runtime pipeline contains duplicate stage IDs: {stage_ids}")
        known = set(stage_ids)
        unknown_nodes = sorted(set(self.dependencies) - known)
        if unknown_nodes:
            raise ValueError(f"runtime DAG declares unknown stage IDs: {unknown_nodes}")
        unknown_profiles = sorted(set(self.execution_profiles) - known)
        if unknown_profiles:
            raise ValueError(f"runtime execution profiles declare unknown stages: {unknown_profiles}")
        for stage_id, profile in self.execution_profiles.items():
            if profile.stage_id != stage_id:
                raise ValueError(
                    f"runtime execution profile key/stage mismatch: {stage_id} != {profile.stage_id}"
                )
        for stage_id, deps in self.dependencies.items():
            unknown = sorted(set(deps) - known)
            if unknown:
                raise ValueError(f"runtime DAG stage {stage_id} depends on unknown stages: {unknown}")
            if stage_id in deps:
                raise ValueError(f"runtime DAG stage {stage_id} cannot depend on itself")
        self.ordered_stages
        self.levels

    @property
    def ordered_stages(self) -> tuple[RuntimeStage, ...]:
        if not self.dependencies:
            return self.stages
        order_index = {stage.stage_id: index for index, stage in enumerate(self.stages)}
        for stage_id, deps in self.dependencies.items():
            for dependency in deps:
                if order_index[dependency] >= order_index[stage_id]:
                    raise ValueError(
                        "runtime DAG stages must be declared in a stable topological order: "
                        f"{stage_id} depends on {dependency}"
                    )
        return self.stages

    @property
    def levels(self) -> tuple[tuple[RuntimeStage, ...], ...]:
        """Stable topological levels; independent nodes share a level."""
        resolved: set[str] = set()
        remaining = list(self.ordered_stages)
        levels: list[tuple[RuntimeStage, ...]] = []
        while remaining:
            ready = tuple(
                stage for stage in remaining
                if set(self.dependencies.get(stage.stage_id, ())).issubset(resolved)
            )
            if not ready:
                raise ValueError("runtime DAG is cyclic or has unresolved dependencies")
            levels.append(ready)
            ready_ids = {stage.stage_id for stage in ready}
            resolved.update(ready_ids)
            remaining = [stage for stage in remaining if stage.stage_id not in ready_ids]
        return tuple(levels)

    def profile_for(self, stage: RuntimeStage) -> StageExecutionProfile:
        return self.execution_profiles.get(
            stage.stage_id,
            StageExecutionProfile(
                stage_id=stage.stage_id,
                parallel_safe=False,
                replay_equivalence_certified=False,
                payload_passthrough=False,
                cancellation_mode=CancellationMode.COOPERATIVE,
                notes="implicit conservative runtime profile",
            ),
        )


@dataclass(slots=True)
class _StageExecution:
    stage: RuntimeStage
    profile: StageExecutionProfile
    result: StageResult
    receipt: RuntimeStageReceipt


class AstroGraphRuntime:
    """Canonical AstroGraph execution runtime with fail-closed scheduling control."""

    runtime_id = "astrograph.runtime"
    runtime_version = "1.3"

    def __init__(
        self,
        pipelines: Iterable[RuntimePipeline],
        *,
        context: RuntimeStageContext | None = None,
    ) -> None:
        items = list(pipelines)
        if not items:
            raise ValueError("AstroGraphRuntime requires at least one pipeline")
        self._pipelines = {item.pipeline_id: item for item in items}
        if len(self._pipelines) != len(items):
            raise ValueError("runtime pipeline IDs must be unique")
        self.context = context or RuntimeStageContext()

    @property
    def pipeline_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._pipelines))

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": "astrograph-runtime-description.v1",
            "runtime_id": self.runtime_id,
            "runtime_version": self.runtime_version,
            "status": "PASS",
            "pipelines": [
                {
                    "pipeline_id": item.pipeline_id,
                    "description": item.description,
                    "levels": [[stage.stage_id for stage in level] for level in item.levels],
                    "stages": [
                        {
                            "stage_id": stage.stage_id,
                            "stage_version": stage.stage_version,
                            "depends_on": list(item.dependencies.get(stage.stage_id, ())),
                            "execution_profile": item.profile_for(stage).model_dump(mode="json"),
                        }
                        for stage in item.ordered_stages
                    ],
                }
                for item in sorted(self._pipelines.values(), key=lambda value: value.pipeline_id)
            ],
            "contracts": STABLE_CONTRACTS.model_dump(mode="json"),
            "provider_qualifications": BUILTIN_PROVIDER_QUALIFICATIONS.snapshot(),
        }

    def _request_context(
        self,
        request: RuntimeRequest,
        policy: RuntimeExecutionPolicy,
        token: CancellationToken,
    ) -> RuntimeStageContext:
        return RuntimeStageContext(
            project_root=self.context.project_root,
            dependencies=self.context.dependencies,
            metadata={**self.context.metadata, **dict(request.metadata)},
            scratch={},
            artifacts=RuntimeArtifactStore(execution_scope=request.request_id),
            cancellation=token,
            execution_policy=policy,
        )

    @staticmethod
    def _error_receipt(
        stage: RuntimeStage,
        *,
        started: float,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        profile: StageExecutionProfile | None = None,
    ) -> _StageExecution:
        effective_profile = profile or StageExecutionProfile(stage_id=stage.stage_id)
        error = RuntimeErrorRecord(
            code=code,
            message=message,
            stage_id=stage.stage_id,
            details=details or {},
        )
        result = StageResult(status=StageStatus.FAILED, error=error)
        return _StageExecution(
            stage=stage,
            profile=effective_profile,
            result=result,
            receipt=RuntimeStageReceipt(
                stage_id=stage.stage_id,
                stage_version=stage.stage_version,
                status=StageStatus.FAILED,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                warning_count=0,
                metadata={
                    "execution_control": {
                        "parallel_safe": effective_profile.parallel_safe,
                        "replay_equivalence_certified": effective_profile.replay_equivalence_certified,
                        "cancellation_mode": effective_profile.cancellation_mode.value,
                        "execution_boundary": effective_profile.execution_boundary.value,
                        "parallel_execution_boundary": effective_profile.parallel_execution_boundary.value,
                        "isolation_adapter_id": effective_profile.isolation_adapter_id,
                        "cancellation_qualification_id": effective_profile.cancellation_qualification_id,
                        "provider_qualification_ids": list(effective_profile.provider_qualification_ids),
                        "resource_units": effective_profile.resource_units,
                        "side_effect": effective_profile.side_effect.value,
                        **(details or {}),
                    }
                },
                error=error,
            ),
        )

    def _execute_stage(
        self,
        stage: RuntimeStage,
        profile: StageExecutionProfile,
        payload: dict[str, Any],
        base_context: RuntimeStageContext,
        *,
        isolated_artifacts: bool,
        upstream_stage_ids: tuple[str, ...] = (),
        force_isolated_process: bool = False,
    ) -> _StageExecution:
        started = perf_counter()
        stage_token = base_context.cancellation.child(timeout_s=profile.timeout_s)
        isolation_metadata: dict[str, Any] = {}
        active_isolated_process = (
            profile.execution_boundary == ExecutionBoundary.ISOLATED_PROCESS
            or force_isolated_process
        )
        active_cancellation_mode = (
            CancellationMode.HARD_ISOLATED if active_isolated_process else profile.cancellation_mode
        )
        try:
            stage_token.checkpoint()
            if active_isolated_process:
                if base_context.project_root is None:
                    raise IsolatedExecutionError(
                        "ASTROGRAPH_ISOLATED_PROJECT_ROOT_REQUIRED",
                        "isolated stage execution requires RuntimeStageContext.project_root",
                    )
                upstream = {
                    stage_id: base_context.artifacts.get(stage_id)
                    for stage_id in upstream_stage_ids
                }
                invocation = IsolatedStageInvocation(
                    adapter_id=profile.isolation_adapter_id or "",
                    stage_id=stage.stage_id,
                    stage_version=stage.stage_version,
                    project_root=str(base_context.project_root),
                    execution_scope=base_context.artifacts.execution_scope,
                    payload=dict(payload),
                    metadata=dict(base_context.metadata),
                    execution_policy=base_context.execution_policy,
                    stage_profile=profile,
                    upstream_artifacts=upstream,
                )
                result, isolation_metadata = execute_isolated_stage(
                    invocation,
                    cancellation=stage_token,
                    termination_grace_s=base_context.execution_policy.isolated_termination_grace_s,
                )
            else:
                artifacts = base_context.artifacts
                if isolated_artifacts:
                    artifacts = RuntimeArtifactStore(execution_scope=base_context.artifacts.execution_scope)
                    if upstream_stage_ids:
                        artifacts.preload({
                            stage_id: base_context.artifacts.get(stage_id)
                            for stage_id in upstream_stage_ids
                        })
                    else:
                        artifacts.preload(base_context.artifacts.snapshot())
                stage_context = RuntimeStageContext(
                    project_root=base_context.project_root,
                    dependencies=base_context.dependencies,
                    metadata=dict(base_context.metadata),
                    scratch={} if isolated_artifacts else base_context.scratch,
                    artifacts=artifacts,
                    cancellation=stage_token,
                    execution_policy=base_context.execution_policy,
                    stage_profile=profile,
                )
                result = stage.execute(dict(payload), stage_context)
                stage_token.checkpoint()
        except IsolatedExecutionError as exc:
            return self._error_receipt(
                stage, started=started, code=exc.code, message=exc.message,
                details={**exc.details, "cancellation_mode": active_cancellation_mode.value},
                profile=profile,
            )
        except RuntimeDeadlineExceeded as exc:
            return self._error_receipt(
                stage,
                started=started,
                code="ASTROGRAPH_RUNTIME_STAGE_TIMEOUT",
                message=str(exc),
                details={"cancellation_mode": active_cancellation_mode.value},
                profile=profile,
            )
        except RuntimeCancelledError as exc:
            return self._error_receipt(
                stage,
                started=started,
                code="ASTROGRAPH_RUNTIME_STAGE_CANCELLED",
                message=str(exc),
                details={"cancellation_mode": active_cancellation_mode.value},
                profile=profile,
            )
        except Exception as exc:  # stage boundary is intentionally fail-closed
            return self._error_receipt(
                stage,
                started=started,
                code="ASTROGRAPH_RUNTIME_STAGE_UNHANDLED_EXCEPTION",
                message=f"{type(exc).__name__}: {exc}",
                profile=profile,
            )
        elapsed = (perf_counter() - started) * 1000.0
        receipt = RuntimeStageReceipt(
            stage_id=stage.stage_id,
            stage_version=stage.stage_version,
            status=result.status,
            elapsed_ms=elapsed,
            warning_count=len(result.warnings),
            metadata={
                **result.metadata,
                "execution_control": {
                    "parallel_safe": profile.parallel_safe,
                    "replay_equivalence_certified": profile.replay_equivalence_certified,
                    "cancellation_mode": active_cancellation_mode.value,
                    "execution_boundary": (
                        ExecutionBoundary.ISOLATED_PROCESS.value
                        if active_isolated_process
                        else profile.execution_boundary.value
                    ),
                    "parallel_execution_boundary": profile.parallel_execution_boundary.value,
                    "isolation_adapter_id": profile.isolation_adapter_id,
                    "cancellation_qualification_id": profile.cancellation_qualification_id,
                    "provider_qualification_ids": list(profile.provider_qualification_ids),
                    "resource_units": profile.resource_units,
                    "side_effect": profile.side_effect.value,
                    **isolation_metadata,
                },
            },
            error=result.error,
            artifact_type=(result.artifact.artifact_type if result.artifact else None),
            artifact_sha256=(result.artifact.content_sha256 if result.artifact else None),
            artifact_semantic_sha256=(result.artifact.semantic_sha256 if result.artifact else None),
        )
        return _StageExecution(stage=stage, profile=profile, result=result, receipt=receipt)

    @staticmethod
    def _parallel_eligible(profile: StageExecutionProfile, policy: RuntimeExecutionPolicy) -> bool:
        if not (profile.parallel_safe and profile.replay_equivalence_certified and profile.payload_passthrough):
            return False
        if profile.provider_qualification_ids:
            try:
                if not BUILTIN_PROVIDER_QUALIFICATIONS.stable_parallel_eligible(
                    profile.provider_qualification_ids
                ):
                    return False
            except KeyError:
                return False
        if policy.require_hard_cancel_for_parallel:
            return profile.hard_cancel_capable
        return policy.allow_cooperative_parallel

    @staticmethod
    def _batches(
        stages: list[tuple[RuntimeStage, StageExecutionProfile]],
        *,
        max_parallel: int,
        max_units: int,
    ) -> list[list[tuple[RuntimeStage, StageExecutionProfile]]]:
        batches: list[list[tuple[RuntimeStage, StageExecutionProfile]]] = []
        current: list[tuple[RuntimeStage, StageExecutionProfile]] = []
        units = 0
        for item in stages:
            required = item[1].resource_units
            if required > max_units:
                raise ValueError(
                    f"stage {item[0].stage_id} resource_units={required} exceeds runtime budget={max_units}"
                )
            if current and (len(current) >= max_parallel or units + required > max_units):
                batches.append(current)
                current = []
                units = 0
            current.append(item)
            units += required
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _failed_result(
        request: RuntimeRequest,
        *,
        payload: dict[str, Any],
        receipts: list[RuntimeStageReceipt],
        warnings: list[str],
        error: RuntimeErrorRecord,
        policy: RuntimeExecutionPolicy,
        scheduler_metadata: dict[str, Any],
    ) -> RuntimeResult:
        return RuntimeResult(
            status=RuntimeStatus.FAILED,
            request_id=request.request_id,
            pipeline_id=request.pipeline_id,
            output=payload,
            stages=receipts,
            warnings=warnings,
            error=error,
            metadata={
                **dict(request.metadata),
                "execution_control": {
                    **policy.model_dump(mode="json"),
                    **scheduler_metadata,
                },
            },
        )

    def execute(self, request: RuntimeRequest) -> RuntimeResult:
        pipeline = self._pipelines.get(request.pipeline_id)
        if pipeline is None:
            error = RuntimeErrorRecord(
                code="ASTROGRAPH_RUNTIME_PIPELINE_NOT_FOUND",
                message=f"Unknown runtime pipeline: {request.pipeline_id}",
                details={"available_pipeline_ids": list(self.pipeline_ids)},
            )
            return RuntimeResult(
                status=RuntimeStatus.FAILED,
                request_id=request.request_id,
                pipeline_id=request.pipeline_id,
                error=error,
                metadata=dict(request.metadata),
            )
        try:
            policy = execution_policy_from_request(request.payload, request.metadata)
        except Exception as exc:
            error = RuntimeErrorRecord(
                code="ASTROGRAPH_RUNTIME_EXECUTION_POLICY_INVALID",
                message=f"{type(exc).__name__}: {exc}",
            )
            return RuntimeResult(
                status=RuntimeStatus.FAILED,
                request_id=request.request_id,
                pipeline_id=request.pipeline_id,
                error=error,
                metadata=dict(request.metadata),
            )

        deadline = (
            time.monotonic() + policy.request_timeout_s
            if policy.request_timeout_s is not None else None
        )
        token = CancellationToken(deadline_monotonic=deadline)
        context = self._request_context(request, policy, token)
        payload = dict(request.payload)
        receipts: list[RuntimeStageReceipt] = []
        warnings: list[str] = []
        scheduler_metadata: dict[str, Any] = {
            "parallel_batches_executed": 0,
            "parallel_stage_count": 0,
            "serial_stage_count": 0,
            "parallel_activation_withheld": False,
        }

        levels = pipeline.levels if policy.scheduler_mode == SchedulerMode.SAFE_PARALLEL else tuple(
            (stage,) for stage in pipeline.ordered_stages
        )
        for level in levels:
            try:
                token.checkpoint()
            except RuntimeCancelledError as exc:
                error = RuntimeErrorRecord(
                    code="ASTROGRAPH_RUNTIME_REQUEST_TIMEOUT",
                    message=str(exc),
                )
                return self._failed_result(
                    request, payload=payload, receipts=receipts, warnings=warnings,
                    error=error, policy=policy, scheduler_metadata=scheduler_metadata,
                )

            items = [(stage, pipeline.profile_for(stage)) for stage in level]
            parallel = [item for item in items if self._parallel_eligible(item[1], policy)]
            serial = [item for item in items if item not in parallel]
            if policy.scheduler_mode == SchedulerMode.SAFE_PARALLEL and len(level) > 1 and len(parallel) < 2:
                scheduler_metadata["parallel_activation_withheld"] = True

            # Parallel nodes are payload-pass-through by contract and receive isolated
            # artifact-store snapshots. Execution may finish out of order, but receipts
            # and warnings are committed in the pipeline's stable declared order.
            level_executions: dict[str, _StageExecution] = {}
            if len(parallel) >= 2:
                for batch in self._batches(
                    parallel,
                    max_parallel=policy.max_parallel_stages,
                    max_units=policy.max_resource_units,
                ):
                    scheduler_metadata["parallel_batches_executed"] += 1
                    scheduler_metadata["parallel_stage_count"] += len(batch)
                    with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="astrograph-dag") as pool:
                        future_map: dict[Future[_StageExecution], tuple[RuntimeStage, StageExecutionProfile]] = {
                            pool.submit(
                                self._execute_stage, stage, profile, payload, context,
                                isolated_artifacts=True,
                                upstream_stage_ids=tuple(pipeline.dependencies.get(stage.stage_id, ())),
                                force_isolated_process=(
                                    profile.parallel_execution_boundary == ExecutionBoundary.ISOLATED_PROCESS
                                ),
                            ): (stage, profile)
                            for stage, profile in batch
                        }
                        executions: list[_StageExecution] = []
                        for future in as_completed(future_map):
                            execution = future.result()
                            executions.append(execution)
                            level_executions[execution.stage.stage_id] = execution
                            if execution.result.status == StageStatus.FAILED:
                                token.cancel(f"parallel sibling failed: {execution.stage.stage_id}")
                    for execution in executions:
                        level_executions[execution.stage.stage_id] = execution
                    # Commit only after every member of this bounded batch has returned.
                    for stage, _profile in batch:
                        execution = level_executions[stage.stage_id]
                        if execution.result.artifact is not None:
                            context.artifacts.put(stage.stage_id, execution.result.artifact)
                        if execution.result.status == StageStatus.FAILED:
                            token.cancel(f"stage failed: {stage.stage_id}")
                            for ordered_stage, _ in items:
                                prior = level_executions.get(ordered_stage.stage_id)
                                if prior is not None:
                                    receipts.append(prior.receipt)
                                    warnings.extend(prior.result.warnings)
                            error = execution.result.error or RuntimeErrorRecord(
                                code="ASTROGRAPH_RUNTIME_STAGE_FAILED",
                                message="Runtime stage failed without an error record.",
                                stage_id=stage.stage_id,
                            )
                            return self._failed_result(
                                request, payload=payload, receipts=receipts, warnings=warnings,
                                error=error, policy=policy, scheduler_metadata=scheduler_metadata,
                            )
                        if execution.result.status == StageStatus.SUCCEEDED and execution.result.payload != payload:
                            error = RuntimeErrorRecord(
                                code="ASTROGRAPH_RUNTIME_PARALLEL_PAYLOAD_MUTATION",
                                message="Parallel stage violated payload_passthrough execution contract.",
                                stage_id=stage.stage_id,
                            )
                            token.cancel(error.message)
                            for ordered_stage, _ in items:
                                prior = level_executions.get(ordered_stage.stage_id)
                                if prior is not None:
                                    receipts.append(prior.receipt)
                                    warnings.extend(prior.result.warnings)
                            return self._failed_result(
                                request, payload=payload, receipts=receipts, warnings=warnings,
                                error=error, policy=policy, scheduler_metadata=scheduler_metadata,
                            )

            # Any stage not admitted to safe parallelism is executed in stable order.
            # A single parallel-eligible node also runs here to avoid pointless threads.
            serial_order = [item for item in items if item in serial or len(parallel) < 2]
            for stage, profile in serial_order:
                scheduler_metadata["serial_stage_count"] += 1
                execution = self._execute_stage(
                    stage, profile, payload, context, isolated_artifacts=False,
                    upstream_stage_ids=tuple(pipeline.dependencies.get(stage.stage_id, ())),
                )
                level_executions[stage.stage_id] = execution
                if execution.result.artifact is not None:
                    context.artifacts.put(stage.stage_id, execution.result.artifact)
                if execution.result.status == StageStatus.FAILED:
                    token.cancel(f"stage failed: {stage.stage_id}")
                    for ordered_stage, _ in items:
                        prior = level_executions.get(ordered_stage.stage_id)
                        if prior is not None:
                            receipts.append(prior.receipt)
                            warnings.extend(prior.result.warnings)
                    error = execution.result.error or RuntimeErrorRecord(
                        code="ASTROGRAPH_RUNTIME_STAGE_FAILED",
                        message="Runtime stage failed without an error record.",
                        stage_id=stage.stage_id,
                    )
                    return self._failed_result(
                        request, payload=payload, receipts=receipts, warnings=warnings,
                        error=error, policy=policy, scheduler_metadata=scheduler_metadata,
                    )
                if execution.result.status == StageStatus.SUCCEEDED:
                    payload = execution.result.payload

            for ordered_stage, _ in items:
                execution = level_executions.get(ordered_stage.stage_id)
                if execution is not None:
                    receipts.append(execution.receipt)
                    warnings.extend(execution.result.warnings)

        stable_order = {stage.stage_id: index for index, stage in enumerate(pipeline.ordered_stages)}
        receipts.sort(key=lambda item: stable_order.get(item.stage_id, len(stable_order)))
        return RuntimeResult(
            status=RuntimeStatus.SUCCEEDED,
            request_id=request.request_id,
            pipeline_id=request.pipeline_id,
            output=payload,
            stages=receipts,
            warnings=warnings,
            metadata={
                **dict(request.metadata),
                "execution_control": {
                    **policy.model_dump(mode="json"),
                    **scheduler_metadata,
                },
            },
        )

    def replay_stage(
        self,
        request: RuntimeRequest,
        *,
        stage_id: str,
        upstream_artifacts: Mapping[str, RuntimeStageArtifact],
    ) -> tuple[Any, RuntimeStageReceipt]:
        pipeline = self._pipelines.get(request.pipeline_id)
        if pipeline is None:
            raise KeyError(f"Unknown runtime pipeline: {request.pipeline_id}")
        stage = next((item for item in pipeline.stages if item.stage_id == stage_id), None)
        if stage is None:
            raise KeyError(f"Unknown runtime stage {stage_id} in {request.pipeline_id}")
        required = tuple(pipeline.dependencies.get(stage_id, ()))
        missing = [dep for dep in required if dep not in upstream_artifacts]
        if missing:
            raise ValueError(f"missing replay upstream artifacts for {stage_id}: {missing}")
        extras = sorted(set(upstream_artifacts) - set(required))
        if extras:
            raise ValueError(f"undeclared replay upstream artifacts for {stage_id}: {extras}")
        policy = execution_policy_from_request(request.payload, request.metadata)
        token = CancellationToken(
            deadline_monotonic=(
                time.monotonic() + policy.request_timeout_s
                if policy.request_timeout_s is not None else None
            )
        )
        context = self._request_context(request, policy, token)
        context.artifacts.preload(upstream_artifacts)
        execution = self._execute_stage(
            stage, pipeline.profile_for(stage), dict(request.payload), context,
            isolated_artifacts=False, upstream_stage_ids=required,
        )
        if execution.result.artifact is not None:
            context.artifacts.put(stage_id, execution.result.artifact)
        return execution.result, execution.receipt
