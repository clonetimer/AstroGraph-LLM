from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.audit import NullAuditStore
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.runtime import (
    AstroGraphRuntime,
    CancellationMode,
    ExecutionBoundary,
    RuntimePipeline,
    RuntimeRequest,
    RuntimeStageContext,
    RuntimeStatus,
    SchedulerMode,
    StageExecutionProfile,
    StageResult,
    StageSideEffect,
    WHOLE_SPACECRAFT_ONLINE_HARD_ISOLATION_CANDIDATES,
    WHOLE_SPACECRAFT_PIPELINE_ID,
    build_application_runtime,
)

ROOT = Path(__file__).resolve().parents[1]


class _ParentOnlyStage:
    stage_version = "1"

    def __init__(self, stage_id: str):
        self.stage_id = stage_id

    def execute(self, payload: dict, context: RuntimeStageContext) -> StageResult:
        raise AssertionError("hard-isolated stage must not execute in the parent process")


def _isolated_runtime(adapter_id: str, *, timeout_s: float | None = None) -> AstroGraphRuntime:
    stage = _ParentOnlyStage("isolated.probe")
    profile = StageExecutionProfile(
        stage_id=stage.stage_id,
        parallel_safe=False,
        replay_equivalence_certified=False,
        payload_passthrough=True,
        cancellation_mode=CancellationMode.HARD_ISOLATED,
        execution_boundary=ExecutionBoundary.ISOLATED_PROCESS,
        isolation_adapter_id=adapter_id,
        cancellation_qualification_id="post8.test.hard-isolation.v1",
        side_effect=StageSideEffect.READ_ONLY,
        timeout_s=timeout_s,
    )
    return AstroGraphRuntime(
        [RuntimePipeline(
            pipeline_id="test.isolated",
            stages=(stage,),
            dependencies={stage.stage_id: ()},
            execution_profiles={stage.stage_id: profile},
        )],
        context=RuntimeStageContext(project_root=ROOT),
    )


def test_hard_isolated_profile_requires_real_process_boundary() -> None:
    with pytest.raises(ValidationError, match="HARD_ISOLATED requires ISOLATED_PROCESS"):
        StageExecutionProfile(
            stage_id="bad",
            cancellation_mode=CancellationMode.HARD_ISOLATED,
            isolation_adapter_id="whole_spacecraft.online.v1",
            cancellation_qualification_id="post8.test.hard-isolation.v1",
        )
    with pytest.raises(ValidationError, match="requires isolation_adapter_id"):
        StageExecutionProfile(
            stage_id="bad",
            cancellation_mode=CancellationMode.HARD_ISOLATED,
            execution_boundary=ExecutionBoundary.ISOLATED_PROCESS,
        )


def test_uncooperative_worker_is_hard_terminated_on_stage_timeout() -> None:
    runtime = _isolated_runtime("_qualification.uncooperative_sleep.v1", timeout_s=0.06)
    started = time.monotonic()
    result = runtime.execute(RuntimeRequest(
        request_id="hard-timeout",
        pipeline_id="test.isolated",
        payload={"sleep_s": 5.0},
    ))
    elapsed = time.monotonic() - started
    assert result.status == RuntimeStatus.FAILED
    assert result.error is not None
    assert result.error.code == "ASTROGRAPH_RUNTIME_STAGE_TIMEOUT"
    assert elapsed < 1.5
    assert result.error.details["execution_boundary"] == "ISOLATED_PROCESS"
    assert result.error.details["termination_mode"] in {"terminate", "kill", "already-exited"}
    assert int(result.error.details["worker_pid"]) > 0


def test_worker_crash_is_contained_and_reported_fail_closed() -> None:
    runtime = _isolated_runtime("_qualification.crash.v1")
    result = runtime.execute(RuntimeRequest(
        request_id="hard-crash",
        pipeline_id="test.isolated",
        payload={"exit_code": 23},
    ))
    assert result.status == RuntimeStatus.FAILED
    assert result.error is not None
    assert result.error.code == "ASTROGRAPH_ISOLATED_WORKER_CRASHED"
    assert result.error.details["returncode"] == 23


def test_reviewed_online_runtime_certifies_three_hard_isolated_candidates() -> None:
    runtime = build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(), audit_store=NullAuditStore(), project_root=ROOT,
    )
    pipeline = next(item for item in runtime.describe()["pipelines"] if item["pipeline_id"] == WHOLE_SPACECRAFT_PIPELINE_ID)
    profiles = {item["stage_id"]: item["execution_profile"] for item in pipeline["stages"]}
    for stage_id in WHOLE_SPACECRAFT_ONLINE_HARD_ISOLATION_CANDIDATES:
        profile = profiles[stage_id]
        assert profile["cancellation_mode"] == "COOPERATIVE"
        assert profile["execution_boundary"] == "IN_PROCESS"
        assert profile["parallel_execution_boundary"] == "ISOLATED_PROCESS"
        assert profile["isolation_adapter_id"] == "whole_spacecraft.online.v1"
        assert profile["cancellation_qualification_id"] == "post8.whole_spacecraft.local-branch.hard-isolation.v1"
        assert profile["parallel_safe"] is True


def test_custom_online_workflow_factory_is_not_falsely_hard_isolation_certified() -> None:
    runtime = build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(), audit_store=NullAuditStore(), project_root=ROOT,
        online_workflow_factory=lambda: object(),
    )
    pipeline = next(item for item in runtime.describe()["pipelines"] if item["pipeline_id"] == WHOLE_SPACECRAFT_PIPELINE_ID)
    profiles = {item["stage_id"]: item["execution_profile"] for item in pipeline["stages"]}
    for stage_id in WHOLE_SPACECRAFT_ONLINE_HARD_ISOLATION_CANDIDATES:
        assert profiles[stage_id]["cancellation_mode"] == "COOPERATIVE"
        assert profiles[stage_id]["execution_boundary"] == "IN_PROCESS"
        assert profiles[stage_id]["parallel_execution_boundary"] == "IN_PROCESS"


def test_stable_hard_cancel_policy_can_admit_certified_online_parallelism(tmp_path: Path) -> None:
    from astrograph.orchestration.models import ExecutionPolicy, PermissionScope, TaskEnvelope, WorkflowContext
    from astrograph.simulation import SimulationRunCompleted

    fixture = ROOT / "tests" / "fixtures" / "online_base"
    protocol = ROOT / "protocols" / "phm_project_a"
    permissions = [
        PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS, PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    event = SimulationRunCompleted(
        event_id="SIM-post8", run_id="heater_stuck_on",
        run_dir=str(fixture / "heater_stuck_on"), nominal_run_dir=str(fixture / "nominal"),
    )
    envelope = TaskEnvelope(
        task_id="TASK-post8", task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(protocol),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "post8"),
            "shadow_mode": True, "reasoner_mode": "rule",
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.6",
        run_id="heater_stuck_on", permissions=permissions,
    )
    runtime = build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(), audit_store=NullAuditStore(), project_root=ROOT,
    )
    result = runtime.execute(RuntimeRequest(
        request_id="post8-stable-parallel", pipeline_id=WHOLE_SPACECRAFT_PIPELINE_ID,
        payload={"envelope": envelope.model_dump(mode="json"), "context": context.model_dump(mode="json")},
        metadata={"execution_control": {
            "scheduler_mode": SchedulerMode.SAFE_PARALLEL.value,
            "max_parallel_stages": 3,
            "max_resource_units": 3,
            "require_hard_cancel_for_parallel": True,
        }},
    ))
    assert result.status == RuntimeStatus.SUCCEEDED
    control = result.metadata["execution_control"]
    assert control["parallel_stage_count"] == 3
    by_stage = {item.stage_id: item for item in result.stages}
    for stage_id in WHOLE_SPACECRAFT_ONLINE_HARD_ISOLATION_CANDIDATES:
        stage_control = by_stage[stage_id].metadata["execution_control"]
        assert stage_control["execution_boundary"] == "ISOLATED_PROCESS"
        assert int(stage_control["worker_pid"]) > 0
    assert result.output["output_payload"]["decision_record"]["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"


def test_parallel_isolated_crash_hard_cancels_uncooperative_sibling() -> None:
    crash = _ParentOnlyStage("crash")
    sleeper = _ParentOnlyStage("sleeper")
    profiles = {
        "crash": StageExecutionProfile(
            stage_id="crash", parallel_safe=True, replay_equivalence_certified=True,
            payload_passthrough=True, cancellation_mode=CancellationMode.HARD_ISOLATED,
            execution_boundary=ExecutionBoundary.ISOLATED_PROCESS,
            isolation_adapter_id="_qualification.crash.v1",
            cancellation_qualification_id="post8.test.hard-isolation.v1",
            side_effect=StageSideEffect.READ_ONLY,
        ),
        "sleeper": StageExecutionProfile(
            stage_id="sleeper", parallel_safe=True, replay_equivalence_certified=True,
            payload_passthrough=True, cancellation_mode=CancellationMode.HARD_ISOLATED,
            execution_boundary=ExecutionBoundary.ISOLATED_PROCESS,
            isolation_adapter_id="_qualification.uncooperative_sleep.v1",
            cancellation_qualification_id="post8.test.hard-isolation.v1",
            side_effect=StageSideEffect.READ_ONLY,
        ),
    }
    runtime = AstroGraphRuntime(
        [RuntimePipeline(
            pipeline_id="test.isolated.parallel",
            stages=(crash, sleeper),
            dependencies={"crash": (), "sleeper": ()},
            execution_profiles=profiles,
        )],
        context=RuntimeStageContext(project_root=ROOT),
    )
    started = time.monotonic()
    result = runtime.execute(RuntimeRequest(
        request_id="hard-sibling-cancel",
        pipeline_id="test.isolated.parallel",
        payload={"sleep_s": 5.0, "exit_code": 23},
        metadata={"execution_control": {
            "scheduler_mode": "SAFE_PARALLEL",
            "max_parallel_stages": 2,
            "max_resource_units": 2,
            "require_hard_cancel_for_parallel": True,
        }},
    ))
    elapsed = time.monotonic() - started
    assert result.status == RuntimeStatus.FAILED
    assert result.error is not None
    assert result.error.code == "ASTROGRAPH_ISOLATED_WORKER_CRASHED"
    assert elapsed < 3.0
    by_stage = {item.stage_id: item for item in result.stages}
    assert by_stage["sleeper"].error is not None
    assert by_stage["sleeper"].error.code == "ASTROGRAPH_RUNTIME_STAGE_CANCELLED"
    assert by_stage["sleeper"].error.details["execution_boundary"] == "ISOLATED_PROCESS"


def test_parallel_isolation_requires_machine_checkable_cancellation_qualification() -> None:
    with pytest.raises(ValidationError, match="cancellation_qualification_id"):
        StageExecutionProfile(
            stage_id="unqualified",
            parallel_safe=True,
            replay_equivalence_certified=True,
            payload_passthrough=True,
            cancellation_mode=CancellationMode.COOPERATIVE,
            execution_boundary=ExecutionBoundary.IN_PROCESS,
            parallel_execution_boundary=ExecutionBoundary.ISOLATED_PROCESS,
            isolation_adapter_id="whole_spacecraft.online.v1",
            side_effect=StageSideEffect.READ_ONLY,
        )


def test_request_metadata_cannot_override_frozen_worker_adapter() -> None:
    runtime = _isolated_runtime("_qualification.uncooperative_sleep.v1", timeout_s=3.0)
    result = runtime.execute(RuntimeRequest(
        request_id="adapter-frozen",
        pipeline_id="test.isolated",
        payload={"sleep_s": 0.01},
        metadata={"isolation_adapter_id": "_qualification.crash.v1"},
    ))
    assert result.status == RuntimeStatus.SUCCEEDED
    stage_control = result.stages[0].metadata["execution_control"]
    assert stage_control["adapter_id"] == "_qualification.uncooperative_sleep.v1"


def test_next_isolated_execution_recovers_stale_dead_parent_workdir(tmp_path: Path) -> None:
    from astrograph.runtime.isolation import cleanup_stale_isolation_directories

    stale = tmp_path / "astrograph-isolated-stage-999999999-dead"
    stale.mkdir()
    (stale / "invocation.json").write_text("{}", encoding="utf-8")
    removed = cleanup_stale_isolation_directories(temp_root=tmp_path)
    assert str(stale.resolve()) in removed
    assert not stale.exists()


def test_no_unqualified_online_provider_is_silently_promoted_to_isolated_parallel() -> None:
    runtime = build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(), audit_store=NullAuditStore(), project_root=ROOT,
    )
    pipeline = next(item for item in runtime.describe()["pipelines"] if item["pipeline_id"] == WHOLE_SPACECRAFT_PIPELINE_ID)
    isolated = {
        item["stage_id"]
        for item in pipeline["stages"]
        if item["execution_profile"]["parallel_execution_boundary"] == "ISOLATED_PROCESS"
    }
    assert isolated == set(WHOLE_SPACECRAFT_ONLINE_HARD_ISOLATION_CANDIDATES)
    for item in pipeline["stages"]:
        if item["stage_id"] in isolated:
            continue
        profile = item["execution_profile"]
        assert profile["cancellation_qualification_id"] is None
        assert profile["parallel_execution_boundary"] == "IN_PROCESS"
