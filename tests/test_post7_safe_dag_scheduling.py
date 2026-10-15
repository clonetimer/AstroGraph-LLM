from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from astrograph.runtime import (
    AstroGraphRuntime,
    CancellationMode,
    RuntimePipeline,
    RuntimeRequest,
    RuntimeStageContext,
    RuntimeStatus,
    SchedulerMode,
    StageExecutionProfile,
    StageSideEffect,
    StageResult,
)


@dataclass
class _ConcurrencyProbe:
    active: int = 0
    maximum: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _CooperativeStage:
    stage_version = "1"

    def __init__(self, stage_id: str, *, delay_s: float = 0.04, mutate_payload: bool = False) -> None:
        self.stage_id = stage_id
        self.delay_s = delay_s
        self.mutate_payload = mutate_payload

    def execute(self, payload: dict, context: RuntimeStageContext) -> StageResult:
        probe: _ConcurrencyProbe | None = context.dependencies.get("probe")
        if probe:
            probe.enter()
        try:
            deadline = time.monotonic() + self.delay_s
            while time.monotonic() < deadline:
                context.cancellation.checkpoint()
                time.sleep(0.002)
            result = dict(payload)
            if self.mutate_payload:
                result[self.stage_id] = True
            return StageResult.succeeded(result)
        finally:
            if probe:
                probe.leave()


def _profile(stage_id: str, *, units: int = 1) -> StageExecutionProfile:
    return StageExecutionProfile(
        stage_id=stage_id,
        parallel_safe=True,
        replay_equivalence_certified=True,
        payload_passthrough=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        side_effect=StageSideEffect.READ_ONLY,
        resource_units=units,
    )


def _runtime(stage_count: int = 2, *, delay_s: float = 0.04, mutate_last: bool = False, units: int = 1):
    probe = _ConcurrencyProbe()
    stages = tuple(
        _CooperativeStage(f"s{i}", delay_s=delay_s, mutate_payload=(mutate_last and i == stage_count - 1))
        for i in range(stage_count)
    )
    profiles = {stage.stage_id: _profile(stage.stage_id, units=units) for stage in stages}
    pipeline = RuntimePipeline(
        pipeline_id="test.parallel",
        stages=stages,
        dependencies={stage.stage_id: () for stage in stages},
        execution_profiles=profiles,
    )
    runtime = AstroGraphRuntime(
        [pipeline],
        context=RuntimeStageContext(dependencies={"probe": probe}),
    )
    return runtime, probe


def _request(**control) -> RuntimeRequest:
    return RuntimeRequest(
        request_id="post7",
        pipeline_id="test.parallel",
        payload={"seed": 1},
        metadata={"execution_control": control},
    )


def test_stable_parallel_policy_withholds_cooperative_only_stages() -> None:
    runtime, probe = _runtime()
    result = runtime.execute(_request(
        scheduler_mode=SchedulerMode.SAFE_PARALLEL.value,
        max_parallel_stages=2,
        max_resource_units=2,
        require_hard_cancel_for_parallel=True,
    ))
    assert result.status == RuntimeStatus.SUCCEEDED
    assert probe.maximum == 1
    control = result.metadata["execution_control"]
    assert control["parallel_stage_count"] == 0
    assert control["parallel_activation_withheld"] is True


def test_explicit_cooperative_research_mode_uses_bounded_parallel_scheduler() -> None:
    runtime, probe = _runtime(stage_count=3)
    result = runtime.execute(_request(
        scheduler_mode=SchedulerMode.SAFE_PARALLEL.value,
        max_parallel_stages=2,
        max_resource_units=2,
        require_hard_cancel_for_parallel=False,
        allow_cooperative_parallel=True,
    ))
    assert result.status == RuntimeStatus.SUCCEEDED
    assert probe.maximum == 2
    control = result.metadata["execution_control"]
    assert control["parallel_stage_count"] == 3
    assert control["parallel_batches_executed"] == 2


def test_resource_budget_prevents_oversubscription() -> None:
    runtime, probe = _runtime(stage_count=3, units=2)
    result = runtime.execute(_request(
        scheduler_mode=SchedulerMode.SAFE_PARALLEL.value,
        max_parallel_stages=3,
        max_resource_units=2,
        require_hard_cancel_for_parallel=False,
        allow_cooperative_parallel=True,
    ))
    assert result.status == RuntimeStatus.SUCCEEDED
    assert probe.maximum == 1
    assert result.metadata["execution_control"]["parallel_batches_executed"] == 3


def test_request_deadline_is_consumed_by_runtime_and_fails_closed() -> None:
    runtime, _ = _runtime(stage_count=1, delay_s=0.2)
    result = runtime.execute(_request(request_timeout_s=0.03))
    assert result.status == RuntimeStatus.FAILED
    assert result.error is not None
    assert result.error.code == "ASTROGRAPH_RUNTIME_STAGE_TIMEOUT"


def test_parallel_payload_mutation_contract_fails_closed() -> None:
    runtime, _ = _runtime(stage_count=2, mutate_last=True)
    result = runtime.execute(_request(
        scheduler_mode=SchedulerMode.SAFE_PARALLEL.value,
        max_parallel_stages=2,
        max_resource_units=2,
        require_hard_cancel_for_parallel=False,
        allow_cooperative_parallel=True,
    ))
    assert result.status == RuntimeStatus.FAILED
    assert result.error is not None
    assert result.error.code == "ASTROGRAPH_RUNTIME_PARALLEL_PAYLOAD_MUTATION"


def test_runtime_description_exposes_execution_profiles_and_levels() -> None:
    runtime, _ = _runtime(stage_count=2)
    description = runtime.describe()["pipelines"][0]
    assert description["levels"] == [["s0", "s1"]]
    assert all(item["execution_profile"]["parallel_safe"] for item in description["stages"])


def test_transactional_online_persistence_preserves_existing_output_on_write_failure(tmp_path, monkeypatch) -> None:
    import astrograph.tools.online as online

    root = tmp_path / "result"
    root.mkdir()
    sentinel = root / "sentinel.txt"
    sentinel.write_text("old", encoding="utf-8")

    def fail_body(payload, staging_root):
        staging_root.mkdir(parents=True)
        (staging_root / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(online, "_persist_online_artifacts_transaction_body", fail_body)
    try:
        online._persist_online_artifacts({"artifact_output_dir": str(root)})
    except RuntimeError as exc:
        assert "injected write failure" in str(exc)
    else:
        raise AssertionError("transaction failure was not propagated")

    assert sentinel.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".result.astrograph-txn-*"))
    assert not list(tmp_path.glob(".result.astrograph-backup-*"))


def test_transactional_online_persistence_retries_transient_windows_commit_denial(tmp_path, monkeypatch) -> None:
    import astrograph.tools.online as online

    if online.os.name != "nt":
        return
    root = tmp_path / "result"

    def write_body(payload, staging_root):
        staging_root.mkdir()
        (staging_root / "receipt.txt").write_text("committed", encoding="utf-8")
        return {}

    original_replace = online.os.replace
    attempts = 0

    def transient_replace(source, target):
        nonlocal attempts
        if Path(source).name.startswith(".result.astrograph-txn-"):
            attempts += 1
            if attempts == 1:
                raise PermissionError("transient Windows commit denial")
        return original_replace(source, target)

    monkeypatch.setattr(online, "_persist_online_artifacts_transaction_body", write_body)
    monkeypatch.setattr(online.os, "replace", transient_replace)
    result = online._persist_online_artifacts({"artifact_output_dir": str(root)})
    assert attempts == 2
    assert (root / "receipt.txt").read_text(encoding="utf-8") == "committed"
    assert result["manifest_path"] == str(root / "manifest.json")
    assert not list(tmp_path.glob(".result.astrograph-txn-*"))


def test_transactional_online_persistence_restores_old_output_after_persistent_denial(tmp_path, monkeypatch) -> None:
    import astrograph.tools.online as online

    if online.os.name != "nt":
        return
    root = tmp_path / "result"
    root.mkdir()
    (root / "sentinel.txt").write_text("old", encoding="utf-8")

    def write_body(payload, staging_root):
        staging_root.mkdir()
        (staging_root / "receipt.txt").write_text("new", encoding="utf-8")
        return {}

    original_replace = online.os.replace
    attempts = 0

    def denied_replace(source, target):
        nonlocal attempts
        if Path(source).name.startswith(".result.astrograph-txn-"):
            attempts += 1
            raise PermissionError("persistent Windows commit denial")
        return original_replace(source, target)

    monkeypatch.setattr(online, "_persist_online_artifacts_transaction_body", write_body)
    monkeypatch.setattr(online.os, "replace", denied_replace)
    monkeypatch.setattr(online.time, "sleep", lambda _: None)
    try:
        online._persist_online_artifacts({"artifact_output_dir": str(root)})
    except PermissionError as exc:
        assert "persistent Windows commit denial" in str(exc)
    else:
        raise AssertionError("persistent commit failure was not propagated")
    assert attempts == 5
    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "old"
    assert not (root / "receipt.txt").exists()
    assert not list(tmp_path.glob(".result.astrograph-txn-*"))
    assert not list(tmp_path.glob(".result.astrograph-backup-*"))



class _FailingStage(_CooperativeStage):
    def execute(self, payload: dict, context: RuntimeStageContext) -> StageResult:
        deadline = time.monotonic() + 0.02
        while time.monotonic() < deadline:
            context.cancellation.checkpoint()
            time.sleep(0.002)
        return StageResult.failed(code="INJECTED_FAILURE", message="boom", stage_id=self.stage_id)


def test_parallel_failure_propagates_cancellation_to_sibling() -> None:
    probe = _ConcurrencyProbe()
    failing = _FailingStage("fail", delay_s=0.02)
    sibling = _CooperativeStage("sibling", delay_s=0.8)
    pipeline = RuntimePipeline(
        pipeline_id="test.cancel",
        stages=(failing, sibling),
        dependencies={"fail": (), "sibling": ()},
        execution_profiles={"fail": _profile("fail"), "sibling": _profile("sibling")},
    )
    runtime = AstroGraphRuntime([pipeline], context=RuntimeStageContext(dependencies={"probe": probe}))
    started = time.monotonic()
    result = runtime.execute(RuntimeRequest(
        request_id="cancel", pipeline_id="test.cancel", payload={},
        metadata={"execution_control": {
            "scheduler_mode": "SAFE_PARALLEL",
            "max_parallel_stages": 2,
            "max_resource_units": 2,
            "require_hard_cancel_for_parallel": False,
            "allow_cooperative_parallel": True,
        }},
    ))
    elapsed = time.monotonic() - started
    assert result.status == RuntimeStatus.FAILED
    assert result.error is not None and result.error.code == "INJECTED_FAILURE"
    assert elapsed < 0.4
    by_stage = {item.stage_id: item for item in result.stages}
    assert by_stage["sibling"].error is not None
    assert by_stage["sibling"].error.code == "ASTROGRAPH_RUNTIME_STAGE_CANCELLED"

def test_online_certified_branches_are_serial_parallel_semantically_equivalent(tmp_path: Path) -> None:
    from astrograph.audit import NullAuditStore
    from astrograph.config import DiagnosisPolicy
    from astrograph.knowledge.repository import CsvKnowledgeRepository
    from astrograph.orchestration.models import ExecutionPolicy, PermissionScope, TaskEnvelope, WorkflowContext
    from astrograph.runtime import (
        WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS,
        WHOLE_SPACECRAFT_PIPELINE_ID,
        build_application_runtime,
    )
    from astrograph.simulation import SimulationRunCompleted

    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "fixtures" / "online_base"
    protocol = root / "protocols" / "phm_project_a"
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    runtime = build_application_runtime(
        repository=CsvKnowledgeRepository(root / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(), audit_store=NullAuditStore(), project_root=root,
    )

    def request(request_id: str, *, parallel: bool) -> RuntimeRequest:
        event = SimulationRunCompleted(
            event_id=f"SIM-{request_id}", run_id="heater_stuck_on",
            run_dir=str(fixture / "heater_stuck_on"), nominal_run_dir=str(fixture / "nominal"),
        )
        envelope = TaskEnvelope(
            task_id=f"TASK-{request_id}", task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(protocol),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(tmp_path / request_id),
                "shadow_mode": True, "reasoner_mode": "rule",
            },
            execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
            permission_scope=permissions,
        )
        context = WorkflowContext(
            workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.6",
            run_id="heater_stuck_on", permissions=permissions,
        )
        metadata = {}
        if parallel:
            metadata = {"execution_control": {
                "scheduler_mode": "SAFE_PARALLEL",
                "max_parallel_stages": 3,
                "max_resource_units": 3,
                "require_hard_cancel_for_parallel": False,
                "allow_cooperative_parallel": True,
            }}
        return RuntimeRequest(
            request_id=request_id, pipeline_id=WHOLE_SPACECRAFT_PIPELINE_ID,
            payload={"envelope": envelope.model_dump(mode="json"), "context": context.model_dump(mode="json")},
            metadata=metadata,
        )

    serial = runtime.execute(request("serial", parallel=False))
    parallel = runtime.execute(request("parallel", parallel=True))
    assert serial.status == parallel.status == RuntimeStatus.SUCCEEDED
    assert [item.stage_id for item in parallel.stages] == list(WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS)
    assert parallel.metadata["execution_control"]["parallel_stage_count"] == 3
    serial_hashes = {item.stage_id: item.artifact_semantic_sha256 for item in serial.stages}
    parallel_hashes = {item.stage_id: item.artifact_semantic_sha256 for item in parallel.stages}
    for stage_id in (
        "online.spatiotemporal.v1",
        "online.verification.v1",
        "online.deterministic_core.v1",
    ):
        assert serial_hashes[stage_id] == parallel_hashes[stage_id]
    assert serial.output["output_payload"]["decision_record"]["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"
    assert parallel.output["output_payload"]["decision_record"]["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"
