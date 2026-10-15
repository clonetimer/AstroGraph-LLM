from __future__ import annotations

from pathlib import Path

from astrograph.audit import NullAuditStore
from astrograph.config import DiagnosisPolicy
from astrograph.knowledge.repository import CsvKnowledgeRepository
from astrograph.orchestration.models import (
    ExecutionPolicy,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.runtime import (
    RuntimeRequest,
    RuntimeStatus,
    StageStatus,
    WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES,
    WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS,
    WHOLE_SPACECRAFT_PIPELINE_ID,
    build_application_runtime,
)
from astrograph.simulation import SimulationRunCompleted


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "online_base"
PROTOCOL = ROOT / "protocols" / "phm_project_a"


def _permissions() -> list[PermissionScope]:
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _runtime():
    return build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(),
        audit_store=NullAuditStore(),
        project_root=ROOT,
    )


def _request(tmp_path: Path, *, request_id: str = "dag-request") -> RuntimeRequest:
    permissions = _permissions()
    event = SimulationRunCompleted(
        event_id=f"SIM-{request_id}",
        run_id="heater_stuck_on",
        run_dir=str(FIXTURE / "heater_stuck_on"),
        nominal_run_dir=str(FIXTURE / "nominal"),
    )
    envelope = TaskEnvelope(
        task_id=f"TASK-{request_id}",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / request_id),
            "shadow_mode": True,
            "reasoner_mode": "rule",
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.6",
        run_id="heater_stuck_on",
        permissions=permissions,
    )
    return RuntimeRequest(
        request_id=request_id,
        pipeline_id=WHOLE_SPACECRAFT_PIPELINE_ID,
        payload={
            "envelope": envelope.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        },
    )


def test_online_pipeline_is_exposed_as_fine_grained_dag() -> None:
    runtime = _runtime()
    pipeline = next(
        item
        for item in runtime.describe()["pipelines"]
        if item["pipeline_id"] == WHOLE_SPACECRAFT_PIPELINE_ID
    )
    assert [stage["stage_id"] for stage in pipeline["stages"]] == list(
        WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS
    )
    assert len(pipeline["stages"]) == 19
    assert all(
        stage["depends_on"] == list(WHOLE_SPACECRAFT_ONLINE_DAG_DEPENDENCIES[stage["stage_id"]])
        for stage in pipeline["stages"]
    )
    assert "compat.whole_spacecraft_online.v1" not in {
        stage["stage_id"] for stage in pipeline["stages"]
    }


def test_online_runtime_executes_all_dag_nodes_and_keeps_extensions_optional(tmp_path: Path) -> None:
    result = _runtime().execute(_request(tmp_path))
    assert result.status == RuntimeStatus.SUCCEEDED
    assert [receipt.stage_id for receipt in result.stages] == list(
        WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS
    )
    status_by_stage = {receipt.stage_id: receipt.status for receipt in result.stages}
    assert status_by_stage["online.evidence.v1"] == StageStatus.SUCCEEDED
    assert status_by_stage["online.reasoning.v1"] == StageStatus.SUCCEEDED
    assert status_by_stage["online.governor.v1"] == StageStatus.SUCCEEDED
    assert status_by_stage["online.extension.control_plane.v1"] == StageStatus.SKIPPED
    assert status_by_stage["online.extension.semantic.v1"] == StageStatus.SKIPPED
    assert status_by_stage["online.extension.mechanism.v1"] == StageStatus.SKIPPED
    assert status_by_stage["online.extension.calibration.v1"] == StageStatus.SKIPPED
    assert "m2n" not in " ".join(status_by_stage).lower()
    assert result.output["output_payload"]["decision_record"]["accepted_fault_id"] == (
        "THERMAL_HEATER_STUCK_ON"
    )


def test_runtime_dag_scratch_state_is_request_scoped(tmp_path: Path) -> None:
    runtime = _runtime()
    first = runtime.execute(_request(tmp_path, request_id="dag-1"))
    second = runtime.execute(_request(tmp_path, request_id="dag-2"))
    assert first.status == RuntimeStatus.SUCCEEDED
    assert second.status == RuntimeStatus.SUCCEEDED
    assert len(first.stages) == len(second.stages) == len(WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS)
    assert first.request_id == "dag-1"
    assert second.request_id == "dag-2"
