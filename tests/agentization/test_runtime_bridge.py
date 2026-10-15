from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.agentization.capabilities import CapabilityEnvironment, build_registry
from astrograph.agentization.protocol import CapabilityProtocolError
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
    WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS,
    WHOLE_SPACECRAFT_PIPELINE_ID,
    build_application_runtime,
)
from astrograph.simulation import SimulationRunCompleted


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "online_base"
PROTOCOL = ROOT / "protocols" / "phm_project_a"


def _runtime():
    return build_application_runtime(
        repository=CsvKnowledgeRepository(ROOT / "data" / "knowledge_seed" / "causes.csv"),
        policy=DiagnosisPolicy(),
        audit_store=NullAuditStore(),
        project_root=ROOT,
    )


@pytest.fixture
def registry(tmp_path: Path):
    env = CapabilityEnvironment(
        project_root=ROOT,
        allowed_roots=(ROOT, tmp_path),
        artifact_dir=tmp_path / "agent-artifacts",
        workflow_root=ROOT / "configs" / "agentization" / "workflows",
        external_audit_roots=(ROOT,),
        runtime_factory=_runtime,
    )
    return build_registry(env)


def _online_request(tmp_path: Path) -> dict:
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    event = SimulationRunCompleted(
        event_id="SIM-agent-runtime",
        run_id="heater_stuck_on",
        run_dir=str(FIXTURE / "heater_stuck_on"),
        nominal_run_dir=str(FIXTURE / "nominal"),
    )
    envelope = TaskEnvelope(
        task_id="TASK-agent-runtime",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "runtime-output"),
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
    return {
        "request_id": "agent-runtime-request",
        "pipeline_id": WHOLE_SPACECRAFT_PIPELINE_ID,
        "payload": {
            "envelope": envelope.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        },
    }


def test_runtime_description_exposes_fixed_atomic_boundary(registry) -> None:
    result = registry.invoke("runtime.describe", "DIAGNOSIS", {})
    assert result["agent_execution_boundary"] == "ATOMIC_RUNTIME_REQUEST"
    assert result["individual_stage_invocation_allowed"] is False
    assert result["policy_selection_allowed"] is False
    online = next(
        pipeline
        for pipeline in result["pipelines"]
        if pipeline["pipeline_id"] == WHOLE_SPACECRAFT_PIPELINE_ID
    )
    assert [stage["stage_id"] for stage in online["stages"]] == list(
        WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS
    )


def test_diagnosis_profile_executes_complete_runtime_dag(registry, tmp_path: Path) -> None:
    result = registry.invoke(
        "runtime.diagnosis.execute",
        "DIAGNOSIS",
        {"request": _online_request(tmp_path)},
    )
    assert result["execution"] == "PERFORMED_BY_DETERMINISTIC_ASTROGRAPH_RUNTIME"
    assert result["runtime_status"] == "SUCCEEDED"
    assert result["stage_count"] == len(WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS)
    assert len(result["request_sha256"]) == len(result["result_sha256"]) == 64
    receipts = result["runtime_result"]["stages"]
    assert [receipt["stage_id"] for receipt in receipts] == list(
        WHOLE_SPACECRAFT_ONLINE_DAG_STAGE_IDS
    )
    assert result["runtime_result"]["output"]["output_payload"]["decision_record"][
        "accepted_fault_id"
    ] == "THERMAL_HEATER_STUCK_ON"


def test_shadow_profile_cannot_execute_diagnosis(registry, tmp_path: Path) -> None:
    with pytest.raises(CapabilityProtocolError) as exc:
        registry.invoke(
            "runtime.diagnosis.execute",
            "SHADOW",
            {"request": _online_request(tmp_path)},
        )
    assert exc.value.code == "CAPABILITY_NOT_ALLOWED"


def test_runtime_bridge_rejects_paths_outside_allowed_roots(registry, tmp_path: Path) -> None:
    request = _online_request(tmp_path)
    request["payload"]["envelope"]["input_payload"]["protocol_root"] = "/etc"
    with pytest.raises(CapabilityProtocolError) as exc:
        registry.invoke(
            "runtime.diagnosis.execute", "DIAGNOSIS", {"request": request}
        )
    assert exc.value.code == "PATH_OUTSIDE_ALLOWED_ROOTS"


def test_reference_workflow_contains_only_registered_atomic_runtime_nodes(registry) -> None:
    workflow = json.loads(
        (ROOT / "configs" / "agentization" / "workflows" / "astrograph-diagnosis-v1.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = {item["name"] for item in registry.manifest()["capabilities"]}
    assert [node["capability"] for node in workflow["nodes"]] == [
        "runtime.describe",
        "runtime.diagnosis.execute",
    ]
    assert all(node["capability"] in capabilities for node in workflow["nodes"])
    assert workflow["runtime_execution_is_atomic"] is True
    assert workflow["individual_stage_invocation_allowed"] is False


def test_all_persistent_workflows_resolve_to_registered_profile_scoped_capabilities(registry) -> None:
    audit = registry.invoke("workflows.audit_capabilities", "MAINTAINER", {})
    assert audit["status"] == "PASS"
    assert audit["invalid_workflows"] == []
    rows = {row["workflow_id"]: row for row in audit["workflows"]}
    assert rows["astrograph-diagnosis-v1"]["status"] == "EXECUTABLE"
    assert rows["astrograph-confirmatory-sealed-v1"]["status"] == (
        "BLOCKED_BY_DECLARED_PREREQUISITES"
    )
    assert all(not row["missing_capabilities"] for row in rows.values())
    assert all(not row["profile_forbidden_capabilities"] for row in rows.values())
