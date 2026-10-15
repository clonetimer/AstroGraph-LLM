from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.integrations.dor_vertical import (
    DORVerticalFreezeAuditRequest,
    DORVerticalFreezeAuditService,
    DORVerticalFreezeStatus,
    vertical_freeze_release_blockers,
)


DOR = Path(os.environ.get("ASTROGRAPH_DOR_EXECUTABLE", "__optional_dor_not_found__"))
TASKS = ["DETECTION", "CLASSIFICATION", "LOCALIZATION", "ROOT_CAUSE", "IDENTIFICATION"]
CORE_GATES = [
    "CONTROL_PLANE_CONTRACTS",
    "CONTEXT_REFERENCE_POLICY",
    "FAULT_SEMANTICS_LOCALIZATION",
    "ROOT_CAUSE_REAL_HARDWARE",
    "IDENTIFICATION_DUAL_ROUTE",
    "MULTITASK_ARBITRATION",
    "REGRESSION_AND_SCHEMA_VERIFICATION",
]


def _write_json(tmp_path: Path, name: str, payload: dict) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(
    tmp_path: Path,
    *,
    weak_task: str | None = None,
    cross_domain_task: str | None = None,
    declared_grade: str = "HIGH_FIDELITY_SYSTEM_SIMULATION",
) -> tuple[Path, str, Path, str, dict[str, str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    case_artifact = tmp_path / "validated-case-report.json"
    case_artifact.write_text('{"status":"PASS","claim_boundary":"test fixture"}\n', encoding="utf-8")
    case_sha = hashlib.sha256(case_artifact.read_bytes()).hexdigest()
    registry = {
        "target_domain": "SATELLITE",
        "requirements": {
            task: {
                "minimum_evidence_grade": declared_grade,
                "require_target_domain": True,
            }
            for task in TASKS
        },
        "cases": [
            {
                "case_id": f"satellite-{task.lower()}",
                "task": task,
                "evidence_grade": (
                    "CONTROLLED_SYNTHETIC" if task == weak_task else declared_grade
                ),
                "domain": "AEROSPACE_TESTBED" if task == cross_domain_task else "SATELLITE",
                "state": "PASS",
                "artifact": str(case_artifact),
                "artifact_sha256": case_sha,
                "provenance": "hash-bound test fixture",
            }
            for task in TASKS
        ],
    }
    registry_path, registry_sha = _write_json(tmp_path, "vertical-registry.json", registry)
    freeze_path, freeze_sha = _write_json(
        tmp_path,
        "freeze-evidence.json",
        {
            "gates": [
                {
                    "gate": gate,
                    "status": "PASS",
                    "reason": "reviewed test fixture",
                    "evidence": [case_sha],
                }
                for gate in CORE_GATES
            ]
        },
    )
    grades = {task: "HIGH_FIDELITY_SYSTEM_SIMULATION" for task in TASKS}
    return registry_path, registry_sha, freeze_path, freeze_sha, grades


def _request(tmp_path: Path, **kwargs: object) -> DORVerticalFreezeAuditRequest:
    registry, registry_sha, freeze, freeze_sha, grades = _inputs(tmp_path, **kwargs)
    return DORVerticalFreezeAuditRequest(
        vertical_registry_path=str(registry),
        expected_vertical_registry_sha256=registry_sha,
        freeze_evidence_path=str(freeze),
        expected_freeze_evidence_sha256=freeze_sha,
        expected_target_domain="SATELLITE",
        minimum_task_grades=grades,
        executable_path=str(DOR),
    )


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_all_five_target_domain_gates_can_reach_freeze_ready(tmp_path: Path) -> None:
    audit = DORVerticalFreezeAuditService().audit(_request(tmp_path))

    assert audit.status == DORVerticalFreezeStatus.READY_FOR_FREEZE
    assert audit.eligible_for_freeze
    assert audit.vertical_status == "FULLY_VALIDATED"
    assert audit.freeze_status == "READY_FOR_1_0_FREEZE"
    assert set(audit.verified_case_artifact_sha256) == {
        f"satellite-{task.lower()}" for task in TASKS
    }
    assert audit.runtime is not None
    assert vertical_freeze_release_blockers(audit) == []


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
@pytest.mark.parametrize(
    ("kwargs", "missing_task"),
    [
        ({"weak_task": "CLASSIFICATION"}, "CLASSIFICATION"),
        ({"cross_domain_task": "LOCALIZATION"}, "LOCALIZATION"),
    ],
)
def test_lower_grade_or_cross_domain_case_cannot_upgrade_vertical_gate(
    tmp_path: Path,
    kwargs: dict[str, str],
    missing_task: str,
) -> None:
    audit = DORVerticalFreezeAuditService().audit(_request(tmp_path, **kwargs))

    assert audit.status == DORVerticalFreezeStatus.BLOCKED
    assert audit.vertical_status == "PARTIALLY_VALIDATED"
    assert missing_task in audit.missing_vertical_gates
    assert "SATELLITE_VERTICAL" in audit.blocking_freeze_gates
    assert vertical_freeze_release_blockers(audit) == ["DOR_VERTICAL_FREEZE_BLOCKED"]


def test_weaker_registry_policy_fails_before_runtime_execution(tmp_path: Path) -> None:
    request = _request(tmp_path, declared_grade="CONTROLLED_SYNTHETIC").model_copy(
        update={"executable_path": str(tmp_path / "missing-dor")}
    )
    audit = DORVerticalFreezeAuditService().audit(request)

    assert audit.status == DORVerticalFreezeStatus.ERROR
    assert "declares weaker evidence" in audit.reason
    assert audit.runtime is None


def test_vertical_registry_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    request = _request(tmp_path).model_copy(
        update={
            "expected_vertical_registry_sha256": "0" * 64,
            "executable_path": str(tmp_path / "missing-dor"),
        }
    )
    audit = DORVerticalFreezeAuditService().audit(request)

    assert audit.status == DORVerticalFreezeStatus.ERROR
    assert "SHA-256 mismatch" in audit.reason
    assert audit.runtime is None


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_online_workflow_persists_blocked_vertical_gate_without_changing_diagnosis(
    tmp_path: Path,
) -> None:
    from astrograph.orchestration.models import (
        ExecutionPolicy,
        ExecutionStatus,
        PermissionScope,
        TaskEnvelope,
        WorkflowContext,
    )
    from astrograph.orchestration.whole_spacecraft_online_workflow import (
        WholeSpacecraftOnlineDiagnosisWorkflow,
    )
    from astrograph.simulation import SimulationRunCompleted

    root = Path(__file__).resolve().parents[1]
    scene_root = root / "tests/fixtures/spacecraft_scenarios"
    vertical_request = _request(tmp_path / "dor-inputs", weak_task="CLASSIFICATION")
    event = SimulationRunCompleted(
        event_id="DOR-VERTICAL-ONLINE",
        run_id="heater_stuck_on",
        run_dir=str(scene_root / "heater_stuck_on"),
        nominal_run_dir=str(scene_root / "nominal"),
    )
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    artifact_dir = tmp_path / "artifacts"
    workflow = WholeSpacecraftOnlineDiagnosisWorkflow()
    result = workflow.invoke(
        TaskEnvelope(
            task_id="DOR-VERTICAL-ONLINE",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(root / "protocols/phm_project_a"),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(artifact_dir),
                "reasoner_mode": "rule",
                "dor_vertical_freeze_audit": vertical_request.model_dump(mode="json"),
            },
            execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="DOR-VERTICAL-ONLINE",
            permissions=permissions,
        ),
    )

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["dor_vertical_freeze_audit"]["status"] == "BLOCKED"
    assert result.output_payload["decision_record"]["accepted_fault_id"] == (
        "THERMAL_HEATER_STUCK_ON"
    )
    policy = result.output_payload["diagnosis_policy_decision"]
    assert "SEMANTIC_QUALIFICATION_NOT_SUPPORTED" in policy["blocking_reasons"]
    assert "DOR_VERTICAL_FREEZE_BLOCKED" in policy["warnings"]
    audit_path = artifact_dir / "audit/dor_vertical_freeze_audit.json"
    assert audit_path.is_file()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "astrograph-online-artifact-manifest.v12"
    assert manifest["files"]["dor_vertical_freeze_audit"]["sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
