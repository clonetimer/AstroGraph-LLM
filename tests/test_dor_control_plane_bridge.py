from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.integrations.dor_control import (
    DORControlPlaneAuditRequest,
    DORControlPlaneAuditService,
    DORControlPlaneStatus,
    control_plane_release_blockers,
)
from astrograph.integrations.dor_runtime import inspect_dor_runtime
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


DOR = Path(os.environ.get("ASTROGRAPH_DOR_EXECUTABLE", "__optional_dor_not_found__"))
DOR_PYTHON = Path(os.environ.get("ASTROGRAPH_DOR_PYTHON", "__optional_dor_python_not_found__"))
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not DOR_PYTHON.is_file(), reason="optional external DOR runtime is not installed")
def test_runtime_identity_rejects_stack_metadata_version_divergence(tmp_path: Path) -> None:
    executable = tmp_path / "dor-version-divergence"
    executable.write_text(
        f"#!{DOR_PYTHON}\n"
        "import json\n"
        "print(json.dumps({"
        "'DOR':{'distribution':'dor-diagnostic-stack','version':'0.0.0-stale','installed':True},"
        "'ISGN':{'distribution':'isgn-toolkit','version':'2.3.0+timer.1','installed':True},"
        "'M2N':{'distribution':'m2n-toolkit','version':'2.16.1+timer.1','installed':True},"
        "'FEC':{'distribution':'fec-toolkit','version':'2.4.2+timer.1','installed':True}}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    with pytest.raises(ValueError, match="DOR version identity mismatch"):
        inspect_dor_runtime(str(executable), timeout_s=10.0)


def _write_control_inputs(tmp_path: Path, *, leak: bool = False) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    roles = ["train"] * 4 + ["val"] * 2 + ["cal"] * 2 + ["test"] * 2
    labels = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    rows = []
    for index, (role, label) in enumerate(zip(roles, labels, strict=True)):
        run_id = f"run-{index}"
        if leak and index == 8:
            run_id = "run-0"
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "run_id": run_id,
                "split": role,
                "fault_label": label,
                "feature": float(index),
            }
        )
    data = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    data_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "contract": "DOR-DATASET-MANIFEST/0.4",
                "dataset_id": "astrograph-control-plane-test",
                "domain": "test",
                "task": "FAULT_CLASSIFICATION",
                "data": {"path": str(data), "sha256": data_sha256, "row_count": len(rows)},
                "identifiers": ["sample_id"],
                "split": {
                    "column": "split",
                    "train_values": ["train"],
                    "validation_values": ["val"],
                    "calibration_values": ["cal"],
                    "test_values": ["test"],
                },
                "supervision": {
                    "label_column": "fault_label",
                    "normal_label": "A",
                    "class_frame": ["A", "B"],
                },
                "context": {
                    "columns": [],
                    "policy": "EXACT",
                    "fallback_policy": "DECLARED_ANCESTOR_ONLY",
                },
                "calibration": {
                    "alpha": 0.1,
                    "min_group_size": 2,
                    "missing_context_policy": "ABSTAIN",
                },
                "leakage_guard": {"group_columns": ["run_id"]},
                "reference_scope": {
                    "kind": "ASSET_LOCAL",
                    "asset_id": "astrograph-control-plane-test-asset",
                },
                "mechanism": {"status": "NOT_PROVIDED", "reason": "unit test"},
                "provenance": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request = tmp_path / "diagnostic_request.json"
    request.write_text(
        json.dumps(
            {
                "contract": "M2N-DIAGNOSTIC-MODEL-REQUEST/1.0",
                "task": "FAULT_CLASSIFICATION",
                "data": {
                    "split_column": "split",
                    "train_values": ["train"],
                    "validation_values": ["val"],
                    "calibration_values": ["cal"],
                    "test_values": ["test"],
                    "id_columns": ["sample_id"],
                },
                "features": {"raw_columns": ["feature"], "mechanism_columns": [], "views": ["RAW"]},
                "supervision": {"label_column": "fault_label"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, request, data


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_control_plane_qualifies_bound_roles_and_records_runtime(tmp_path: Path) -> None:
    manifest, request, data = _write_control_inputs(tmp_path)
    audit = DORControlPlaneAuditService().audit(
        DORControlPlaneAuditRequest(
            dataset_manifest_path=str(manifest),
            diagnostic_request_path=str(request),
            data_path=str(data),
            executable_path=str(DOR),
            expected_reference_scope={
                "kind": "ASSET_LOCAL",
                "asset_id": "astrograph-control-plane-test-asset",
            },
        )
    )

    assert audit.status == DORControlPlaneStatus.QUALIFIED
    assert audit.eligible_for_diagnostic_execution
    assert audit.plan_contract == "DOR-DIAGNOSTIC-PLAN/0.4"
    assert audit.reference_safety is not None
    assert audit.reference_safety["test_labels_used"] is False
    assert audit.reference_safety["test_scores_used"] is False
    assert audit.reference_safety["implicit_pooling"] is False
    assert audit.declared_reference_scope == {
        "kind": "ASSET_LOCAL",
        "asset_id": "astrograph-control-plane-test-asset",
    }
    assert audit.runtime is not None
    assert audit.runtime.dor_version == audit.runtime.stack_info["DOR"]["version"]
    assert set(audit.runtime.components) == {"DOR", "ISGN", "M2N", "FEC"}
    assert all(
        component.stack_version == component.metadata_version == component.module_version
        for component in audit.runtime.components.values()
    )
    assert len(audit.runtime.stack_fingerprint_sha256) == 64
    assert audit.runtime.distribution_source_sha256 is not None
    assert audit.runtime.distribution_source_hash_scope == "dor-package-python-files"
    assert control_plane_release_blockers(audit) == []


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_control_plane_rejects_cross_role_group_leakage(tmp_path: Path) -> None:
    manifest, request, data = _write_control_inputs(tmp_path, leak=True)
    audit = DORControlPlaneAuditService().audit(
        DORControlPlaneAuditRequest(
            dataset_manifest_path=str(manifest),
            diagnostic_request_path=str(request),
            data_path=str(data),
            executable_path=str(DOR),
        )
    )

    assert audit.status == DORControlPlaneStatus.UNSUPPORTED
    assert not audit.eligible_for_diagnostic_execution
    assert audit.task_qualification is not None
    assert audit.task_qualification["checks"]["leakage_guard"]["cross_role_group_count"] == 1
    assert control_plane_release_blockers(audit) == ["DOR_CONTROL_PLANE_UNSUPPORTED"]


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_control_plane_rejects_reference_scope_conflicting_with_profile(tmp_path: Path) -> None:
    manifest, request, data = _write_control_inputs(tmp_path)
    audit = DORControlPlaneAuditService().audit(
        DORControlPlaneAuditRequest(
            dataset_manifest_path=str(manifest),
            diagnostic_request_path=str(request),
            data_path=str(data),
            executable_path=str(DOR),
            expected_reference_scope={
                "kind": "ASSET_LOCAL",
                "asset_id": "another-profile-asset",
            },
        )
    )

    assert audit.status == DORControlPlaneStatus.ERROR
    assert not audit.eligible_for_diagnostic_execution
    assert "does not match the resolved Dataset Profile" in audit.reason


def test_control_plane_missing_runtime_fails_closed(tmp_path: Path) -> None:
    manifest, request, data = _write_control_inputs(tmp_path)
    audit = DORControlPlaneAuditService().audit(
        DORControlPlaneAuditRequest(
            dataset_manifest_path=str(manifest),
            diagnostic_request_path=str(request),
            data_path=str(data),
            executable_path=str(tmp_path / "missing-dor"),
        )
    )

    assert audit.status == DORControlPlaneStatus.ERROR
    assert not audit.eligible_for_diagnostic_execution
    assert control_plane_release_blockers(audit) == ["DOR_CONTROL_PLANE_ERROR"]


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_online_workflow_persists_qualified_control_plane_audit(tmp_path: Path) -> None:
    manifest, diagnostic_request, data = _write_control_inputs(tmp_path / "control")
    fixture = ROOT / "tests/fixtures/online_base"
    event = SimulationRunCompleted(
        event_id="DOR-CONTROL-ONLINE",
        run_id="heater_stuck_on",
        run_dir=str(fixture / "heater_stuck_on"),
        nominal_run_dir=str(fixture / "nominal"),
    )
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    artifact_dir = tmp_path / "artifacts"
    envelope = TaskEnvelope(
        task_id="DOR-CONTROL-ONLINE",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(ROOT / "protocols/phm_project_a"),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(artifact_dir),
            "shadow_mode": True,
            "reasoner_mode": "rule",
            "dor_control_plane_audit": {
                "dataset_manifest_path": str(manifest),
                "diagnostic_request_path": str(diagnostic_request),
                "data_path": str(data),
                "executable_path": str(DOR),
            },
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    result = WholeSpacecraftOnlineDiagnosisWorkflow().invoke(
        envelope,
        WorkflowContext(
            workflow_id=WholeSpacecraftOnlineDiagnosisWorkflow.workflow_id,
            run_id="DOR-CONTROL-ONLINE",
            permissions=permissions,
        ),
    )

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["dor_control_plane_audit"]["status"] == "QUALIFIED"
    plan = result.output_payload["diagnosis_execution_plan"]
    enabled = {item["stage"] for item in plan["steps"] if item["enabled"]}
    assert "QUALIFY_DIAGNOSTIC_CONTROL_PLANE" in enabled
    persisted = artifact_dir / "audit/dor_control_plane_audit.json"
    assert persisted.is_file()
    artifact_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert artifact_manifest["schema_version"] == "astrograph-online-artifact-manifest.v12"
    assert artifact_manifest["files"]["dor_control_plane_audit"]["sha256"] == hashlib.sha256(
        persisted.read_bytes()
    ).hexdigest()
