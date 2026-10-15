from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.integrations.dor_multitask import (
    DORArtifactBinding,
    DORIdentificationRunRequest,
    DORMultitaskAuditRequest,
    DORMultitaskAuditService,
    DORMultitaskStatus,
    multitask_release_blockers,
)


DOR = Path(os.environ.get("ASTROGRAPH_DOR_EXECUTABLE", "__optional_dor_not_found__"))


def _write_json(tmp_path: Path, name: str, payload: dict) -> DORArtifactBinding:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return DORArtifactBinding(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _write_csv(tmp_path: Path) -> DORArtifactBinding:
    path = tmp_path / "evidence.csv"
    path.write_text("evidence_id,validity\ne1,VALID\n", encoding="utf-8")
    return DORArtifactBinding(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _artifacts(tmp_path: Path, *, detection: str = "ABNORMAL") -> dict[str, DORArtifactBinding]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return {
        "detection": _write_json(
            tmp_path,
            "detection.json",
            {"status": "SUPPORTED", "diagnosis": detection},
        ),
        "classification": _write_json(
            tmp_path,
            "classification.json",
            {"status": "SUPPORTED", "prediction_set": ["A__failed_off"]},
        ),
        "localization": _write_json(
            tmp_path,
            "localization.json",
            {
                "contract": "DOR-LOCALIZATION-RESULT/0.5",
                "status": "SUPPORTED",
                "fault_hypotheses": [
                    {
                        "fault_id": "A__failed_off",
                        "fault_mode": "failed_off",
                        "matched_evidence": ["e1"],
                    }
                ],
            },
        ),
        "root_cause": _write_json(
            tmp_path,
            "root-cause.json",
            {
                "contract": "DOR-ROOT-CAUSE-RESULT/0.6",
                "status": "SUPPORTED",
                "root_cause_set": ["A__failed_off"],
            },
        ),
        "ontology": _write_json(
            tmp_path,
            "ontology.json",
            {
                "contract": "DOR-FAULT-ONTOLOGY/0.5",
                "ontology_id": "astrograph-multitask-test",
                "domain": "test",
                "components": [{"component_id": "A", "kind": "PHYSICAL"}],
                "sensors": [],
                "commands": [],
                "topology_edges": [],
                "faults": [
                    {
                        "fault_id": "A__failed_off",
                        "target_component_id": "A",
                        "fault_mode": "failed_off",
                        "category": "DISCRETE",
                        "exclusion_group": "A__state",
                        "expected_evidence": [
                            {
                                "scope_type": "COMPONENT",
                                "scope_id": "A",
                                "expected_state": "FAILED_OFF",
                                "weight": 1.0,
                            }
                        ],
                    }
                ],
                "provenance": {"sources": ["test-fixture"]},
            },
        ),
        "evidence_csv": _write_csv(tmp_path),
    }


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_full_chain_runs_identification_then_multitask_without_new_ranking(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    audit = DORMultitaskAuditService().audit(
        DORMultitaskAuditRequest(
            detection=artifacts["detection"],
            classification=artifacts["classification"],
            localization=artifacts["localization"],
            root_cause=artifacts["root_cause"],
            identification_run=DORIdentificationRunRequest(
                ontology=artifacts["ontology"],
                evidence_csv=artifacts["evidence_csv"],
            ),
            case_id="case-a",
            executable_path=str(DOR),
        )
    )

    assert audit.status == DORMultitaskStatus.SUPPORTED
    assert audit.diagnostic_depth == "IDENTIFICATION"
    assert audit.strongest_fault_ids == ["A__failed_off"]
    assert audit.physical_change_claimed
    assert audit.generated_identification_sha256 is not None
    assert audit.runtime is not None
    assert audit.dor_result is not None
    assert audit.dor_result["claims"]["classification"]["prediction_set"] == [
        "A__failed_off"
    ]
    assert multitask_release_blockers(audit, "A__failed_off") == []
    assert multitask_release_blockers(audit, "B__failed_off")[0].startswith(
        "DOR_MULTITASK_FAULT_MISMATCH"
    )


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_normal_detection_conflicting_with_downstream_faults_abstains(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, detection="NORMAL")
    audit = DORMultitaskAuditService().audit(
        DORMultitaskAuditRequest(
            detection=artifacts["detection"],
            localization=artifacts["localization"],
            root_cause=artifacts["root_cause"],
            executable_path=str(DOR),
        )
    )

    assert audit.status == DORMultitaskStatus.ABSTAIN
    assert audit.diagnostic_depth == "NONE"
    assert audit.contradictions
    assert not audit.eligible_for_formal_fault_release
    assert multitask_release_blockers(audit, "A__failed_off") == [
        "DOR_MULTITASK_CONTRADICTION"
    ]


def test_artifact_hash_mismatch_fails_before_runtime_execution(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    bad_detection = artifacts["detection"].model_copy(update={"sha256": "0" * 64})
    audit = DORMultitaskAuditService().audit(
        DORMultitaskAuditRequest(
            detection=bad_detection,
            localization=artifacts["localization"],
            executable_path=str(tmp_path / "missing-dor"),
        )
    )

    assert audit.status == DORMultitaskStatus.ERROR
    assert "SHA-256 mismatch" in audit.reason
    assert audit.runtime is None


def test_identification_run_requires_localization() -> None:
    binding = DORArtifactBinding(path="placeholder", sha256="0" * 64)
    with pytest.raises(ValueError, match="requires a localization"):
        DORMultitaskAuditRequest(
            detection=binding,
            identification_run=DORIdentificationRunRequest(
                ontology=binding,
                evidence_csv=binding,
            ),
        )


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
def test_online_workflow_persists_multitask_gate_without_rewriting_diagnosis(
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
    artifacts = _artifacts(tmp_path / "dor-inputs")
    request = DORMultitaskAuditRequest(
        detection=artifacts["detection"],
        classification=artifacts["classification"],
        localization=artifacts["localization"],
        root_cause=artifacts["root_cause"],
        identification_run=DORIdentificationRunRequest(
            ontology=artifacts["ontology"],
            evidence_csv=artifacts["evidence_csv"],
        ),
        executable_path=str(DOR),
    )
    event = SimulationRunCompleted(
        event_id="DOR-MULTITASK-ONLINE",
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
            task_id="DOR-MULTITASK-ONLINE",
            task_type="simulation.online_diagnosis",
            input_payload={
                "simulation_event": event.model_dump(mode="json"),
                "protocol_root": str(root / "protocols/phm_project_a"),
                "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
                "artifact_output_dir": str(artifact_dir),
                "reasoner_mode": "rule",
                "dor_multitask_audit": request.model_dump(mode="json"),
            },
            execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
            permission_scope=permissions,
        ),
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="DOR-MULTITASK-ONLINE",
            permissions=permissions,
        ),
    )

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["dor_multitask_audit"]["status"] == "SUPPORTED"
    assert result.output_payload["decision_record"]["accepted_fault_id"] == (
        "THERMAL_HEATER_STUCK_ON"
    )
    assert "SEMANTIC_QUALIFICATION_NOT_SUPPORTED" in (
        result.output_payload["diagnosis_policy_decision"]["blocking_reasons"]
    )
    assert any(
        reason.startswith("DOR_MULTITASK_FAULT_MISMATCH")
        for reason in result.output_payload["diagnosis_policy_decision"]["warnings"]
    )
    audit_path = artifact_dir / "audit/dor_multitask_audit.json"
    assert audit_path.is_file()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "astrograph-online-artifact-manifest.v12"
    assert manifest["files"]["dor_multitask_audit"]["sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
