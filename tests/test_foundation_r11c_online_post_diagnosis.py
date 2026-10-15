"""R11-C synthetic online post-result explanation, never upstream fault detection."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from astrograph.evaluation import audit_artifact_dir
from astrograph.orchestration.models import (
    ExecutionPolicy, ExecutionStatus, PermissionScope, TaskEnvelope, WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
from astrograph.orchestration.online_post_diagnosis import (
    freeze_online_diagnosis, run_online_post_diagnosis_explanation,
)
from astrograph.simulation import SimulationRunCompleted

ROOT = Path(__file__).resolve().parents[1]


class Spy:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def enrich(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("SECRET_INTERNAL_MODEL_HOST_ERROR")
        return {
            "status": "SUCCEEDED", "generation": {"status": "NOT_CONFIGURED"},
            "diagnostic_candidate_can_be_overridden": False,
        }


def execute(tmp_path, service=None, review=None):
    event = SimulationRunCompleted(
        event_id="SYNTHETIC-R11C-HEATER", run_id="heater_stuck_on",
        run_dir=str(ROOT / "tests/fixtures/online_base/heater_stuck_on"),
        nominal_run_dir=str(ROOT / "tests/fixtures/online_base/nominal"),
    )
    permissions = [
        PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS, PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    envelope = TaskEnvelope(
        task_id="R11C-SYNTH-HEATER", task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(ROOT / "protocols/phm_project_a"),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "online"),
            "shadow_mode": True, "reasoner_mode": "rule",
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.6",
        run_id="heater_stuck_on", permissions=permissions,
    )
    workflow = WholeSpacecraftOnlineDiagnosisWorkflow(
        post_diagnosis_knowledge_service=service,
        post_diagnosis_material_review=review,
    )
    result = workflow.invoke(envelope, context)
    assert result.status == ExecutionStatus.SUCCEEDED
    return result


def test_online_freeze_only_uses_validated_host_output(tmp_path):
    result = execute(tmp_path)
    run_id, snapshot, receipt = freeze_online_diagnosis(result)
    assert run_id == "heater_stuck_on"
    assert snapshot["candidate_fault_id"] == result.output_payload["decision_record"]["accepted_fault_id"]
    assert snapshot["candidate_fault_ids"] == [snapshot["candidate_fault_id"]]
    assert receipt["structured_output_sha256"] == result.output_payload["structured_diagnosis_output"]["output_sha256"]
    assert receipt["diagnostic_vote_allowed"] is False
    assert snapshot["labels_used"] is False and snapshot["truth_used"] is False


def test_online_post_result_records_sidecar_without_changing_core_artifacts(tmp_path):
    result = execute(tmp_path)
    orig = deepcopy(result)
    spy = Spy()
    receipt = run_online_post_diagnosis_explanation(
        result=result, knowledge_service=spy, output_root=tmp_path / "online",
    )
    assert receipt["status"] == "SUCCEEDED"
    assert receipt["generation_status"] == "NOT_CONFIGURED"
    assert result == orig
    assert spy.calls[0]["diagnosis_snapshot"]["candidate_fault_id"] == result.output_payload["decision_record"]["accepted_fault_id"]
    assert spy.calls[0]["material_review_context"] is None
    assert receipt["review_binding"]["status"] == "NOT_AVAILABLE"
    assert (tmp_path / "online/post_diagnosis/online_explanation_receipt.json").exists()
    assert audit_artifact_dir(tmp_path / "online").manifest_valid


def test_opt_in_runtime_entrypoint_calls_after_persist_without_changing_result(tmp_path):
    spy = Spy()
    result = execute(tmp_path, service=spy)
    assert len(spy.calls) == 1
    assert result.output_payload["structured_diagnosis_output"]["output_sha256"]
    assert (tmp_path / "online/post_diagnosis/online_explanation_receipt.json").exists()
    assert audit_artifact_dir(tmp_path / "online").manifest_valid


def test_tampered_structured_output_and_run_identity_never_reach_model(tmp_path):
    result = execute(tmp_path)
    for alteration in ("run", "decision", "hash"):
        bad = deepcopy(result)
        if alteration == "run":
            bad.metadata["run_id"] = "FOREIGN-RUN"
        elif alteration == "decision":
            bad.output_payload["decision_record"]["accepted_fault_id"] = "FOREIGN-FAULT"
        else:
            bad.output_payload["structured_diagnosis_output"]["output_sha256"] = "0" * 64
        spy = Spy()
        receipt = run_online_post_diagnosis_explanation(
            result=bad, knowledge_service=spy, output_root=tmp_path / alteration,
        )
        assert receipt["status"] == "FAILED_CLOSED"
        assert not spy.calls
        assert "SECRET_" not in json.dumps(receipt)


def test_failed_model_call_is_isolated_from_successful_online_result(tmp_path):
    spy = Spy(fail=True)
    result = execute(tmp_path, service=spy)
    assert len(spy.calls) == 1
    receipt = json.loads(
        (tmp_path / "online/post_diagnosis/online_explanation_receipt.json").read_text()
    )
    assert receipt["status"] == "FAILED_CLOSED"
    assert receipt["reason_code"] == "OPTIONAL_EXPLANATION_FAILED"
    assert "SECRET_INTERNAL_MODEL_HOST_ERROR" not in json.dumps(receipt)
    assert result.status == ExecutionStatus.SUCCEEDED
    assert audit_artifact_dir(tmp_path / "online").manifest_valid
