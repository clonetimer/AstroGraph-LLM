from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from astrograph.investigation import HypothesisSet
from astrograph.orchestration.llm_diagnostic_loop import DiagnosticActionProposal
from astrograph.orchestration.models import (
    ExecutionPolicy,
    ExecutionStatus,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
from astrograph.simulation import SimulationRunCompleted
from astrograph.tools.online import build_online_tool_registry

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "online_base"
PROTOCOL = ROOT / "protocols" / "phm_project_a"


class EchoReferenceBackend:
    model_id = "fake-online-reasoner"
    model_version = "test-v1"

    def __init__(self) -> None:
        self.last_call_metadata: dict[str, Any] = {}

    def generate_structured(self, *, agent_id: str, output_model, payload: dict[str, Any]):
        self.last_call_metadata = {"status": "SUCCEEDED", "mode": "echo-reference"}
        return output_model.model_validate(payload["reference_hypothesis_set"])


class FailingBackend(EchoReferenceBackend):
    def generate_structured(self, *, agent_id: str, output_model, payload: dict[str, Any]):
        raise TimeoutError("simulated vLLM timeout")


class ControlledLoopBackend(EchoReferenceBackend):
    supports_diagnostic_deliberation = True

    def generate_structured(self, *, agent_id, output_model, payload):
        if output_model is HypothesisSet:
            assert payload["llm_diagnostic_deliberation"]["status"] == "COMPLETED"
            return super().generate_structured(
                agent_id=agent_id, output_model=output_model, payload=payload
            )
        assert output_model is DiagnosticActionProposal
        observations = payload["observations"]
        if not observations:
            value = {"capability_id": "inspect_evidence_status", "objective": "inspect"}
        elif len(observations) == 1:
            value = {
                "capability_id": "inspect_deterministic_core",
                "objective": "cross-check",
                "cited_observation_ids": [observations[-1]["observation_id"]],
            }
        else:
            value = {
                "capability_id": "synthesize_hypotheses",
                "objective": "synthesize",
                "cited_observation_ids": [item["observation_id"] for item in observations],
            }
        return output_model.model_validate(value)


class ControlledLoopSynthesisFailureBackend(ControlledLoopBackend):
    def generate_structured(self, *, agent_id, output_model, payload):
        if output_model is HypothesisSet:
            raise TimeoutError("simulated final synthesis timeout")
        return super().generate_structured(
            agent_id=agent_id, output_model=output_model, payload=payload
        )


def _permissions():
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _invoke(tmp_path: Path, mode: str, backend=None):
    event = SimulationRunCompleted(
        event_id="SIM-TEST-HEATER",
        run_id="heater_stuck_on",
        run_dir=str(FIXTURE / "heater_stuck_on"),
        nominal_run_dir=str(FIXTURE / "nominal"),
    )
    permissions = _permissions()
    envelope = TaskEnvelope(
        task_id=f"TEST-{mode}",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / mode),
            "shadow_mode": True,
            "reasoner_mode": mode,
        },
        execution_policy=ExecutionPolicy(
            deterministic_only=mode == "rule",
            allow_llm=mode != "rule",
        ),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.5",
        run_id="heater_stuck_on",
        permissions=permissions,
    )
    return WholeSpacecraftOnlineDiagnosisWorkflow(reasoning_backend=backend).invoke(envelope, context)


def test_online_registry_contains_all_guarded_tools():
    specs = {item.tool_id: item for item in build_online_tool_registry().list_specs()}
    assert "online.runtime_evidence.build" in specs
    assert "diagnosis.invoke_online_core" in specs
    assert "online.evidence_graph.build" in specs
    assert "online.artifacts.persist" in specs
    assert specs["online.artifacts.persist"].required_permissions == [
        PermissionScope.WRITE_ARTIFACTS,
        PermissionScope.WRITE_AUDIT,
    ]


def test_rule_online_flow_uses_tool_ids_and_independent_core(tmp_path):
    result = _invoke(tmp_path, "rule")
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["decision_record"]["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"
    core = result.output_payload["deterministic_core_result"]
    assert core["reference_source"] == "deterministic evidence evaluation"
    trace = result.output_payload["agent_trace"]
    tool_ids = {item["tool_id"] for item in trace["events"] if item.get("tool_id")}
    assert {
        "semantics.validate",
        "online.runtime_evidence.build",
        "diagnosis.invoke_online_core",
        "online.evidence_graph.build",
        "online.artifacts.persist",
    }.issubset(tool_ids)
    artifact = tmp_path / "rule" / "diagnosis" / "deterministic_core_result.json"
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8"))["predicted_case_id"] == "thermal_heater_stuck_on"
    policy = result.output_payload["diagnosis_policy_decision"]
    assert policy["status"] == "UNKNOWN"
    activation = result.output_payload["reasoning_activation"]
    assert activation["llm_invoked"] is False
    assert activation["deterministic_path_completed_first"] is True
    assert result.output_payload["structured_diagnosis_output"]["execution_context_summary"]["reasoning_activation"] == activation
    assert "MISSING_MODEL_EVIDENCE" in policy["blocking_reasons"]
    assert "MISSING_KG_EVIDENCE" in policy["blocking_reasons"]
    assert "UPSTREAM_FORMAL_RELEASE_NOT_ALLOWED" in policy["blocking_reasons"]
    policy_path = tmp_path / "rule" / "diagnosis" / "diagnosis_policy_decision.json"
    bundle_path = tmp_path / "rule" / "diagnosis" / "diagnosis_evidence_bundle.json"
    assert policy_path.is_file()
    assert bundle_path.is_file()
    manifest = json.loads((tmp_path / "rule" / "manifest.json").read_text(encoding="utf-8"))
    assert "diagnosis_policy_decision" in manifest["files"]
    assert "diagnosis_evidence_bundle" in manifest["files"]
    assert "structured_diagnosis_output" in manifest["files"]
    structured_path = tmp_path / "rule" / "report" / "structured_diagnosis_output.json"
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    assert structured["final_decision"]["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"
    assert structured["safety_and_release_summary"]["policy_status"] == "UNKNOWN"
    assert "decision_record" in structured["lineage_sha256"]


def test_llm_timeout_falls_back_to_deterministic_advice(tmp_path):
    result = _invoke(tmp_path, "hybrid", FailingBackend())
    assert result.status == ExecutionStatus.SUCCEEDED
    activation = result.output_payload["reasoning_activation"]
    assert "LLM_ERROR_DETERMINISTIC_FALLBACK" in activation["trigger_codes"]
    assert result.output_payload["llm_runtime_snapshot"]["runtime_snapshot"]["llm_error_type"] == "TimeoutError"


def test_hybrid_online_flow_accepts_structured_backend(tmp_path):
    result = _invoke(tmp_path, "hybrid", EchoReferenceBackend())
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_payload["reasoner_mode"] == "hybrid"
    assert result.output_payload["agent_trace"]["model_id"] == "fake-online-reasoner"


def test_hybrid_online_flow_persists_substantive_controlled_llm_loop(tmp_path):
    result = _invoke(tmp_path, "hybrid", ControlledLoopBackend())
    trajectory = result.output_payload["llm_diagnostic_trajectory"]
    assert trajectory["status"] == "COMPLETED"
    assert trajectory["substance_audit"]["status"] == "SUBSTANTIVE_METHOD_PATH"
    assert trajectory["substance_audit"]["executed_diagnostic_capability_count"] == 2
    structured = result.output_payload["structured_diagnosis_output"]
    assert structured["execution_context_summary"]["llm_diagnostic_trajectory"] == trajectory
    assert structured["lineage_sha256"]["llm_diagnostic_trajectory"]
    trajectory_path = (
        tmp_path / "hybrid" / "diagnosis" / "llm_diagnostic_trajectory.json"
    )
    assert json.loads(trajectory_path.read_text(encoding="utf-8")) == trajectory


def test_final_llm_failure_invalidates_substance_claim_and_uses_reference(tmp_path):
    result = _invoke(tmp_path, "hybrid", ControlledLoopSynthesisFailureBackend())
    trajectory = result.output_payload["llm_diagnostic_trajectory"]
    assert trajectory["status"] == "POLICY_BLOCKED"
    assert trajectory["final_synthesis_consumes_trajectory"] is False
    assert trajectory["substance_audit"]["status"] == "WRAPPER_RISK"
    assert trajectory["termination_reason"] == "FINAL_SYNTHESIS_FAILED:TimeoutError"
    assert "LLM_ERROR_DETERMINISTIC_FALLBACK" in result.output_payload[
        "reasoning_activation"
    ]["trigger_codes"]
    assert result.output_payload["decision_record"]["comparison_status"] == "MATCH"
    assert result.output_payload["decision_record"]["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"


def test_reasoner_mode_requires_correct_backend_policy(tmp_path):
    llm_without_backend = _invoke(tmp_path, "llm", None)
    assert llm_without_backend.status == ExecutionStatus.FAILED
    assert llm_without_backend.error.code == "ASTROGRAPH_REASONER_BACKEND_REQUIRED"

    rule_with_backend = _invoke(tmp_path, "rule", EchoReferenceBackend())
    assert rule_with_backend.status == ExecutionStatus.FAILED
    assert rule_with_backend.error.code == "ASTROGRAPH_REASONER_MODE_MISMATCH"
