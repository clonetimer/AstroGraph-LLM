from __future__ import annotations

from pathlib import Path

from astrograph.agents.agentic import DiagnosticReasonerAgent, StructuredReasoningBackend
from astrograph.investigation import HypothesisSet, HypothesisType
from astrograph.orchestration.models import (
    ExecutionPolicy,
    ExecutionStatus,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
from astrograph.simulation import SimulationRunCompleted, WholeSpacecraftRuntimeEvidenceBuilder

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "online_faults"
PROTOCOL = ROOT / "protocols" / "phm_project_a"

EXPECTED = {
    "reaction_wheel_jam": "ADCS_RW_JAM",
    "battery_internal_resistance_increase": "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE",
    "solar_array_string_failure": "EPS_SOLAR_ARRAY_STRING_FAILURE",
    "gyro_bias_drift": "ADCS_GYRO_BIAS_DRIFT",
    "transmitter_output_power_loss": "COMM_TRANSMITTER_OUTPUT_POWER_LOSS",
    "storage_write_failure": "DATA_STORAGE_WRITE_FAILURE",
}


def _permissions():
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _invoke_rule(tmp_path: Path, scenario: str):
    permissions = _permissions()
    event = SimulationRunCompleted(
        event_id=f"SIM-V057-{scenario}",
        run_id=scenario,
        run_dir=str(FIXTURE / scenario),
        nominal_run_dir=str(FIXTURE / "nominal"),
    )
    envelope = TaskEnvelope(
        task_id=f"V057-RULE-{scenario}",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "rule" / scenario),
            "shadow_mode": True,
            "reasoner_mode": "rule",
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.7",
        run_id=scenario,
        permissions=permissions,
    )
    return WholeSpacecraftOnlineDiagnosisWorkflow().invoke(envelope, context)


def test_v057_first_six_faults_are_accepted_by_rule_and_core(tmp_path):
    for scenario, fault_id in EXPECTED.items():
        result = _invoke_rule(tmp_path, scenario)
        assert result.status == ExecutionStatus.SUCCEEDED, scenario
        decision = result.output_payload["decision_record"]
        assert decision["disposition"] == "ACCEPT", scenario
        assert decision["accepted_fault_id"] == fault_id, scenario
        assert decision["comparison_status"] == "MATCH", scenario
        bundle = result.output_payload["episode_evidence_bundle"]
        assert fault_id in bundle["analysis_metadata"]["registered_mechanisms_detected"]


def test_v057_radiator_plus_heater_uses_composite_member_status(tmp_path):
    result = _invoke_rule(tmp_path, "radiator_plus_heater")
    assert result.status == ExecutionStatus.SUCCEEDED
    decision = result.output_payload["decision_record"]
    assert decision["disposition"] == "HUMAN_REVIEW"
    assert decision["comparison_status"] == "COMPOSITE_MEMBER_EVIDENCE_ONLY"
    assert decision["requires_human_review"] is True


class _VariableNoFaultBackend(StructuredReasoningBackend):
    model_id = "test-variable-no-fault"
    model_version = "v1"

    def generate_structured(self, *, agent_id, output_model, payload):
        reference = HypothesisSet.model_validate(payload["reference_hypothesis_set"])
        changed = []
        for item in reference.hypotheses:
            if item.hypothesis_type == HypothesisType.NO_REGISTERED_FAULT:
                changed.append(item.model_copy(update={
                    "supporting_evidence_fact_ids": [],
                    "reasoning_summary": "LLM omitted deterministic no-fault evidence references.",
                }))
            else:
                changed.append(item)
        return reference.model_copy(update={"hypotheses": changed})


def test_v057_no_registered_fault_structure_is_deterministically_normalized():
    event = SimulationRunCompleted(
        event_id="SIM-V057-NOMINAL",
        run_id="nominal",
        run_dir=str(FIXTURE / "nominal"),
        nominal_run_dir=str(FIXTURE / "nominal"),
    )
    bundle, gate = WholeSpacecraftRuntimeEvidenceBuilder(PROTOCOL).build(event)
    assert gate.passed
    reference = DiagnosticReasonerAgent.reference_hypotheses(bundle)
    candidate = DiagnosticReasonerAgent(backend=_VariableNoFaultBackend()).reason(bundle)
    ref_nf = next(x for x in reference.hypotheses if x.hypothesis_type == HypothesisType.NO_REGISTERED_FAULT)
    got_nf = next(x for x in candidate.hypotheses if x.hypothesis_type == HypothesisType.NO_REGISTERED_FAULT)
    assert got_nf.model_dump(exclude={"created_at"}) == ref_nf.model_dump(exclude={"created_at"})
    assert candidate.validation_metadata["no_registered_fault_normalized"] is True
