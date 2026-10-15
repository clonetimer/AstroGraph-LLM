from __future__ import annotations

import json
from pathlib import Path

from astrograph.investigation import ClaimType, HypothesisStatus, HypothesisType
from astrograph.orchestration.models import ExecutionPolicy, ExecutionStatus, PermissionScope, TaskEnvelope, WorkflowContext
from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
from astrograph.simulation import SimulationRunCompleted

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "online_extended"
PROTOCOL = ROOT / "protocols" / "phm_project_a"

EXPECTED = {
    "battery_capacity_loss": "EPS_BATTERY_CAPACITY_LOSS",
    "solar_efficiency_loss": "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS",
    "radiator_rejection_loss": "THERMAL_RADIATOR_REJECTION_LOSS",
    "downlink_service_loss": "COMM_DOWNLINK_SERVICE_LOSS",
    "temperature_sensor_bias": "THERMAL_TEMPERATURE_SENSOR_BIAS",
    "temperature_sensor_drift": "THERMAL_TEMPERATURE_SENSOR_DRIFT",
}
COMPOSITES = {
    "eps_capacity_plus_solar": {"EPS_BATTERY_CAPACITY_LOSS", "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS"},
    "radiator_plus_heater": {"THERMAL_RADIATOR_REJECTION_LOSS", "THERMAL_HEATER_STUCK_ON"},
    "rw_friction_plus_star_tracker": {"ADCS_RW_FRICTION_INCREASE", "ADCS_STAR_TRACKER_DROPOUT"},
    "transmitter_plus_storage_write": {"COMM_TRANSMITTER_OUTPUT_POWER_LOSS", "DATA_STORAGE_WRITE_FAILURE"},
}


def _permissions():
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _invoke(tmp_path: Path, scenario: str):
    permissions = _permissions()
    event = SimulationRunCompleted(
        event_id=f"SIM-V058-{scenario}",
        run_id=scenario,
        run_dir=str(FIXTURE / scenario),
        nominal_run_dir=str(FIXTURE / "nominal"),
        metadata={"scenario_name_for_evaluation_only": scenario},
    )
    envelope = TaskEnvelope(
        task_id=f"V058-RULE-{scenario}",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / scenario),
            "shadow_mode": True,
            "reasoner_mode": "rule",
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id="simulation.whole_spacecraft.online_diagnosis.v0.5.8",
        run_id=scenario,
        permissions=permissions,
    )
    return WholeSpacecraftOnlineDiagnosisWorkflow().invoke(envelope, context)


def test_remaining_six_single_faults_are_formally_accepted(tmp_path):
    for scenario, fault_id in EXPECTED.items():
        result = _invoke(tmp_path, scenario)
        assert result.status == ExecutionStatus.SUCCEEDED, scenario
        decision = result.output_payload["decision_record"]
        assert decision["disposition"] == "ACCEPT", scenario
        assert decision["accepted_fault_id"] == fault_id, scenario
        assert decision["comparison_status"] == "MATCH", scenario
        hypotheses = result.output_payload["hypothesis_set"]["hypotheses"]
        supported = [item for item in hypotheses if item["target_fault_id"] == fault_id]
        assert len(supported) == 1
        assert supported[0]["status"] == "SUPPORTED"


def test_temperature_sensor_bias_and_drift_are_not_cross_accepted(tmp_path):
    bias = _invoke(tmp_path, "temperature_sensor_bias").output_payload["hypothesis_set"]["hypotheses"]
    drift = _invoke(tmp_path, "temperature_sensor_drift").output_payload["hypothesis_set"]["hypotheses"]
    bias_map = {item.get("target_fault_id"): item["status"] for item in bias}
    drift_map = {item.get("target_fault_id"): item["status"] for item in drift}
    assert bias_map["THERMAL_TEMPERATURE_SENSOR_BIAS"] == "SUPPORTED"
    assert bias_map["THERMAL_TEMPERATURE_SENSOR_DRIFT"] != "SUPPORTED"
    assert drift_map["THERMAL_TEMPERATURE_SENSOR_DRIFT"] == "SUPPORTED"
    assert drift_map["THERMAL_TEMPERATURE_SENSOR_BIAS"] != "SUPPORTED"


def test_solar_efficiency_and_string_failure_are_distinguished(tmp_path):
    result = _invoke(tmp_path, "solar_efficiency_loss")
    hypotheses = result.output_payload["hypothesis_set"]["hypotheses"]
    status = {item.get("target_fault_id"): item["status"] for item in hypotheses}
    assert status["EPS_SOLAR_ARRAY_EFFICIENCY_LOSS"] == "SUPPORTED"
    assert status.get("EPS_SOLAR_ARRAY_STRING_FAILURE") != "SUPPORTED"


def test_composites_emit_unverified_research_hypotheses_and_human_review(tmp_path):
    for scenario, expected_members in COMPOSITES.items():
        result = _invoke(tmp_path, scenario)
        assert result.status == ExecutionStatus.SUCCEEDED, scenario
        decision = result.output_payload["decision_record"]
        assert decision["disposition"] == "HUMAN_REVIEW", scenario
        assert decision["requires_human_review"] is True
        assert decision["comparison_status"] == "COMPOSITE_MEMBER_EVIDENCE_ONLY"
        hypotheses = result.output_payload["hypothesis_set"]["hypotheses"]
        research = [item for item in hypotheses if item["hypothesis_type"] == HypothesisType.RESEARCH_HYPOTHESIS.value]
        assert len(research) == 1
        item = research[0]
        assert item["status"] == HypothesisStatus.UNDER_VERIFICATION.value
        assert item["claim_type"] == ClaimType.HYPOTHETICAL.value
        assert set(item["member_fault_ids"]) == expected_members
        assert len(item["verification_requirements"]) >= 3
        assert item["interaction_hypothesis"]
        artifact = tmp_path / scenario / "diagnosis" / "research_hypotheses.json"
        assert artifact.is_file()
        document = json.loads(artifact.read_text(encoding="utf-8"))
        assert len(document["hypotheses"]) == 1


def test_research_hypothesis_cannot_be_formally_accepted_by_model_validation():
    from astrograph.investigation import DiagnosticHypothesis
    import pytest

    with pytest.raises(ValueError):
        DiagnosticHypothesis(
            hypothesis_id="H-INVALID",
            hypothesis_type=HypothesisType.RESEARCH_HYPOTHESIS,
            target_fault_id=None,
            status=HypothesisStatus.SUPPORTED,
            claim_type=ClaimType.HYPOTHETICAL,
            reasoning_summary="invalid",
            member_fault_ids=["A", "B"],
        )
