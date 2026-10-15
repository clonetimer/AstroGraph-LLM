from __future__ import annotations

from pathlib import Path

from astrograph.evaluation import (
    all_expected_scenarios,
    audit_artifact_dir,
    build_stability_report,
    scenario_expectation,
    validate_summary,
)
from astrograph.orchestration.models import (
    ExecutionPolicy,
    ExecutionStatus,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
from astrograph.simulation import SimulationRunCompleted

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "online_base"
PROTOCOL = ROOT / "protocols" / "phm_project_a"


def _permissions():
    return [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]


def _run_rule(tmp_path: Path):
    permissions = _permissions()
    event = SimulationRunCompleted(
        event_id="SIM-V056-HEATER",
        run_id="heater_stuck_on",
        run_dir=str(FIXTURE / "heater_stuck_on"),
        nominal_run_dir=str(FIXTURE / "nominal"),
    )
    envelope = TaskEnvelope(
        task_id="V056-RULE-HEATER",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "rule" / "heater_stuck_on"),
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
    return WholeSpacecraftOnlineDiagnosisWorkflow().invoke(envelope, context)


def test_v056_expectation_catalog_has_all_21_routes():
    scenarios = all_expected_scenarios()
    assert len(scenarios) == 21
    assert scenario_expectation("heater_stuck_on").expected_fault_id == "THERMAL_HEATER_STUCK_ON"
    assert scenario_expectation("battery_internal_resistance_increase").expected_fault_id == "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE"
    assert scenario_expectation("rw_friction_plus_star_tracker").expected_disposition == "HUMAN_REVIEW"


def test_v056_artifact_audit_validates_manifest_tools_and_core(tmp_path):
    result = _run_rule(tmp_path)
    assert result.status == ExecutionStatus.SUCCEEDED
    evidence_bundle = result.output_payload["diagnosis_evidence_bundle"]
    capability_axes = evidence_bundle["capability_axes"]
    assert capability_axes["capability_manifest_id"].startswith("RUNTIME-ACTIVITY-AXES-V0529")
    assert "seed" in capability_axes["supported_axes"]
    assert capability_axes["unsupported_axes"] == []
    assert capability_axes["requested_axes"] == ["seed"]
    audit = audit_artifact_dir(tmp_path / "rule" / "heater_stuck_on")
    assert audit.manifest_valid
    assert not audit.missing_tool_ids
    assert audit.core_comparison_status == "MATCH"
    assert audit.disposition == "ACCEPT"
    assert audit.accepted_fault_id == "THERMAL_HEATER_STUCK_ON"
    assert audit.hypothesis_structure_sha256
    assert audit.decision_structure_sha256


def test_v056_summary_acceptance_and_stability(tmp_path):
    result = _run_rule(tmp_path)
    assert result.status == ExecutionStatus.SUCCEEDED
    row = {
        "scenario": "heater_stuck_on",
        "mode": "rule",
        "status": "SUCCEEDED",
        "disposition": "ACCEPT",
        "accepted_fault_id": "THERMAL_HEATER_STUCK_ON",
        "comparison_status": "MATCH",
        "requires_human_review": False,
        "repair_attempt_count": 0,
        "degraded_to_reference": False,
        "error": None,
    }
    summary = {"reasoner_mode": "rule", "rows": [row]}
    first = validate_summary(summary, artifact_root=tmp_path)
    second = validate_summary(summary, artifact_root=tmp_path)
    assert first.workflow_success_rate == 1.0
    assert first.expected_route_match_rate == 1.0
    assert first.independent_core_match_rate == 1.0
    assert first.tool_registry_valid_count == 1
    stability = build_stability_report([first, second], reasoner_mode="rule")
    assert stability.exact_signature_stability_mean == 1.0
    assert stability.minimum_scenario_stability == 1.0
    assert stability.aggregate_acceptance.expected_route_match_rate == 1.0


def test_v056_simulation_source_audit_uses_manifest_and_telemetry():
    from astrograph.evaluation import audit_simulation_output_root

    audit = audit_simulation_output_root(
        FIXTURE,
        scenarios=["nominal", "heater_stuck_on"],
    )
    assert audit.passed
    assert audit.expected_scenario_count == 2
    assert audit.passed_count == 2
    assert all(row["telemetry_sha256"] for row in audit.rows)


def test_v056_policy_evidence_reads_dynamic_run_manifest_activity_axes(tmp_path):
    run_dir = tmp_path / "run"
    nominal_dir = tmp_path / "nominal"
    run_dir.mkdir()
    nominal_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        """
{
  "schema_version": "phm.minimal_runtime.run.v1",
  "scenario": "battery_capacity_loss",
  "seed": 20260722,
  "activity_axes": {
    "seed": {"controlled": true, "value": 20260722},
    "fault_timing": {"controlled": true, "fault_start_s": 10.0, "fault_duration_s": 30.0},
    "severity": {"controlled": true, "fault_severity": "high"},
    "noise": {"controlled": false, "sensor_noise_scale": 1.0}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    event = SimulationRunCompleted(
        event_id="SIM-DYNAMIC-AXES",
        run_id="battery_capacity_loss",
        run_dir=str(run_dir),
        nominal_run_dir=str(nominal_dir),
    )

    axes = WholeSpacecraftOnlineDiagnosisWorkflow._capability_axes_evidence(
        simulation_event=event,
    )

    assert axes.requested_axes == ["fault_timing", "seed", "severity"]
    assert axes.unsupported_axes == []
    assert "severity" in axes.supported_axes
