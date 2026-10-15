from __future__ import annotations

from astrograph.diagnosis.dxc_reduced_order_physics import rank_dxc_reduced_order_physics
from astrograph.investigation.diagnostic_probes import (
    DiagnosticProbeExecutor,
    DiagnosticProbeKind,
    build_dxc_reduced_order_simulation_probe_catalog,
)


def _catalog():
    return {
        "xml_sha256": "1" * 64,
        "components": [
            {"component_id": "W", "component_type": "Wire", "kind": "", "fault_modes": []},
            {"component_id": "R60", "component_type": "Relay", "kind": "", "fault_modes": ["StuckOpen"]},
            {"component_id": "P60", "component_type": "PositionSensor", "kind": "sensor", "fault_modes": ["Stuck"]},
            {"component_id": "L60", "component_type": "LightBulb60W", "kind": "load", "fault_modes": ["FailedOff"]},
            {"component_id": "R55", "component_type": "Relay", "kind": "", "fault_modes": ["StuckOpen"]},
            {"component_id": "P55", "component_type": "PositionSensor", "kind": "sensor", "fault_modes": ["Stuck"]},
            {"component_id": "L55", "component_type": "LightBulb55W", "kind": "load", "fault_modes": ["FailedOff"]},
            {"component_id": "T55", "component_type": "TemperatureSensor", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit": "DegF"},
            {"component_id": "I", "component_type": "ACCurrentTransmitter12", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit": "Amps"},
            {"component_id": "V", "component_type": "ACVoltageSensor", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit": "VAC"},
        ],
        "connections": [
            {"c1": "L60", "c2": "W"}, {"c1": "W", "c2": "R60"}, {"c1": "R60", "c2": "P60"},
            {"c1": "L55", "c2": "W"}, {"c1": "W", "c2": "R55"}, {"c1": "R55", "c2": "P55"}, {"c1": "L55", "c2": "T55"},
            {"c1": "W", "c2": "I"}, {"c1": "W", "c2": "V"},
        ],
        "command_bindings": [
            {"command_id": "R60_CL", "component_id": "R60"},
            {"command_id": "R55_CL", "component_id": "R55"},
        ],
    }


def _records(*, commanded_off: bool = False):
    rows = []
    for second in range(0, 101):
        if commanded_off and second == 40:
            rows.append({"kind": "command", "time_tick": second * 1000, "command_id": "R60_CL", "value": False})
        # L60 fails off at the model transition: branch current loses 60/120 = 0.5 A.
        current = 1.5 + 0.002 * ((second % 5) - 2) if second < 40 else 1.0 + 0.002 * ((second % 5) - 2)
        # L55 remains healthy and keeps heating, contradicting L55:FailedOff.
        temp55 = 200.0 + 0.2 * second
        rows.append({
            "kind": "sensors",
            "time_tick": second * 1000,
            "values": {"I": current, "V": 120.0, "P60": True, "P55": True, "T55": temp55},
        })
    rows.sort(key=lambda row: row["time_tick"])
    return rows


def test_r29_reduced_order_simulation_resolves_unique_physical_load_candidate():
    decision = rank_dxc_reduced_order_physics(
        records=_records(),
        catalog=_catalog(),
        base_fault_ids=(),
        base_unresolved_reason_codes=("SENSOR_OFFSET_UNCORROBORATED_PHYSICAL_RESPONSE_UNRESOLVED",),
        first_alarm_time_s=40.0,
    )
    assert decision.selected_fault_ids == ("L60:FailedOff",)
    assert decision.application_status == "RESOLVED_BY_REDUCED_ORDER_PHYSICS"
    assert decision.numerical_time_series_simulation is True
    assert decision.continuous_network_solver is False
    supported = [item for item in decision.candidate_fits if item.fit_status == "SUPPORTED"]
    assert len(supported) == 1
    assert supported[0].fault_ids == ("L60:FailedOff",)
    assert supported[0].expected_current_drop_a == 0.5
    assert supported[0].simulation_trace_sha256


def test_r29_related_relay_command_blocks_fault_resolution():
    decision = rank_dxc_reduced_order_physics(
        records=_records(commanded_off=True),
        catalog=_catalog(),
        base_fault_ids=(),
        base_unresolved_reason_codes=("SENSOR_OFFSET_UNCORROBORATED_PHYSICAL_RESPONSE_UNRESOLVED",),
        first_alarm_time_s=40.0,
    )
    assert decision.selected_fault_ids == ()
    assert decision.application_status.startswith("UNRESOLVED")
    l60 = next(item for item in decision.candidate_fits if item.fault_ids == ("L60:FailedOff",))
    assert l60.command_contradiction_codes
    assert l60.fit_status == "REJECTED"


def test_r29_resolved_r28_fault_set_passes_through_without_post_contact_rerank():
    decision = rank_dxc_reduced_order_physics(
        records=_records(),
        catalog=_catalog(),
        base_fault_ids=("SOME_PLANT:FailedOff",),
        base_unresolved_reason_codes=(),
        first_alarm_time_s=40.0,
    )
    assert decision.selected_fault_ids == ("SOME_PLANT:FailedOff",)
    assert decision.application_status == "PASSTHROUGH_R28_RESOLVED"
    assert decision.candidate_fits == ()


def test_r29_agent_probe_is_read_only_mirror_of_core_decision():
    decision = rank_dxc_reduced_order_physics(
        records=_records(),
        catalog=_catalog(),
        base_fault_ids=(),
        base_unresolved_reason_codes=("SENSOR_OFFSET_UNCORROBORATED_PHYSICAL_RESPONSE_UNRESOLVED",),
        first_alarm_time_s=40.0,
    )
    before = decision.selected_fault_ids
    catalog = build_dxc_reduced_order_simulation_probe_catalog(
        episode_id="EP-R29",
        decision=decision,
        source_artifact_sha256="a" * 64,
    )
    spec = catalog.probes[0]
    assert spec.kind == DiagnosticProbeKind.DXC_REDUCED_ORDER_SIMULATION_REVIEW
    result = DiagnosticProbeExecutor().execute(catalog=catalog, probe_id=spec.probe_id)
    assert tuple(result.metrics["selected_fault_ids"]) == before
    assert result.metrics["review_semantics"] == "READ_ONLY_MIRROR_OF_DETERMINISTIC_REDUCED_ORDER_PHYSICS_DECISION"
    assert result.formal_evidence_generated is False
    assert result.formal_evidence_eligible is False
    assert decision.selected_fault_ids == before


def test_r29_physics_candidates_use_neutral_simulation_hypothesis_contract(monkeypatch):
    from astrograph.simulation import diagnostic_provider as simulation_provider

    seen = []
    original = simulation_provider.build_counterfactual_request

    def capture(candidate, **kwargs):
        seen.append(candidate)
        return original(candidate, **kwargs)

    monkeypatch.setattr(simulation_provider, "build_counterfactual_request", capture)
    # dxc_reduced_order_physics imports compare_counterfactual_diagnoses, whose
    # global build_counterfactual_request is the module function patched above.
    decision = rank_dxc_reduced_order_physics(
        records=_records(),
        catalog=_catalog(),
        base_fault_ids=(),
        base_unresolved_reason_codes=("SENSOR_OFFSET_UNCORROBORATED_PHYSICAL_RESPONSE_UNRESOLVED",),
        first_alarm_time_s=40.0,
    )
    assert decision.selected_fault_ids == ("L60:FailedOff",)
    assert seen
    assert all(
        getattr(candidate, "candidate_semantics", None)
        == "HOST_BOUND_SIMULATION_HYPOTHESIS_NOT_DIAGNOSIS_PROOF"
        for candidate in seen
    )
    assert all(not hasattr(candidate, "hits_all_conflicts") for candidate in seen)


def test_r29_core_source_has_no_truth_or_case_identity_dependency():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/astrograph/simulation/adapt_reduced_order.py",
        "src/astrograph/diagnosis/dxc_reduced_order_physics.py",
        "src/astrograph/evaluation/dxc_r29_reduced_order.py",
    ):
        text = (root / relative).read_text(encoding="utf-8").lower()
        assert "true_fault" not in text
        assert "ground_truth" not in text
        assert "case_id" not in text
