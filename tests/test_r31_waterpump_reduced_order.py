from __future__ import annotations

import math

from astrograph.diagnosis.dxc_pump_reduced_order import evaluate_dxc_water_pump_physics
from astrograph.investigation.diagnostic_probes import (
    DiagnosticProbeExecutor,
    DiagnosticProbeKind,
    build_dxc_waterpump_reduced_order_simulation_probe_catalog,
)
from astrograph.simulation.adapt_pump_reduced_order import (
    ADAPTWaterPumpTypeParameters,
    extract_water_pump_type_parameters,
)


def _catalog():
    return {
        "xml_sha256": "a" * 64,
        "components": [
            {"component_id": "PUMP", "component_type": "WaterPump", "kind": "load", "fault_modes": ["FailedOff", "FlowBlocked"]},
            {"component_id": "FT", "component_type": "FlowTransmitter", "kind": "sensor", "unit": "GPH", "fault_modes": ["Offset", "Stuck"]},
            {"component_id": "W1", "component_type": "Wire", "kind": "", "fault_modes": []},
            {"component_id": "R", "component_type": "Relay", "kind": "", "fault_modes": ["StuckOpen", "StuckClosed"]},
            {"component_id": "W2", "component_type": "Wire", "kind": "", "fault_modes": []},
            {"component_id": "I", "component_type": "ACCurrentTransmitter12", "kind": "sensor", "unit": "Amps", "fault_modes": ["Offset", "Stuck"]},
            {"component_id": "V", "component_type": "ACVoltageSensor", "kind": "sensor", "unit": "VAC", "fault_modes": ["Offset", "Stuck"]},
        ],
        "connections": [
            {"c1": "PUMP", "c2": "FT"}, {"c1": "PUMP", "c2": "W1"}, {"c1": "W1", "c2": "R"},
            {"c1": "R", "c2": "W2"}, {"c1": "W2", "c2": "I"}, {"c1": "W2", "c2": "V"},
        ],
        "command_bindings": [{"command_id": "R_CL", "component_id": "R"}],
    }


def _params():
    return ADAPTWaterPumpTypeParameters(
        rated_power_w=35.0,
        nominal_flow_gph=250.0,
        source_xml_sha256="a" * 64,
    )


def _records(mode: str, *, commanded: bool = False):
    rows = []
    onset = 10.0
    for index in range(41):
        t = index * 0.5
        if commanded and math.isclose(t, onset):
            rows.append({"kind": "command", "time_tick": int(t * 1000), "command_id": "R_CL", "value": False})
        v = 120.0
        background = 1.0
        pump_current = 35.0 / v
        if t < onset:
            current = background + pump_current
            flow = 250.0
        else:
            elapsed = t - onset
            if mode == "FailedOff":
                current = background
                flow = 250.0 * math.exp(-elapsed / 1.0)
            else:
                current = background + pump_current
                flow = 250.0 * (0.4 + 0.6 * math.exp(-elapsed / 2.0))
        rows.append({
            "kind": "sensors",
            "time_tick": int(t * 1000),
            "values": {"I": current, "V": v, "FT": flow},
        })
    rows.sort(key=lambda row: row["time_tick"])
    return rows


def test_r31_failed_off_current_flow_model_corroborates_base_without_override():
    decision = evaluate_dxc_water_pump_physics(
        records=_records("FailedOff"),
        catalog=_catalog(),
        type_parameters=_params(),
        base_fault_ids=("PUMP:FailedOff",),
        transition_time_s=10.0,
    )
    assert decision.selected_fault_ids == ("PUMP:FailedOff",)
    assert decision.physics_preferred_fault_ids == ("PUMP:FailedOff",)
    assert decision.supports_base_mode is True
    assert decision.application_status == "PUMP_PHYSICS_CORROBORATES_BASE_MODE"
    assert decision.auto_override_enabled is False
    assert decision.numerical_time_series_simulation is True


def test_r31_flowblocked_model_corroborates_electrically_energized_low_flow_mode():
    decision = evaluate_dxc_water_pump_physics(
        records=_records("FlowBlocked"),
        catalog=_catalog(),
        type_parameters=_params(),
        base_fault_ids=("PUMP:FlowBlocked",),
        transition_time_s=10.0,
    )
    assert decision.physics_preferred_fault_ids == ("PUMP:FlowBlocked",)
    assert decision.supports_base_mode is True
    by_mode = {item.fault_mode: item for item in decision.candidate_fits}
    assert by_mode["FlowBlocked"].fitted_flow_asymptote_ratio > 0.2
    assert by_mode["FlowBlocked"].normalized_current_rmse < by_mode["FailedOff"].normalized_current_rmse


def test_r31_physics_can_flag_mode_contradiction_without_auto_rewriting_diagnosis():
    decision = evaluate_dxc_water_pump_physics(
        records=_records("FlowBlocked"),
        catalog=_catalog(),
        type_parameters=_params(),
        base_fault_ids=("PUMP:FailedOff",),
        transition_time_s=10.0,
    )
    assert decision.selected_fault_ids == ("PUMP:FailedOff",)
    assert decision.physics_preferred_fault_ids == ("PUMP:FlowBlocked",)
    assert decision.supports_base_mode is False
    assert decision.application_status == "PUMP_PHYSICS_CONTRADICTS_BASE_MODE_NO_AUTO_OVERRIDE"


def test_r31_command_near_transition_makes_physics_nondiagnostic():
    decision = evaluate_dxc_water_pump_physics(
        records=_records("FailedOff", commanded=True),
        catalog=_catalog(),
        type_parameters=_params(),
        base_fault_ids=("PUMP:FailedOff",),
        transition_time_s=10.0,
    )
    assert decision.application_status == "PUMP_PHYSICS_NONDIAGNOSTIC_COMMAND_NEAR_TRANSITION"
    assert decision.physics_preferred_fault_ids == ()


def test_r31_agent_probe_is_read_only_mirror():
    decision = evaluate_dxc_water_pump_physics(
        records=_records("FailedOff"),
        catalog=_catalog(),
        type_parameters=_params(),
        base_fault_ids=("PUMP:FailedOff",),
        transition_time_s=10.0,
    )
    before = decision.selected_fault_ids
    catalog = build_dxc_waterpump_reduced_order_simulation_probe_catalog(
        episode_id="EP-R31-PUMP",
        decision=decision,
        source_artifact_sha256="b" * 64,
    )
    spec = catalog.probes[0]
    assert spec.kind == DiagnosticProbeKind.DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW
    result = DiagnosticProbeExecutor().execute(catalog=catalog, probe_id=spec.probe_id)
    assert tuple(result.metrics["selected_fault_ids"]) == before
    assert tuple(result.metrics["physics_preferred_fault_ids"]) == ("PUMP:FailedOff",)
    assert result.formal_evidence_generated is False
    assert result.formal_evidence_eligible is False
    assert decision.selected_fault_ids == before


def test_r31_xml_parameter_extraction_is_exact_and_content_bound():
    xml = b'''<?xml version="1.0"?><systemCatalog xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <componentTypeCatalog><componentType xsi:type="load"><name>WaterPump</name><modesRef>WaterPump</modesRef><nominalWattage>35</nominalWattage></componentType></componentTypeCatalog>
    <modeCatalog><modeGroup><name>WaterPump</name><mode xsi:type="mode"><name>Nominal</name><description>When supplied with electricity, pumps water at approximately 250GPH.</description></mode></modeGroup></modeCatalog>
    </systemCatalog>'''
    params = extract_water_pump_type_parameters(xml)
    assert params.rated_power_w == 35.0
    assert params.nominal_flow_gph == 250.0
    assert len(params.source_xml_sha256) == 64


def test_r31_core_source_has_no_truth_or_case_identity_dependency():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/astrograph/simulation/adapt_pump_reduced_order.py",
        "src/astrograph/diagnosis/dxc_pump_reduced_order.py",
    ):
        text = (root / relative).read_text(encoding="utf-8").lower()
        assert "true_fault" not in text
        assert "ground_truth" not in text
        assert "case_id" not in text
