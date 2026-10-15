from __future__ import annotations

from astrograph.diagnosis.dxc_forward_response import DXCForwardResponseConfig, rank_dxc_forward_response
from astrograph.diagnosis.dxc_expected_response import DXCExpectedResponseConfig
from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.dxc_discrete_consistency import DXCDiscreteConsistencyConfig, diagnose_discrete_state_consistency
from astrograph.evaluation.dxc_hybrid_model_based import DXCHybridPrediction
from astrograph.evaluation.dxc_r27_expected_response import assemble_dxc_r27_prediction
from astrograph.evaluation.dxc_r28_forward_simulation import assemble_dxc_r28_prediction
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog
from astrograph.investigation.diagnostic_probes import (
    DiagnosticProbeExecutor,
    DiagnosticProbeKind,
    build_dxc_forward_simulation_probe_catalog,
)


def _light_catalog():
    components = [{"component_id": "W", "component_type": "Wire", "kind": ""}]
    connections = []
    for index in range(1, 4):
        components.extend([
            {"component_id": f"L{index}", "component_type": "LightBulb25W", "kind": "load", "fault_modes": ["FailedOff"]},
            {"component_id": f"T{index}", "component_type": "TemperatureSensor", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit_status": "VERIFIED_FROM_XML"},
        ])
        connections.extend([
            {"c1": f"L{index}", "c2": "W"},
            {"c1": f"L{index}", "c2": f"T{index}"},
            {"c1": f"L{index}", "c2": "LS"},
        ])
    components.append({"component_id": "LS", "component_type": "LightSensor", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit_status": "VERIFIED_FROM_XML"})
    return {"components": components, "connections": connections}


def test_forward_model_resolves_multi_light_failed_off_from_local_thermal_response():
    catalog = _light_catalog()
    records = []
    for second in range(0, 101):
        if second < 40:
            t1 = t2 = t3 = 100.0 + 0.1 * second
            light = 50.0
        else:
            t1 = 104.0 - 0.30 * (second - 40)
            t2 = 104.0 + 0.10 * (second - 40)
            t3 = 104.0 - 0.22 * (second - 40)
            light = 30.0
        records.append({"kind": "sensors", "time_tick": second * 1000, "values": {"T1": t1, "T2": t2, "T3": t3, "LS": light}})
    decision = rank_dxc_forward_response(
        records=records,
        catalog=catalog,
        base_fault_ids=("L2:FailedOff",),
        first_alarm_time_s=40.0,
        config=DXCForwardResponseConfig(),
    )
    assert decision.selected_fault_ids == ("L1:FailedOff", "L3:FailedOff")
    assert "L2:FailedOff" in decision.vetoed_fault_ids
    assert all(item.transition.transition_kind == "PLANT_COMPONENT_MODE_TRANSITION" for item in decision.fits if item.fault_id.startswith("L"))
    assert decision.continuous_physics_simulator is False
    assert decision.formal_evidence_eligible is False


def test_forward_model_direct_stuck_replaces_uncorroborated_neighbor_offsets():
    catalog = {
        "components": [
            {"component_id": "S1", "component_type": "DCCurrentTransmitter12", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit_status": "VERIFIED_FROM_XML"},
            {"component_id": "S2", "component_type": "DCVoltageSensor", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit_status": "VERIFIED_FROM_XML"},
        ],
        "connections": [],
    }
    records = []
    for i in range(120):
        s1 = 3.0 + 0.02 * ((i % 5) - 2) if i < 80 else 0.0
        s2 = 120.0 + 0.05 * ((i % 3) - 1)
        records.append({"kind": "sensors", "time_tick": i * 1000, "values": {"S1": s1, "S2": s2}})
    decision = rank_dxc_forward_response(
        records=records,
        catalog=catalog,
        base_fault_ids=("S2:Offset",),
        first_alarm_time_s=35.0,
    )
    assert decision.selected_fault_ids == ("S1:Stuck",)
    assert "S2:Offset" in decision.vetoed_fault_ids
    assert any(item.fault_id == "S1:Stuck" and item.transition.to_mode == "STUCK_OBSERVATION" for item in decision.fits)


def test_forward_model_sensor_only_offset_must_be_reproduced_or_becomes_unresolved():
    catalog = {
        "components": [
            {"component_id": "I1", "component_type": "DCCurrentTransmitter50", "kind": "sensor", "fault_modes": ["Offset", "Stuck"], "unit_status": "VERIFIED_FROM_XML"},
        ],
        "connections": [],
    }
    records = []
    for i in range(120):
        # A modest physical operating-level change, not an admitted large additive Offset signature.
        value = 20.0 + 0.02 * ((i % 4) - 1.5) if i < 80 else 17.5 + 0.02 * ((i % 4) - 1.5)
        records.append({"kind": "sensors", "time_tick": i * 1000, "values": {"I1": value}})
    decision = rank_dxc_forward_response(
        records=records,
        catalog=catalog,
        base_fault_ids=("I1:Offset",),
        first_alarm_time_s=80.0,
    )
    assert decision.selected_fault_ids == ()
    assert decision.vetoed_fault_ids == ("I1:Offset",)
    assert decision.unresolved_reason_codes == ("SENSOR_OFFSET_UNCORROBORATED_PHYSICAL_RESPONSE_UNRESOLVED",)


XML = b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<modeCatalog>
  <modeGroup><name>RelayMode</name><mode xsi:type="mode"><name>NominalClosed</name></mode><mode xsi:type="faultMode"><name>StuckOpen</name></mode></modeGroup>
  <modeGroup><name>PositionMode</name><mode xsi:type="mode"><name>Nominal</name></mode><mode xsi:type="faultMode"><name>Stuck</name></mode></modeGroup>
</modeCatalog>
<componentTypeCatalog>
  <componentType xsi:type=""><name>Relay</name><modesRef>RelayMode</modesRef></componentType>
  <componentType xsi:type="sensor"><name>PositionSensor</name><modesRef>PositionMode</modesRef><sensorValue><dataType>bool</dataType></sensorValue></componentType>
</componentTypeCatalog>
<systems><system><components>
  <component><name>R1</name><componentType>Relay</componentType><command><name>R1_CL</name><commandValue><dataType>bool</dataType></commandValue></command></component>
  <component><name>P1</name><componentType>PositionSensor</componentType></component>
</components><connections><connection><c1>R1</c1><c2>P1</c2></connection></connections></system></systems>
</root>'''


def test_agent_forward_simulation_probe_is_read_only_and_core_prediction_is_agent_independent():
    records, _ = parse_scenario("\n".join(["sensors @0 {P1=true};", "sensors @1000 {P1=false};"]))
    catalog = system_catalog(XML)
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["r28-probe"]):
        discrete = diagnose_discrete_state_consistency(records, catalog, DXCDiscreteConsistencyConfig(startup_s=0.0))
    continuous = DXCHybridPrediction(
        predicted_fault_ids=("P1:Stuck",),
        first_alarm_time_s=0.5,
        final_alarm_time_s=1.0,
        final_component_ids=("P1",),
        model_sha256="1" * 64,
        graph_sha256="2" * 64,
    )
    r27 = assemble_dxc_r27_prediction(
        records=records,
        catalog=catalog,
        continuous=continuous,
        discrete=discrete,
        expected_response_config=DXCExpectedResponseConfig(startup_s=0.1),
    )
    r28 = assemble_dxc_r28_prediction(records=records, catalog=catalog, r27_prediction=r27)
    before = r28.predicted_fault_ids
    assert r28.agent_required is False

    probe_catalog = build_dxc_forward_simulation_probe_catalog(
        episode_id="EP-R28",
        decision=r28.forward_response,
        source_artifact_sha256="a" * 64,
    )
    spec = probe_catalog.probes[0]
    assert spec.kind == DiagnosticProbeKind.DXC_FORWARD_SIMULATION_REVIEW
    result = DiagnosticProbeExecutor().execute(catalog=probe_catalog, probe_id=spec.probe_id)
    assert tuple(result.metrics["selected_fault_ids"]) == before
    assert result.metrics["review_semantics"] == "READ_ONLY_MIRROR_OF_DETERMINISTIC_FORWARD_MODEL_DECISION"
    assert result.formal_evidence_generated is False
    assert result.formal_evidence_eligible is False
    assert r28.predicted_fault_ids == before
