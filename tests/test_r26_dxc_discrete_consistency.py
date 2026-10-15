from __future__ import annotations

import pytest

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.dxc_discrete_consistency import (
    DXCDiscreteConsistencyConfig,
    diagnose_discrete_state_consistency,
)
from astrograph.evaluation.dxc_hybrid_fusion_r26 import DXCFusionPolicy, fuse_dxc_r26
from astrograph.evaluation.dxc_hybrid_model_based import DXCHybridPrediction
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog


XML = b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<modeCatalog>
  <modeGroup><name>RelayMode</name><mode xsi:type="mode"><name>NominalClosed</name></mode><mode xsi:type="mode"><name>NominalOpen</name></mode><mode xsi:type="faultMode"><name>StuckClosed</name></mode><mode xsi:type="faultMode"><name>StuckOpen</name></mode></modeGroup>
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


def _score(lines: list[str]):
    records, _ = parse_scenario("\n".join(lines))
    catalog = system_catalog(XML)
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["r26-discrete"]):
        return diagnose_discrete_state_consistency(records, catalog, DXCDiscreteConsistencyConfig(startup_s=0.0))


def test_uncommanded_position_open_transition_maps_to_adjacent_relay_mode():
    result = _score([
        "sensors @0 {P1=true};",
        "sensors @1000 {P1=false};",
    ])
    assert result.predicted_fault_ids == ("R1:StuckOpen",)
    assert result.events[0].sensor_component_id == "P1"
    assert result.formal_evidence_eligible is False


def test_exact_component_command_suppresses_nominal_position_transition():
    result = _score([
        "sensors @0 {P1=true};",
        "command @500 R1_CL = false;",
        "sensors @1000 {P1=false};",
    ])
    assert result.predicted_fault_ids == ()
    assert result.events == ()


def test_fusion_discrete_first_uses_exact_state_fault_and_continuous_is_only_fallback():
    discrete = _score(["sensors @0 {P1=true};", "sensors @1000 {P1=false};"])
    continuous = DXCHybridPrediction(
        predicted_fault_ids=("P1:Stuck", "R1:StuckClosed"),
        first_alarm_time_s=0.5,
        final_alarm_time_s=2.0,
        final_component_ids=("P1", "R1"),
        model_sha256="1" * 64,
        graph_sha256="2" * 64,
    )
    fused = fuse_dxc_r26(continuous, discrete, DXCFusionPolicy.DISCRETE_FIRST_SINGLE_CONTINUOUS)
    assert fused.predicted_fault_ids == ("R1:StuckOpen",)
    assert fused.first_alarm_time_s == 1.0
    assert fused.formal_evidence_eligible is False


def test_fusion_single_continuous_fallback_preserves_ranked_component_not_lexicographic_fault():
    empty = _score(["sensors @0 {P1=true};", "sensors @1000 {P1=true};"])
    continuous = DXCHybridPrediction(
        predicted_fault_ids=("A:Offset", "Z:Stuck"),
        first_alarm_time_s=5.0,
        final_alarm_time_s=6.0,
        final_component_ids=("Z", "A"),
        model_sha256="3" * 64,
        graph_sha256="4" * 64,
    )
    fused = fuse_dxc_r26(continuous, empty, DXCFusionPolicy.DISCRETE_FIRST_SINGLE_CONTINUOUS)
    assert fused.predicted_fault_ids == ("Z:Stuck",)


def test_non_bool_position_observation_fails_closed():
    catalog = system_catalog(XML)
    records = [{"kind": "sensors", "time_tick": 0, "values": {"P1": 1.0}}]
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["r26-discrete-bad"]):
        with pytest.raises(ValueError, match="VALUE_NOT_BOOL"):
            diagnose_discrete_state_consistency(records, catalog, DXCDiscreteConsistencyConfig(startup_s=0.0))
