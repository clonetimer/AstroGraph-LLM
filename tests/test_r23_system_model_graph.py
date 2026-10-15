from __future__ import annotations

from astrograph.evaluation.nasa_adapt_adapter import system_catalog
from astrograph.knowledge.system_model_graph import (
    ObservationMismatch,
    build_system_model_graph,
    structural_conflicts_from_mismatches,
)


XML = b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<modeCatalog>
  <modeGroup><name>Breaker</name><mode xsi:type="mode"><name>Nominal</name></mode><mode xsi:type="faultMode"><name>FailedOpen</name></mode></modeGroup>
  <modeGroup><name>Sense</name><mode xsi:type="mode"><name>Nominal</name></mode><mode xsi:type="faultMode"><name>Offset</name></mode></modeGroup>
</modeCatalog>
<componentTypeCatalog>
  <componentType xsi:type="switch"><name>BreakerType</name><modesRef>Breaker</modesRef></componentType>
  <componentType xsi:type="sensor"><name>SensorType</name><engUnits>VDC</engUnits><modesRef>Sense</modesRef><sensorValue><dataType>real</dataType></sensorValue></componentType>
</componentTypeCatalog>
<systems><system><components>
  <component><name>CB1</name><componentType>BreakerType</componentType><command><name>CB1_OPEN</name><commandValue><dataType>bool</dataType></commandValue></command></component>
  <component><name>E1</name><componentType>SensorType</componentType></component>
</components><connections><connection><c1>CB1</c1><c2>E1</c2></connection></connections></system></systems>
</root>'''


def test_catalog_binds_instance_command_and_separates_nominal_fault_modes():
    catalog = system_catalog(XML)
    assert catalog["command_bindings"] == [{
        "command_id": "CB1_OPEN", "component_id": "CB1", "component_type": "BreakerType",
        "data_type": "bool", "binding_source": "COMPONENT_INSTANCE_XML",
    }]
    assert catalog["ambiguous_command_ids"] == []
    assert catalog["component_types"]["BreakerType"]["fault_modes"] == ["FailedOpen"]
    assert catalog["component_types"]["BreakerType"]["nominal_modes"] == ["Nominal"]


def test_system_model_graph_uses_command_and_topology_for_structural_conflict():
    graph = build_system_model_graph(system_catalog(XML))
    assert graph.command_target("CB1_OPEN") == "component:CB1"
    conflict = structural_conflicts_from_mismatches(
        graph,
        [ObservationMismatch(mismatch_id="M1", sensor_component_id="E1", command_id="CB1_OPEN")],
        max_topology_hops=1,
    )[0]
    assert conflict.fault_ids == ("CB1:FailedOpen", "E1:Offset")
