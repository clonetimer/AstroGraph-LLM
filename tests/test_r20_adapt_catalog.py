import pytest
from astrograph.evaluation.nasa_adapt_adapter import system_catalog,require_formal_unit
XML=b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><componentTypeCatalog><componentType xsi:type="sensor"><name>Voltage</name><engUnits>VDC</engUnits></componentType><componentType xsi:type="sensor"><name>Unknown</name></componentType></componentTypeCatalog><systems><system><components><component><name>misleading_temperature_name</name><componentType>Voltage</componentType></component><component><name>E123</name><componentType>Unknown</componentType></component></components></system></systems></root>'''

def test_units_from_xml_never_channel_spelling():
    c=system_catalog(XML);assert c['components'][0]['unit']=='VDC'
    require_formal_unit(c['components'][0])
    with pytest.raises(ValueError,match='UNIT_UNVERIFIED'):require_formal_unit(c['components'][1])


def test_r23_instance_command_binding_is_explicit_not_name_inferred():
    xml=b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><componentTypeCatalog><componentType xsi:type="switch"><name>Breaker</name></componentType></componentTypeCatalog><systems><system><components><component><name>CB1</name><componentType>Breaker</componentType><command><name>EY1_OP</name><commandValue><dataType>bool</dataType></commandValue></command></component></components></system></systems></root>'''
    c=system_catalog(xml)
    assert c['command_bindings'][0]['command_id']=='EY1_OP'
    assert c['command_bindings'][0]['component_id']=='CB1'
    assert c['components'][0]['command_ids']==['EY1_OP']
