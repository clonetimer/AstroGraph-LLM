from astrograph.evaluation.nasa_adapt_adapter import system_catalog


def test_all_independently_declared_mode_groups_enter_fault_universe():
    xml=b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <modeCatalog><modeGroup><name>BasicLoad</name><mode xsi:type="faultMode"><name>FailedOff</name></mode></modeGroup>
    <modeGroup><name>Resistor</name><mode xsi:type="faultMode"><name>ResistanceDrift</name></mode></modeGroup></modeCatalog>
    <componentTypeCatalog><componentType xsi:type="load"><name>ACResistor</name><modesRef>BasicLoad</modesRef><modesRef>Resistor</modesRef></componentType></componentTypeCatalog>
    <systems><system><component><name>AC483</name><componentType>ACResistor</componentType></component></system></systems></root>'''
    catalog=system_catalog(xml)
    assert catalog['target_universe']==['AC483:FailedOff','AC483:ResistanceDrift']
    assert catalog['components'][0]['unit_status']=='UNIT_UNVERIFIED'
