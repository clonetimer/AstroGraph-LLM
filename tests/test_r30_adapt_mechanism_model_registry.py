from astrograph.simulation.adapt_component_model_registry import (
    ModelFidelity, MechanismModelClass, build_adapt_mechanism_model_registry,
)


def _catalog():
    return {
        'components':[
            {'component_id':'L1','component_type':'LightBulb60W','kind':'component','fault_modes':['FailedOff'],'command_ids':[]},
            {'component_id':'R1','component_type':'Relay','kind':'component','fault_modes':['StuckOpen','StuckClosed'],'command_ids':['CMD_R1']},
            {'component_id':'P1','component_type':'WaterPump','kind':'component','fault_modes':['FailedOff','FlowBlocked'],'command_ids':['CMD_P1']},
            {'component_id':'S1','component_type':'FlowTransmitter','kind':'sensor','fault_modes':['Offset','Stuck'],'command_ids':[]},
            {'component_id':'I1','component_type':'ACCurrentTransmitter12','kind':'sensor','fault_modes':['Offset','Stuck'],'command_ids':[]},
            {'component_id':'V1','component_type':'ACVoltageSensor','kind':'sensor','fault_modes':['Offset','Stuck'],'command_ids':[]},
        ],
        'connections':[{'c1':'L1','c2':'I1'},{'c1':'L1','c2':'V1'},{'c1':'R1','c2':'L1'},{'c1':'P1','c2':'S1'}],
    }


def test_registry_separates_numerical_from_qualitative_mechanism_coverage():
    r=build_adapt_mechanism_model_registry(_catalog())
    bulb=next(x for x in r.models if x.fault_id=='L1:FailedOff')
    assert bulb.model_class==MechanismModelClass.RESISTIVE_LOAD_FAILED_OFF
    assert bulb.fidelity==ModelFidelity.NUMERICAL_REDUCED_ORDER
    assert bulb.executable_provider_id=='adapt-resistive-lightbulb-reduced-order-v1'
    pump=next(x for x in r.models if x.fault_id=='P1:FlowBlocked')
    assert pump.fidelity==ModelFidelity.NUMERICAL_REDUCED_ORDER
    assert pump.executable_provider_id=='adapt-waterpump-reduced-order-v1'
    assert r.coverage['modeled_fault_mode_coverage']==1.0
    assert r.coverage['numerical_reduced_order_coverage_of_fault_universe'] < 1.0
