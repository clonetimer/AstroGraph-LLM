import pytest
from astrograph.evaluation.dxc_workbook_truth import parse_annotation_rows,verify_injection_agreement
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario

def test_explicit_nominal_and_multi_fault_continuations():
    rows=[['','Fault Description','',''],['File Name','Component','Mode','Time (sec)'],['unknown_name.txt','N/A','',''],['nominal_sounding.txt','BAT1','Degraded',10.],['','IT1','Offset',20.]]
    truth=parse_annotation_rows(rows)
    assert truth['unknown_name']['explicit_nominal']
    assert truth['nominal_sounding']['true_fault_ids']==['BAT1:Degraded','IT1:Offset']
    _,scn=parse_scenario('sensors @0 { X = 1 };\nfaultInjection @10000 isInjection = true, fault = { BAT1 = Degraded }, parameters = { };\nfaultInjection @20000 isInjection = true, fault = { IT1 = Offset }, parameters = { };')
    assert verify_injection_agreement(truth['nominal_sounding'],scn)['verified_fault_event_count']==2
    scn['true_fault_ids']=['BAT1:Degraded']
    with pytest.raises(ValueError,match='FAULT_SET_MISMATCH'):verify_injection_agreement(truth['nominal_sounding'],scn)

def test_missing_explicit_nominal_not_manufactured():
    with pytest.raises(ValueError,match='INCOMPLETE'):parse_annotation_rows([['File Name','Component','Mode','Time (sec)'],['case','','','']])
