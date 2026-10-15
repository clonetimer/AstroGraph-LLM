import pytest
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario
SCN="sensors @0 { E1 = 10, switch = true };\ncommand @2 cmd = false;\nfaultInjection @10 isInjection = true, fault = { BAT1 = Degraded, IT1 = Offset }, parameters = { Offset = ' 1.0' };\nsensors @20 { E1 = 0 };"

def test_separate_truth_multi_fault_and_ticks():
    online,truth=parse_scenario(SCN)
    assert truth['true_fault_ids']==['BAT1:Degraded','IT1:Offset']
    assert truth['injection_time']==10 and truth['time_unit']=='UNVERIFIED_RAW_TICK'
    assert all('fault' not in str(r).lower() for r in online)
    changed=SCN.replace('BAT1 = Degraded','BAT2 = Failed')
    assert parse_scenario(changed)[0]==online

def test_unknown_record_is_not_silently_ignored():
    with pytest.raises(ValueError,match='UNPARSED'):parse_scenario(SCN+'\nsecretTruth @0 F1;')

def test_time_reversal_is_rejected():
    with pytest.raises(ValueError,match='TIMESTAMP'):parse_scenario('sensors @2 { X = 1 };\nsensors @1 { X = 0 };')

def test_same_tick_preserves_ordered_observations():
    online,_=parse_scenario('sensors @2 { X = 1 };\nsensors @2 { X = 0 };')
    assert len(online)==2 and [r['values']['X'] for r in online]==[1,0]
