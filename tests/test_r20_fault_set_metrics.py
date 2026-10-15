import pytest
from astrograph.evaluation.fault_set_metrics import evaluate_fault_set

def test_multifault_top_one_is_failure_not_success():
    m=evaluate_fault_set(['F1','F2'],['F1'])
    assert not m['exact_fault_set_match'] and m['set_recall']==.5 and m['false_negative_fault_count']==1
    with pytest.raises(ValueError,match='SET_VALUED'):evaluate_fault_set('F1',['F1'])

def test_delay_and_false_faults_are_explicit():
    m=evaluate_fault_set(['F1'],['F2'],injection_time=10,first_detection_time=13)
    assert m['first_detection_delay']==3 and not m['fault_isolated']
    assert m['false_positive_fault_count']==m['false_negative_fault_count']==1
