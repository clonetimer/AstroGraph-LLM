import pytest
from astrograph.evaluation.esa_anomaly_adapter import require_evaluation_prerequisites
from astrograph.evaluation.esa_event_metrics import evaluate_events

def test_no_event_metrics_before_predecessor_gates():
    # No claimed corrected eventwise metric implementation until DXC gate passes.
    with pytest.raises(ValueError,match='PREREQUISITES'):require_evaluation_prerequisites(ops_sat_passed=True,dxc_passed=False)

def truth(aid,start,end,category='ANOMALY'):
    return dict(event_id=aid,start_ns=int(start*1e9),end_ns=int(end*1e9),category=category)

def alarm(start,end,closed=False):return dict(start_ns=int(start*1e9),end_ns=int(end*1e9),end_closed=closed)

def test_always_on_is_penalized_by_nominal_duration():
    m=evaluate_events([truth('a',10,20)],[alarm(0,100,True)],full_range=(0,int(100e9)))
    assert m['TP_events']==1 and m['event_recall']==1 and m['corrected_eventwise_F0.5']==0.

def test_endpoint_and_rare_challenges_keep_reference_semantics():
    m=evaluate_events([truth('a',10,20),truth('r',30,40,'RARE_NOMINAL'),truth('g',50,60,'COMMUNICATION_GAP')],
      [alarm(0,10),alarm(30,35),alarm(50,55)],full_range=(0,int(100e9)))
    assert m['TP_events']==0 and m['FP_events']==1
    assert m['rare_nominal_false_positive_count']==m['communication_gap_false_positive_count']==1

def test_one_multichannel_id_counts_as_one_event_and_redundant_alarms():
    m=evaluate_events([truth('a',10,20),truth('a',15,25)],[alarm(10,12),alarm(18,21)],full_range=(0,int(100e9)))
    assert m['TP_events']==1 and m['redundant_detections']==1
