import pytest
from astrograph.evaluation.selective_risk import *
H="b"*64

def pred(i,g,p): return FrozenPrediction(str(i),g,{"A":p,"B":1-p},"A" if p>=.5 else "B",H)
def truth(i,g,y): return TruthRecord(str(i),g,y)
def test_metrics_and_group_threshold_guard():
 ps=[pred(1,"3",.9),pred(2,"8",.4)]; ts=[truth(1,"3","A"),truth(2,"8","B")]
 m=evaluate_probabilities(ps,ts); assert m["accuracy"]==1
 r=select_confidence_threshold(ps,ts,allowed_groups={"3","8"},forbidden_groups={"4","9"},max_selective_risk=.1); assert r["coverage"]==1
 with pytest.raises(ValueError): select_confidence_threshold([pred(1,"4",.9)],[truth(1,"4","A")],allowed_groups={"4"},forbidden_groups={"4","9"},max_selective_risk=.1)

def test_equal_confidence_block_cannot_be_split_by_correctness():
 from astrograph.evaluation.selective_risk import risk_coverage_curve
 ps=[pred(1,"g1",.8),pred(2,"g2",.8)]
 ts=[truth(1,"g1","A"),truth(2,"g2","B")]
 # pred() predicts A; second row is therefore wrong at the same confidence.
 curve=risk_coverage_curve(ps,ts)
 assert len(curve)==1
 assert curve[0].coverage==1.0 and curve[0].risk==0.5
 receipt=select_confidence_threshold(ps,ts,allowed_groups={"g1","g2"},forbidden_groups=set(),max_selective_risk=0.0)
 assert receipt["threshold"] is None and receipt["coverage"]==0.0
