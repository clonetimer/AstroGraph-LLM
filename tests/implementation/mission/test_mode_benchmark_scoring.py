"""Scoring must not turn abstention/unavailability into success or discard rows."""
import importlib.util
from pathlib import Path
import pytest
import numpy as np
spec=importlib.util.spec_from_file_location('mode_benchmark_score',Path(__file__).resolve().parents[3]/'scripts/score_mode_benchmark.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)

def test_fault_abstention_retained_but_component_set_can_agree():
    truth=[{'anomaly':1,'component':0,'fault':0}]
    pred=[{'prediction':{'kind':'SET','ids':['F1','F2']},'component':'C','fault':'F1'}]
    m=score.score_predictions(truth,pred,faults=['F1','F2'],components=['C'],fault_component={'F1':'C','F2':'C'})
    assert m['classification_system']['abstained']==1
    assert m['classification_system']['macro_f1']==0
    assert m['localization_system_derived']['correct']==1
    assert m['detection_all']['correct']==1
    assert m['classification_raw_head']['correct']==1

def test_unavailable_remains_in_all_window_denominator():
    t=[{'anomaly':0,'component':-1,'fault':-1},{'anomaly':1,'component':0,'fault':0}]
    p=[{'prediction':{'kind':'UNAVAILABLE','ids':[]}},{'prediction':{'kind':'CLASS','ids':['F']},'component':'C','fault':'F'}]
    m=score.score_predictions(t,p,faults=['F'],components=['C'],fault_component={'F':'C'})
    assert m['detection_all']['n']==2 and m['detection_all']['unavailable']==1
    assert m['detection_executable']['n']==1
    assert m['fault_counts']['correct_class']==1

def test_probability_shape_normalization_and_finite_are_checked():
    for v in ([[.5,.6]],[[float('nan'),0.]],[[1.,0.,0.]],[[-.1,1.1]]):
        with pytest.raises(ValueError):score.checked_probs(v,1,2)
    assert score.checked_probs([[.1,.9]],1,2).shape==(1,2)

def test_missed_detection_is_classification_failure_not_removed():
    m=score.score_predictions([{'anomaly':1,'component':0,'fault':0}],
        [{'prediction':{'kind':'NORMAL','ids':[]},'component':'C','fault':'F'}],
        faults=['F'],components=['C'],fault_component={'F':'C'})
    assert m['classification_system']['correct']==0 and m['fault_counts']['missed_as_normal']==1
    assert m['classification_raw_head']['correct']==1
    assert m['localization_independent_head']['correct']==1
