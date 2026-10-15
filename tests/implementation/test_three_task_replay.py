import copy
import random
import pytest
from astrograph.evaluation.three_task_replay import (
    classification_metrics,evaluate_three_task_replay,ABSTAIN,UNAVAILABLE,NORMAL)


@pytest.mark.parametrize('seed',range(12))
def test_fixed_labels_match_sklearn_with_abstentions(seed):
    from sklearn.metrics import f1_score
    r=random.Random(seed);labels=['A','B','C'];t=[r.choice(labels) for _ in range(61)]
    p=[r.choice(labels+[ABSTAIN,UNAVAILABLE,NORMAL]) for _ in t]
    m=classification_metrics(t,p,labels=labels)
    assert m['macro_f1']==pytest.approx(f1_score(t,p,labels=labels,average='macro',zero_division=0))
    assert sum(map(sum,m['confusion']['counts']))==len(t)


def test_abstention_and_unavailable_are_not_removed():
    m=classification_metrics(['A','A','B'],['A',ABSTAIN,UNAVAILABLE],labels=['A','B'])
    assert m['per_class']['A']['fn']==1 and m['per_class']['B']['fn']==1
    assert m['macro_f1']==pytest.approx(1/3)
    assert m['accuracy_all']==1/3 and m['execution_coverage']==2/3


@pytest.mark.parametrize('truth,pred,labels',[
    (['A'],[],['A']),(['A'],['NEW'],['A']),(['NEW'],['A'],['A']),
    (['A'],['A'],['A','A']),(['A'],['A'],[]),([ABSTAIN],[ABSTAIN],[ABSTAIN])])
def test_reject_misaligned_or_unknown(truth,pred,labels):
    with pytest.raises(ValueError):classification_metrics(truth,pred,labels=labels)


def rows():
    return [dict(case_id='r',endpoint_s=1,target='A',target_component='x',
                 heads=dict(direct_component_probabilities={'x':.1,'y':.9},
                            derived_component_probabilities={'x':.9,'y':.1},
                            component_probabilities={'x':.7,'y':.3},
                            raw_class_probabilities={'a':.1,'b':.9},
                            dataset_class_probabilities={'a':.8,'b':.2}),
                 data=dict(kind='CLASS',ids=['A']), fused=dict(kind='SET',ids=['A','B']))]


def score(r):
    return evaluate_three_task_replay(r,fault_to_component={'A':'x','B':'y'},
                                     dataset_class_to_fault={'a':'A','b':'B'},branches=['data','fused'])


def test_direct_head_is_not_replaced_by_fault_mapping():
    m=score(rows())
    assert m['model_heads']['localization_direct_head']['correct']==0
    assert m['model_heads']['localization_deployed_blend']['correct']==1
    assert m['model_heads']['classification_raw_head']['correct']==0
    assert m['model_heads']['classification_deployed_policy']['correct']==1
    assert m['branches']['data']['localization_system_derived']['correct']==1
    assert m['branches']['fused']['classification_system']['abstained']==1
    assert m['branches']['fused']['detection_executable']['correct']==1


def test_legacy_missing_heads_are_not_fabricated():
    r=rows();r[0]['heads']=None;m=score(r)
    assert m['model_heads']['localization_direct_head']['unavailable']==1
    assert m['model_heads']['localization_direct_head']['correct']==0


def test_component_can_be_resolved_while_fault_set_remains():
    r=rows();r[0]['heads']=None
    m=evaluate_three_task_replay(r,fault_to_component={'A':'x','B':'x'},dataset_class_to_fault={'a':'A','b':'B'},branches=['fused'])
    assert m['branches']['fused']['classification_system']['abstained']==1
    assert m['branches']['fused']['localization_system_derived']['correct']==1


def test_duplicate_and_bad_component_truth_fail():
    r=rows()
    with pytest.raises(ValueError):score(r+r)
    r[0]['target_component']='y'
    with pytest.raises(ValueError):score(r)


def test_normal_rows_do_not_enter_fault_classification():
    r=rows();n=copy.deepcopy(r[0]);n.update(case_id='normal',target=None,target_component=None)
    n['data']={'kind':'UNAVAILABLE','ids':[]};n['fused']={'kind':'NORMAL','ids':[]}
    m=score(r+[n])
    assert m['branches']['data']['classification_system']['n']==1
    assert m['branches']['data']['detection_all']['n']==2
    assert m['branches']['data']['detection_executable']['n']==1


def test_invalid_probability_or_fault_on_normal_window_fails():
    r=rows();r[0]['heads']['direct_component_probabilities']['x']=float('nan')
    with pytest.raises(ValueError):score(r)
    r=rows();r[0].update(target=None,target_component=None);r[0]['fused']={'kind':'SET','ids':['NEW']}
    with pytest.raises(ValueError):score(r)
