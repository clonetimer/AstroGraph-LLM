from copy import deepcopy
from astrograph.integrations.brb_diagnosis_trial import packet,predict,specificity_only


def summary():
    return {'diagnostic_decision':{'status':'SUPPORTED_SINGLETON','candidate_fault_ids':['ADCS_GYRO_BIAS_DRIFT'],'evidence_state_matrix':[{'fault_id':'ADCS_GYRO_BIAS_DRIFT','state':'SATISFIED'}],'blocking_conflicts':[],'observation_contract':{'paired_reference_identical':False}},'three_task_results':{'detection':{'detected':True},'head_trace':{'raw_class_probabilities':{'gyro':.2,'wheel':.8},'class_to_runtime_fault':{'gyro':'ADCS_GYRO_BIAS_DRIFT','wheel':'RW'}}},'candidates':[{'fault_id':'ADCS_GYRO_BIAS_DRIFT','eligible_for_explanation':True,'supporting_evidence_ids':['E'],'contradicting_evidence_ids':[]}]}


def test_specificity_repair_demotes_to_set_without_inventing_observation():
    s=summary();old=deepcopy(s);p=packet(s,repair_specificity=True)
    assert p['candidates']['ADCS_GYRO_BIAS_DRIFT']['mechanism']=='NONSPECIFIC'
    assert specificity_only(s)=={'kind':'SET','ids':['ADCS_GYRO_BIAS_DRIFT','RW']}
    assert s==old


def test_evaluation_metadata_cannot_affect_packet():
    s=summary();a=packet(s,repair_specificity=True)
    s.update(target='RW',case_id='train-99',evaluation_metadata={'truth':'gyro'})
    assert packet(s,repair_specificity=True)==a


def test_unseen_rules_abstain_from_specific_class():
    m={'method':'BRB','repair_specificity':True,'rules':[[0.,0.]]*12,'threshold':.2,'margin':0.}
    result=predict(summary(),m)
    assert result['prediction']['kind']=='SET'
    assert all(x['ignorance']==1 for x in result['scores'].values())


def test_confirmed_normal_reference_wins_over_high_rule_support():
    m={'method':'BRB','repair_specificity':True,'rules':[[1.,0.]]*12,'threshold':.2,'margin':0.}
    s=summary();s['diagnostic_decision']['observation_contract']['paired_reference_identical']=True
    assert predict(s,m)['prediction']=={'kind':'NORMAL','ids':[]}
