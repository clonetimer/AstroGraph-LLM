"""Experiment contracts, not model-quality assertions."""
from pathlib import Path
import sys,json,copy
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/ablation'),str(ROOT/'scripts')]
from common import intervene,sha_obj,read,write,event,SCORE_ARMS
from astrograph.oag.contracts import OAGInput
from astrograph.oag.kernel import EvidenceDiagnosisKernel
from astrograph.oag.agent import AutonomousDiagnosisService
from astrograph.oag.evidence_tools import MeasurementBank,tools_from_measurements
from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.evaluation.agent_quality.fixtures import build_numeric_cases
from astrograph.orchestration.models import WorkflowContext,PermissionScope

@pytest.fixture
def experiment(tmp_path):
    cases=build_numeric_cases(ROOT,tmp_path/'cases');req=read(cases[0]['request_path'])
    data=OAGInput.model_validate(read(req['artifacts']['input']))
    bank=MeasurementBank.model_validate(read(req['artifacts']['measurements']))
    kernel=EvidenceDiagnosisKernel(req['artifacts']['graph_dir'])
    ctx=WorkflowContext(workflow_id='test',run_id='test',actor='test',permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE])
    full=AutonomousDiagnosisService(kernel,tools=tools_from_measurements(bank,data.bundle),max_checks=2).run(data,context=ctx)
    return data,kernel,EpisodeEvidenceBundle.model_validate(full['effective_evidence_bundle']),full

@pytest.mark.parametrize('arm',list(SCORE_ARMS)+['no_executable_knowledge','explicit_counter_rules_off'])
def test_interventions_do_not_mutate_root_or_effective(experiment,arm):
    data,kernel,bundle,full=experiment
    d=sha_obj(data.model_dump(mode='json'));b=sha_obj(bundle.model_dump(mode='json'))
    result=intervene(data,bundle,kernel,arm=arm)
    assert result['diagnosis']['root_cause_confirmed'] is False
    assert result['diagnosis']['control_actions_authorized'] is False
    assert d==sha_obj(data.model_dump(mode='json'))
    assert b==sha_obj(bundle.model_dump(mode='json'))
    for row in result['fusion']['birrf_result']['candidates']:
        for key,value in SCORE_ARMS.get(arm,{}).items():
            source={'data_weight':'DATA_DRIVEN','forward_weight':'KG_FORWARD','backward_weight':'KG_BACKWARD'}.get(key)
            if source and value==0:assert row['reciprocal_contributions'].get(source,0)==0

def test_no_knowledge_is_not_merely_zero_scores(experiment):
    data,kernel,bundle,full=experiment
    none=intervene(data,bundle,kernel,arm='no_executable_knowledge')['diagnosis']
    assert none['status']=='MODEL_ONLY' and none['selected_fault_id']=='A'
    assert all(not r['supporting_evidence_ids'] and not r['contradicting_evidence_ids'] for r in none['candidate_explanations'])
    assert full['diagnosis']['selected_fault_id']=='B'

def test_explicit_counter_removal_has_causal_effect_in_fixture(experiment):
    data,kernel,bundle,full=experiment
    res=intervene(data,bundle,kernel,arm='explicit_counter_rules_off')
    assert res['diagnosis']['selected_fault_id'] is None
    assert res['diagnosis']['status']=='CANDIDATE_SET'
    assert full['diagnosis']['status']=='SUPPORTED_SINGLETON'


def test_unknown_arm_fails_closed(experiment):
    data,kernel,bundle,_=experiment
    with pytest.raises(ValueError,match='UNREGISTERED'):intervene(data,bundle,kernel,arm='made-up')


def test_immutable_json_and_ledger(tmp_path):
    p=tmp_path/'a.json';write(p,{'n':1})
    with pytest.raises(FileExistsError):write(p,{'n':2})
    event(tmp_path,'START');event(tmp_path,'DONE');rows=[json.loads(x) for x in (tmp_path/'ledger.jsonl').read_text(encoding="utf-8").splitlines()]
    assert rows[1]['previous_sha256']==rows[0]['event_sha256']
    for row in rows:
        digest=row.pop('event_sha256');assert digest==sha_obj(row)


def test_score_masks_are_not_zero_all_branches():
    from astrograph.fusion import BiRRFConfig
    for mask in SCORE_ARMS.values():
        cfg=BiRRFConfig.model_validate({**BiRRFConfig().model_dump(),**mask})
        assert cfg.data_weight+cfg.forward_weight+cfg.backward_weight>0


def test_macro_f1_keeps_abstention_unavailable():
    from astrograph.evaluation.three_task_replay import classification_metrics,ABSTAIN,UNAVAILABLE
    m=classification_metrics(['A','B','B'],['A',ABSTAIN,UNAVAILABLE],labels=['A','B'])
    assert m['n']==3 and m['correct']==1 and m['unavailable']==1 and m['abstained']==1
    assert m['per_class']['B']['fn']==2 and m['macro_f1']==.5
