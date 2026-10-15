from datetime import datetime,timedelta,timezone
import hashlib,json
from pathlib import Path
import pytest
from astrograph.memory.service import CaseMemoryService,MemoryPolicy,PendingCase,MemoryConflictError
from astrograph.memory.ppl_bridge import PPLBridge
from astrograph.orchestration.models import WorkflowContext,PermissionScope

ROOT=Path(__file__).resolve().parents[2]
T=datetime(2026,9,13,tzinfo=timezone.utc)

def ctx(actor='agent',permissions=None):
    return WorkflowContext(workflow_id='memory',run_id='r',actor=actor,permissions=permissions or [PermissionScope.WRITE_MEMORY,PermissionScope.READ_MEMORY])

def policy():return MemoryPolicy.model_validate_json((ROOT/'configs/memory/diagnostic.policy.json').read_text(encoding="utf-8"))

def case(case_id='c1',**changes):
    data=dict(case_id=case_id,source_group_id='old-group',producer_actor='agent',symptom_ids=('speed-low',),operating_mode='tracking',
              proposed_fault_id='maybe-wrong',evidence_ids=('e1',),observed_end=T,end_reason='completed',output_valid=True)
    data.update(changes);return PendingCase(**data)

def confirm(store,**changes):
    data=dict(case_id='c1',event_id='review1',verified_fault_id='RW_JAM',truth_source_ref='verified-artifact-1',
              verified_at=T+timedelta(seconds=1),available_at=T+timedelta(seconds=2),expected_snapshot=0,
              context=ctx('ground-truth-reviewer',[PermissionScope.REVIEW_MEMORY]))
    data.update(changes);return store.confirm_case(**data)

def retrieve(store,**changes):
    data=dict(operating_mode='tracking',snapshot=store.snapshot(),cutoff_at=T+timedelta(seconds=3),
              exclude_case_id='new-case',exclude_source_group_id='new-group',context=ctx())
    data.update(changes);return store.retrieve(('speed-low',),**data)

def test_pending_not_truth_independent_confirmation_is_idempotent(tmp_path):
    s=CaseMemoryService(tmp_path/'memory.db',policy());s.stage_case(case(),event_id='stage1',context=ctx())
    assert retrieve(s)==[]
    first=confirm(s);assert first['snapshot']==1
    assert confirm(s)==first and s.snapshot()==1
    assert len(s.pending_outbox())==1
    result=retrieve(s);assert result[0]['case']['verified_fault_id']=='RW_JAM'
    assert result[0]['diagnostic_vote_allowed'] is False
    with pytest.raises(MemoryConflictError):confirm(s,verified_fault_id='different')

def test_snapshot_time_and_source_isolation(tmp_path):
    s=CaseMemoryService(tmp_path/'memory.db',policy());s.stage_case(case(),event_id='stage1',context=ctx());confirm(s)
    assert retrieve(s,snapshot=0)==[]
    assert retrieve(s,cutoff_at=T+timedelta(seconds=1))==[]
    assert retrieve(s,exclude_case_id='c1')==[]
    assert retrieve(s,exclude_source_group_id='old-group')==[]
    with pytest.raises(ValueError):retrieve(s,snapshot=20)

@pytest.mark.parametrize('changes',[{'end_reason':'max-tokens'},{'end_reason':'error'},{'output_valid':False}])
def test_truncated_failed_invalid_cases_never_confirm(tmp_path,changes):
    s=CaseMemoryService(tmp_path/'memory.db',policy());s.stage_case(case(**changes),event_id='s',context=ctx())
    with pytest.raises(ValueError,match='incomplete'):confirm(s)
    assert s.snapshot()==0 and s.pending_outbox()==[]

def test_permissions_producer_and_cas(tmp_path):
    s=CaseMemoryService(tmp_path/'memory.db',policy());s.stage_case(case(),event_id='s',context=ctx())
    with pytest.raises(PermissionError):confirm(s,context=ctx())
    with pytest.raises(MemoryConflictError,match='compare'):confirm(s,expected_snapshot=10)
    p=policy().model_copy(update={'approved_reviewers':('agent',)})
    same=CaseMemoryService(tmp_path/'memory.db',p)
    with pytest.raises(PermissionError,match='self-confirm'):confirm(same,context=ctx('agent',[PermissionScope.REVIEW_MEMORY]))

def test_confirmation_rolls_back_all_writes_on_receipt_failure(tmp_path,monkeypatch):
    s=CaseMemoryService(tmp_path/'memory.db',policy());s.stage_case(case(),event_id='s',context=ctx())
    def fail(*a):raise RuntimeError('storage failure')
    monkeypatch.setattr(s,'_receipt',fail)
    with pytest.raises(RuntimeError):confirm(s)
    assert s.snapshot()==0 and not s.pending_outbox() and not retrieve(s)

def bridge():return PPLBridge(ROOT,profile_sha256=policy().profile_sha256,approved_reviewers=frozenset(policy().approved_reviewers))

def test_real_canonical_ppl_completed_and_truncated_resolution():
    b=bridge();event={'type':'QUESTION_DEFINED','payload':{'question':'Why is wheel speed low?'}}
    complete=b.resolve(event=event,context=ctx(),end_reason='completed')
    assert complete['committed'] is True
    assert complete['state']['research']['question']=='Why is wheel speed low?'
    for reason in ('max-tokens','error'):
        truncated=b.resolve(event=event,context=ctx(),end_reason=reason)
        assert truncated['committed'] is False
        assert truncated['state']['research']['question']==''

def test_ppl_validation_requires_host_review_permission():
    with pytest.raises(PermissionError):bridge().resolve(event={'type':'VALIDATION_RESULT','payload':{}},context=ctx())

def test_ppl_state_transaction_replay_and_cas(tmp_path):
    s=CaseMemoryService(tmp_path/'memory.db',policy(),bridge=bridge())
    event={'type':'QUESTION_DEFINED','payload':{'question':'Q'}}
    first=s.apply_profile_event(event,event_id='p1',expected_version=0,end_reason='completed',context=ctx())
    assert first['version']==1
    second=s.apply_profile_event(event,event_id='p1',expected_version=0,end_reason='completed',context=ctx())
    assert first==second
    with pytest.raises(MemoryConflictError):s.apply_profile_event(event,event_id='p2',expected_version=0,end_reason='completed',context=ctx())
    result=s.apply_profile_event(event,event_id='p3',expected_version=1,end_reason='max-tokens',context=ctx())
    assert result['version']==1 and result['committed'] is False
