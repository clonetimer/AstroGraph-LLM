from dataclasses import replace
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import pytest
from astrograph.oag.agent import AutonomousDiagnosisService, CaseContext
from astrograph.oag.evidence_tools import FeatureSeries,MeasurementBank,tools_from_measurements
from astrograph.oag.kernel import EvidenceDiagnosisKernel
from astrograph.oag.cli import main
from astrograph.memory.service import CaseMemoryService,MemoryPolicy
from astrograph.orchestration.models import WorkflowContext,PermissionScope

ROOT=Path(__file__).resolve().parents[2]
T=datetime(2026,9,12,tzinfo=timezone.utc)

def ctx(actor='diagnostic-agent'):
    return WorkflowContext(workflow_id='agent-test',run_id='r',actor=actor,permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE,PermissionScope.READ_MEMORY,PermissionScope.WRITE_MEMORY])

def tools(data):
    bank=MeasurementBank(episode_id=data.bundle.episode.episode_id,series=tuple(
        FeatureSeries(feature_definition_id='F_'+name,field='sensor.'+name,unit='1',time_s=(0,1,2),current=(2,2,2),available_at_s=2)
        for name in ('b','counter')))
    return tools_from_measurements(bank,data.bundle)

def test_real_numeric_checks_change_final_decision_and_explanation(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=None,counter=None)
    before=data.model_dump_json()
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=tools(data)).run(data,context=ctx())
    assert out['initial_diagnosis']['status']=='MODEL_ONLY'
    assert out['diagnosis']['selected_fault_id']=='B'
    assert out['diagnosis']['status']=='SUPPORTED_SINGLETON'
    assert out['checks_executed']==2 and len(out['evidence_revisions'])==2
    assert out['explanation']['evidence_ids']
    assert data.model_dump_json()==before
    assert not out['control_actions_authorized']
    assert all(not r['original_model_receipt_rewritten'] for r in out['evidence_revisions'])


def test_good_decision_does_not_call_unnecessary_tools(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=2)
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=tools(data)).run(data,context=ctx())
    assert out['checks_executed']==0


def test_empty_tools_stops_with_explicit_gap(make_diagnostic_case):
    data,graph=make_diagnostic_case()
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph)).run(data,context=ctx())
    assert out['stopping_reason']=='NO_AVAILABLE_DISCRIMINATING_CHECK'
    assert out['diagnosis']['status']=='MODEL_ONLY'


def test_zero_budget_does_not_call_tool(make_diagnostic_case):
    data,graph=make_diagnostic_case()
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=tools(data),max_checks=0).run(data,context=ctx())
    assert out['checks_executed']==0 and out['stopping_reason']=='CHECK_BUDGET_EXHAUSTED'


def test_empty_tool_result_not_repeated_or_counted_as_support(make_diagnostic_case):
    data,graph=make_diagnostic_case()
    tool=replace(tools(data)[0],handler=lambda _:[])
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=(tool,),max_checks=3).run(data,context=ctx())
    assert out['checks_executed']==1 and not out['evidence_revisions']
    assert out['planning_trace'][0]['outcome']=='NO_CHANGE'


@pytest.mark.parametrize('change',[{'unit':'A'},{'window_end_s':3},{'input_data_sha256':'b'*64},{'sample_count':0},{'quality_flags':['INVALID']},{'value':float('nan')},{'episode_id':'other'}])
def test_bad_tool_evidence_cannot_change_diagnosis(make_diagnostic_case,change):
    data,graph=make_diagnostic_case()
    tool=tools(data)[0];original=tool.handler
    bad=replace(tool,handler=lambda request:[f.model_copy(update=change) for f in original(request)])
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=(bad,)).run(data,context=ctx())
    assert out['planning_trace'][0]['outcome']=='FAILED'
    assert not out['evidence_revisions']
    assert out['diagnosis']==out['initial_diagnosis']


def test_published_knowledge_cannot_change_mid_session(make_diagnostic_case):
    data,graph=make_diagnostic_case()
    tool=tools(data)[0];original=tool.handler
    def changing(request):
        with (graph/'edges.csv').open('a') as f:f.write('\n')
        return original(request)
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=(replace(tool,handler=changing),)).run(data,context=ctx())
    assert out['planning_trace'][0]['outcome']=='FAILED'
    assert out['diagnosis']==out['initial_diagnosis'] and not out['evidence_revisions']


def test_memory_requires_confirmation_and_replay_is_idempotent(make_diagnostic_case,tmp_path):
    data,graph=make_diagnostic_case()
    store=CaseMemoryService(tmp_path/'memory.db',MemoryPolicy(profile_sha256='0'*64,approved_reviewers=('reviewer',)))
    service=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=tools(data),memory=store)
    case=CaseContext('case-1','source-1','tracking',T)
    first=service.run(data,context=ctx(),case=case)
    assert first['memory']['write']=='STAGED_PENDING_INDEPENDENT_CONFIRMATION'
    assert store.snapshot()==0
    again=service.run(data,context=ctx(),case=case)
    assert again['memory']['write']=='STAGED_PENDING_INDEPENDENT_CONFIRMATION'
    receipt=store.confirm_case(case_id='case-1',event_id='review-1',verified_fault_id='B',truth_source_ref='independent-fixture-review',
                              verified_at=T+timedelta(seconds=1),available_at=T+timedelta(seconds=2),expected_snapshot=0,
                              context=WorkflowContext(workflow_id='review',run_id='review-1',actor='reviewer',permissions=[PermissionScope.REVIEW_MEMORY]))
    assert receipt['snapshot']==1
    new_case=CaseContext('case-2','source-2','tracking',T+timedelta(seconds=3))
    after=service.run(data,context=ctx(),case=new_case)
    assert after['memory']['retrieved_case_ids']==['case-1']
    assert any(t['plan']['memory_advised'] for t in after['planning_trace'])
    assert after['diagnosis']['selected_fault_id']=='B' and not after['confirmed_memory_is_diagnostic_vote']


def test_cli_runs_actual_kernel_and_numerical_checks(make_diagnostic_case,tmp_path):
    data,graph=make_diagnostic_case()
    inp=tmp_path/'input.json';inp.write_text(data.model_dump_json())
    bank=MeasurementBank(episode_id='episode',series=tuple(FeatureSeries(feature_definition_id='F_'+name,field='sensor.'+name,unit='1',
         time_s=(0,1,2),current=(2,2,2),available_at_s=2) for name in ('b','counter')))
    bp=tmp_path/'measurements.json';bp.write_text(bank.model_dump_json())
    out=tmp_path/'result.json'
    assert main(['diagnose-agent','--input',str(inp),'--graph-dir',str(graph),'--measurements',str(bp),'--output',str(out)])==0
    assert json.loads(out.read_text(encoding="utf-8"))['diagnosis']['selected_fault_id']=='B'


def test_permissions_are_not_taken_from_telemetry(make_diagnostic_case):
    data,graph=make_diagnostic_case()
    with pytest.raises(PermissionError):AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph)).run(data,context=WorkflowContext(workflow_id='x',run_id='r'))


def test_working_preference_does_not_stop_discriminating_check(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=None)
    counter_tool=next(t for t in tools(data) if 'counter' in t.tool_id)
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=(counter_tool,)).run(data,context=ctx())
    assert out['initial_diagnosis']['working_hypothesis_fault_id']=='A'
    assert out['initial_diagnosis']['selected_fault_id'] is None
    assert out['checks_executed']==1
    assert out['diagnosis']['selected_fault_id']=='B'


def test_unknown_measurement_contract_is_rejected():
    with pytest.raises(ValueError):MeasurementBank(episode_id='episode',schema_version='unknown.v9')
