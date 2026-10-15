"""Synthetic orchestration fixtures, not trained-model accuracy benchmarks."""
from datetime import datetime, timedelta, timezone
import threading
import pytest
from astrograph.semantics.online_contracts import ChannelBinding,DiagnosticRequest,ObservationFrame,ReleaseRef
from astrograph.orchestration.dual_source_dag import DualSourceDAGWorkflow
from astrograph.orchestration.models import WorkflowContext,PermissionScope,SideEffectLevel
from astrograph.tools.registry import ToolRegistry,ToolDefinition,ToolSpec
from astrograph.tools.plugins import PluginSnapshot
from astrograph.fusion.models import FaultRanking,RankedFault,RankingSource

T=datetime(2026,9,13,tzinfo=timezone.utc)
PIN=ReleaseRef(release_id='fixture',sha256='a'*64)

def request():
    frame=ObservationFrame.create(artifact_id='obs_fixture',channels=[ChannelBinding(channel_id='speed',semantic_id='rw.speed',unit='rad/s')],
        observed_at=[T],available_at=[T],values=[[2]],observed_mask=[[True]])
    return DiagnosticRequest(request_id='fixture',asset_id='fixture',operating_mode='tracking',cutoff_at=T,
        observation=frame,semantic_release=PIN,model_release=PIN,knowledge_release=PIN,plugin_release=PIN)

def distribution(ids,values):return {'status':'AVAILABLE','class_ids':ids,'values':values,'score_kind':'PROBABILITY'}

def data(payload):
    req=payload['request']
    return {'model_release':req['model_release'],'input_sha256':req['observation']['payload_sha256'],
        'anomaly':distribution(['NORMAL','ANOMALY'],[.1,.9]),'component':distribution(['wheel'],[1.]),
        'fault':distribution(['F1','F2'],[.2,.8] if payload['derived_observations'] else [.8,.2])}

def knowledge(payload,top='F1'):
    req=payload['request'];ids=[top,'F2' if top=='F1' else 'F1']
    def rank(source):
        return FaultRanking(ranking_id=source.value,episode_id=req['request_id'],source=source,producer_id='fixture',producer_version='1',
            candidates=[RankedFault(fault_id=f,rank=i+1,score=.8-i*.6,source=source) for i,f in enumerate(ids)]).model_dump(mode='json')
    return {'input_sha256':req['observation']['payload_sha256'],'knowledge_sha256':req['knowledge_release']['sha256'],
            'forward':rank(RankingSource.KG_FORWARD),'backward':rank(RankingSource.KG_BACKWARD)}

def observation(payload,**changes):
    v=dict(evidence_id='ev1',source_input_sha256=payload['request']['observation']['payload_sha256'],
        metric_id='registered.slope',unit='rad/s2',value=.5,available_at=T.isoformat(),source_evidence_ids=['obs_fixture'])
    v.update(changes);return {'observations':[v]}

def workflow(d=data,k=knowledge,inspect=None,**options):
    reg=ToolRegistry()
    for name,handler in [('data',d),('knowledge',k),('inspect',inspect)]:
        if handler:
            reg.register(ToolDefinition(ToolSpec(tool_id=name,side_effect_level=SideEffectLevel.READ_ONLY),handler))
    reg.seal()
    snapshot=PluginSnapshot('a'*64,{x.tool_id:x.tool_version for x in reg.list_specs()},reg)
    return DualSourceDAGWorkflow(snapshot,data_tool='data',knowledge_tool='knowledge',fault_universe=('F1','F2'),
        investigation_tools=('inspect',) if inspect else (),
        investigation_contracts={'inspect':{'registered.slope':'rad/s2'}} if inspect else {},**options)

def context():return WorkflowContext(workflow_id='fixture',run_id='fixture',permissions=[PermissionScope.RUN_DIAGNOSIS])

def test_parallel_independent_branches_and_real_fusion():
    barrier=threading.Barrier(2,timeout=4)
    def d(p):barrier.wait();return data(p)
    def k(p):
        assert set(p)=={'request','registered_fault_ids','derived_observations'}
        assert 'model_top' not in p
        barrier.wait();return knowledge(p)
    result=workflow(d,k).invoke(request(),context())
    assert result['stopping_reason']=='BRANCHES_AGREE'
    assert result['final_fusion']['top_fault_id']=='F1'
    assert result['formal_decision'] is None
    outcomes=result['rounds'][0]
    assert outcomes['Fuse']['started_ns']>=max(outcomes[x]['finished_ns'] for x in ('Data','Knowledge'))
    assert result['tool_calls']==2

def test_real_registered_observation_changes_model_ranking_then_refuses_to_claim_governor():
    result=workflow(k=lambda p:knowledge(p,'F2'),inspect=observation).invoke(request(),context())
    assert result['tool_calls']==5
    assert result['planning_trace'][0]['data_top']=='F1'
    assert result['planning_trace'][1]['data_top']=='F2'
    assert result['final_fusion']['top_fault_id']=='F2'
    assert result['stopping_reason']=='BRANCHES_AGREE'
    assert result['formal_decision'] is None

@pytest.mark.parametrize('inspect,reason',[
    (lambda p:{'observations':[]},'NO_NEW_EVIDENCE'),
    (lambda p:observation(p,available_at=(T+timedelta(seconds=1)).isoformat()),'INVESTIGATION_FAILED'),
    (lambda p:observation(p,source_input_sha256='b'*64),'INVESTIGATION_FAILED'),
])
def test_stop_without_new_admissible_evidence(inspect,reason):
    result=workflow(k=lambda p:knowledge(p,'F2'),inspect=inspect).invoke(request(),context())
    assert result['stopping_reason']==reason
    assert result['tool_calls']==3
    assert result['derived_observations']==[]

def test_budget_prevents_unusable_investigation():
    result=workflow(k=lambda p:knowledge(p,'F2'),inspect=observation,max_tool_calls=4).invoke(request(),context())
    assert result['stopping_reason']=='TOOL_BUDGET_EXHAUSTED' and result['tool_calls']==2

def test_both_failed_branches_are_unavailable_not_normal():
    def fail(p):raise RuntimeError('fixture')
    result=workflow(fail,fail).invoke(request(),context())
    assert result['status']=='UNAVAILABLE'
    assert result['stopping_reason']=='NO_AVAILABLE_CANDIDATES'
    assert result['formal_decision'] is None

def test_release_and_permission_checked_before_execution():
    with pytest.raises(PermissionError):workflow().invoke(request(),context().model_copy(update={'permissions':[]}))
    req=request().model_copy(update={'plugin_release':ReleaseRef(release_id='other',sha256='b'*64)})
    with pytest.raises(ValueError,match='snapshot'):workflow().invoke(req,context())

def test_branch_result_from_other_input_rejected_not_counted_as_support():
    def wrong(p):v=data(p);v['input_sha256']='b'*64;return v
    result=workflow(d=wrong).invoke(request(),context())
    assert result['rounds'][0]['Data']['status']=='FAILED'
    assert result['planning_trace'][0]['data_top'] is None
    assert result['planning_trace'][0]['conflict'] is True

def test_dynamic_tools_cannot_write_state():
    reg=ToolRegistry();reg.register(ToolDefinition(ToolSpec(tool_id='write',side_effect_level=SideEffectLevel.MUTATE_STATE),lambda p:{}))
    with pytest.raises(PermissionError,match='read-only'):
        DualSourceDAGWorkflow(PluginSnapshot('a'*64,{},reg),data_tool='write',knowledge_tool='write',fault_universe=('F1',))


def test_canonical_runtime_wraps_opt_in_subgraph_without_new_default():
    from astrograph.orchestration.dual_source_dag import build_dual_source_candidate_runtime,CANDIDATE_PIPELINE_ID
    from astrograph.runtime import RuntimeRequest,RuntimeStatus
    runtime=build_dual_source_candidate_runtime(workflow(),context())
    result=runtime.execute(RuntimeRequest(request_id='fixture',pipeline_id=CANDIDATE_PIPELINE_ID,payload=request().model_dump(mode='json')))
    assert result.status==RuntimeStatus.SUCCEEDED
    assert result.output['formal_decision'] is None
    assert result.stages[0].stage_id=='dual_source.candidate_subgraph.v1'

@pytest.mark.parametrize('changes',[{'metric_id':'invented.metric'},{'unit':'rpm'},{'source_evidence_ids':['unseen']}])
def test_unregistered_metric_unit_or_provenance_does_not_enter_fusion(changes):
    result=workflow(k=lambda p:knowledge(p,'F2'),inspect=lambda p:observation(p,**changes)).invoke(request(),context())
    assert result['stopping_reason']=='INVESTIGATION_FAILED'
    assert result['derived_observations']==[]
