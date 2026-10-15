"""Mode acceptance: same kernel, host-owned budgets, quarantine and provenance."""
import json
from pathlib import Path
import pytest
from astrograph.agents.mission.models import AgentRequest, AgentSettings, ModeBudget, ExplorationProposal, CaseIdentity
from astrograph.agents.mission.modes import resolve_mode, allowed_tools
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.planner import Planner
from .test_agent import setup_case, ScriptedBackend

@pytest.mark.parametrize('mode',['normal','thinking','creative'])
def test_modes_preserve_same_diagnosis_without_new_evidence(tmp_path,make_diagnostic_case,mode):
    s,req,*_=setup_case(tmp_path,make_diagnostic_case)
    normal=MissionAgent(s).run(req)
    other=MissionAgent(s).run(req.model_copy(update={'task_id':'mode-'+mode,'work_mode':mode}))
    assert other.status=='SUCCEEDED',other.model_dump()
    for key in ('diagnosis','fusion','explanation'):
        assert normal.output['diagnosis_result'][key]==other.output['diagnosis_result'][key]
    assert other.work_mode==mode and other.mode_receipt['effective_mode']==mode
    assert not other.mode_receipt['mode_grants_permissions']
    if mode!='normal':assert other.output['investigation']['not_a_new_diagnosis']
    if mode=='creative':
        assert other.output['exploration']['status']=='RESEARCH_AGENDA_ONLY'
        assert other.output['exploration']['memory_write']=='NONE'
    else:assert 'exploration' not in other.output

@pytest.mark.parametrize('mode',['normal','thinking','creative'])
def test_mode_does_not_grant_training_permission(tmp_path,mode):
    req=AgentRequest(task_id='noauth',task='train_model',work_mode=mode)
    s=AgentSettings(read_roots=[str(tmp_path)],output_root=str(tmp_path/'runs'))
    with pytest.raises(ValueError,match='training_manifest'):MissionAgent(s).run(req)
    assert 'model.train' not in allowed_tools('diagnose',mode)

@pytest.mark.parametrize('mode',['thinking','creative'])
def test_mode_budget_inherits_legacy_until_host_explicitly_overrides(mode):
    r=AgentRequest(task_id='x',work_mode=mode)
    a=AgentSettings(max_checks=2,max_optional_actions=1,max_llm_calls=4)
    s,receipt=resolve_mode(a,r,backend_available=False)
    assert (s.max_checks,s.max_optional_actions,s.max_llm_calls)==(2,1,4)
    a=AgentSettings(max_checks=2,work_mode_budgets={mode:ModeBudget(max_checks=5,max_proposals=2)})
    s,receipt=resolve_mode(a,r,backend_available=False)
    assert s.max_checks==5 and receipt['max_proposals']==2
    _,normal=resolve_mode(a,AgentRequest(task_id='n'),backend_available=False)
    assert normal['budgets']['max_checks']==2

def test_request_cannot_raise_budgets_or_change_mode_to_arbitrary():
    for kwargs in ({'work_mode':'god'},{'max_checks':20},{'work_mode_budgets':{}},{'work_mode':'creative','allow_external_literature':True}):
        with pytest.raises(ValueError):AgentRequest(task_id='x',**kwargs)

def test_normal_cannot_be_overridden_by_mode_profile():
    with pytest.raises(ValueError):AgentSettings(work_mode_budgets={'normal':{'max_checks':10}})

def test_host_mode_deny_happens_before_workspace(tmp_path):
    s=AgentSettings(allowed_work_modes=['normal'],output_root=str(tmp_path/'runs'))
    with pytest.raises(PermissionError,match='WORK_MODE'):MissionAgent(s).run(AgentRequest(task_id='x',work_mode='creative'))
    assert not (tmp_path/'runs'/'x').exists()

def test_unavailable_llm_is_reported_not_hidden():
    _,r=resolve_mode(AgentSettings(planner='llm_assisted'),AgentRequest(task_id='x',work_mode='thinking'),backend_available=False)
    assert r['planner_effective']=='deterministic' and r['capability_degradations']
    assert r['effective_mode']=='thinking'  # deterministic thinking strategy, not fabricated LLM work

class CreativeBackend(ScriptedBackend):
    def generate_structured(self,**kw):
        model=kw['output_model'];p=kw['payload']
        if model is ExplorationProposal:
            return model(hypotheses=[{'hypothesis':'A competing sensor mechanism requires further testing.',
                'citation_ids':[p['sources_untrusted'][0]['citation_id']],
                'required_observations':['independent sensor residual'],
                'falsification_test':'A reliable independent residual consistent with nominal would reject this proposed explanation.'}])
        return super().generate_structured(**kw)

def test_creative_llm_proposal_quarantined_and_core_unchanged(tmp_path,make_diagnostic_case):
    s,r,*_=setup_case(tmp_path,make_diagnostic_case,planner='llm_assisted')
    b=CreativeBackend();normal=MissionAgent(s,backend=b).run(r)
    creative=MissionAgent(s,backend=b).run(r.model_copy(update={'task_id':'creative','work_mode':'creative'}))
    assert creative.output['diagnosis_result']['diagnosis']==normal.output['diagnosis_result']['diagnosis']
    e=creative.output['exploration']
    assert len(e['hypotheses'])==1 and e['hypotheses'][0]['status']=='UNVERIFIED'
    assert not e['diagnostic_vote_allowed'] and not e['automatic_execution'] and e['memory_write']=='NONE'
    assert (Path(s.output_root)/'creative'/'exploration.json').is_file()
    assert all(x['tool_id'] not in {'model.train','graph.build','memory.confirm'} for x in creative.tool_receipts)

@pytest.mark.parametrize('attack',['citation','confirmation','execution','quota','exception'])
def test_bad_exploration_never_changes_core(tmp_path,make_diagnostic_case,attack):
    class Bad(CreativeBackend):
        def generate_structured(self,**kw):
            if kw['output_model'] is ExplorationProposal:
                if attack=='exception':raise TimeoutError('fixture only')
                v=super().generate_structured(**kw).model_dump()
                if attack=='citation':v['hypotheses'][0]['citation_ids']=['FAKE']
                if attack=='confirmation':v['confirmed_memory_write_allowed']=True
                if attack=='execution':v['execute_code']='arbitrary_code'
                if attack=='quota':v['hypotheses']*=4
                return v
            return super().generate_structured(**kw)
    s,r,*_=setup_case(tmp_path,make_diagnostic_case,planner='llm_assisted')
    out=MissionAgent(s,backend=Bad()).run(r.model_copy(update={'work_mode':'creative'}))
    # dev7: diagnosis is preserved, failed LLM capability is no longer hidden.
    assert out.capability_status['core_diagnosis']=='SUCCEEDED'
    assert out.status=='PARTIAL' and out.output['diagnosis_result']['diagnosis']['selected_fault_id']=='B'
    assert out.output['exploration']['status']=='RESEARCH_AGENDA_ONLY'

def test_same_id_different_mode_cannot_reuse_completed_result(tmp_path,make_diagnostic_case):
    s,r,*_=setup_case(tmp_path,make_diagnostic_case);a=MissionAgent(s);a.run(r)
    with pytest.raises(ValueError,match='different request'):a.run(r.model_copy(update={'work_mode':'thinking'}))

def test_explicit_normal_equals_omitted_normal(tmp_path,make_diagnostic_case):
    s,r,*_=setup_case(tmp_path,make_diagnostic_case);a=MissionAgent(s)
    assert a.run(r).model_dump()==a.run(AgentRequest(**r.model_dump())).model_dump()

@pytest.mark.parametrize('mode',['thinking','creative'])
def test_budget_zero_no_context_calls(tmp_path,make_diagnostic_case,mode):
    s,r,*_=setup_case(tmp_path,make_diagnostic_case,max_checks=0,work_mode_budgets={mode:ModeBudget(max_optional_actions=0)})
    p=tmp_path/'doc.md';p.write_text('Sensor fault troubleshooting and counterevidence.')
    out=MissionAgent(s).run(r.model_copy(update={'work_mode':mode,'documents':[str(p)]}))
    assert not any(x['tool_id'] in {'rag.retrieve','literature.search'} for x in out.tool_receipts)
    assert out.output['diagnosis_result']['checks_executed']==0

def test_mode_and_device_operating_mode_distinct():
    from datetime import datetime,timezone
    r=AgentRequest(task_id='x',work_mode='thinking',case=CaseIdentity(case_id='x',source_group_id='g',operating_mode='eclipse',observed_end=datetime.now(timezone.utc)))
    assert r.case.operating_mode=='eclipse' and r.work_mode=='thinking'

def test_normal_planner_cannot_generate_exploration():
    with pytest.raises(PermissionError):Planner().explore(diagnosis={},sources=[])

def test_thinking_repeated_tool_queries_must_be_distinct():
    from astrograph.agents.mission.models import InvestigationPlan
    class Duplicate:
        def generate_structured(self,**kw):
            return InvestigationPlan(task='diagnose',actions=[{'tool_id':'rag.retrieve','query':'same','rationale':'x'}]*2,rationale='x')
    p=Planner(Duplicate(),mode='llm_assisted',work_mode='thinking')
    v=p.prepare(AgentRequest(task_id='x'),['rag.retrieve'],max_actions=2)
    assert p.records[-1]['outcome']=='DETERMINISTIC_FALLBACK'

def test_creative_without_documents_or_llm_does_not_invent_findings(tmp_path):
    s=AgentSettings(output_root=str(tmp_path/'r'))
    out=MissionAgent(s).run(AgentRequest(task_id='empty',task='research',work_mode='creative'))
    assert out.status=='PARTIAL'
    assert out.output['exploration']['hypotheses']==[]

def test_cli_mode_override_changes_only_requested_strategy(tmp_path,make_diagnostic_case,capsys):
    from astrograph.agents.mission.cli import main
    s,r,*_=setup_case(tmp_path,make_diagnostic_case)
    sp=tmp_path/'settings.json';rp=tmp_path/'request.json'
    sp.write_text(s.model_dump_json());rp.write_text(r.model_dump_json())
    assert main(['run','--settings',str(sp),'--request',str(rp),'--work-mode','thinking'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['work_mode']=='thinking'
    actual=json.loads(Path(result['result_path']).read_text(encoding="utf-8"))
    assert actual['output']['diagnosis_result']['diagnosis']['selected_fault_id']=='B'

def test_cli_denied_mode_does_not_elevate_host(tmp_path,capsys):
    from astrograph.agents.mission.cli import main
    s=AgentSettings(allowed_work_modes=['normal'],output_root=str(tmp_path/'run'))
    sp=tmp_path/'settings.json';rp=tmp_path/'request.json'
    sp.write_text(s.model_dump_json());rp.write_text(AgentRequest(task_id='denied').model_dump_json())
    assert main(['run','--settings',str(sp),'--request',str(rp),'--work-mode','creative'])==2
    assert 'WORK_MODE_NOT_AUTHORIZED' in capsys.readouterr().out
    assert not (tmp_path/'run'/'denied').exists()

@pytest.mark.parametrize('work_mode',['normal','creative'])
def test_backend_cannot_mutate_kernel_through_planning_payload(tmp_path,make_diagnostic_case,work_mode):
    class Mutates(CreativeBackend):
        def generate_structured(self,**kw):
            value=super().generate_structured(**kw)
            # A provider implementation must not receive mutable references to
            # kernel output, even when it is an in-process backend rather than HTTP.
            if kw['payload'].get('fixed_diagnosis'):
                kw['payload']['fixed_diagnosis']['selected_fault_id']='FORGED'
            return value
    s,r,*_=setup_case(tmp_path,make_diagnostic_case,planner='llm_assisted')
    out=MissionAgent(s,backend=Mutates()).run(r.model_copy(update={'work_mode':work_mode}))
    assert out.status=='SUCCEEDED'
    assert out.output['diagnosis_result']['diagnosis']['selected_fault_id']=='B'
    assert 'FORGED' not in json.dumps(out.output['diagnosis_result'])
