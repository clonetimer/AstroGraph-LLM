"""AR audit findings turned into assertions of correct behavior, not defect reproduction.
All network endpoints bind loopback. Scripted backends are not domain LLM claims.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from pydantic import BaseModel
from astrograph.agents.mission.models import (AgentRequest,AgentSettings,InvestigationPlan,CheckChoice,AgentNarrative,ExplorationProposal)
from astrograph.agents.mission.service import MissionAgent, _TaskRun
from astrograph.agents.mission.planner import Planner
from astrograph.agents.mission.context import ContextBuilder, ContextBudgetExceeded
from astrograph.agents.mission.storage import Workspace,sha,file_sha
from astrograph.agents.mission.documents import DocumentTools
from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend,OutputBudgetError
from astrograph.integrations.execution_budget import ExecutionBudget,execution_scope,AttemptBudgetExceeded
from astrograph.integrations.bounded_http import request_json
from astrograph.oag.contracts import OntologyProposal
from astrograph.orchestration.models import WorkflowContext,PermissionScope
from astrograph.tools.registry import ToolRegistry,ToolDefinition,ToolSpec

ROOT=Path(__file__).resolve().parents[3]

class Scripted:
    def __init__(self):self.payloads=[]
    def generate_structured(self,*,agent_id,output_model,payload):
        self.payloads.append((output_model.__name__,copy.deepcopy(payload)))
        if output_model is InvestigationPlan:return output_model(task=payload['fixed_task'],actions=[],rationale='approved tools only')
        if output_model is CheckChoice:return output_model(tool_id=payload['eligible_checks'][0]['tool_id'],rationale='discriminate candidates')
        if output_model is AgentNarrative:return output_model(candidate_fault_id=payload['required_candidate_fault_id'],excerpt_ids=[payload['sources_untrusted'][0]['excerpt_id']])
        if output_model is ExplorationProposal:return output_model(hypotheses=[])
        if output_model is OntologyProposal:return output_model(triples=[])
        raise ValueError(output_model.__name__)

def setup(tmp_path,**kw):
    req=AgentRequest(task_id='repair',artifacts={k:str(ROOT/'examples/core_diagnosis'/v) for k,v in {'input':'input.json','graph_dir':'graph','measurements':'measurements.json'}.items()})
    settings=AgentSettings(project_root=str(ROOT),read_roots=[str(ROOT),str(tmp_path)],output_root=str(tmp_path/'runs'),**kw)
    return settings,req

@pytest.fixture
def http_server():
    servers=[]
    def start(handler):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler);server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();servers.append((server,thread))
        return f'http://127.0.0.1:{server.server_port}'
    yield start
    for server,thread in servers:server.shutdown();server.server_close();thread.join(timeout=1)

class Quiet(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

@pytest.mark.parametrize('model',[InvestigationPlan,CheckChoice,AgentNarrative,ExplorationProposal,OntologyProposal])
@pytest.mark.parametrize('profile',['legacy_v051','evidence_protocol_v052','evidence_protocol_v059_compact'])
def test_task_prompt_does_not_inherit_hypothesis_protocol(model,profile):
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='fixture',prompt_profile=profile))
    # Prompt routing is tested with a satisfiable host contract; an empty check
    # catalogue is independently tested to reject before any network request.
    payload={"eligible_checks":[{"tool_id":"fixture.only"}]} if model is CheckChoice else {}
    messages=backend._messages(output_model=model,payload=payload)
    assert 'NO_REGISTERED_FAULT' not in messages[0]['content']
    assert 'hypothesis set' not in json.loads(messages[1]['content'])['task'].lower()
    assert 'output_json_schema' in json.loads(messages[1]['content'])

@pytest.mark.skipif(os.name!='posix',reason='POSIX interrupt unit test; portable guarantee is supervisor')
def test_inprocess_sleep_interrupted_before_side_effect():
    seen=[];reg=ToolRegistry()
    def slow(p):time.sleep(.20);seen.append('too late');return {'status':'OK'}
    reg.register(ToolDefinition(ToolSpec(tool_id='slow',tool_version='1',timeout_s=.03),slow))
    context=WorkflowContext(workflow_id='test',run_id='r',permissions=[PermissionScope.READ_ARTIFACTS])
    start=time.monotonic()
    with pytest.raises(TimeoutError,match='DEADLINE'):reg.invoke('slow',{},context)
    assert time.monotonic()-start<.16 and not seen
    assert reg.executor.receipts()[0]['python_interruptible_timeout']
    assert not reg.executor.receipts()[0]['hard_timeout_enforced']  # not claiming C interruptibility


def test_slow_drip_has_total_http_deadline(http_server):
    class Drip(Quiet):
        def do_POST(self):
            self.send_response(200);self.end_headers()
            try:
                for byte in b'{"ok":true}':self.wfile.write(bytes([byte]));self.wfile.flush();time.sleep(.08)
            except (BrokenPipeError,ConnectionResetError):pass
    url=http_server(Drip);start=time.monotonic()
    with pytest.raises(TimeoutError,match='DEADLINE'):request_json(url,payload={},timeout_s=.23)
    assert time.monotonic()-start<.65

@pytest.mark.parametrize('declared',[True,False])
def test_http_response_size_is_bounded(http_server,declared):
    class Large(Quiet):
        def do_POST(self):
            self.send_response(200)
            if declared:self.send_header('Content-Length','10000')
            self.end_headers()
            try:self.wfile.write(b'x'*10000)
            except BrokenPipeError:pass
    with pytest.raises(ValueError,match='TOO_LARGE'):request_json(http_server(Large),payload={},timeout_s=2,max_response_bytes=1024)


def test_http_error_does_not_echo_server_secrets(http_server):
    class Bad(Quiet):
        def do_POST(self):self.send_response(500);self.end_headers();self.wfile.write(b'PRIVATE_TOKEN_SENTINEL')
    with pytest.raises(RuntimeError) as exc:request_json(http_server(Bad),payload={},timeout_s=2)
    assert '500' in str(exc.value) and 'PRIVATE_TOKEN_SENTINEL' not in str(exc.value)


def test_http_redirect_does_not_forward_provider_authorization(http_server):
    received = []
    class Destination(Quiet):
        def do_GET(self):
            received.append(self.headers.get('Authorization'))
            self.send_response(200);self.end_headers()
            self.wfile.write(b'{"ok":true}')
        do_POST = do_GET
    destination = http_server(Destination)
    class Redirecting(Quiet):
        def do_POST(self):
            self.send_response(307)
            self.send_header('Location', destination+'/steal')
            self.end_headers()
    with pytest.raises(RuntimeError,match='HTTP_REDIRECT_NOT_ALLOWED'):
        request_json(http_server(Redirecting),payload={'ping':True},
                     headers={'Authorization':'Bearer local-provider-secret'},timeout_s=3)
    assert received == []


def test_all_formats_share_physical_request_budget():
    b=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='fixture',repair_attempts=1,tokenizer_mode='disabled'))
    sent=[]
    def post(body):sent.append(body);raise ValueError('controlled failure')
    b._post=post
    budget=ExecutionBudget(5,max_http_attempts=2)
    with execution_scope(budget),pytest.raises(AttemptBudgetExceeded):
        b.generate_structured(agent_id='test',output_model=InvestigationPlan,payload={'fixed_task':'research'})
    assert len(sent)==budget.http_attempts==2 and b.last_call_metadata['attempts']==2


def test_retry_expansion_rechecks_prompt_plus_output():
    b=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='fixture',response_format_strategy='json_schema',
        repair_attempts=1,tokenizer_mode='disabled',max_tokens=600,context_length=1024,budget_policy='auto_expand'))
    b._measure_prompt_tokens=lambda **kw:{'selected_prompt_tokens':100,'guarded_prompt_tokens':132}
    sent=[]
    def post(body):
        sent.append(body)
        return {'choices':[{'message':{'content':json.dumps({'task':'research','actions':[],'rationale':'fixture'})},'finish_reason':'length'}]}
    b._post=post
    with pytest.raises(OutputBudgetError,match='RETRY_CONTEXT'):b.generate_structured(agent_id='test',output_model=InvestigationPlan,payload={})
    assert len(sent)==1

@pytest.mark.parametrize('mode',['normal','thinking','creative'])
def test_zero_optional_budget_means_zero_queries_in_all_modes(tmp_path,mode):
    s,r=setup(tmp_path,max_optional_actions=0,max_checks=0)
    doc=tmp_path/'source.md';doc.write_text('Sensor troubleshooting and counterevidence.')
    out=MissionAgent(s).run(r.model_copy(update={'work_mode':mode,'documents':[str(doc)]}))
    assert not any(x['tool_id'] in {'rag.retrieve','literature.search'} for x in out.tool_receipts)


def test_total_llm_failure_is_visible_and_core_preserved(tmp_path):
    class Fails:
        def generate_structured(self,**kw):raise TimeoutError('fixture')
    s,r=setup(tmp_path,planner='llm_assisted');out=MissionAgent(s,backend=Fails()).run(r)
    assert out.status=='PARTIAL' and out.mode_receipt['planner_effective']=='deterministic_fallback'
    assert out.capability_status['llm']['accepted']==0 and out.limitations
    assert out.capability_status['core_diagnosis']=='SUCCEEDED' and out.output['diagnosis_result']


def test_late_output_failure_returns_committed_core(tmp_path,monkeypatch):
    def broken(self):raise RuntimeError('fixture output failure')
    monkeypatch.setattr(_TaskRun,'_structured_output',broken)
    s,r=setup(tmp_path);out=MissionAgent(s).run_inline(r)
    assert out.status=='PARTIAL' and out.output['diagnosis_result']
    root=Path(s.output_root)/r.task_id
    assert json.loads((root/'core-result.json').read_text(encoding="utf-8"))==out.output['diagnosis_result']
    assert file_sha(root/'core-result.json')==json.loads((root/'core-result-receipt.json').read_text(encoding="utf-8"))['sha256']


def test_context_preserves_values_units_negation_and_receipts(tmp_path):
    b=ContextBuilder(max_tokens=6000,output_reserve=1000)
    mandatory={'selected_fault_id':None,'value':-0.0005,'unit':'rad/s','status':'NOT_SATISFIED',
               'contradicting_evidence_ids':['E-negative'],'observed_at_s':12.5}
    sources=[{'citation_id':'short','text':'Not confirmed. Missing independent sensor observation.'},
             {'citation_id':'huge','text':'x'*120000}]
    out=b.build(CheckChoice,{'diagnosis':mandatory,'eligible_checks':[],'sources_untrusted':sources})
    assert out['diagnosis']==mandatory
    assert out['sources_untrusted'][0]['text']==sources[0]['text']
    assert b.records[-1]['omitted_sources'] and not b.records[-1]['protected_facts_rewritten']
    json.loads(json.dumps(out))


def test_mandatory_context_too_large_refuses_instead_of_truncating():
    b=ContextBuilder(max_tokens=4096,output_reserve=1000)
    with pytest.raises(ContextBudgetExceeded):b.build(CheckChoice,{'diagnosis':{'counterevidence':'must keep '*5000}})
    assert b.records[-1]['status']=='MANDATORY_CONTEXT_TOO_LARGE'


def test_context_duplicate_source_does_not_become_extra_evidence():
    b=ContextBuilder()
    out=b.build(CheckChoice,{'sources_untrusted':[{'citation_id':'a','text':'not confirmed'},{'citation_id':'b','text':'not confirmed'}]})
    assert len(out['sources_untrusted'])==1
    assert b.records[-1]['omitted_sources'][0]['reason']=='DUPLICATE_CONTENT'


def test_exploration_context_is_whole_json_and_has_check_history(tmp_path):
    s,r=setup(tmp_path,planner='llm_assisted');b=Scripted()
    out=MissionAgent(s,backend=b).run(r.model_copy(update={'work_mode':'creative'}))
    payload=next(v for name,v in b.payloads if name=='ExplorationProposal')
    anchors=[json.loads(s['text']) for s in payload['sources_untrusted']]
    # dev9 splits the old oversized TASK JSON into independently cited facts.
    assert any('stopping_reason' in a for a in anchors)
    assert any('root_cause_confirmed' in a for a in anchors)
    source_ids={s['citation_id'] for s in payload['sources_untrusted']}
    assert set(payload['allowed_citation_ids'])==source_ids
    assert out.output['exploration']['memory_write']=='NONE'


def test_fabricated_numerical_narrative_rejected():
    class Fabricates:
        def generate_structured(self,**kw):return AgentNarrative(candidate_fault_id='A',paragraphs=[{'text':'Voltage is 999 V.','citation_ids':['real']}])
    p=Planner(Fabricates(),mode='llm_assisted')
    out=p.explain({'selected_fault_id':'A'},[{'citation_id':'real','text':'No voltage observation is available.'}])
    assert out['status']=='DETERMINISTIC_EXPLANATION_ONLY'
    assert 'NARRATIVE_NOT_EXACT' in p.records[-1]['reason']


def test_exact_quoted_explanation_allowed_but_not_claimed_as_truth():
    text='Current evidence does not distinguish the two candidates.'
    class Quotes:
        def generate_structured(self,**kw):return AgentNarrative(candidate_fault_id=None,paragraphs=[{'text':text,'citation_ids':['real']}])
    out=Planner(Quotes(),mode='llm_assisted').explain({'selected_fault_id':None},[{'citation_id':'real','text':text}])
    assert out['assertion_policy']=='SOURCE_EXTRACTIVE_ONLY'
    assert not out['semantic_truth_independently_verified']


def test_check_selector_gets_history_and_no_repeat(tmp_path):
    s,r=setup(tmp_path,planner='llm_assisted');b=Scripted()
    out=MissionAgent(s,backend=b).run(r)
    payloads=[v for name,v in b.payloads if name=='CheckChoice']
    assert payloads and all('planning_trace' in p and 'remaining_checks' in p for p in payloads)
    ids=[x['tool_id'] for x in out.output['diagnosis_result']['planning_trace']]
    assert len(ids)==len(set(ids))
    if len(payloads)>1:assert payloads[1]['planning_trace']


def test_rag_index_and_query_cache_invalidate_on_document_change(tmp_path):
    ws=Workspace(tmp_path,[tmp_path],'rag');ws.root.mkdir()
    t=DocumentTools(ws,AgentSettings(read_roots=[str(tmp_path)]))
    t.documents=[{'document_id':'D','text':'# sensor\nsensor evidence','source_sha256':'a'*64,'source_uri':'fixture:///d'}]
    a=t.retrieve('sensor');b=t.retrieve('sensor')
    assert a==b and t.cache_stats['index_builds']==1 and t.cache_stats['query_hits']==1
    t.documents[0]['text']+='\n\nNew observation.'
    t.retrieve('sensor');assert t.cache_stats['index_builds']==2


def test_dense_corpus_encoded_once_queries_separately(tmp_path):
    import numpy as np
    ws=Workspace(tmp_path,[tmp_path],'dense');ws.root.mkdir()
    t=DocumentTools(ws,AgentSettings(read_roots=[str(tmp_path)],embedding_model_path=str(tmp_path)))
    t.documents=[{'document_id':'D','text':'# sensor\nsensor evidence','source_sha256':'a'*64,'source_uri':'fixture:///d'}]
    class Encoder:
        def encode(self,texts,**kw):return np.ones((len(texts),2))/2**.5
    t._dense=Encoder()
    t.retrieve('sensor');t.retrieve('evidence');t.retrieve('sensor')
    assert t.cache_stats['corpus_encodes']==1 and t.cache_stats['query_encodes']==2


def test_large_graph_extraction_batches_are_accounted_for(tmp_path):
    ws=Workspace(tmp_path,[tmp_path],'graph');ws.root.mkdir()
    settings=AgentSettings(ontology_batch_chars=1000,ontology_max_batches=2)
    t=DocumentTools(ws,settings,planner=Planner(Scripted(),mode='llm_assisted',max_calls=2))
    t.documents=[{'document_id':'D','text':'Gyro bias affects attitude.\n'*200,'source_sha256':'a'*64}]
    result=t.build_graph()
    assert len(result['batch_coverage'])>2 and not result['extraction_complete']
    assert any(x['status']=='BATCH_BUDGET_EXHAUSTED' for x in result['batch_coverage'])
    assert not result['active_kg_modified']


def test_context_artifacts_are_saved_hash_bound(tmp_path):
    s,r=setup(tmp_path,planner='llm_assisted');MissionAgent(s,backend=Scripted()).run(r)
    root=Path(s.output_root)/r.task_id
    receipts=json.loads((root/'context-receipts.json').read_text(encoding="utf-8"))
    assert receipts
    for entry in receipts:
        artifact=entry['full_payload_artifact'];assert file_sha(root/artifact['name'])==artifact['sha256']


def test_process_supervised_and_inline_share_diagnosis(tmp_path):
    s,r=setup(tmp_path)
    inline=MissionAgent(s).run_inline(r)
    supervised=MissionAgent(s).run_supervised(r.model_copy(update={'task_id':'supervised'}))
    assert supervised.output['diagnosis_result']['diagnosis']==inline.output['diagnosis_result']['diagnosis']
    assert supervised.execution_budget['enforcement']=='PROCESS_SUPERVISOR'


def test_supervisor_recovery_keeps_core_after_uninterruptible_worker(tmp_path,monkeypatch):
    """Parent kills a worker stuck in C sleep; not just a Python timeout mock."""
    import subprocess
    from astrograph.agents.mission import supervisor
    s,r=setup(tmp_path,task_timeout_s=.40)
    root=Path(s.output_root)/r.task_id;ws=Workspace(s.output_root,s.read_roots,r.task_id)
    ws.begin({'request_sha256':sha(r.model_dump(mode='json')),'settings_sha256':sha(s.model_dump(mode='json'))})
    core={'diagnosis':{'status':'MODEL_HYPOTHESIS','selected_fault_id':'A'},'fixture':True}
    p=ws.write_json('core-result.json',core);ws.write_json('core-result-receipt.json',{'sha256':file_sha(p)})
    real_popen=subprocess.Popen;spawned=[]
    def stuck(args,**kw):
        cmd=[sys.executable,'-c','import ctypes; ctypes.CDLL(None).sleep(20)'] if os.name=='posix' else [sys.executable,'-c','import time;time.sleep(20)']
        p=real_popen(cmd,**kw);spawned.append(p);return p
    monkeypatch.setattr(supervisor.subprocess,'Popen',stuck)
    start=time.monotonic();out=supervisor.run_supervised(s,r)
    assert time.monotonic()-start<1.5 and all(p.poll() is not None for p in spawned)
    assert out.status=='PARTIAL' and out.output['diagnosis_result']==core
    assert out.capability_status['supervisor']['reason']=='TASK_TOTAL_DEADLINE_EXCEEDED'


def test_supervisor_cancellation_preserves_diagnosis(tmp_path,monkeypatch):
    from astrograph.agents.mission import supervisor
    s,r=setup(tmp_path,task_timeout_s=10);ws=Workspace(s.output_root,s.read_roots,r.task_id)
    ws.begin({'request_sha256':sha(r.model_dump(mode='json')),'settings_sha256':sha(s.model_dump(mode='json'))})
    e=threading.Event();e.set()
    out=supervisor.run_supervised(s,r,cancel=e)
    assert out.status=='FAILED' and out.capability_status['supervisor']['reason']=='TASK_CANCELLED'


def test_resume_refuses_uncertain_side_effect(tmp_path):
    ws=Workspace(tmp_path,[tmp_path],'resume');identity={'id':1};ws.begin(identity)
    ws.event('TOOL_STARTED',step=1,tool_id='model.train',input_sha256=sha({}))
    with pytest.raises(RuntimeError,match='REQUIRES_REVIEW'):ws.begin(identity,resume_read_only=True)


def test_explicit_safe_read_prefix_resume_skips_completed_step(tmp_path,monkeypatch):
    s,r=setup(tmp_path,resume_read_only=True)
    original=_TaskRun._graph_reason
    def interrupt(self):raise KeyboardInterrupt('simulated process interruption before a read-only step completed')
    monkeypatch.setattr(_TaskRun,'_graph_reason',interrupt)
    with pytest.raises(KeyboardInterrupt):MissionAgent(s).run_inline(r)
    monkeypatch.setattr(_TaskRun,'_graph_reason',original)
    out=MissionAgent(s).run_inline(r)
    assert out.status=='SUCCEEDED' and out.output['diagnosis_result']
    events=[json.loads(x) for x in (Path(s.output_root)/r.task_id/'events.jsonl').read_text(encoding="utf-8").splitlines()]
    assert any(x['event']=='READ_ONLY_STEP_REUSED' and x['tool_id']=='model.infer' for x in events)


def test_ranking_debug_lineage_is_projected_not_hard_rejected():
    b=Scripted();p=Planner(b,mode='llm_assisted',work_mode='thinking')
    ranking={'candidates':[{'fault_id':'A','score':.72,'source_evidence_ids':['E1'],
                           'metadata':{'lineage':'x'*121000,'unit':'V','signed_value':-3}}]}
    p.prepare(AgentRequest(task_id='projection'),['rag.retrieve'],diagnostic_context={'data_ranking':ranking})
    assert p.calls==1 and b.payloads
    candidate=b.payloads[0][1]['diagnostic_context']['data_ranking']['candidates'][0]
    assert candidate['score']==.72 and candidate['source_evidence_ids']==['E1']
    assert candidate['metadata']['signed_value']==-3 and candidate['metadata']['lineage']['requires_lookup']
    assert p.context_builder.records[-1]['externalized_nonsemantic_fields']


def test_read_only_resume_checkpoint_tampering_rejected(tmp_path,monkeypatch):
    s,r=setup(tmp_path,resume_read_only=True)
    monkeypatch.setattr(_TaskRun,'_graph_reason',lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):MissionAgent(s).run_inline(r)
    path=Path(s.output_root)/r.task_id/'step-001.json';v=json.loads(path.read_text(encoding="utf-8"));v['result']['mode']='forged';path.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='integrity'):MissionAgent(s).run_inline(r)


def test_memory_sql_topk_matches_exhaustive_jaccard(tmp_path):
    from datetime import datetime,timezone
    from astrograph.memory.service import CaseMemoryService,MemoryPolicy
    import random
    store=CaseMemoryService(tmp_path/'case.db',MemoryPolicy(profile_sha256='0'*64,approved_reviewers=('reviewer',),max_cases=5))
    rng=random.Random(74);all_cases=[]
    with store._db() as db:
        for i in range(60):
            case={'case_id':f'c{i:03}','symptom_ids':rng.sample(['a','b','c','d','e','f'],rng.randrange(1,6)),
                  'operating_mode':'tracking' if i%3 else 'slew','knowledge_version':'v1' if i%4 else 'v2'}
            all_cases.append(case)
            db.execute('INSERT INTO confirmed VALUES(?,?,?,?,?)',(case['case_id'],i+1,json.dumps(case),'2020-01-01T00:00:00+00:00','g'+str(i)))
        db.execute('UPDATE head SET version=60')
    ctx=WorkflowContext(workflow_id='test',run_id='memory',permissions=[PermissionScope.READ_MEMORY])
    result=store.retrieve(('a','c','e'),operating_mode='tracking',snapshot=60,cutoff_at=datetime.now(timezone.utc),
        exclude_case_id='new',exclude_source_group_id='new',knowledge_version='v1',context=ctx)
    scored=[]
    for case in all_cases:
        q={'a','c','e'};items=set(case['symptom_ids']);score=len(q&items)/len(q|items)
        if score and case['operating_mode']=='tracking' and case['knowledge_version']=='v1':scored.append((case['case_id'],score))
    expected=sorted(scored,key=lambda x:(-x[1],x[0]))[:5]
    assert [(x['case']['case_id'],x['retrieval_score']) for x in result]==expected


def test_task_lease_prevents_concurrent_same_identity(tmp_path):
    from astrograph.agents.mission.run_lock import task_lease
    with task_lease(tmp_path,'a'):
        with pytest.raises(RuntimeError,match='ALREADY_RUNNING'):
            with task_lease(tmp_path,'a'):pass
        with task_lease(tmp_path,'b'):pass
    with task_lease(tmp_path,'a'):pass


def test_model_cache_pin_detects_midtask_replacement(tmp_path):
    import numpy as np
    model=tmp_path/'model';model.mkdir();weight=model/'weights.bin';weight.write_bytes(b'one')
    ws=Workspace(tmp_path,[tmp_path],'cache-pin');ws.root.mkdir()
    t=DocumentTools(ws,AgentSettings(read_roots=[str(tmp_path)],embedding_model_path=str(model)))
    t.documents=[{'document_id':'D','text':'sensor','source_sha256':'a'*64,'source_uri':'fixture:///d'}]
    class Encoder:
        def encode(self,texts,**kw):return np.ones((len(texts),2))
    t._dense=Encoder();t.retrieve('sensor');weight.write_bytes(b'different')
    with pytest.raises(ValueError,match='MODEL_CHANGED'):t.retrieve('sensor')


def test_retry_repair_note_also_counted_in_context():
    b=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='fixture',response_format_strategy='json_schema',
        repair_attempts=1,max_tokens=600,context_length=1024,tokenizer_mode='disabled'))
    def measure(**kw):
        repaired='repair_instruction' in kw['messages'][1]['content']
        return {'selected_prompt_tokens':600 if repaired else 100,'guarded_prompt_tokens':600 if repaired else 100}
    b._measure_prompt_tokens=measure;calls=[]
    def post(body):calls.append(body);return {'choices':[{'message':{'content':'invalid'},'finish_reason':'stop'}]}
    b._post=post
    with pytest.raises(OutputBudgetError):b.generate_structured(agent_id='test',output_model=InvestigationPlan,payload={})
    assert len(calls)==1


def test_real_http_transport_and_supervised_planning_fixture(tmp_path,http_server):
    class ContractServer(Quiet):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            user=json.loads(body['messages'][1]['content']);payload=user['payload']
            model=body['response_format']['json_schema']['name']
            if model=='InvestigationPlan':value={'task':payload['fixed_task'],'actions':[],'rationale':'fixture'}
            elif model=='CheckChoice':value={'tool_id':payload['eligible_checks'][0]['tool_id'],'rationale':'fixture'}
            elif model=='AgentNarrative':value={'candidate_fault_id':payload['required_candidate_fault_id'],'excerpt_ids':[payload['sources_untrusted'][0]['excerpt_id']]}
            elif model=='ExplorationProposal':value={'hypotheses':[]}
            else:raise AssertionError(model)
            response=json.dumps({'choices':[{'message':{'content':json.dumps(value)},'finish_reason':'stop'}]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)
    config=OpenAICompatibleConfig(model='SCRIPTED_HTTP_FIXTURE_NOT_LLM',base_url=http_server(ContractServer),
        response_format_strategy='json_schema',repair_attempts=0,timeout_s=3,tokenizer_mode='disabled',context_length=16384)
    path=tmp_path/'llm.json';data=config.model_dump(mode='json');data['api_key']='fixture';path.write_text(json.dumps(data))
    s,r=setup(tmp_path,planner='llm_assisted',llm_config=str(path),max_http_attempts=8)
    out=MissionAgent(s).run_supervised(r.model_copy(update={'work_mode':'thinking'}))
    assert out.status=='SUCCEEDED',out.model_dump()
    assert out.capability_status['llm']['status']=='SUCCEEDED'
    assert out.execution_budget['http_attempts']>0
    assert out.execution_budget['enforcement']=='PROCESS_SUPERVISOR'


def test_requested_mode_does_not_mutate_policy_or_diagnosis(tmp_path):
    s,r=setup(tmp_path)
    normal=MissionAgent(s).run(r)
    creative=MissionAgent(s).run(r.model_copy(update={'task_id':'creative','work_mode':'creative'}))
    assert normal.output['diagnosis_result']['diagnosis']==creative.output['diagnosis_result']['diagnosis']
    assert creative.output['exploration']['memory_write']=='NONE'


def test_partial_ontology_never_claims_complete_coverage(tmp_path):
    ws=Workspace(tmp_path,[tmp_path],'partial-build');ws.root.mkdir()
    class Partial:
        def __init__(self):self.n=0
        def generate_structured(self,**kw):
            self.n+=1
            if self.n>1:raise TimeoutError('fixture second batch fails')
            doc=kw['payload']['documents_untrusted'][0]
            return OntologyProposal(triples=[{'subject':'gyro','predicate':'affects','object':'residual',
                'source_document_id':doc['document_id'],'source_quote':'Gyro affects residual.'}])
    t=DocumentTools(ws,AgentSettings(ontology_batch_chars=512),planner=Planner(Partial(),mode='llm_assisted'))
    t.documents=[{'document_id':'D','text':'Gyro affects residual.\n'+'z'*900,'source_sha256':'a'*64}]
    out=t.build_graph()
    assert out['status']=='PARTIAL_REVIEW_REQUIRED' and not out['extraction_complete']
    assert out['edges'] and not out['active_kg_modified']

@pytest.mark.parametrize('task',['research','build_knowledge','train_model','infer_candidate'])
def test_non_diagnostic_planning_accepts_none_diagnostic_context(task):
    b=Scripted();p=Planner(b,mode='llm_assisted')
    plan=p.prepare(AgentRequest(task_id='nondiagnostic',task=task),[],diagnostic_context=None)
    assert plan.task==task and p.calls==1 and p.records[-1]['outcome']=='ACCEPTED'


def test_unvalidated_task_id_cannot_escape_supervisor_lease(tmp_path):
    s,r=setup(tmp_path)
    bad=r.model_copy(update={'task_id':'../escape'})
    with pytest.raises(ValueError):MissionAgent(s).run_supervised(bad)
    assert not (Path(s.output_root)/'escape.lock').exists()
