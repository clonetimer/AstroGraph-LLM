"""dev9: grounding, source visibility, canonical projection and exact wire audit.
All generated responses below are explicit fixtures, not live LLM quality tests.
"""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest
from astrograph.agents.mission.context import ContextBuilder, ContextBudgetExceeded
from astrograph.agents.mission.sources import diagnostic_sources, normalize_sources, visible_ids
from astrograph.agents.mission.models import AgentNarrative, ExplorationProposal, InvestigationPlan, AgentRequest
from astrograph.agents.mission.planner import Planner
from astrograph.agents.mission.storage import Workspace, canonical, sha
from astrograph.llm import OpenAICompatibleConfig,OpenAICompatibleStructuredBackend
from astrograph.llm.task_prompts import task_output_schema
from astrograph.llm.trace_verification import verify_task


def core():
    return {'diagnosis':{'status':'MODEL_ONLY','selected_fault_id':'A','candidate_fault_ids':['A'],
        'detected':True,'knowledge_verified':False,'root_cause_confirmed':False,
        'requires_discriminating_evidence':True,'candidate_explanations':[
            {'fault_id':'A','selected':True,'knowledge_state':'UNKNOWN',
             'contradicting_evidence_ids':['NEGATIVE-1'],'missing_required_definition_ids':['missing-1']} ]},
        'explanation':{'conclusion':'Not independently confirmed.','candidate_explanations':['x'*20000]},
        'planning_trace':[{'tool_id':'check-A','outcome':'NO_CHANGE'}],
        'evidence_revisions':[{'before':'v1','after':'v2'}],'stopping_reason':'BUDGET'}


def store(tmp_path):
    ws=Workspace(tmp_path,[tmp_path],'ground');ws.root.mkdir()
    def writer(digest,payload):
        path=ws.root/('context-'+digest+'.json')
        if not path.exists():ws.write_json(path.name,payload)
        return {'name':path.name,'sha256':digest}
    return ws,writer


def test_mapping_source_is_json_not_python_repr():
    value={'valid':False,'unit':'rad/s','value':0,'absent':None}
    output=normalize_sources([{'citation_id':'s','text':value}])[0]
    assert json.loads(output['text'])==value
    assert output['text_format']=='CANONICAL_JSON'
    assert 'False' not in output['text'] and 'None' not in output['text']


@pytest.mark.parametrize('text',[None,17,True,object()])
def test_nontext_sources_rejected(text):
    with pytest.raises(ValueError):normalize_sources([{'citation_id':'s','text':text}])


def test_same_source_id_conflicting_text_rejected():
    with pytest.raises(ValueError,match='COLLISION'):
        normalize_sources([{'citation_id':'same','text':'a'},{'citation_id':'same','text':'b'}])


def test_anchor_is_short_but_core_and_negative_facts_untouched():
    original=core();before=canonical(original)
    sources=diagnostic_sources(original,exploration=True)
    anchor=sources[0]
    assert anchor['required_for_grounding'] and len(anchor['text'])<1600
    assert json.loads(anchor['text'])['root_cause_confirmed'] is False
    assert all(s['core_sha256']==sha(original) for s in sources)
    assert any('NEGATIVE-1' in s['text'] for s in sources)
    assert any('NO_CHANGE' in s['text'] for s in sources)
    assert canonical(original)==before


def test_required_anchor_reserved_before_large_optional_source(tmp_path):
    ws,writer=store(tmp_path);b=ContextBuilder(artifact_writer=writer)
    sources=[{'citation_id':'huge','text':'word '*10000}]+diagnostic_sources(core())
    view=b.build(AgentNarrative,{'fixed_diagnosis':core()['diagnosis'],'sources_untrusted':sources},
        budget_measure=lambda p:{'fits':len(canonical(p))<4500})
    required=sources[1]['citation_id']
    assert required in view['allowed_citation_ids'] and 'huge' not in view['allowed_citation_ids']
    assert b.records[-1]['visible_citation_ids']==visible_ids(view)
    assert (ws.root/b.records[-1]['projected_artifact']['name']).read_bytes()==canonical(view)


def test_required_anchor_cannot_be_silently_omitted():
    with pytest.raises(ContextBudgetExceeded):
        ContextBuilder().build(AgentNarrative,{'sources_untrusted':[{'citation_id':'need',
            'text':'x'*10000,'required_for_grounding':True}]},budget_measure=lambda p:{'fits':len(canonical(p))<500})


class Response:
    def __init__(self,fn):self.fn=fn;self.calls=0;self.payload=None
    def generate_structured(self,**kw):
        self.calls+=1;self.payload=kw['payload'];return self.fn(kw['payload'])


def test_excerpt_selection_renders_source_not_model_prose():
    b=Response(lambda p:{'candidate_fault_id':'A','excerpt_ids':[p['sources_untrusted'][0]['excerpt_id']]})
    p=Planner(b,mode='llm_assisted')
    out=p.explain({'selected_fault_id':'A'},[{'citation_id':'s','text':'Not confirmed; voltage was NOT measured.'}])
    assert out['content']['paragraphs']==[{'text':'Not confirmed; voltage was NOT measured.','citation_ids':['s']}]
    assert out['assertion_policy']=='SOURCE_EXTRACTIVE_ONLY'
    assert not out['semantic_truth_independently_verified']


@pytest.mark.parametrize('attack',['GQF-valid-but-not-citable','omitted','made-up'])
def test_unknown_or_omitted_source_rejected(attack):
    b=Response(lambda p:{'candidate_fault_id':'A','paragraphs':[{'text':'fake','citation_ids':[attack]}]})
    cb=ContextBuilder();p=Planner(b,mode='llm_assisted',context_builder=cb)
    out=p.explain({'selected_fault_id':'A','query_plan_id':'GQF-valid-but-not-citable'},[
        {'citation_id':'ok','text':'Not confirmed.'}, {'citation_id':'omitted','text':'z'*100000}])
    assert out['status']=='DETERMINISTIC_EXPLANATION_ONLY'
    assert 'ACTUALLY_VISIBLE_SOURCES' in p.records[-1]['reason']
    assert 'omitted' not in b.payload['allowed_citation_ids']


@pytest.mark.parametrize('attack',['strip_negation','system_prose','extra_citation'])
def test_citation_text_not_laundered(attack):
    text='Root cause is not confirmed.'
    outtext={'strip_negation':'confirmed.','system_prose':'The fusion kernel, not your prose, decides diagnoses.',
             'extra_citation':text}[attack]
    refs=['a','b'] if attack=='extra_citation' else ['a']
    b=Response(lambda p:{'candidate_fault_id':'A','paragraphs':[{'text':outtext,'citation_ids':refs}]})
    p=Planner(b,mode='llm_assisted')
    out=p.explain({'selected_fault_id':'A'},[{'citation_id':'a','text':text},{'citation_id':'b','text':'Unrelated data.'}])
    assert out['status']=='DETERMINISTIC_EXPLANATION_ONLY'
    assert 'NARRATIVE_NOT_EXACT' in p.records[-1]['reason']


@pytest.mark.parametrize('mode',['explain','explore'])
def test_empty_visible_sources_never_generates(mode):
    b=Response(lambda p: (_ for _ in ()).throw(AssertionError('MUST_NOT_GENERATE')))
    p=Planner(b,mode='llm_assisted',work_mode='creative')
    sources=[{'citation_id':'too-large','text':'a'*100000}]
    if mode=='explain':p.explain({'selected_fault_id':None},sources)
    else:p.explore(diagnosis={},sources=sources)
    assert b.calls==p.calls==0
    assert 'NO_VISIBLE_CITABLE_SOURCE' in p.records[-1]['reason']


@pytest.mark.parametrize('mode',['explain','explore'])
def test_explicit_abstention_not_success(mode):
    b=Response(lambda p:{'candidate_fault_id':None,'paragraphs':[]} if mode=='explain' else {'hypotheses':[]})
    p=Planner(b,mode='llm_assisted',work_mode='creative')
    sources=[{'citation_id':'s','text':'No evidence.'}]
    out=p.explain({'selected_fault_id':None},sources) if mode=='explain' else p.explore(diagnosis={},sources=sources)
    assert out['status']=='LLM_ABSTAINED'
    assert p.records[-1]['outcome']=='ABSTAINED'


def test_schema_enums_use_visible_namespace_only():
    payload={'sources_untrusted':[{'citation_id':'s','excerpt_id':'Q1','text':'Not confirmed.'}],
             'required_candidate_fault_id':'A','fixed_diagnosis':{'query_plan_id':'GQF-not-a-source'}}
    s=task_output_schema(AgentNarrative,payload)
    assert s['properties']['excerpt_ids']['items']['enum']==['Q1']
    assert s['properties']['paragraphs']['maxItems']==0
    assert s['properties']['candidate_fault_id']['enum']==['A']
    t=task_output_schema(ExplorationProposal,payload)
    assert t['$defs']['ExplorationHypothesis']['properties']['citation_ids']['items']['enum']==['s']


def test_key_order_cannot_change_projection_hash_or_messages(tmp_path):
    ws,w=store(tmp_path)
    original={'z':1,'diagnosis':core()['diagnosis'],'sources_untrusted':diagnostic_sources(core())}
    def reverse(v):
        if isinstance(v,dict):return {k:reverse(x) for k,x in reversed(list(v.items()))}
        if isinstance(v,list):return [reverse(x) for x in v]
        return v
    b1=ContextBuilder(artifact_writer=w);b2=ContextBuilder(artifact_writer=w)
    a=b1.build(AgentNarrative,original);b=b2.build(AgentNarrative,reverse(original))
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='fixture',tokenizer_mode='disabled'))
    assert canonical(a)==canonical(b)
    assert b1.records[-1]['projected_sha256']==b2.records[-1]['projected_sha256']
    assert backend._messages(output_model=AgentNarrative,payload=a)==backend._messages(output_model=AgentNarrative,payload=b)


@pytest.fixture
def endpoint():
    servers=[]
    def make(content,code=200):
        capture=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                capture.append(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(code);self.send_header('Content-Length',str(len(content)));self.end_headers()
                self.wfile.write(content)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start();servers.append((server,t))
        return f'http://127.0.0.1:{server.server_port}/v1',capture
    yield make
    for s,t in servers:s.shutdown();s.server_close();t.join()


def make_http(tmp_path,endpoint,body,*,code=200,trace_mode='full'):
    url,capture=endpoint(body,code)
    b=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='LOOPBACK_FIXTURE_NOT_LLM',base_url=url,
        api_key='PRIVATE_HEADER_ONLY',context_length=16384,max_tokens=1000,tokenizer_mode='disabled',
        response_format_strategy='json_schema',repair_attempts=0,timeout_s=5))
    ws,w=store(tmp_path)
    p=Planner(b,mode='llm_assisted',context_builder=ContextBuilder(artifact_writer=w),
              audit_writer=ws.write_bytes,trace_mode=trace_mode)
    return p,ws,capture


def test_exact_wire_prompt_and_response_persist_before_validation(tmp_path,endpoint):
    raw=b'{  "choices" : [{"message":{"content":"{\\"task\\":\\"research\\",\\"rationale\\":\\"fixture\\",\\"actions\\":[]}"},"finish_reason":"stop"}], "usage":{"prompt_tokens":17} }\n'
    p,ws,capture=make_http(tmp_path,endpoint,raw)
    p.prepare(AgentRequest(task_id='x',task='research'),[])
    assert p.records[-1]['outcome']=='ACCEPTED' and len(capture)==1
    assert next(ws.root.glob('*.request.json')).read_bytes()==capture[0]
    assert next(ws.root.glob('*.response.bin')).read_bytes()==raw
    assert all(b'PRIVATE_HEADER_ONLY' not in x.read_bytes() for x in ws.root.iterdir())
    report=verify_task(ws.root)
    assert report['status']=='VERIFIED' and report['rows'][0]['transport_observed']
    assert report['generation_requests_sent']==0


@pytest.mark.parametrize('raw,code',[(b'not-json',200),(b'{"reason":"server unavailable"}',503),
        (b'{"choices":[{"message":{"content":"{bad}"},"finish_reason":"stop"}]}',200)])
def test_invalid_response_retained_as_data_not_accepted(tmp_path,endpoint,raw,code):
    p,ws,capture=make_http(tmp_path,endpoint,raw,code=code)
    p.prepare(AgentRequest(task_id='x'),[])
    assert p.records[-1]['outcome']=='DETERMINISTIC_FALLBACK'
    assert next(ws.root.glob('*.response.bin')).read_bytes()==raw
    assert verify_task(ws.root)['status']=='VERIFIED'


def test_hashes_only_mode_does_not_persist_body(tmp_path,endpoint):
    p,ws,_=make_http(tmp_path,endpoint,b'bad',trace_mode='hashes')
    p.prepare(AgentRequest(task_id='x',goal='SENSITIVE_TASK_MARKER'),[])
    assert not list(ws.root.glob('*.request.json')) and not list(ws.root.glob('*.response.bin'))
    # Full context is an existing artifact; hashes-only affects new wire trace only.
    for x in ws.root.glob('llm-call-*'):assert b'SENSITIVE_TASK_MARKER' not in x.read_bytes()
    assert verify_task(ws.root)['rows'][0]['request']['status']=='HASHES_ONLY_NO_BYTE_REPLAY'


def test_trace_tampering_rejected(tmp_path,endpoint):
    p,ws,_=make_http(tmp_path,endpoint,b'bad')
    p.prepare(AgentRequest(task_id='x'),[])
    next(ws.root.glob('*.request.json')).write_bytes(b'{}')
    with pytest.raises(ValueError,match='HASH_MISMATCH'):verify_task(ws.root)


@pytest.mark.parametrize('kind',['narrative','exploration'])
def test_real_adcs_report_has_visible_anchor_within_same_16k_budget(tmp_path,kind):
    # Captured already-used ADCS report, NOT an independent accuracy benchmark.
    raw=json.loads((Path(__file__).resolve().parents[2]/'fixtures/mission/grounding_adcs_core.json').read_text(encoding='utf-8'))
    from astrograph.agents.mission.modes import MODE_OBJECTIVES
    from astrograph.agents.mission.planner import INSTRUCTION
    ws,w=store(tmp_path)
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='OFFLINE_NOT_LLM',
        tokenizer_mode='disabled',context_length=16384,max_tokens=4096,response_format_strategy='json_schema'))
    model=AgentNarrative if kind=='narrative' else ExplorationProposal
    mode='thinking' if kind=='narrative' else 'creative'
    value={'fixed_diagnosis':raw['diagnosis'],'sources_untrusted':diagnostic_sources(raw,exploration=kind=='exploration')}
    if kind=='narrative':value['required_candidate_fault_id']=raw['diagnosis']['selected_fault_id']
    else:value.update(max_proposals=3,required_output='Separate unverified hypotheses with falsification tests and required observations. Do not assert they are confirmed.')
    builder=ContextBuilder(max_tokens=16384,output_reserve=4096,artifact_writer=w)
    view=builder.build(model,value,budget_measure=lambda x:backend.measure_structured_context(output_model=model,payload={
        'instruction':INSTRUCTION,'mode_objective':MODE_OBJECTIVES[mode],'work_mode':mode,**x}))
    assert view['sources_untrusted']
    assert value['sources_untrusted'][0]['citation_id'] in view['allowed_citation_ids']
    assert builder.records[-1]['budget_measurement']['fits']
    assert json.loads(view['sources_untrusted'][0]['text'])['root_cause_confirmed'] is False


def test_logical_invalid_citation_has_prompt_response_and_validation_trace(tmp_path,endpoint):
    raw=json.dumps({'choices':[{'message':{'content':json.dumps({'candidate_fault_id':'A',
        'paragraphs':[{'text':'Not confirmed.','citation_ids':['GQF-not-source']}]})},'finish_reason':'stop'}]}).encode()
    p,ws,_=make_http(tmp_path,endpoint,raw)
    out=p.explain({'selected_fault_id':'A','query_plan_id':'GQF-not-source'},[{'citation_id':'s','text':'Not confirmed.'}])
    assert out['status']=='DETERMINISTIC_EXPLANATION_ONLY'
    assert next(ws.root.glob('*.response.bin')).read_bytes()==raw
    record=json.loads(next(ws.root.glob('planner-record-*.json')).read_text(encoding="utf-8"))
    assert record['trace_prefix'] and 'ACTUALLY_VISIBLE_SOURCES' in record['reason']
    assert verify_task(ws.root)['rows'][0]['visible_citation_ids']==['s']


def test_unknown_excerpt_never_substitutes_nearest_valid_one():
    b=Response(lambda p:{'candidate_fault_id':'A','excerpt_ids':['Q999']})
    p=Planner(b,mode='llm_assisted')
    assert p.explain({'selected_fault_id':'A'},[{'citation_id':'s','text':'Real.'}])['status']=='DETERMINISTIC_EXPLANATION_ONLY'
    assert 'UNKNOWN_VISIBLE_EXCERPT' in p.records[-1]['reason']


def test_failed_audit_commit_does_not_leak_task_context():
    from astrograph.llm.audit import audit_scope,current_audit
    class Broken:
        def index(self):raise OSError('fixture disk full')
    assert current_audit() is None
    with pytest.raises(OSError):
        with audit_scope(Broken()):assert current_audit() is not None
    assert current_audit() is None
