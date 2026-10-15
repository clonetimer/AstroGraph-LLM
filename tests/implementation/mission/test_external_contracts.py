"""Local transport/CLI fixtures validate adapters, not live-service model quality."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import pytest
from astrograph.agents.mission.models import AgentSettings,AgentRequest,InvestigationPlan
from astrograph.agents.mission.planner import Planner
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.documents import DocumentTools
from astrograph.agents.mission.storage import Workspace
from astrograph.llm import OpenAICompatibleConfig,OpenAICompatibleStructuredBackend
from astrograph.integrations.platform_knowledge import Anything2OntologyCLIAdapter

ROOT=Path(__file__).resolve().parents[3]

@contextmanager
def server(response):
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path,body))
            raw=json.dumps(response(body)).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):pass
    s=ThreadingHTTPServer(('127.0.0.1',0),Handler);t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
    try:yield f'http://127.0.0.1:{s.server_port}',seen
    finally:s.shutdown();s.server_close();t.join(timeout=2)


def test_existing_llm_backend_accepts_new_planning_schema_over_http():
    answer={'task':'research','actions':[],'rationale':'Use source-bound retrieval.'}
    def response(body):return {'choices':[{'message':{'content':json.dumps(answer)},'finish_reason':'stop'}]}
    with server(response) as (url,seen):
        backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(base_url=url+'/v1',model='LOCAL_TRANSPORT_FIXTURE',
            response_format_strategy='json_schema',repair_attempts=0,tokenizer_mode='disabled',max_tokens=1500,timeout_s=3))
        p=Planner(backend,mode='llm_assisted');plan=p.prepare(AgentRequest(task_id='x',task='research'),[])
        assert plan.rationale==answer['rationale']
        assert p.records[0]['outcome']=='ACCEPTED'
        assert seen[0][0]=='/v1/chat/completions'
        assert seen[0][1]['response_format']['type']=='json_schema'


def test_truncated_llm_response_falls_back_without_calling_arbitrary_tools():
    def response(body):return {'choices':[{'message':{'content':'{"task":"research"'},'finish_reason':'length'}]}
    with server(response) as (url,seen):
        backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(base_url=url,model='LOCAL_TRANSPORT_FIXTURE',
            response_format_strategy='json_schema',repair_attempts=0,tokenizer_mode='disabled',max_tokens=1500,timeout_s=3))
        p=Planner(backend,mode='llm_assisted');plan=p.prepare(AgentRequest(task_id='x',task='research'),[])
        assert not plan.actions and p.records[0]['outcome']=='DETERMINISTIC_FALLBACK'


def test_paperseek_real_adapter_local_http_contract(tmp_path):
    def response(body):return {'ranked':[{'title':'Fixture mechanism note','abstract':'Gyro bias requires independent residual checks.',
        'url':'https://example.org/fixture-mechanism'}]}
    with server(response) as (url,seen):
        s=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/'runs'),
            paperseek_url=url,allow_external_literature=True)
        result=MissionAgent(s).run(AgentRequest(task_id='research',task='research',goal='gyro bias mechanism'))
        assert result.status=='SUCCEEDED'
        assert seen[0][0]=='/api/search'
        assert result.output['sources'][0]['source_uri']=='https://example.org/fixture-mechanism'
        assert not result.output['sources'][0]['diagnostic_vote_allowed']


def test_bad_paperseek_response_reports_partial_no_fake_sources(tmp_path):
    with server(lambda body:{'invented':'not a ranked response'}) as (url,seen):
        s=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/'runs'),
            paperseek_url=url,allow_external_literature=True)
        result=MissionAgent(s).run(AgentRequest(task_id='research',task='research',goal='gyro bias'))
        assert result.status=='PARTIAL' and not result.output['sources']
        assert any('literature.search' in x for x in result.limitations)


def test_mineru_cli_contract_retains_provenance_and_review_status(tmp_path):
    # This executable is explicitly a CLI contract fixture, not the MinerU model.
    exe=tmp_path/'mineru-fixture';exe.write_text('#!'+sys.executable+'\nimport argparse,pathlib\np=argparse.ArgumentParser();p.add_argument("-p");p.add_argument("-o");a=p.parse_args()\npathlib.Path(a.o,"fixture.md").write_text("# Figure\\nGenerated fixture description.")\n')
    exe.chmod(0o755)
    source=tmp_path/'source.pdf';source.write_bytes(b'CONTRACT_FIXTURE_NOT_PDF_CONTENT')
    ws=Workspace(tmp_path/'runs',[tmp_path],'mineru');ws.root.mkdir(parents=True)
    settings=AgentSettings(allow_mineru=True,mineru_executable=str(exe))
    result=DocumentTools(ws,settings).convert([str(source)])
    document=result['documents'][0]
    assert document['parser']=='MINERU_CLI' and document['review_status']=='UNREVIEWED'
    assert document['source_sha256'] and not document['diagnostic_vote_allowed']
    assert list(ws.root.rglob('*.md'))


def test_a2o_existing_four_stage_adapter_in_fresh_workspace(tmp_path):
    # The supplied adapter is real; the four external CLI modules are import/IO fixtures.
    source=tmp_path/'a2o-source'
    for command,module,args in Anything2OntologyCLIAdapter.stages:
        package=source/'src'/module.split('.')[0];package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        code='import os,pathlib\n'
        if command=='skus2ontology':
            code+='p=pathlib.Path(os.environ["ONTOLOGY_DIR"]);p.mkdir(parents=True,exist_ok=True);(p/"ontology.json").write_text(\'{"fixture": "review-only"}\')\n'
        (package/'cli.py').write_text(code)
    ws=Workspace(tmp_path/'runs',[tmp_path],'a2o');ws.root.mkdir(parents=True)
    text=tmp_path/'text.md';text.write_text('Gyro bias affects attitude residual.')
    tools=DocumentTools(ws,AgentSettings());tools.clean(tools.convert([str(text)])['documents'])
    adapter=Anything2OntologyCLIAdapter(source_root=source,timeout_s=5)
    out=tools.build_graph(a2o=adapter)
    assert out['status']=='REVIEW_REQUIRED' and len(out['commands'])==4
    assert not out['active_kg_modified'] and out['input_source_bindings']
