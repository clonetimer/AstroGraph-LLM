import json
from pathlib import Path
from fastapi.testclient import TestClient
from astrograph.api import create_app
from astrograph.integrations.reviewed_context import ContextStore,ContextError
from astrograph.integrations.upload_report import UploadReportJobStore
class Backend:
    model_id='platform-model'
    def __init__(self):self.payloads=[]
    def generate_structured(self,*,agent_id,output_model,payload):
        self.payloads.append(payload);sid=payload['sources'][0]['source_id']
        return output_model.model_validate({'summary':'summary','findings':[{'text':'finding','citation_ids':[sid]}], 'hypotheses':[],'information_gaps':[],'recommended_review_actions':[]})
from astrograph.integrations.platform_knowledge import PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService
class Service(PlatformKnowledgeEnrichmentService):
    def __init__(self):
        super().__init__(PlatformKnowledgeConfig(), llm_backend=Backend())
def prepare(store):
    c=ContextStore(store.root/'reviewed_context');w=c.create_workspace(name='A',device='A',event='A')
    e=c.create_entry(w['workspace_id'],kind='user_note',title='note',body='check units',actor='op')
    c.review_entry(w['workspace_id'],e['entry_id'],expected_revision=1,reviewed=True,actor='reviewer')
    return {'workspace_id':w['workspace_id'],'entries':[{'entry_id':e['entry_id'],'revision':2}]}
def test_platform_context_reaches_model_and_report(tmp_path):
    service=Service();store=UploadReportJobStore(tmp_path/'jobs',knowledge_service=service)
    j=store.create("说明外部诊断");jid=j['job_id'];store.put_file(jid,'diagnosis.txt',b'Wheel speed anomaly');store.put_file(jid,'analysis_input.json',json.dumps({'schema_version':'astrograph-result-log-review.v1','analysis_mode':'EXTERNAL_DIAGNOSIS_ANALYSIS','diagnosis_results':['diagnosis.txt']}).encode())
    store.bind_reviewed_context(jid,prepare(store))
    r=store.run(jid)
    assert r['analysis']['knowledge_enrichment']['generation']['status']=='SUCCEEDED'
    assert service.llm_backend.payloads[0]['reviewed_context']==r['reviewed_context']
    try:store.bind_reviewed_context(jid,prepare(store))
    except (ContextError,ValueError):pass
    else:raise AssertionError('must freeze task context')
def test_context_routes_permissions_and_actor_not_client_controlled(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','bearer')
    monkeypatch.setenv('ASTROGRAPH_API_TOKENS_JSON',json.dumps({'operator-token':['operator'],'viewer-token':['viewer'],'reviewer-token':['reviewer']}))
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'jobs'))
    app=create_app()
    with TestClient(app) as client:
        def headers(token):return {'Authorization':'Bearer '+token}
        url='/v1/platform/workspaces'
        assert client.post(url,json={'name':'A'},headers=headers('viewer-token')).status_code==403
        r=client.post(url,json={'name':'A'},headers=headers('operator-token'));assert r.status_code==200,r.text
        wid=r.json()['workspace_id'];ep=url+'/'+wid+'/entries'
        r=client.post(ep,json={'kind':'user_note','title':'t','body':'b'},headers=headers('operator-token'));assert r.status_code==200,r.text
        eid=r.json()['entry_id'];review=ep+'/'+eid+'/review'
        assert client.post(review,json={'expected_revision':1,'reviewed':True},headers=headers('operator-token')).status_code==403
        assert client.post(review+'/',json={'expected_revision':1,'reviewed':True},headers=headers('operator-token')).status_code==403
        assert client.post(review,json={'expected_revision':1,'reviewed':True},headers=headers('reviewer-token')).status_code==200
        assert client.post(ep,json={'kind':'user_note','title':'t','body':'b','actor':'spoof'},headers=headers('operator-token')).status_code==409

def test_context_assets_resolve_and_legacy_flow_rejects_selected_context(tmp_path):
    import re
    with TestClient(create_app()) as client:
        for page in ('/platform/mission', '/platform/legacy'):
            html = client.get(page).text
            links = re.findall(r'(?:src|href)="([^"]*workspace_view\.(?:js|css))"', html)
            assert len(links) == 2
            for url in links:
                assert client.get(url).status_code == 200, url
    store=UploadReportJobStore(tmp_path/'jobs',knowledge_service=Service())
    j=store.create('legacy context must not be silently ignored');jid=j['job_id']
    store.put_file(jid,'diagnosis.txt',b'Wheel speed anomaly')
    store.put_file(jid,'analysis_input.json',json.dumps({'schema_version':'astrograph-result-log-review.v1',
        'analysis_mode':'MATERIAL_REVIEW','diagnosis_results':['diagnosis.txt']}).encode())
    store.bind_reviewed_context(jid,prepare(store))
    check=store.preflight(jid)
    assert not check['can_start']
    assert any(x['code']=='CONTEXT_REQUIRES_EXTERNAL_ANALYSIS' for x in check['blockers'])

def test_context_binding_refuses_active_worker_lock_and_restores_own_lock(tmp_path):
    import pytest
    store=UploadReportJobStore(tmp_path/'jobs',knowledge_service=Service())
    jid=store.create('race boundary')['job_id'];root=store._job_root(jid)
    (root/'.run.lock').write_text('worker')
    with pytest.raises(ContextError):
        store.bind_reviewed_context(jid,prepare(store))
    assert not (root/'context_snapshot.json').exists()
    assert (root/'.run.lock').read_text()=='worker'
    assert not (root/'.launch.lock').exists()
