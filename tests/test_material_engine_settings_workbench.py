"""Knowledge-engine settings integration, auth, versioning and runtime handoff."""
from __future__ import annotations
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from astrograph.api import create_app
from astrograph.integrations.material_engine_settings import (
    MaterialEngineRegistry, MaterialEngineSettings, load_active_engine_environment,
)


@pytest.fixture(autouse=True)
def isolate_process_env():
    original = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(tmp_path/'platform'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','local')
    return TestClient(create_app(),base_url='http://localhost')


def payload(**kw):
    return {'settings':{**{'rag_documents':9,'rag_sections':17,'rag_hits':6,
                            'ocr_engine':'tesseract','ocr_languages':'chi_sim+eng',
                            'community_llm_enabled':True},**kw},'expected_revision':None}


def test_save_applies_runtime_and_restores_from_disk(client, tmp_path, monkeypatch):
    url='/v1/platform/settings/knowledge-engine'
    initial=client.get(url)
    assert initial.status_code==200 and initial.json()['revision'] is None
    first=client.put(url,json=payload())
    assert first.status_code==200,first.text
    revision=first.json()['revision']
    assert revision.startswith('revision-')
    assert os.environ['ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET']=='9'
    assert os.environ['ASTROGRAPH_MATERIAL_COMMUNITY_LLM_ENABLED']=='1'
    assert os.environ['ASTROGRAPH_MATERIAL_OCR_ENGINE']=='tesseract'
    assert 'qdrant_api_key' not in first.json()['settings']
    assert 'qdrant_secret_ref' not in first.json()
    assert client.get(url+'/revisions').json()['revisions'][0]['revision']==revision
    os.environ.pop('ASTROGRAPH_MATERIAL_OCR_ENGINE')
    os.environ.pop('ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET')
    load_active_engine_environment(tmp_path/'platform')
    assert os.environ['ASTROGRAPH_MATERIAL_OCR_ENGINE']=='tesseract'
    assert os.environ['ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET']=='9'
    # New FastAPI process reads the previously saved settings.
    other=TestClient(create_app(),base_url='http://localhost')
    assert other.get(url).json()['revision']==revision
    # The default entry now redirects to the task-first Mission console;
    # the original settings UI remains accessible only under /platform/legacy.
    assert '/platform/mission' in client.get('/platform').text
    assert '模型与引擎' in client.get('/platform/mission').text
    assert '检索与多模态配置' in client.get('/platform/legacy').text
    assert '大模型配置' in client.get('/platform/legacy').text
    assert client.get('/platform/assets/v0.21.1/platform_engine_settings.js').status_code==200


def test_revision_conflict_rollback_and_history(client):
    url='/v1/platform/settings/knowledge-engine'
    first=client.put(url,json=payload()).json()['revision']
    data=payload(community_llm_enabled=False,ocr_engine='disabled')
    data['expected_revision']=first
    changed=client.put(url,json=data)
    assert changed.status_code==200,changed.text
    second=changed.json()['revision']
    assert second!=first
    assert 'ASTROGRAPH_MATERIAL_OCR_ENGINE' not in os.environ
    stale=client.put(url,json=data)
    assert stale.status_code==409
    res=client.post(url+'/rollback',json={'expected_revision':second,'target_revision':first})
    assert res.status_code==200,res.text
    third=res.json()['revision']
    assert third not in (first,second)
    assert res.json()['restored_from']==first
    assert res.json()['settings']['ocr_engine']=='tesseract'
    assert os.environ['ASTROGRAPH_MATERIAL_COMMUNITY_LLM_ENABLED']=='1'
    assert client.post(url+'/rollback',json={'expected_revision':second,'target_revision':first}).status_code==409
    assert client.post(url+'/rollback',json={'expected_revision':third,'target_revision':'../../secret'}).status_code==422


def test_validation_and_no_unapproved_egress(client, tmp_path):
    url='/v1/platform/settings/knowledge-engine'
    assert client.put(url,json=payload(vector_backend='qdrant',embedding_model=str(tmp_path),
            qdrant_url='http://danger.example:6333')).status_code==422
    assert client.put(url,json=payload(vector_backend='qdrant',embedding_model=str(tmp_path),
            qdrant_url='http://localhost:6333@evil.local:6333')).status_code==422
    assert client.put(url,json=payload(embedding_model='relative/path')).status_code==422
    assert client.put(url,json=payload(vision_enabled=True,vision_model='model',vision_pages=1,
            vision_endpoint='https://remote.example/v1/chat/completions')).status_code==422
    assert client.put(url,json=payload(ocr_languages='eng;curl')).status_code==422
    # Inactive connection settings are still persisted and echoed to admins.
    # Reject unsafe dormant endpoints and keys which would be orphaned in the
    # keyring after saving a SQLite revision.
    assert client.put(url,json=payload(vector_backend='sqlite',
        qdrant_url='http://attacker.example:6333')).status_code==422
    assert client.put(url,json=payload(vector_backend='sqlite',
        qdrant_url='http://username:password@127.0.0.1:6333')).status_code==422
    assert client.put(url,json=payload(vector_backend='sqlite',
        qdrant_api_key='unused-key')).status_code==422
    assert client.put(url,json=payload(rag_hits=90)).status_code==422
    assert client.put(url,json={'settings':{**payload()['settings'],'extra_root':'evil'}}).status_code==422
    response=client.post(url+'/probe',json=payload()['settings'])
    assert response.status_code==200,response.text
    assert response.headers['cache-control']=='no-store'
    assert response.json()['checks']['ocr']['engine']=='tesseract'
    assert response.json()['production_qualified'] is False


def test_bearer_role_permissions(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'platform'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','bearer')
    monkeypatch.setenv('ASTROGRAPH_API_TOKENS_JSON',json.dumps({'admin-secret':['admin'],'viewer-secret':['viewer']}))
    client=TestClient(create_app())
    url='/v1/platform/settings/knowledge-engine'
    assert client.get(url).status_code==401
    assert client.get(url,headers={'Authorization':'Bearer viewer-secret'}).status_code==403
    assert client.put(url,json=payload(),headers={'Authorization':'Bearer viewer-secret'}).status_code==403
    assert client.put(url,json=payload(),headers={'Authorization':'Bearer admin-secret'}).status_code==200
    assert client.get(url,headers={'Authorization':'Bearer admin-secret'}).status_code==200


class FakeSecrets:
    values={}
    @staticmethod
    def make_ref(kind,revision):return kind+':'+revision
    def set(self,ref,key):self.values[ref]=key;return True
    def get(self,ref):return self.values.get(ref)


def test_secret_never_persisted_in_json_or_response(client, tmp_path):
    app=client.app
    fake=FakeSecrets();fake.values={}
    app.state.platform_secret_store=fake
    url='/v1/platform/settings/knowledge-engine'
    data=payload(vector_backend='qdrant',embedding_model=str(tmp_path),
                 qdrant_url='http://127.0.0.1:6333',qdrant_api_key='super-secret-value')
    response=client.put(url,json=data)
    assert response.status_code==200,response.text
    assert response.json()['qdrant_api_key_configured'] is True
    assert 'super-secret-value' not in response.text
    root=tmp_path/'platform'/'.engine_settings'
    assert not any('super-secret-value' in p.read_text() for p in root.glob('*.json'))
    assert 'qdrant_secret_ref' not in response.json()
    assert 'super-secret-value' not in client.get(url).text
    os.environ.pop('ASTROGRAPH_MATERIAL_QDRANT_API_KEY')
    MaterialEngineRegistry(tmp_path/'platform',fake).activate()
    assert os.environ['ASTROGRAPH_MATERIAL_QDRANT_API_KEY']=='super-secret-value'


def test_fail_closed_without_keyring(client,tmp_path):
    class NoKeyring(FakeSecrets):
        def set(self,ref,key):return False
    client.app.state.platform_secret_store=NoKeyring()
    data=payload(vector_backend='qdrant',embedding_model=str(tmp_path),
                 qdrant_url='http://127.0.0.1:6333',qdrant_api_key='cannot-persist')
    x=client.put('/v1/platform/settings/knowledge-engine',json=data)
    assert x.status_code==422
    assert client.get('/v1/platform/settings/knowledge-engine').json()['revision'] is None


def test_worker_loads_saved_version_before_running(client,tmp_path):
    saved=client.put('/v1/platform/settings/knowledge-engine',json=payload(rag_documents=23))
    assert saved.status_code==200
    os.environ.pop('ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET',None)
    from astrograph.integrations.upload_report import UploadReportJobStore
    store=UploadReportJobStore(tmp_path/'platform')
    with pytest.raises(KeyError):store.run('JOB-'+'0'*24)
    assert os.environ['ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET']=='23'


def test_advanced_multimodal_cli_uses_saved_defaults(client,tmp_path,monkeypatch):
    client.put('/v1/platform/settings/knowledge-engine',json=payload(
        vision_enabled=True,vision_endpoint='http://127.0.0.1:8001/v1/chat/completions',
        vision_model='local-visual-test',vision_pages=2))
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'platform'))
    import astrograph.retrieval.advanced_multimodal as mm
    captured={}
    def fake_ocr(path,**kw):
        captured['ocr']=kw
        return {'status':'OCR_UNREVIEWED','page_count':2,'elements':[]}
    def fake_render(*args,**kwargs):
        filename=Path(args[2]);filename.write_bytes(b'\x89PNG\r\n\x1a\n');return filename
    def fake_vlm(png,*,endpoint,model):
        captured['vlm']=(endpoint,model);return {'status':'VISION_CANDIDATE_REQUIRES_EXPERT_REVIEW'}
    monkeypatch.setattr(mm,'ocr_scanned_pdf',fake_ocr)
    monkeypatch.setattr(mm,'render_page_png',fake_render)
    monkeypatch.setattr(mm,'describe_diagram_local',fake_vlm)
    assert mm.main(['manual.pdf','--output',str(tmp_path/'out.json')])==0
    assert captured['vlm']==('http://127.0.0.1:8001/v1/chat/completions','local-visual-test')
    assert captured['ocr']['languages']=='chi_sim+eng'
    assert json.loads((tmp_path/'out.json').read_text())['visual_candidates'][0]['status'].endswith('REVIEW')


def test_reject_invalid_secret_payload_without_echoing_value(client,tmp_path):
    token='private-config-token-should-not-echo'
    invalid=payload(qdrant_api_key=token,vector_backend='qdrant',
                    embedding_model=str(tmp_path),qdrant_url='http://evil.invalid')
    res=client.put('/v1/platform/settings/knowledge-engine',json=invalid)
    assert res.status_code==422
    assert token not in res.text
    assert 'qdrant_api_key' not in res.text


def test_qdrant_key_not_reused_for_new_destination_or_disabled_backend(client,tmp_path):
    fake=FakeSecrets();fake.values={}
    client.app.state.platform_secret_store=fake
    url='/v1/platform/settings/knowledge-engine'
    first=client.put(url,json=payload(vector_backend='qdrant',embedding_model=str(tmp_path),
                  qdrant_url='http://127.0.0.1:6333',qdrant_api_key='for-original-destination'))
    assert first.status_code==200,first.text
    second=payload(vector_backend='qdrant',embedding_model=str(tmp_path),
                   qdrant_url='http://localhost:6333')
    second['expected_revision']=first.json()['revision']
    changed=client.put(url,json=second)
    assert changed.status_code==200,changed.text
    assert changed.json()['qdrant_api_key_configured'] is False
    assert not os.environ.get('ASTROGRAPH_MATERIAL_QDRANT_API_KEY')
    third=payload(vector_backend='sqlite')
    third['expected_revision']=changed.json()['revision']
    disabled=client.put(url,json=third)
    assert disabled.status_code==200,disabled.text
    assert disabled.json()['qdrant_api_key_configured'] is False
    assert not os.environ.get('ASTROGRAPH_MATERIAL_QDRANT_API_KEY')
