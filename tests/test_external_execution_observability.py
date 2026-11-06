"""Same canonical recorder is used by the full platform, not only the portable UI."""
import json
import pytest
from fastapi.testclient import TestClient
from astrograph.api import create_app
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.integrations.platform_knowledge import PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService
from astrograph.integrations.execution_observability import execution_snapshot


class Backend:
    def generate_structured(self, *, payload, output_model, **kw):
        return output_model.model_validate({'summary':'模型分析',
            'findings':[{'text':'上游报告记录温度升高','citation_ids':[payload['sources'][0]['source_id']]}],
            'hypotheses':[], 'information_gaps':[], 'recommended_review_actions':[]})


def stage(root):
    store=UploadReportJobStore(root, knowledge_service=PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(), llm_backend=Backend()))
    job=store.create('分析诊断报告');job_id=job['job_id']
    store.put_file(job_id,'diagnosis.txt','报告记录温度升高。'.encode('utf-8'))
    store.put_file(job_id,'analysis_input.json',json.dumps({
        'schema_version':'astrograph-result-log-review.v1','analysis_mode':'EXTERNAL_DIAGNOSIS_ANALYSIS',
        'diagnosis_results':['diagnosis.txt']}).encode())
    assert store.preflight(job_id)['can_start']
    return store,job_id


def test_full_platform_external_analysis_publishes_same_run_receipts(tmp_path):
    store,job_id=stage(tmp_path/'jobs');report=store.run(job_id)
    snapshot=execution_snapshot(store._job_root(job_id),job_id)
    assert not snapshot['legacy']
    run=snapshot['runs'][0]
    assert run['status']=='SUCCEEDED'
    assert {a['name'] for a in run['artifacts']}=={'report.json','report.md'}
    assert run['materials'][0]['filename']=='diagnosis.txt'
    assert snapshot['usage']['input_tokens'] is None  # fake backend, not measured network usage


def test_full_platform_does_not_overwrite_prior_run_artifacts(tmp_path):
    store,job_id=stage(tmp_path/'jobs');store.run(job_id)
    root=store._job_root(job_id)
    snap=execution_snapshot(root,job_id)
    assert len(snap['runs'])==1
    archived=root/'execution/runs'/snap['runs'][0]['run_id']/'artifacts/report.json'
    assert archived.read_bytes()==(root/'output/report.json').read_bytes()


def test_platform_execution_api_obeys_auth_and_artifact_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','bearer')
    monkeypatch.setenv('ASTROGRAPH_API_TOKENS_JSON',json.dumps({'viewer-key':['viewer']}))
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'platform'))
    store,job_id=stage(tmp_path/'jobs');store.run(job_id)
    app=create_app();app.state.upload_report_store=store
    with TestClient(app) as client:
        url='/v1/platform/jobs/'+job_id+'/execution'
        assert client.get(url).status_code==401
        response=client.get(url,headers={'Authorization':'Bearer viewer-key'})
        assert response.status_code==200
        assert response.headers['Cache-Control']=='no-store'
        run=response.json()['runs'][0]
        artifact='/v1/platform/jobs/'+job_id+'/runs/'+run['run_id']+'/report.json'
        assert client.get(artifact).status_code==401
        response=client.get(artifact,headers={'Authorization':'Bearer viewer-key'})
        assert response.status_code==200
        assert response.content==(store._job_root(job_id)/'output/report.json').read_bytes()
        assert client.get(artifact.replace('report.json','job.json'),headers={'Authorization':'Bearer viewer-key'}).status_code==404


def test_full_platform_followup_records_new_run_not_idempotent_replays(tmp_path):
    from astrograph.integrations.external_diagnosis_conversation import ask_follow_up, get_conversation
    class WithFollowUp(Backend):
        model_id='test-model'
        def generate_structured(self, *, payload, output_model, agent_id, **kw):
            if agent_id=='external-diagnosis-follow-up':
                return output_model.model_validate({'answer':[{'text':'建议复核温度传感器',
                    'citation_ids':[payload['sources'][0]['source_id']]}], 'information_gaps':[]})
            return super().generate_structured(payload=payload,output_model=output_model,**kw)
    store,job_id=stage(tmp_path/'jobs')
    store.knowledge_service.llm_backend=WithFollowUp()
    store.run(job_id)
    before=(store._job_root(job_id)/'output/report.json').read_bytes()
    request=dict(question='下一步核查什么？',input_sha256=get_conversation(store,job_id)['input_sha256'],
                 request_id='followup-test-001')
    result=ask_follow_up(store,job_id,**request)
    assert result['turns'][-1]['status']=='SUCCEEDED'
    snap=execution_snapshot(store._job_root(job_id),job_id)
    assert len(snap['runs'])==2
    turn_run=next(r for r in snap['runs'] if r['kind']=='follow_up')
    assert turn_run['status']=='SUCCEEDED'
    assert turn_run['model']=='test-model'
    assert [x['name'] for x in turn_run['artifacts']]==['conversation.json']
    assert ask_follow_up(store,job_id,**request)==result
    assert len(execution_snapshot(store._job_root(job_id),job_id)['runs'])==2
    assert (store._job_root(job_id)/'output/report.json').read_bytes()==before


def test_execution_shared_assets_served_by_both_platform_shells(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'platform'))
    from astrograph.integrations.upload_report import PLATFORM_UI_ASSET_VERSION
    app=create_app()
    with TestClient(app) as client:
        for prefix in ['/platform/mission/assets/', '/platform/assets/'+PLATFORM_UI_ASSET_VERSION+'/']:
            for name in ['execution_view.js','execution_view.css']:
                response=client.get(prefix+name)
                assert response.status_code==200,(prefix,name,response.text)
                assert len(response.content)>100
