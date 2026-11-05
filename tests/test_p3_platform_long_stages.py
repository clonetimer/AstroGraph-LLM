"""Full platform long-report stages must retain citation and model pins."""
import json

from astrograph.integrations import upload_report
from astrograph.integrations.platform_knowledge import PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService
from astrograph.integrations.upload_report import UploadReportJobStore, PlatformJobCreateRequest


class StageBackend:
    def __init__(self, model_id):
        self.model_id = model_id
        self.calls = []

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append((agent_id, payload))
        sid = payload['sources'][0]['source_id']
        return output_model.model_validate({
            'summary': '尚需人工复核', 'findings': [{'text': '已记录异常', 'citation_ids': [sid]}],
            'hypotheses': [], 'information_gaps': [], 'recommended_review_actions': [],
        })


def make_store(tmp_path):
    primary = StageBackend('primary')
    store = UploadReportJobStore(tmp_path/'jobs', knowledge_service=PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(), llm_backend=primary))
    return store, primary


def make_profile(store, name):
    return store.state_repository.save_service_profile(
        service_kind='llm', name=name, validation_status='PASS',
        config={'enabled': True, 'base_url': 'http://127.0.0.1:1234/v1',
                'model': name, 'provider': 'lmstudio'},
    )


def stage_file(store, job, *, text):
    jid = job['job_id']
    store.put_file(jid, 'diagnosis.txt', text.encode('utf-8'))
    store.put_file(jid, 'analysis_input.json', json.dumps({
        'schema_version': 'astrograph-result-log-review.v1',
        'analysis_mode': 'EXTERNAL_DIAGNOSIS_ANALYSIS',
        'diagnosis_results': ['diagnosis.txt'],
    }).encode())
    return jid


def test_platform_phase_contract_accepts_chunk_synthesis_followup():
    request = PlatformJobCreateRequest(phase_engine_ids={
        'chunk': 'llm:CHUNK', 'synthesis': 'llm:SYNTH', 'follow_up': 'llm:QA',
    })
    assert set(request.phase_engine_ids) == {'chunk', 'synthesis', 'follow_up'}


def test_long_report_uses_selected_chunk_and_synthesis_models_and_preserves_original_citations(tmp_path, monkeypatch):
    store, primary = make_store(tmp_path)
    engines = {'chunk': StageBackend('chunk'), 'synthesis': StageBackend('synthesis')}
    profiles = {key: make_profile(store, key) for key in engines}
    monkeypatch.setattr(upload_report, 'OpenAICompatibleStructuredBackend', lambda conf: engines[conf.model])
    phases = {key: 'llm:' + profile['profile_id'] for key, profile in profiles.items()}
    job = store.create('解释根因及反证', phase_engine_ids=phases)
    jid = stage_file(store, job, text='反作用轮电流和温度异常，需要核验。\n' * 4000)
    result = store.run(jid)
    assert result['status'] == 'SUCCEEDED'
    explanation = result['analysis']['knowledge_enrichment']['model_explanation']
    assert result['analysis']['knowledge_enrichment']['generation']['status'] == 'SUCCEEDED'
    assert explanation['processing']['mode'] == 'HIERARCHICAL_CHUNK_SYNTHESIS'
    assert len(engines['chunk'].calls) > 1
    assert len(engines['synthesis'].calls) == 1
    assert primary.calls == []
    ids = {source['source_id'] for source in explanation['sources']}
    assert set(explanation['findings'][0]['citation_ids']) <= ids
    assert (store._job_root(jid)/'output/knowledge/checkpoint.json').exists()


def test_short_report_keeps_primary_single_call_with_long_phase_configuration(tmp_path, monkeypatch):
    store, primary = make_store(tmp_path)
    engines = {'chunk': StageBackend('chunk'), 'synthesis': StageBackend('synthesis')}
    profiles = {key: make_profile(store, key) for key in engines}
    monkeypatch.setattr(upload_report, 'OpenAICompatibleStructuredBackend', lambda conf: engines[conf.model])
    job = store.create('分析短报告', phase_engine_ids={key: 'llm:' + p['profile_id'] for key,p in profiles.items()})
    result = store.run(stage_file(store, job, text='反作用轮电流异常。'))
    assert result['analysis']['knowledge_enrichment']['generation']['status'] == 'SUCCEEDED'
    assert len(primary.calls) == 1
    assert not engines['chunk'].calls and not engines['synthesis'].calls


def test_changed_chunk_profile_blocks_preflight_before_model_call(tmp_path):
    store, primary = make_store(tmp_path)
    profile = make_profile(store, 'chunk')
    jid = stage_file(store, store.create('解释报告',phase_engine_ids={'chunk': 'llm:' + profile['profile_id']}),
                     text='反作用轮电流异常。' * 300)
    store.state_repository.save_service_profile(
        service_kind='llm',name='chunk',validation_status='PASS',
        config={'enabled': True, 'base_url': 'http://127.0.0.1:7654/v1', 'model': 'changed'},
    )
    check = store.preflight(jid)
    assert not check['can_start']
    assert any(item['code']=='PHASE_MODEL_PROFILE_CHANGED' for item in check['blockers'])
    assert primary.calls == []
