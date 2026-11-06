"""Full-platform P3: explicit profile selection for follow-up, no silent fallback."""
from __future__ import annotations

import json

import pytest
from astrograph.integrations import upload_report
from astrograph.integrations.external_diagnosis_conversation import ask_follow_up
from astrograph.integrations.platform_knowledge import PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.integrations.execution_observability import execution_snapshot


class CaptureBackend:
    def __init__(self, model_id='primary'):
        self.model_id = model_id
        self.calls = []

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append((agent_id, payload))
        sid = payload['sources'][0]['source_id']
        if agent_id == 'external-diagnosis-follow-up':
            return output_model.model_validate({'answer': [{'text': '来源仍需核对', 'citation_ids': [sid]}], 'information_gaps': []})
        return output_model.model_validate({
            'summary': '仅分析已上传的报告', 'findings': [{'text': '输入材料记录异常', 'citation_ids': [sid]}],
            'hypotheses': [], 'information_gaps': [], 'recommended_review_actions': [],
        })


def profile(store, *, name='qa-model', validation_status='PASS'):
    return store.state_repository.save_service_profile(
        service_kind='llm', name=name, validation_status=validation_status,
        config={'enabled': True, 'base_url': 'http://127.0.0.1:1234/v1', 'model': name, 'provider': 'lmstudio'},
    )


def staged(store, **kwargs):
    job = store.create('解释外部诊断结果', **kwargs)
    store.put_file(job['job_id'], 'diagnosis.txt', '反作用轮轮速出现异常，尚不能确定原因。'.encode())
    store.put_file(job['job_id'], 'analysis_input.json', json.dumps({
        'schema_version': 'astrograph-result-log-review.v1', 'analysis_mode': 'EXTERNAL_DIAGNOSIS_ANALYSIS',
        'diagnosis_results': ['diagnosis.txt'],
    }).encode())
    return job['job_id']


def test_phase_profile_stored_as_ids_and_hash_but_no_endpoint(tmp_path):
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=CaptureBackend()))
    p = profile(store)
    job = store.create('test', phase_engine_ids={'follow_up': 'llm:' + p['profile_id']})
    assert job['phase_engine_ids'] == {'follow_up': 'llm:' + p['profile_id']}
    assert len(job['phase_profile_fingerprints']['follow_up']) == 64
    assert '127.0.0.1' not in json.dumps(job['phase_profile_fingerprints'])
    with pytest.raises(ValueError):
        store.create('bad', phase_engine_ids={'not_a_phase': 'llm:' + p['profile_id']})
    verified = store.create('verification phase', phase_engine_ids={'verification': 'llm:' + p['profile_id']})
    assert verified['phase_engine_ids']['verification'] == 'llm:' + p['profile_id']
    assert len(verified['phase_profile_fingerprints']['verification']) == 64
    with pytest.raises(ValueError):
        store.create('bad', phase_engine_ids={'follow_up': 'llm:does-not-exist'})
    with pytest.raises(ValueError):
        store.create('bad', analysis_engine_id='rules', phase_engine_ids={'follow_up': 'llm:' + p['profile_id']})
    q = profile(store, name='unverified', validation_status='FAILED')
    with pytest.raises(ValueError):
        store.create('bad', phase_engine_ids={'follow_up': 'llm:' + q['profile_id']})


def test_full_platform_keeps_analysis_model_and_routes_followup_without_leaking_secret(tmp_path, monkeypatch):
    primary = CaptureBackend('primary')
    service = PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=primary)
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=service)
    p = profile(store)
    alt = CaptureBackend('qa-model')
    monkeypatch.setattr(upload_report, 'OpenAICompatibleStructuredBackend', lambda config: alt)
    jid = staged(store, phase_engine_ids={'follow_up': 'llm:' + p['profile_id']})
    result = store.run(jid)
    assert result['analysis']['knowledge_enrichment']['generation']['status'] == 'SUCCEEDED'
    assert len(primary.calls) == 1 and not alt.calls
    snapshot = json.loads((store._job_root(jid) / 'output/external_diagnosis_input.json').read_text())
    turns = ask_follow_up(store, jid, question='下一步该核对什么？', input_sha256=snapshot['input_sha256'], request_id='test-phase-0001')
    assert turns['turns'][0]['status'] == 'SUCCEEDED'
    assert len(alt.calls) == 1 and alt.calls[0][0] == 'external-diagnosis-follow-up'
    assert len(primary.calls) == 1
    assert result['job_id'] == jid
    assert '127.0.0.1' not in json.dumps(store.get(jid)['phase_profile_fingerprints'])


def test_changed_phase_profile_fails_closed_instead_of_falling_back(tmp_path, monkeypatch):
    primary = CaptureBackend('primary')
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=primary))
    p = profile(store)
    jid = staged(store, phase_engine_ids={'follow_up': 'llm:' + p['profile_id']})
    store.run(jid)
    p = store.state_repository.save_service_profile(
        service_kind='llm', name='qa-model', validation_status='PASS',
        config={'enabled': True, 'base_url': 'http://127.0.0.1:4321/v1', 'model': 'changed-model'},
    )
    payload = json.loads((store._job_root(jid) / 'output/external_diagnosis_input.json').read_text())
    out = ask_follow_up(store, jid, question='状态如何？', input_sha256=payload['input_sha256'], request_id='test-phase-0002')
    assert out['turns'][0]['reason_code'] == 'MODEL_CONFIGURATION_UNAVAILABLE'
    assert len(primary.calls) == 1


def test_api_schema_and_mission_ui_surface_profile_selection():
    from astrograph.api import PlatformJobCreateRequest
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'src/astrograph/resources'
    assert PlatformJobCreateRequest(phase_engine_ids={'follow_up': 'llm:PROFILE-EXAMPLE'}).phase_engine_ids
    html = (root / 'mission_console.html').read_text(encoding='utf-8')
    js = (root / 'mission_console.js').read_text(encoding='utf-8')
    assert 'id="mission-followup-engine"' in html
    assert 'id="mission-verification-engine"' in html
    assert "verificationEngineId" in js
    report_js = (root / "platform_report.js").read_text(encoding="utf-8")
    assert "主张与证据核验" in report_js
    assert "模型原始引用（未独立核验）" in report_js
    assert 'phase_engine_ids' in js
    assert 'mission-followup-engine' in js


def test_phase_selection_only_works_for_external_report_and_stale_profiles_block_preflight(tmp_path):
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=CaptureBackend()))
    p = profile(store)
    phases = {'follow_up': 'llm:' + p['profile_id']}
    jid = staged(store, phase_engine_ids=phases)
    assert store.preflight(jid)['can_start'] is True
    store.state_repository.save_service_profile(service_kind='llm', name='qa-model',
        validation_status='PASS', config={'enabled': True, 'base_url': 'http://127.0.0.1:5678/v1', 'model': 'changed'})
    check = store.preflight(jid)
    assert check['can_start'] is False
    assert any(x['code']=='PHASE_MODEL_PROFILE_CHANGED' for x in check['blockers'])
    failed = store.run(jid)
    assert failed['status'] == 'FAILED'
    assert failed['error']['type'] == 'PreflightBlockedError'
    # The same optional selection must never influence legacy diagnosis workflows.
    old = store.create('old workflow', phase_engine_ids={})
    assert old['phase_engine_ids'] == {}


def test_default_jobs_keep_preexisting_single_model_behavior(tmp_path):
    original = CaptureBackend()
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=original))
    jid = staged(store)
    report = store.run(jid)
    assert report['analysis']['knowledge_enrichment']['generation']['status'] == 'SUCCEEDED'
    bundle = json.loads((store._job_root(jid)/'output/external_diagnosis_input.json').read_text())
    state = ask_follow_up(store, jid, question='解释原因', input_sha256=bundle['input_sha256'], request_id='test-default-001')
    assert state['turns'][0]['status'] == 'SUCCEEDED'
    assert [x[0] for x in original.calls] == ['external-diagnosis-analysis', 'external-diagnosis-follow-up']


def test_existing_selected_primary_profile_cannot_change_after_stage_binding(tmp_path):
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=CaptureBackend()))
    primary = profile(store, name='main-model')
    alt = profile(store, name='follow-model')
    job = store.create('analysis', analysis_engine_id='llm:' + primary['profile_id'],
                       phase_engine_ids={'follow_up': 'llm:' + alt['profile_id']})
    store.state_repository.save_service_profile(service_kind='llm', name='main-model',
        validation_status='PASS', config={'enabled': True, 'base_url': 'https://example.org/v1', 'model': 'modified'})
    with pytest.raises(ValueError, match='changed'):
        store._knowledge_service_for_job(store.get(job['job_id']))


def test_auto_base_backend_cannot_change_silently_across_store_restart(tmp_path):
    service = PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=CaptureBackend('original'))
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=service)
    alt = profile(store)
    job = store.create('analysis', phase_engine_ids={'follow_up': 'llm:' + alt['profile_id']})
    changed = UploadReportJobStore(store.root, knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=CaptureBackend('changed')))
    with pytest.raises(ValueError, match='changed'):
        changed._knowledge_service_for_job(changed.get(job['job_id']))


def test_platform_http_accepts_stage_selection_with_existing_profiles(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from astrograph.api import create_app
    from astrograph.integrations.platform_state import PlatformStateRepository
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(tmp_path / 'platform'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE', 'local')
    state = PlatformStateRepository.from_environment(platform_root=tmp_path / 'platform')
    main = state.save_service_profile(service_kind='llm', name='main-http', validation_status='PASS',
        config={'enabled': True, 'base_url': 'http://127.0.0.1:1234/v1', 'model': 'main-http'})
    alt = state.save_service_profile(service_kind='llm', name='qa-http', validation_status='PASS',
        config={'enabled': True, 'base_url': 'http://127.0.0.1:1234/v1', 'model': 'qa-http'})
    with TestClient(create_app()) as client:
        result = client.post('/v1/platform/jobs', json={
            'task': 'interpret', 'analysis_engine_id': 'llm:' + main['profile_id'],
            'phase_engine_ids': {'follow_up': 'llm:' + alt['profile_id']},
        })
        assert result.status_code == 200, result.text
        job = result.json()
        assert job['phase_engine_ids']['follow_up'] == 'llm:' + alt['profile_id']
        assert '127.0.0.1' not in json.dumps(job.get('phase_profile_fingerprints'))
        bad = client.post('/v1/platform/jobs', json={'phase_engine_ids': {'invalid': 'llm:' + alt['profile_id']}})
        assert bad.status_code == 422
        missing = client.post('/v1/platform/jobs', json={
            'analysis_engine_id': 'llm:' + main['profile_id'],
            'phase_engine_ids': {'follow_up': 'llm:PROFILE-DOES-NOT-EXIST'},
        })
        assert missing.status_code == 422
        assert '127.0.0.1' not in missing.text
