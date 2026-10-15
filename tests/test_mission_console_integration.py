"""v2.1 design-to-runtime integration: endpoints, role separation, demo materials."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from astrograph.api import create_app


def _app(monkeypatch, tmp_path: Path, *, bearer=False):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE', 'bearer' if bearer else 'local')
    if bearer:
        monkeypatch.setenv('ASTROGRAPH_API_TOKENS_JSON', json.dumps({
            'read-secret-abc': {'subject': 'visitor', 'roles': ['viewer']},
            'ops-secret-abc': {'subject': 'operator1', 'roles': ['operator']},
            'admin-secret-abc': {'subject': 'admin1', 'roles': ['admin']},
        }))
    else:
        monkeypatch.delenv('ASTROGRAPH_API_TOKENS_JSON', raising=False)
    return create_app()


def test_mission_console_assets_are_allowlisted(monkeypatch, tmp_path):
    with TestClient(_app(monkeypatch, tmp_path)) as client:
        page = client.get('/platform/mission')
        assert page.status_code == 200
        assert '智能助理' in page.text and '三类材料' in page.text
        for name, kind in [('mission_console.js','javascript'), ('mission_console.css','css'), ('mission_demo_cases.json','json')]:
            r = client.get('/platform/mission/assets/'+name)
            assert r.status_code == 200 and kind in r.headers['content-type']
        assert client.get('/platform/mission/assets/../../api.py').status_code in (400,404)
        assert client.get('/platform/mission/assets/platform_ui.html').status_code == 404
        identity = client.get('/v1/platform/mission/identity').json()
        assert identity['mode'] == 'local'
        assert identity['permissions'] == {'read':True, 'analyze':True,'settings':True}
        assert client.get('/platform').status_code == 200  # defaults to new workbench
        assert '/platform/mission' in client.get('/platform').text
        assert client.get('/platform/legacy').status_code == 200  # legacy available when needed


def test_mission_identity_reflects_real_server_role(monkeypatch, tmp_path):
    with TestClient(_app(monkeypatch, tmp_path, bearer=True)) as client:
        assert client.get('/v1/platform/mission/identity').status_code == 401
        def auth(token): return {'Authorization':'Bearer '+token}
        viewer = client.get('/v1/platform/mission/identity', headers=auth('read-secret-abc'))
        assert viewer.status_code == 200
        assert viewer.json()['roles'] == ['viewer']
        assert viewer.json()['permissions'] == {'read':True,'analyze':False,'settings':False}
        assert client.post('/v1/platform/jobs',json={'task':'test','work_mode':'normal'},headers=auth('read-secret-abc')).status_code == 403
        assert client.get('/v1/platform/settings/llm',headers=auth('ops-secret-abc')).status_code == 403
        admin = client.get('/v1/platform/mission/identity',headers=auth('admin-secret-abc'))
        assert admin.status_code == 200 and admin.json()['permissions']['settings'] is True


def test_mission_builtin_cases_pass_actual_preflight(monkeypatch, tmp_path):
    with TestClient(_app(monkeypatch, tmp_path)) as client:
        catalog = client.get('/platform/mission/assets/mission_demo_cases.json').json()
        assert catalog['synthetic'] is True
        assert len(catalog['cases']) == 4
        one = catalog['cases'][0]
        job = client.post('/v1/platform/jobs',json={'task':'根据输入材料分析原因与处置','work_mode':'normal'}).json()
        job_id=job['job_id']
        roles={}
        for role, entries in one['files'].items():
            roles[role]=[]
            for file in entries:
                response=client.put(f'/v1/platform/jobs/{job_id}/files/{file["name"]}',content=file['content'].encode('utf8'))
                assert response.status_code == 200, response.text
                roles[role].append(file['name'])
        manifest={'schema_version':'astrograph-result-log-review.v1',**roles,'equipment_scope':one['equipment_scope']}
        uploaded=client.put(f'/v1/platform/jobs/{job_id}/files/analysis_input.json',content=json.dumps(manifest).encode('utf8'))
        assert uploaded.status_code == 200, uploaded.text
        preflight=client.post(f'/v1/platform/jobs/{job_id}/preflight')
        assert preflight.status_code == 200, preflight.text
        assert preflight.json()['can_start'] is True, preflight.text


def test_mission_interface_has_no_fake_model_role_switch():
    resource = Path(__file__).parents[1]/'src/astrograph/resources/mission_console.js'
    code = resource.read_text(encoding='utf8')
    assert '/v1/platform/mission/identity' in code
    assert '/v1/platform/jobs' in code
    assert '/v1/platform/material-knowledge/report-reviews/' in code
    assert 'modelReplyFromFake' not in code
    assert 'role-select' not in code
    assert 'localStorage.setItem(\'token' not in code


def test_mission_real_demo_execution_report_and_review_state(monkeypatch, tmp_path):
    """The new report/chat is backed by actual generated fault evidence, not fixture answers."""
    with TestClient(_app(monkeypatch, tmp_path)) as client:
        demo = client.get('/platform/mission/assets/mission_demo_cases.json').json()['cases'][0]
        response = client.post('/v1/platform/jobs', json={'task': '分析真实输入材料的原因与处置', 'work_mode': 'normal'})
        assert response.status_code == 200
        job_id = response.json()['job_id']
        manifest = {'schema_version': 'astrograph-result-log-review.v1', 'equipment_scope': demo['equipment_scope']}
        for role, files in demo['files'].items():
            manifest[role] = [f['name'] for f in files]
            for file in files:
                upload = client.put(f'/v1/platform/jobs/{job_id}/files/{file["name"]}',
                                    content=file['content'].encode('utf-8'))
                assert upload.status_code == 200, upload.text
        assert client.put(f'/v1/platform/jobs/{job_id}/files/analysis_input.json',
                          content=json.dumps(manifest).encode('utf-8')).status_code == 200
        preflight = client.post(f'/v1/platform/jobs/{job_id}/preflight')
        assert preflight.json()['can_start'] is True
        execution = client.post(f'/v1/platform/jobs/{job_id}/run')
        assert execution.status_code == 200 and execution.json()['status'] == 'SUCCEEDED'
        report = client.get(f'/v1/platform/jobs/{job_id}/report').json()
        review = client.get(f'/v1/platform/jobs/{job_id}/manual-review/status').json()
        block = report['analysis']['result_log_review']
        assert report['artifact_kind'] == 'RESULT_LOG_REVIEW'
        assert block['candidates'] and block['evidence']
        assert block['candidates'][0]['cause_hypotheses']
        assert block['candidates'][0]['handling_measures']
        assert review['status'] == 'PENDING'
        assert review['current_root_cause_confirmed'] is False
        assert review['automatic_control_authorized'] is False


def test_mission_console_linked_from_legacy_and_invalidates_changed_task():
    project = Path(__file__).parents[1]
    html = (project/'src/astrograph/resources/platform_ui.html').read_text(encoding='utf-8')
    js = (project/'src/astrograph/resources/mission_console.js').read_text(encoding='utf-8')
    assert 'href="/platform/mission"' in html
    assert "['mission-task','mission-scope']" in js
    assert "S.preflightId='';$('#mission-start').disabled=true" in js
    assert '/v1/platform/analysis-engines' in js
    assert 'analysis_engine_id:S.analysisEngineId' in js
    assert 'model_explanation' in js
    assert 'role-select' not in js


def test_mission_llm_persistent_profile_restart_and_role_boundary(monkeypatch, tmp_path):
    """Model editor's 'save as default' maps to real persistent, admin-only API."""
    import astrograph.integrations.platform_llm_probe as probe_module

    monkeypatch.setattr(probe_module, 'probe_openai_compatible_service',
                        lambda **kwargs: {'status': 'READY', 'transport': 'test_stub'})
    with TestClient(_app(monkeypatch, tmp_path, bearer=True)) as client:
        data = {
            'name': '航天知识分析本地模型',
            'make_default': True,
            'activate': True,
            'settings': {'provider': 'local-openai-compatible',
                         'base_url': 'http://127.0.0.1:18000/v1',
                         'model': 'astrograph-local-test', 'enabled': True},
        }
        assert client.post('/v1/platform/settings/llm/profiles', json=data,
                           headers={'Authorization':'Bearer read-secret-abc'}).status_code == 403
        assert client.post('/v1/platform/settings/llm/profiles', json=data,
                           headers={'Authorization':'Bearer ops-secret-abc'}).status_code == 403
        saved = client.post('/v1/platform/settings/llm/profiles', json=data,
                            headers={'Authorization':'Bearer admin-secret-abc'})
        assert saved.status_code == 200, saved.text
        assert saved.json()['active'] is True
        profile_id = saved.json()['profile']['profile_id']
        assert profile_id
        catalogue = client.get('/v1/platform/analysis-engines',
                               headers={'Authorization':'Bearer ops-secret-abc'})
        assert catalogue.status_code == 200
        choices = {item['engine_id']: item for item in catalogue.json()['engines']}
        assert choices['rules']['kind'] == 'rules'
        assert choices[f'llm:{profile_id}']['model'] == 'astrograph-local-test'
        selected_job = client.post(
            '/v1/platform/jobs',
            json={'task':'使用指定模型分析材料','work_mode':'normal',
                  'analysis_engine_id':f'llm:{profile_id}'},
            headers={'Authorization':'Bearer ops-secret-abc'},
        )
        assert selected_job.status_code == 200, selected_job.text
        assert selected_job.json()['analysis_engine']['profile_id'] == profile_id
        assert selected_job.json()['analysis_engine']['model'] == 'astrograph-local-test'
        running = client.get('/v1/platform/settings/llm',
                             headers={'Authorization':'Bearer admin-secret-abc'}).json()
        assert running['persists_across_restart'] is True
        assert running['model'] == 'astrograph-local-test'

    # Simulate process startup with no manually exported model variables.
    for key in ('ASTROGRAPH_PLATFORM_LLM_BASE_URL', 'ASTROGRAPH_PLATFORM_LLM_MODEL',
                'ASTROGRAPH_PLATFORM_LLM_PROVIDER', 'ASTROGRAPH_PLATFORM_LLM_ACTIVE_PROFILE_ID'):
        monkeypatch.delenv(key, raising=False)
    with TestClient(create_app()) as client:
        restored = client.get('/v1/platform/settings/llm',
                              headers={'Authorization':'Bearer admin-secret-abc'})
        assert restored.status_code == 200
        assert restored.json()['model'] == 'astrograph-local-test'
        assert restored.json()['persists_across_restart'] is True
