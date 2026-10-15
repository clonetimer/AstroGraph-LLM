"""Result/log reports must reach the command center with machine-readable ABSTAIN."""
import json
from fastapi.testclient import TestClient
from astrograph.api import create_app


def test_material_abstain_is_pending_event_not_pseudo_confirmed(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(tmp_path / 'platform'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE', 'local')
    with TestClient(create_app()) as client:
        cases=client.get('/platform/mission/assets/mission_demo_cases.json').json()['cases']
        one=cases[0]
        job=client.post('/v1/platform/jobs',json={'task':'分析故障','work_mode':'normal'}).json()['job_id']
        manifest={'schema_version':'astrograph-result-log-review.v1','equipment_scope':one['equipment_scope']}
        for role,files in one['files'].items():
            manifest[role]=[]
            for f in files:
                manifest[role].append(f['name'])
                assert client.put(f'/v1/platform/jobs/{job}/files/{f["name"]}',content=f['content'].encode()).status_code==200
        assert client.put(f'/v1/platform/jobs/{job}/files/analysis_input.json',content=json.dumps(manifest).encode()).status_code==200
        assert client.post(f'/v1/platform/jobs/{job}/preflight').json()['can_start']
        assert client.post(f'/v1/platform/jobs/{job}/run').json()['status']=='SUCCEEDED'
        overview=client.get('/v1/platform/overview').json()
        event=overview['recent_jobs'][0]
        assert event['artifact_kind']=='RESULT_LOG_REVIEW'
        assert event['diagnosis_disposition']=='ABSTAIN'
        assert event['manual_review_status']=='PENDING'
        assert event['requires_human_review'] is True
        assert event['fault_name']
        assert overview['review_or_attention_count']==1


def test_review_completion_updates_task_queue_without_confirming_root_cause(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(tmp_path / 'platform'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE', 'local')
    with TestClient(create_app()) as client:
        case=client.get('/platform/mission/assets/mission_demo_cases.json').json()['cases'][0]
        job=client.post('/v1/platform/jobs',json={'task':'核查材料','work_mode':'normal'}).json()['job_id']
        manifest={'schema_version':'astrograph-result-log-review.v1','equipment_scope':case['equipment_scope']}
        for role,files in case['files'].items():
            manifest[role]=[]
            for f in files:
                assert client.put(f'/v1/platform/jobs/{job}/files/{f["name"]}',content=f['content'].encode()).status_code==200
                manifest[role].append(f['name'])
        assert client.put(f'/v1/platform/jobs/{job}/files/analysis_input.json',content=json.dumps(manifest).encode()).status_code==200
        assert client.post(f'/v1/platform/jobs/{job}/preflight').json()['can_start']
        assert client.post(f'/v1/platform/jobs/{job}/run').json()['status']=='SUCCEEDED'
        report=client.get(f'/v1/platform/jobs/{job}/report').json()
        from astrograph.integrations.manual_report_review import ManualReportReviewStore
        store=ManualReportReviewStore(tmp_path/'platform'/'material_knowledge'/'manual_report_reviews.sqlite3')
        initial=store.detail(job,report,include_history=True)
        assert initial['status']=='PENDING'
        for item in initial['candidates']:
            initial=store.record(job_id=job,report=report,report_sha256=initial['report_sha256'],
                fault_id=item['fault_id'],decision='FOLLOW_UP_HYPOTHESIS',
                evidence_ref='TEST-EVIDENCE-INDEPENDENT-20261011',note='复核证据仅支持继续排查，仍未确诊根因',
                actor='expert01',expected_revision=initial['revision'])
        assert initial['status']=='REVIEWED'
        assert initial['current_root_cause_confirmed'] is False
        event=client.get('/v1/platform/overview').json()['recent_jobs'][0]
        assert event['manual_review_status']=='REVIEWED'
        assert event['requires_human_review'] is False
        assert event['diagnosis_disposition']=='ABSTAIN'
