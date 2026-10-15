"""Report-level expert review: actionable UI, immutable evidence, authorization and concurrency."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.audit.store import sha256_json
from astrograph.integrations.manual_report_review import ManualReportReviewStore
from astrograph.integrations.upload_report import UploadReportJobStore, PLATFORM_UI_ASSET_VERSION

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'result_log_review'


def new_job(root: Path):
    store = UploadReportJobStore(root)
    job_id = store.create('校验材料人工复核工作流')['job_id']
    for file in EXAMPLE.iterdir():
        if file.is_file():
            store.put_file(job_id, file.name, file.read_bytes())
    assert store.preflight(job_id)['can_start']
    assert store.run(job_id)['status'] == 'SUCCEEDED'
    return store, job_id


def test_expert_can_act_from_result_without_mutating_source_report(tmp_path, monkeypatch):
    root = tmp_path / 'platform'
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(root))
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN', 'test-material-manual-review-token-123456')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER', 'expert-reviewer')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS', 'expert-reviewer')
    store, job_id = new_job(root)
    before = store.report(job_id)
    digest = sha256_json(before)
    with TestClient(create_app(), base_url='http://localhost') as client:
        endpoint = f'/v1/platform/jobs/{job_id}/manual-review/status'
        private = f'/v1/platform/material-knowledge/report-reviews/{job_id}'
        empty = client.get(endpoint)
        assert empty.status_code == 200
        assert empty.json()['status'] == 'PENDING'
        assert empty.json()['report_sha256'] == digest
        assert 'history' not in empty.json()
        assert client.get(private).status_code == 403
        assert client.get('/v1/platform/material-knowledge/report-reviews').status_code == 403
        assert client.post(private + '/decisions', json={'report_sha256':digest,'fault_id':before['analysis']['result_log_review']['candidates'][0]['fault_id'], 'decision':'EVIDENCE_INSUFFICIENT','note':'证据不足，建议补充时间日志信息','expected_revision':0}).status_code == 403
        login = client.post('/v1/platform/material-knowledge/review-session',json={
            'credential':'test-material-manual-review-token-123456'})
        assert login.status_code == 200
        csrf = {'X-Material-CSRF': login.json()['csrf']}
        detail = client.get(private).json()
        fault_id = detail['candidates'][0]['fault_id']
        assert any(j['job_id'] == job_id for j in client.get('/v1/platform/material-knowledge/report-reviews').json()['items'])
        payload = {
            'report_sha256':digest,'fault_id':fault_id,'decision':'FOLLOW_UP_HYPOTHESIS',
            'evidence_ref':'LAB-CHECK-2026-001','note':'经独立核对电机电流和轮速，建议继续排查摩擦异常',
            'expected_revision':0,
        }
        assert client.post(private + '/decisions',json=payload).status_code == 403
        bad = client.post(private+'/decisions',headers=csrf,json={**payload,'report_sha256':'0'*64})
        assert bad.status_code == 409
        missing = client.post(private+'/decisions',headers=csrf,json={**payload,'fault_id':'NOT_IN_REPORT'})
        assert missing.status_code == 422
        assert client.post(private+'/decisions',headers=csrf,json={**payload,'evidence_ref':''}).status_code == 422
        accepted = client.post(private+'/decisions',headers=csrf,json=payload)
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()['status']=='REVIEWED'
        assert accepted.json()['candidates'][0]['review']['actor']=='expert-reviewer'
        assert not accepted.json()['current_root_cause_confirmed']
        assert not accepted.json()['automatic_control_authorized']
        assert client.get(endpoint).json()['status'] == 'REVIEWED'
        assert client.post(private+'/decisions',headers=csrf,json=payload).status_code == 409
        newer={**payload,'decision':'EVIDENCE_INSUFFICIENT','evidence_ref':'',
               'note':'再次复核发现关键时间窗口缺少证据，请补充更多日志','expected_revision':accepted.json()['revision']}
        revision = client.post(private+'/decisions',headers=csrf,json=newer)
        assert revision.status_code == 200, revision.text
        assert revision.json()['status']=='REVIEWED_WITH_GAPS'
        assert len(revision.json()['history']) == 2
        assert store.report(job_id) == before
        assert sha256_json(store.report(job_id)) == digest
        assert client.get(endpoint).json().get('history') is None
        assert client.delete('/v1/platform/material-knowledge/review-session',headers=csrf).status_code == 200
        assert client.get(private).status_code == 403


def test_two_candidates_partial_progress_and_missing_candidate(tmp_path):
    report={
        'artifact_kind':'RESULT_LOG_REVIEW',
        'analysis':{'result_log_review':{'candidates':[
            {'fault_id':'EPS_A','fault_name':'电源 A'},
            {'fault_id':'ADCS_B','fault_name':'姿控 B'},
        ]}},
    }
    db=ManualReportReviewStore(tmp_path/'review.sqlite3')
    before=db.detail('JOB-'+'A'*24,report)
    assert before['total_count'] == 2 and before['status'] == 'PENDING'
    changed=db.record(job_id='JOB-'+'A'*24,report=report,report_sha256=before['report_sha256'],
        fault_id='EPS_A',decision='EVIDENCE_INSUFFICIENT',evidence_ref='',
        note='当前证据不足，需要补充功率相关观测量',actor='expert',expected_revision=0)
    assert changed['status']=='IN_PROGRESS' and changed['reviewed_count']==1
    changed=db.record(job_id='JOB-'+'A'*24,report=report,report_sha256=before['report_sha256'],
        fault_id='ADCS_B',decision='NOT_SUPPORTED',evidence_ref='INDEPENDENT-LOG-002',
        note='外部工程台架记录不支持所述姿控异常',actor='expert',expected_revision=changed['revision'])
    assert changed['status']=='REVIEWED_WITH_GAPS' and changed['reviewed_count']==2
    assert len(changed['history'])==2
    other={**report,'analysis':{'result_log_review':{'candidates':[]}}}
    no=db.detail('JOB-'+'B'*24,other,include_history=True)
    assert no['total_count']==1 and no['candidates'][0]['fault_id']=='__REPORT_NO_CANDIDATE__'


def test_report_navigation_assets_have_action_and_do_not_expose_bootstrap_secret():
    from astrograph.integrations.upload_report import platform_asset, platform_html
    report, _ = platform_asset('platform_report.js')
    graph, _ = platform_asset('platform_graph_workbench.js')
    shell = platform_html()
    assert 'report-manual-review-open' in report
    assert 'focusManualReportReview(report.job_id)' in report
    assert 'data-review-kind="manual"' in shell
    assert 'data-manual-review-submit' in graph
    assert 'X-Material-CSRF' in graph
    assert PLATFORM_UI_ASSET_VERSION in shell
