"""Report-level expert review: actionable UI, immutable evidence, authorization and concurrency."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.audit.store import sha256_json
from astrograph.integrations.manual_report_review import ManualReportReviewStore, ManualReviewConflict
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
    assert 'AUDIT_INTEGRITY_FAILED' in graph
    assert '审计链异常' in graph
    assert 'data-claim-feedback-export' in graph
    assert PLATFORM_UI_ASSET_VERSION in shell


def governed_claim_report():
    return {
        "artifact_kind": "RESULT_LOG_REVIEW",
        "analysis": {
            "result_log_review": {"candidates": [
                {"fault_id": "ADCS_RW_FRICTION_INCREASE", "fault_name": "飞轮摩擦增大"},
            ]},
            "knowledge_enrichment": {"model_explanation": {"claim_governance": {
                "schema_version": "astrograph-claim-governance.v1",
                "claims": [{
                    "claim_id": "claim-1234567890abcdef1234567890abcdef",
                    "claim_type": "HYPOTHESIS",
                    "text": "遥测链路延迟可能影响轮速异常的时间对齐。",
                    "epistemic_status": "INFERRED",
                    "evidence_refs": [{"source_id": "SRC-1", "filename": "diagnosis.txt",
                                       "locator": "line:12", "sha256": "a" * 64}],
                    "opposing_evidence_refs": [],
                    "verification_status": "SUPPORTED",
                    "verification_evidence_refs": [{"source_id": "SRC-2", "filename": "events.log",
                                                    "locator": "line:8", "sha256": "b" * 64}],
                    "verification_counter_evidence_refs": [],
                }],
            }}},
        },
    }


def test_claim_level_expert_feedback_is_append_only_and_revision_is_an_overlay(tmp_path):
    report = governed_claim_report()
    job_id = "JOB-" + "C" * 24
    store = ManualReportReviewStore(tmp_path / "review.sqlite3")
    before = store.detail(job_id, report, include_history=True)
    assert before["claim_review_status"] == "PENDING"
    assert before["claim_total_count"] == 1
    claim_id = before["claims"][0]["claim_id"]
    revised = store.record_claim(
        job_id=job_id, report=report, report_sha256=before["report_sha256"],
        claim_id=claim_id, decision="REVISE", evidence_ref="LAB-RECORD-2026-017",
        note="根据独立时标比对，原表述应限定为时间对齐风险。",
        actor="expert-reviewer", expected_revision=0,
        revised_text="日志时间基准差异可能造成轮速异常与事件记录的错位。",
        revised_epistemic_status="INFERRED",
    )
    assert revised["claim_review_status"] == "REVIEWED"
    review = revised["claims"][0]["review"]
    assert review["decision"] == "REVISE"
    assert review["revised_text"].startswith("日志时间基准差异")
    assert revised["claims"][0]["text"] == "遥测链路延迟可能影响轮速异常的时间对齐。"
    assert len(revised["claim_history"]) == 1
    assert report == governed_claim_report()
    with pytest.raises(ManualReviewConflict):
        store.record_claim(
            job_id=job_id, report=report, report_sha256=before["report_sha256"],
            claim_id=claim_id, decision="ACCEPT", evidence_ref="LAB-RECORD-2026-018",
            note="重复提交旧版本，必须被并发保护拒绝。", actor="expert-reviewer",
            expected_revision=0,
        )
    pending = store.record_claim(
        job_id=job_id, report=report, report_sha256=before["report_sha256"],
        claim_id=claim_id, decision="NEEDS_EVIDENCE", evidence_ref="",
        note="需要补充同一时间窗口的独立星务时钟校准记录。",
        actor="expert-reviewer", expected_revision=revised["claim_revision"],
    )
    assert pending["claim_review_status"] == "REVIEWED_WITH_GAPS"
    assert len(pending["claim_history"]) == 2
    assert report["analysis"]["knowledge_enrichment"]["model_explanation"]["claim_governance"]["claims"][0]["text"] == "遥测链路延迟可能影响轮速异常的时间对齐。"


def test_claim_level_review_api_requires_expert_csrf_and_preserves_report(tmp_path, monkeypatch):
    root = tmp_path / "platform"
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(root))
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEW_TOKEN", "test-material-claim-review-token-123456")
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEWER", "claim-expert")
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEWERS", "claim-expert")
    store = UploadReportJobStore(root)
    job_id = store.create("校验主张级人工复核")["job_id"]
    report = governed_claim_report()
    monkeypatch.setattr(store, "report", lambda requested_job_id: report)
    digest = sha256_json(report)
    claim_id = report["analysis"]["knowledge_enrichment"]["model_explanation"]["claim_governance"]["claims"][0]["claim_id"]
    app = create_app()
    app.state.upload_report_store = store
    with TestClient(app, base_url="http://localhost") as client:
        private = f"/v1/platform/material-knowledge/report-reviews/{job_id}"
        endpoint = f"{private}/claims/{claim_id}/decisions"
        assert client.get(private).status_code == 403
        login = client.post("/v1/platform/material-knowledge/review-session", json={
            "credential": "test-material-claim-review-token-123456",
        })
        assert login.status_code == 200
        csrf = {"X-Material-CSRF": login.json()["csrf"]}
        detail = client.get(private)
        assert detail.status_code == 200
        assert detail.json()["claim_total_count"] == 1
        payload = {
            "report_sha256": digest, "decision": "ACCEPT",
            "evidence_ref": "INDEPENDENT-LAB-CLAIM-002",
            "note": "已核对原始日志与独立试验记录，支持将该内容保留为待核验主张。",
            "expected_revision": 0,
        }
        assert client.post(endpoint, json=payload).status_code == 403
        accepted = client.post(endpoint, headers=csrf, json=payload)
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["claim_review_status"] == "REVIEWED"
        assert accepted.json()["claims"][0]["review"]["actor"] == "claim-expert"
        assert accepted.json()["claims"][0]["review"]["decision"] == "ACCEPT"
        stale = client.post(endpoint, headers=csrf, json=payload)
        assert stale.status_code == 409
        assert report == governed_claim_report()
        assert client.get(private).json()["claim_review_status"] == "REVIEWED"
        assert client.delete("/v1/platform/material-knowledge/review-session", headers=csrf).status_code == 200




def test_claim_feedback_hash_chain_is_append_only_and_exportable(tmp_path):
    import sqlite3
    report = governed_claim_report()
    job_id = "JOB-" + "D" * 24
    path = tmp_path / "review.sqlite3"
    store = ManualReportReviewStore(path)
    detail = store.detail(job_id, report, include_history=True)
    claim = detail["claims"][0]
    claim_id = claim["claim_id"]
    first = store.record_claim(
        job_id=job_id, report=report, report_sha256=detail["report_sha256"],
        claim_id=claim_id, decision="ACCEPT", evidence_ref="INDEPENDENT-LAB-RECORD-2026-01",
        note="已对照独立台架记录和原始日志，保留为经专家审阅的解释主张。",
        actor="reviewer-a", expected_revision=0,
    )
    second = store.record_claim(
        job_id=job_id, report=report, report_sha256=detail["report_sha256"],
        claim_id=claim_id, decision="REVISE", evidence_ref="INDEPENDENT-LAB-RECORD-2026-02",
        note="专家根据额外时序证据缩小主张适用范围。",
        actor="reviewer-b", expected_revision=first["claim_revision"],
        revised_text="时间基准差异可能造成遥测轮速异常与事件记录错位。",
        revised_epistemic_status="INFERRED",
    )
    assert second["claim_history_integrity"] == "VERIFIED"
    assert [x["snapshot_binding_status"] for x in second["claim_history"]] == ["BOUND", "BOUND"]
    assert all(len(x["event_sha256"]) == 64 for x in second["claim_history"])
    # History is newest-first for the review UI; the newest event links to the older hash.
    assert second["claim_history"][0]["previous_event_sha256"] == second["claim_history"][1]["event_sha256"]

    exported = store.export_claim_feedback(job_id=job_id, report=report)
    assert exported["history_integrity"] == "VERIFIED"
    assert exported["event_count"] == 2
    assert exported["automatic_training_authorized"] is False
    assert exported["claim_snapshots"][0]["claim_id"] == claim_id

    with sqlite3.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="APPEND_ONLY"):
            db.execute("UPDATE report_claim_review_events SET note='tampered' WHERE event_id=?",
                       (second["claim_revision"],))
        with pytest.raises(sqlite3.IntegrityError, match="APPEND_ONLY"):
            db.execute("DELETE FROM report_claim_review_events WHERE event_id=?",
                       (second["claim_revision"],))


def test_claim_feedback_tampering_is_detected_and_blocks_writes_and_export(tmp_path):
    import sqlite3
    report = governed_claim_report()
    job_id = "JOB-" + "E" * 24
    path = tmp_path / "review.sqlite3"
    store = ManualReportReviewStore(path)
    detail = store.detail(job_id, report, include_history=True)
    claim_id = detail["claims"][0]["claim_id"]
    recorded = store.record_claim(
        job_id=job_id, report=report, report_sha256=detail["report_sha256"],
        claim_id=claim_id, decision="ACCEPT", evidence_ref="INDEPENDENT-TEST-RECORD-2026",
        note="已通过独立记录复核该主张，保留为经专家审阅内容。",
        actor="reviewer", expected_revision=0,
    )
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER report_claim_review_events_no_update")
        db.execute("UPDATE report_claim_review_events SET note='direct database tampering' WHERE event_id=?",
                   (recorded["claim_revision"],))
    failed = store.detail(job_id, report, include_history=True)
    assert failed["claim_history_integrity"] == "FAILED"
    assert failed["claim_review_status"] == "AUDIT_INTEGRITY_FAILED"
    with pytest.raises(ManualReviewConflict, match="audit chain integrity"):
        store.export_claim_feedback(job_id=job_id, report=report)
    with pytest.raises(ManualReviewConflict, match="audit chain integrity"):
        store.record_claim(
            job_id=job_id, report=report, report_sha256=detail["report_sha256"],
            claim_id=claim_id, decision="ACCEPT", evidence_ref="INDEPENDENT-TEST-RECORD-2027",
            note="第二次写入在审计链异常时必须被拒绝。", actor="reviewer",
            expected_revision=recorded["claim_revision"],
        )


def test_claim_review_schema_migrates_legacy_events_as_unbound_but_verifiable(tmp_path):
    import sqlite3
    path = tmp_path / "legacy.sqlite3"
    report = governed_claim_report()
    detail = ManualReportReviewStore(path).detail("JOB-" + "F" * 24, report, include_history=True)
    claim_id = detail["claims"][0]["claim_id"]
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER report_claim_review_events_no_update")
        db.execute("DROP TRIGGER report_claim_review_events_no_delete")
        db.execute("DROP TABLE report_claim_review_events")
        db.execute("""CREATE TABLE report_claim_review_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL, report_sha256 TEXT NOT NULL, claim_id TEXT NOT NULL,
            decision TEXT NOT NULL, evidence_ref TEXT NOT NULL, note TEXT NOT NULL,
            revised_text TEXT NOT NULL, revised_epistemic_status TEXT NOT NULL,
            actor TEXT NOT NULL, recorded_at TEXT NOT NULL
        )""")
        db.execute("""INSERT INTO report_claim_review_events(
            job_id,report_sha256,claim_id,decision,evidence_ref,note,revised_text,
            revised_epistemic_status,actor,recorded_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)""", (
            "JOB-" + "F" * 24, detail["report_sha256"], claim_id, "ACCEPT",
            "LEGACY-EXPERT-RECORD-001", "旧版复核记录，缺少主张快照哈希但报告哈希仍绑定。",
            "", "", "legacy-reviewer", "2026-01-01T00:00:00+00:00",
        ))
    migrated = ManualReportReviewStore(path).detail("JOB-" + "F" * 24, report, include_history=True)
    assert migrated["claim_history_integrity"] == "VERIFIED"
    assert migrated["claims"][0]["review"]["snapshot_binding_status"] == "LEGACY_UNBOUND"
    assert migrated["claim_review_status"] == "REVIEWED"


def test_claim_feedback_export_api_requires_authenticated_expert_session(tmp_path, monkeypatch):
    root = tmp_path / "platform"
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(root))
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEW_TOKEN", "test-material-export-token-123456")
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEWER", "export-expert")
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEWERS", "export-expert")
    store = UploadReportJobStore(root)
    job_id = store.create("专家反馈导出 API")["job_id"]
    report = governed_claim_report()
    monkeypatch.setattr(store, "report", lambda requested_job_id: report)
    app = create_app()
    app.state.upload_report_store = store
    with TestClient(app, base_url="http://localhost") as client:
        endpoint = f"/v1/platform/material-knowledge/report-reviews/{job_id}/claims/export"
        assert client.get(endpoint).status_code == 403
        login = client.post("/v1/platform/material-knowledge/review-session",
                            json={"credential": "test-material-export-token-123456"})
        assert login.status_code == 200
        response = client.get(endpoint)
        assert response.status_code == 200, response.text
        assert response.json()["event_count"] == 0
        assert response.json()["automatic_training_authorized"] is False
