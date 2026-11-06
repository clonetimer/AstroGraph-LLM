"""External-text import, actual model transport, and exported report contracts."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from pathlib import Path
import threading
from zipfile import ZipFile

from fastapi.testclient import TestClient
import pytest

from astrograph.api import create_app
from astrograph.integrations.platform_cli import main as cli_main
from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService,
)
from astrograph.integrations.result_log_review import load_manifest, read_lines
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.llm.openai_compatible import (
    OpenAICompatibleConfig, OpenAICompatibleStructuredBackend,
)


def manifest(**extra):
    return {"schema_version": "astrograph-result-log-review.v1",
            "analysis_mode": "EXTERNAL_DIAGNOSIS_ANALYSIS",
            "diagnosis_results": ["diagnosis.txt"], **extra}


def stage(store, text="冷却组件的响应变慢，具体原因尚未确定。", *, job_options=None, **extra):
    job = store.create("分析外部报告中的可能原因及需要补充的信息。", **(job_options or {}))
    store.put_file(job["job_id"], "diagnosis.txt", text.encode("utf-8"))
    store.put_file(job["job_id"], "analysis_input.json", json.dumps(manifest(**extra)).encode())
    return job["job_id"]


def answer(payload):
    sid = payload["sources"][0]["source_id"]
    return {
        "summary": "模型分析：响应变慢可能与散热或传感链路有关，目前无法区分。",
        "findings": [{"text": "报告记录了响应变慢，但没有提供事件时间窗。", "epistemic_status": "REPORTED", "citation_ids": [sid]}],
        "hypotheses": [{"title": "传感链路延迟", "analysis": "这是可核查的解释，需对照独立读数。", "citation_ids": [sid]}],
        "information_gaps": ["缺少独立读数与事件时间。"],
        "recommended_review_actions": ["比对独立读数与响应时间。"],
    }


def test_manifest_only_requires_diagnosis_and_explicit_missing_optional_still_fails(tmp_path):
    (tmp_path / "diagnosis.txt").write_text("外部报告原文", encoding="utf-8")
    path = tmp_path / "analysis_input.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    value, roles = load_manifest(tmp_path, path)
    assert value.analysis_mode == "EXTERNAL_DIAGNOSIS_ANALYSIS"
    assert roles["log_reports"] == roles["references"] == []
    path.write_text(json.dumps(manifest(log_reports=["missing.log"])), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        load_manifest(tmp_path, path)
    path.write_text(json.dumps(manifest(diagnosis_results=[])), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(tmp_path, path)


@pytest.mark.parametrize("suffix,raw", [
    ("txt", "长行" * 1600 + "关键尾部"),
    ("csv", "字段," + "数据" * 400 + ",关键尾部"),
])
def test_parsed_text_does_not_silently_discard_tail(tmp_path, suffix, raw):
    source = tmp_path / ("source." + suffix)
    source.write_text(raw, encoding="utf-8")
    lines = read_lines(source, "diagnosis_results")
    assert "关键尾部" in lines[0].text
    assert raw.split(",")[1] in lines[0].text if suffix == "csv" else lines[0].text == raw


@pytest.mark.parametrize("blocks,expected", [
    ([('heading', 'DIAGNOSIS: coolant pump failure'), ('text', 'Investigate the condition described above.')],
     ['DIAGNOSIS: coolant pump failure', 'Investigate the condition described above.']),
    ([('heading', 'DIAGNOSIS: coolant pump failure')], ['DIAGNOSIS: coolant pump failure']),
    ([('text', 'External diagnosis report'), ('heading', 'First diagnosis'), ('table', '50'),
      ('heading', 'Second diagnosis'), ('text', 'Review both findings.')],
     ['External diagnosis report', 'First diagnosis', '| 50 |', 'Second diagnosis', 'Review both findings.']),
])
def test_docx_headings_and_body_reach_model_once_in_original_order(tmp_path, blocks, expected):
    body = []
    for kind, text in blocks:
        if kind == 'table':
            body.append(f'<w:tbl><w:tr><w:tc><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')
        else:
            style = '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>' if kind == 'heading' else ''
            body.append(f'<w:p>{style}<w:r><w:t>{text}</w:t></w:r></w:p>')
    namespace = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('word/document.xml', f'<w:document xmlns:w="{namespace}"><w:body>{"".join(body)}</w:body></w:document>')
        archive.writestr('word/styles.xml', f'<w:styles xmlns:w="{namespace}"><w:style w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style></w:styles>')
    backend = CaptureBackend()
    store = UploadReportJobStore(tmp_path / 'jobs', knowledge_service=
        PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=backend))
    job_id = store.create('Analyse the external diagnosis.')['job_id']
    store.put_file(job_id, 'diagnosis.docx', buffer.getvalue())
    store.put_file(job_id, 'analysis_input.json', json.dumps(manifest(diagnosis_results=['diagnosis.docx'])).encode())
    assert store.preflight(job_id)['can_start'] is True
    report = store.run(job_id)
    assert report['analysis']['knowledge_enrichment']['generation']['status'] == 'SUCCEEDED'
    sources = backend.calls[0][1]['sources']
    assert [source['text'] for source in sources] == expected
    assert len({source['locator'] for source in sources}) == len(expected)
    for index, (kind, _text) in enumerate(blocks, 1):
        if kind != 'table':
            assert sources[index - 1]['locator'] == f'word/document.xml#paragraph:{index}'


class CaptureBackend:
    def __init__(self, model_id="primary"):
        self.model_id = model_id
        self.provider = "test-provider"
        self.calls = []

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append((agent_id, payload))
        return output_model.model_validate(answer(payload))


class CaptureVerifierBackend:
    model_id = "reviewer-model"
    provider = "test-provider"

    def __init__(self):
        self.calls = []

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append((agent_id, payload))
        source_id = payload["sources"][0]["source_id"]
        return output_model.model_validate({"assessments": [{
            "claim_id": claim["claim_id"],
            "support_status": "SUPPORTED",
            "rationale": "独立审阅在原始来源中找到了对该表述的直接支持。",
            "supporting_source_ids": [source_id],
            "counter_evidence_source_ids": [],
            "information_gaps": [],
        } for claim in payload["claims"]]})


def test_external_mode_bypasses_diagnosis_and_graph_and_keeps_model_output(tmp_path, monkeypatch):
    import astrograph.integrations.result_log_review as legacy
    import astrograph.integrations.material_reasoning as material

    def forbidden(*args, **kwargs):
        raise AssertionError("External text analysis invoked legacy diagnosis/graph processing")

    monkeypatch.setattr(legacy, "analyze_result_log_review", forbidden)
    monkeypatch.setattr(material, "analyze_material_knowledge", forbidden)
    backend = CaptureBackend()
    service = PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=backend)
    store = UploadReportJobStore(tmp_path / "jobs", knowledge_service=service)
    original = "未登记组件响应变慢；无故障编码。\n还需要补充独立时序。"
    job_id = stage(store, original)
    assert store.preflight(job_id)["can_start"] is True
    report = store.run(job_id)
    assert report["status"] == "SUCCEEDED"
    assert report["analysis"]["analysis_mode"] == "EXTERNAL_DIAGNOSIS_ANALYSIS"
    assert report["analysis"]["result_log_review"]["candidates"] == []
    assert report["analysis"]["effect_guard"]["raw_telemetry_detection_executed"] is False
    assert len(backend.calls) == 1 and backend.calls[0][0] == "external-diagnosis-analysis"
    assert "\n".join(s["text"] for s in backend.calls[0][1]["sources"]) == original
    model = report["analysis"]["knowledge_enrichment"]["model_explanation"]
    assert model["status"] == "AVAILABLE_REQUIRES_EXPERT_REVIEW"
    assert model["summary"] == answer(backend.calls[0][1])["summary"]
    assert report == store.report(job_id)
    md = store.report_markdown_path(job_id).read_text(encoding="utf-8")
    for text in [model["summary"], model["hypotheses"][0]["analysis"], original.splitlines()[0],
                 model["information_gaps"][0], model["recommended_review_actions"][0]]:
        assert text in md
    assert model["findings"][0]["citation_ids"][0] in md


def test_no_model_preserves_import_and_exposes_actual_status(tmp_path):
    service = PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig())
    store = UploadReportJobStore(tmp_path / "jobs", knowledge_service=service)
    job_id = stage(store)
    report = store.run(job_id)
    assert report["status"] == "SUCCEEDED"  # Import/report completion is not model success.
    knowledge = report["analysis"]["knowledge_enrichment"]
    assert knowledge["generation"]["status"] == "NOT_CONFIGURED"
    assert knowledge["model_explanation"]["status"] == "NOT_CONFIGURED"
    assert "NOT_CONFIGURED" in store.report_markdown_path(job_id).read_text(encoding="utf-8")
    assert store.diagnostics(job_id)["llm_call_status"] == "NOT_CONFIGURED"


def test_preflight_and_overview_identify_external_mode_and_model_status(tmp_path):
    store = UploadReportJobStore(tmp_path / "jobs", knowledge_service=
        PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig()))
    job_id = stage(store)
    details = store.preflight(job_id)["input_contract"]["details"]
    assert details["analysis_mode"] == "EXTERNAL_DIAGNOSIS_ANALYSIS"
    assert details["material_review_only"] is False
    report = store.run(job_id)
    summary = store._overview_report_summary(report)
    assert summary["llm_call_status"] == "NOT_CONFIGURED"
    assert summary["diagnosis_disposition"] == "MODEL_NOT_CONFIGURED"


def test_cli_short_summary_keeps_model_status_distinct_from_saved_report(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ASTROGRAPH_PLATFORM_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ASTROGRAPH_PLATFORM_LLM_MODEL", raising=False)
    source = tmp_path / "diagnosis.txt"
    source.write_text("未登记组件响应变慢。", encoding="utf-8")
    assert cli_main(["--root", str(tmp_path / "jobs"), "analyze-report", "--result", str(source)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "SUCCEEDED"
    assert summary["llm_call_status"] == "NOT_CONFIGURED"
    assert summary["analysis_mode"] == "EXTERNAL_DIAGNOSIS_ANALYSIS"


def test_analyze_report_cli_accepts_one_external_text(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ASTROGRAPH_PLATFORM_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ASTROGRAPH_PLATFORM_LLM_MODEL", raising=False)
    source = tmp_path / "report.md"
    source.write_text("# 诊断结果\n观察到未登记组件响应变慢。", encoding="utf-8")
    rc = cli_main(["--root", str(tmp_path / "jobs"), "analyze-report", "--result", str(source), "--full"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert report["analysis"]["analysis_mode"] == "EXTERNAL_DIAGNOSIS_ANALYSIS"
    assert report["analysis"]["knowledge_enrichment"]["generation"]["status"] == "NOT_CONFIGURED"


@contextmanager
def model_wire():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.loads(request["messages"][-1]["content"])["payload"]
            calls.append({"path": self.path, "request": request, "payload": payload})
            body = json.dumps({"choices": [{"message": {"content": json.dumps(answer(payload), ensure_ascii=False)},
                                            "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_api_upload_to_actual_http_transport_to_json_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "disabled")
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    with model_wire() as (url, calls):
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=url, model="scripted-protocol-test-not-a-real-model",
            response_format_strategy="json_object", tokenizer_mode="disabled",
            repair_attempts=0, timeout_s=10, budget_preflight=False,
        ))
        store = UploadReportJobStore(tmp_path / "jobs", knowledge_service=
            PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=backend))
        app = create_app()
        app.state.upload_report_store = store
        with TestClient(app) as client:
            created = client.post("/v1/platform/jobs", json={"task": "分析这份外部诊断文本。"})
            assert created.status_code == 200
            job_id = created.json()["job_id"]
            prefix = f"/v1/platform/jobs/{job_id}"
            original = "未登记组件响应变慢。" + "数据" * 1400 + "关键末尾线索。"
            for name, blob in {"diagnosis.txt": original.encode(),
                               "analysis_input.json": json.dumps(manifest()).encode()}.items():
                assert client.put(prefix + "/files/" + name, content=blob).status_code == 200
            assert client.post(prefix + "/preflight").json()["can_start"] is True
            executed = client.post(prefix + "/run")
            assert executed.status_code == 200 and executed.json()["status"] == "SUCCEEDED"
            report = client.get(prefix + "/report").json()
            markdown = client.get(prefix + "/report.md")
            assert markdown.status_code == 200
            assert len(calls) == 1 and calls[0]["path"] == "/v1/chat/completions"
            assert calls[0]["payload"]["sources"][0]["text"] == original
            expected = answer(calls[0]["payload"])
            model = report["analysis"]["knowledge_enrichment"]["model_explanation"]
            assert model["summary"] == expected["summary"]
            assert expected["hypotheses"][0]["analysis"] in markdown.text
            assert model["sources"][0]["source_id"] in markdown.text
            assert store.diagnostics(job_id)["llm_call_status"] == "SUCCEEDED"


def test_external_claim_registry_flows_into_append_only_expert_review(tmp_path):
    from astrograph.integrations.manual_report_review import ManualReportReviewStore, report_claims

    backend = CaptureBackend()
    store = UploadReportJobStore(
        tmp_path / "platform",
        knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=backend),
    )
    job_id = stage(store, "报告明确记录响应变慢，但尚无独立时序。")
    report = store.run(job_id)
    model = report["analysis"]["knowledge_enrichment"]["model_explanation"]
    governance = model["claim_governance"]

    assert governance["status"] == "READY_FOR_REVIEW"
    assert governance["verification"]["status"] == "NOT_CONFIGURED"
    assert governance["verification"]["fact_truth_certified"] is False
    claims = report_claims(report)
    assert claims
    assert any(item["epistemic_status"] == "REPORTED" for item in claims)
    assert all(item["expert_review_status"] == "PENDING" for item in governance["claims"])

    review_store = ManualReportReviewStore(store.root / "material_knowledge" / "manual_report_reviews.sqlite3")
    detail = review_store.detail(job_id, report, include_history=True)
    assert detail["claim_total_count"] == len(claims)
    claim = next(item for item in detail["claims"] if item["epistemic_status"] == "REPORTED")
    before = json.dumps(report, ensure_ascii=False, sort_keys=True)

    reviewed = review_store.record_claim(
        job_id=job_id, report=report, report_sha256=detail["report_sha256"],
        claim_id=claim["claim_id"], decision="REVISE",
        evidence_ref="INDEPENDENT-TEST-LOG-2026-11",
        note="专家核对了独立日志，修订该主张的适用时间范围。",
        actor="domain-reviewer", expected_revision=0,
        revised_text="原始报告在给定时间窗内记录了响应变慢。",
        revised_epistemic_status="REPORTED",
    )
    assert reviewed["claim_review_status"] == "IN_PROGRESS"
    assert reviewed["claim_reviewed_count"] == 1
    assert reviewed["claim_history_integrity"] == "VERIFIED"
    reviewed_claim = next(item for item in reviewed["claims"] if item["claim_id"] == claim["claim_id"])
    assert reviewed_claim["review"]["decision"] == "REVISE"
    assert json.dumps(report, ensure_ascii=False, sort_keys=True) == before

    feedback = review_store.export_claim_feedback(job_id=job_id, report=report)
    assert feedback["history_integrity"] == "VERIFIED"
    assert feedback["automatic_training_authorized"] is False
    assert feedback["event_count"] == 1


def test_external_claim_review_api_uses_authenticated_expert_session(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "platform"))
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEW_TOKEN", "external-claim-review-token-2026")
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEWER", "domain-expert")
    monkeypatch.setenv("ASTROGRAPH_MATERIAL_REVIEWERS", "domain-expert")

    from astrograph.integrations import upload_report

    backend = CaptureBackend("analysis-model")
    verifier = CaptureVerifierBackend()
    store = UploadReportJobStore(
        tmp_path / "platform",
        knowledge_service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(), llm_backend=backend),
    )
    profile = store.state_repository.save_service_profile(
        service_kind="llm", name="independent-reviewer", validation_status="PASS",
        config={"enabled": True, "base_url": "http://127.0.0.1:1234/v1",
                "model": "reviewer-model", "provider": "test-provider"},
    )
    monkeypatch.setattr(upload_report, "OpenAICompatibleStructuredBackend", lambda config: verifier)
    job_id = stage(
        store, "报告记录组件响应异常，尚待独立验证。",
        job_options={"phase_engine_ids": {"verification": "llm:" + profile["profile_id"]}},
    )
    report = store.run(job_id)
    governance = report["analysis"]["knowledge_enrichment"]["model_explanation"]["claim_governance"]
    assert governance["verification"]["status"] == "COMPLETED_REQUIRES_EXPERT_REVIEW"
    assert governance["verification"]["verifier_model"] == "reviewer-model"
    assert all(item["expert_review_status"] == "PENDING" for item in governance["claims"])
    assert verifier.calls and verifier.calls[0][0] == "external-diagnosis-evidence-verifier"
    app = create_app()
    app.state.upload_report_store = store

    with TestClient(app, base_url="http://localhost") as client:
        private = f"/v1/platform/material-knowledge/report-reviews/{job_id}"
        assert client.get(private).status_code == 200
        detail = client.get(private).json()
        assert detail["claim_total_count"] >= 1
        claim_id = detail["claims"][0]["claim_id"]
        login = client.post("/v1/platform/material-knowledge/review-session", json={
            "credential": "external-claim-review-token-2026",
        })
        assert login.status_code == 200
        headers = {"X-Material-CSRF": login.json()["csrf"]}
        accepted = client.post(
            f"{private}/claims/{claim_id}/decisions",
            headers=headers,
            json={
                "report_sha256": detail["report_sha256"],
                "decision": "ACCEPT",
                "evidence_ref": "INDEPENDENT-TEST-REPORT-2026",
                "note": "已对照独立试验记录，允许保留为经专家审阅的解释主张。",
                "expected_revision": 0,
            },
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["claim_review_status"] in {"IN_PROGRESS", "REVIEWED"}
        refreshed = client.get(private).json()
        assert refreshed["claim_history_integrity"] == "VERIFIED"
        assert any(item.get("review", {}).get("decision") == "ACCEPT" for item in refreshed["claims"])
        assert store.report(job_id) == report
