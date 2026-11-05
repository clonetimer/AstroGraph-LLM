"""Portable-subset tests; no full project dependencies and no external model calls."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import zipfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
STAGE = tempfile.TemporaryDirectory(prefix="astrograph-portable-tests-")
spec = importlib.util.spec_from_file_location("portable_build", HERE / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
builder.stage_app(HERE.parent, Path(STAGE.name))
sys.path.insert(0, STAGE.name)
PRELOADED_MODULES = set(sys.modules)
import workbench as app


def file_input(text="外部诊断：设备温度偏高。\n日志：2026-11-01 温度报警。", name="诊断.txt", encoding="utf-8"):
    raw = text.encode(encoding)
    return {"filename": name, "role": "diagnosis_results",
            "content_base64": base64.b64encode(raw).decode("ascii"),
            "encoding": "gb18030" if encoding == "gb18030" else "auto"}


def binary_input(raw, name, role="diagnosis_results"):
    return {"filename": name, "role": role,
            "content_base64": base64.b64encode(raw).decode("ascii"), "encoding": "auto"}


def make_docx():
    import io
    document = b'''<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Diagnosis heading</w:t></w:r></w:p><w:p><w:r><w:t>Body evidence paragraph.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Temperature</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>87</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'''
    styles = b'''<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style></w:styles>'''
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
    return output.getvalue()


def make_pdf(text="Temperature anomaly at page one."):
    """Create a real, minimal PDF without ReportLab's post-3.8 hash API."""
    import io
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        literal = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        stream.set_data(('BT /F1 10 Tf 30 250 Td (' + literal + ') Tj ET').encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@contextmanager
def model_server(mode="good", delay=0, ipv6=False):
    state = {"calls": 0, "bodies": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_json({"data": [{"id": "test-model"}]})

        def send_json(self, value):
            raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"] += 1
            state["bodies"].append(body)
            time.sleep(delay)
            request = json.loads(body["messages"][-1]["content"])
            payload = request["payload"]
            sid = payload["sources"][0]["source_id"]
            if mode == "bad_citation":
                sid = "source-not-in-this-input"
            if request["task"] == "ExternalDiagnosisFollowUp":
                answer = {"answer": [{"text": "需要结合原文进一步核查。", "citation_ids": [sid]}],
                          "information_gaps": ["缺少相邻时刻日志"]}
            else:
                answer = {"summary": "报告记录了温度异常。<script>alert(1)</script>",
                          "findings": [{"text": "上游报告称温度偏高。", "citation_ids": [sid]}],
                          "hypotheses": [], "information_gaps": ["未提供温度基线"],
                          "recommended_review_actions": ["核对原始记录"], "review_actions": []}
                diagnosis_files = {source["filename"] for source in payload["sources"]
                                   if source.get("role") == "diagnosis_results"}
                if len(diagnosis_files) > 1:
                    answer.update(
                        common_findings=[{"text": "两份报告均记录温度异常。", "citation_ids": [sid]}],
                        report_disagreements=[{"text": "两份报告的异常发生时间不同，需分开核对。", "citation_ids": [sid]}],
                        complementary_evidence=[{"text": "一份报告提供温度，另一份提供姿态状态。", "citation_ids": [sid]}],
                        event_alignment=[], integrated_conclusion="现有材料支持存在温度异常，但不能据此合并不同时间事件。")
            content = "not-json" if mode == "bad_json" else json.dumps(answer, ensure_ascii=False)
            message = {"role": "assistant", "content": content}
            if mode == "refusal":
                message["refusal"] = "declined"
            self.send_json({"choices": [{"message": message,
                                          "finish_reason": "length" if mode == "truncated" else "stop"}]})

    class IPv6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    try:
        server = IPv6Server(("::1", 0), Handler) if ipv6 else ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError:
        if ipv6:
            raise unittest.SkipTest("IPv6 loopback unavailable on this host")
        raise
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = "[::1]" if ipv6 else "127.0.0.1"
        state["base_url"] = f"http://{host}:{server.server_port}/v1"
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def connection(base_url="http://127.0.0.1:1234/v1"):
    return {"mode": "local", "base_url": base_url, "model": "test-model",
            "api_key": "private-key-must-not-persist", "max_tokens": 4096,
            "timeout_s": 30, "response_format_strategy": "plain_json"}


def wait_job(store, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = store.read_job(job_id)
        if response["job"]["status"] != "RUNNING":
            return response
        time.sleep(0.02)
    raise AssertionError("job did not complete")


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="诊断 workbench ")
        self.store = app.Workbench(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, conn=None, files=None):
        return {"task": "解释诊断报告，保留不确定性", "files": files or [file_input()],
                "connection": conn or connection()}

    def test_minimal_imports_do_not_load_engine_or_ml(self):
        forbidden = ["numpy", "pandas", "torch", "transformers", "langgraph", "astrograph.diagnosis"]
        newly_loaded = set(sys.modules) - PRELOADED_MODULES
        self.assertFalse([name for name in forbidden if name in newly_loaded])

    def test_offline_self_test_completes_using_real_worker(self):
        # The shared test runner may preload analytics packages globally. The
        # portable process starts from an isolated embedded runtime instead.
        held = {name: sys.modules.pop(name) for name in ("numpy", "pandas") if name in sys.modules}
        try:
            result = app.self_test()
        finally:
            sys.modules.update(held)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["real_model_called"])
        self.assertFalse(result["internet_required"])

    def test_local_and_online_endpoint_validation(self):
        self.assertEqual(app.parse_connection(connection())["mode"], "local")
        bad = [
            {"base_url": "http://example.com/v1"},
            {"base_url": "file:///tmp/secret"},
            {"base_url": "http://user:pass@127.0.0.1/v1"},
            {"base_url": "http://127.0.0.1/v1?api_key=hidden"},
            {"mode": "online", "base_url": "http://example.com/v1"},
            {"timeout_s": 0}, {"max_tokens": True}, {"model": " "},
        ]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(app.WorkbenchError):
                app.parse_connection({**connection(), **change})
        self.assertEqual(app.parse_connection({**connection(), "mode": "online",
                                              "base_url": "https://example.com/v1"})["mode"], "online")

    def test_unsafe_and_duplicate_filenames_rejected(self):
        for name in ["../x.txt", "CON.txt", "lpt1.md", "a:b.txt", "a?.txt", "x.txt.", "bad\x00.txt"]:
            with self.subTest(name=name), self.assertRaises(app.WorkbenchError):
                self.store.create_job(self.payload(files=[file_input(name=name)]))
        with self.assertRaises(app.WorkbenchError):
            self.store.create_job(self.payload(files=[file_input(name="A.txt"), file_input(name="a.TXT")]))

    def test_manifest_filename_is_reserved_case_insensitively(self):
        for name in ("analysis_input.json", "ANALYSIS_INPUT.JSON", "Analysis_Input.Json"):
            with self.subTest(name=name), self.assertRaises(app.WorkbenchError):
                app._safe_filename(name)

    def test_empty_invalid_json_and_oversize_are_rejected_before_model(self):
        for material in [file_input(" "), file_input("{bad", "invalid.json"),
                         file_input("x" * 180001), file_input("abc", "sample.pdf")]:
            with self.subTest(name=material["filename"]), self.assertRaises(app.WorkbenchError):
                self.store.create_job(self.payload(files=[material]))
        invalid = file_input()
        invalid["content_base64"] = "not base64"
        with self.assertRaises(app.WorkbenchError):
            self.store.create_job(self.payload(files=[invalid]))

    def test_native_docx_import_preserves_heading_body_and_table_locators(self):
        with model_server() as model:
            original_bytes = make_docx()
            item = binary_input(original_bytes, "diagnosis.docx")
            job = self.store.create_job(self.payload(connection(model["base_url"]), [item]))
            result = wait_job(self.store, job["job_id"])
            self.assertEqual(result["job"]["status"], "SUCCEEDED")
            sources = result["report"]["analysis"]["knowledge_enrichment"]["model_explanation"]["sources"]
            text = "\n".join(source["text"] for source in sources)
            self.assertIn("Diagnosis heading", text)
            self.assertIn("Body evidence paragraph.", text)
            self.assertIn("Temperature", text)
            self.assertEqual((self.store._job_root(job["job_id"]) / "input_original" / "diagnosis.docx").read_bytes(), original_bytes)

    def test_pdf_import_extracts_page_locators_and_scanned_pdf_requires_ocr(self):
        with model_server() as model:
            item = binary_input(make_pdf(), "diagnosis.pdf")
            job = self.store.create_job(self.payload(connection(model["base_url"]), [item]))
            result = wait_job(self.store, job["job_id"])
            self.assertEqual(result["job"]["status"], "SUCCEEDED")
            sources = result["report"]["analysis"]["knowledge_enrichment"]["model_explanation"]["sources"]
            self.assertTrue(any("pdf#page:1" in source["locator"] for source in sources))
            with self.assertRaises(app.WorkbenchError) as error:
                self.store.create_job(self.payload(connection(model["base_url"]), [binary_input(make_pdf(""), "scan.pdf")]))
            self.assertEqual(error.exception.code, "DOCUMENT_NEEDS_OCR")

    def test_encrypted_pdf_is_rejected_before_model_call(self):
        from pypdf import PdfWriter
        import io
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret")
        output = io.BytesIO()
        writer.write(output)
        with self.assertRaises(app.WorkbenchError):
            self.store.create_job(self.payload(files=[binary_input(output.getvalue(), "encrypted.pdf")]))

    def test_long_report_chunk_planner_covers_every_character_and_keeps_source_locators(self):
        from astrograph.integrations.long_report_analysis import plan_analysis_chunks
        source = {"source_id": "source-original", "role": "diagnosis_results", "filename": "long.txt",
                  "locator": "line:1", "sha256": "a" * 64, "text": ("段落甲。" * 1800) + "\n" + ("段落乙。" * 1800)}
        chunks = plan_analysis_chunks([source], max_chars=8000, overlap_chars=300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.source_id == "source-original" for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 8000 for chunk in chunks))
        covered = [False] * len(source["text"])
        for chunk in chunks:
            for index in range(chunk.start_offset, chunk.end_offset):
                covered[index] = True
        self.assertTrue(all(covered))
        self.assertTrue(all(chunk.text_hash for chunk in chunks))

    def test_long_report_resume_reuses_completed_chunks_after_interruption(self):
        from astrograph.integrations.long_report_analysis import run_hierarchical_analysis
        from astrograph.integrations.external_diagnosis_analysis import _digest, _source_id
        text = "温度异常与姿态偏差需要分别核对。" * 1100
        source = {"source_id": "source-original", "role": "diagnosis_results", "filename": "长报告.txt",
                  "locator": "line:1-1100", "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
        source["source_id"] = _source_id("resume-task", source)
        bundle = {"schema_version": "astrograph-external-diagnosis-input.v1", "task_id": "resume-task",
                  "input_sha256": _digest([source]), "sources": [source], "character_count": len(text)}
        class FlakyBackend:
            def __init__(self): self.calls = {}; self.failed_once = False
            def generate_structured(self, *, agent_id, output_model, payload):
                task = payload["task"]
                chunk = next((i for i in range(1, 10) if f"第 {i}/" in task), None)
                if chunk:
                    self.calls[chunk] = self.calls.get(chunk, 0) + 1
                    if chunk == 2 and not self.failed_once:
                        self.failed_once = True
                        raise RuntimeError("simulated interruption")
                sid = payload["sources"][0]["source_id"]
                return {"summary": "片段分析完成。", "findings": [{"text": "报告记录温度异常。", "citation_ids": [sid]}],
                        "hypotheses": [], "information_gaps": [], "recommended_review_actions": [],
                        "review_actions": [], "common_findings": [], "report_disagreements": [],
                        "complementary_evidence": [], "event_alignment": [], "integrated_conclusion": ""}
        backend = FlakyBackend(); output = Path(self.temp.name) / "resumable-analysis"
        first = run_hierarchical_analysis(task_id="resume-task", task="分析长报告", input_bundle=bundle,
                                          llm_backend=backend, output_root=output)
        self.assertEqual(first["generation"]["status"], "FAILED")
        self.assertEqual(backend.calls.get(1), 1)
        second = run_hierarchical_analysis(task_id="resume-task", task="分析长报告", input_bundle=bundle,
                                           llm_backend=backend, output_root=output)
        self.assertEqual(second["generation"]["status"], "SUCCEEDED")
        self.assertEqual(backend.calls.get(1), 1, "completed first chunk must not be called again")
        self.assertEqual(backend.calls.get(2), 2, "failed chunk is retried")

    def test_long_report_end_to_end_uses_chunks_and_final_original_citations(self):
        with model_server() as model:
            text = ("外部诊断记录：温度上升，需要结合上下文分析。\n" * 3200)
            item = file_input(text, "long-diagnosis.txt")
            job = self.store.create_job(self.payload(connection(model["base_url"]), [item]))
            result = wait_job(self.store, job["job_id"])
            self.assertEqual(result["job"]["status"], "SUCCEEDED")
            explanation = result["report"]["analysis"]["knowledge_enrichment"]["model_explanation"]
            self.assertEqual(explanation["citation_reference_validation"], "PASS")
            self.assertGreater(explanation["processing"]["chunk_count"], 1)
            allowed = {source["source_id"] for source in explanation["sources"]}
            for finding in explanation["findings"]:
                self.assertTrue(set(finding["citation_ids"]).issubset(allowed))
            self.assertGreater(model["calls"], explanation["processing"]["chunk_count"])
            conversation = self.store.follow_up(job["job_id"], {
                "question": "综合分析中哪些片段支持温度异常？", "input_sha256": job["input_sha256"],
                "request_id": "long-report-follow-up-1", "connection": connection(model["base_url"])})
            self.assertEqual(conversation["turns"][0]["status"], "SUCCEEDED")

    def test_multiple_reports_return_comparison_sections_with_citations(self):
        with model_server() as model:
            files = [file_input("2026-10-01：温度上升，持续 10 分钟。", "thermal.txt"),
                     file_input("2026-10-03：姿态异常，与温度记录时间不同。", "adcs.txt")]
            job = self.store.create_job(self.payload(connection(model["base_url"]), files))
            result = wait_job(self.store, job["job_id"])
            explanation = result["report"]["analysis"]["knowledge_enrichment"]["model_explanation"]
            self.assertEqual(explanation["citation_reference_validation"], "PASS")
            self.assertTrue(explanation["common_findings"])
            self.assertTrue(explanation["report_disagreements"])
            self.assertTrue(explanation["complementary_evidence"])
            self.assertIn("不能据此合并", explanation["integrated_conclusion"])
            allowed = {source["source_id"] for source in explanation["sources"]}
            for field in ("common_findings", "report_disagreements", "complementary_evidence", "event_alignment"):
                for item in explanation[field]:
                    self.assertTrue(set(item["citation_ids"]).issubset(allowed))
            markdown = self.store.export(job["job_id"], "report.md")[0].decode("utf-8")
            self.assertIn("跨报告共识", markdown)
            self.assertIn("报告间分歧", markdown)

    def test_real_transport_report_followup_and_no_persisted_credentials(self):
        with model_server() as model:
            conn = connection(model["base_url"])
            job = self.store.create_job(self.payload(conn))
            response = wait_job(self.store, job["job_id"])
            knowledge = response["report"]["analysis"]["knowledge_enrichment"]
            self.assertEqual(knowledge["generation"]["status"], "SUCCEEDED")
            explanation = knowledge["model_explanation"]
            self.assertEqual(explanation["citation_reference_validation"], "PASS")
            self.assertIn("温度偏高", explanation["sources"][0]["text"])
            self.assertNotIn("<script>", self.store.export(job["job_id"], "report.md")[0].decode())
            request = {"question": "还缺什么材料？", "input_sha256": explanation["input_sha256"],
                       "request_id": "test-request-0001", "connection": conn}
            state = self.store.follow_up(job["job_id"], request)
            self.assertEqual(state["turns"][0]["status"], "SUCCEEDED")
            calls = model["calls"]
            self.assertEqual(self.store.follow_up(job["job_id"], request), state)
            self.assertEqual(model["calls"], calls)
            changed = {**request, "question": "different"}
            with self.assertRaises(app.WorkbenchError):
                self.store.follow_up(job["job_id"], changed)
            restored = app.Workbench(Path(self.temp.name))
            self.assertEqual(restored.conversation(job["job_id"]), state)
            for path in Path(self.temp.name).rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"private-key-must-not-persist", path.read_bytes(), str(path))
            self.assertEqual(self.store.probe(conn)["models"], [{"id": "test-model"}])

    def test_chinese_gb18030_original_bytes_and_utf16_bom(self):
        with model_server() as model:
            for encoding in ["gb18030", "utf-16"]:
                text = "外部诊断报告：转速下降，原因待核查。"
                item = file_input(text, encoding=encoding)
                job = self.store.create_job(self.payload(connection(model["base_url"]), [item]))
                result = wait_job(self.store, job["job_id"])
                source = result["report"]["analysis"]["knowledge_enrichment"]["model_explanation"]["sources"][0]
                self.assertEqual(source["text"], text)
                raw = (self.store._job_root(job["job_id"]) / "input_original" / item["filename"]).read_bytes()
                self.assertEqual(raw, text.encode(encoding))
                self.assertEqual(source["sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest())

    def test_invalid_model_output_never_appears_as_success(self):
        for mode in ["bad_citation", "bad_json", "refusal", "truncated"]:
            with self.subTest(mode=mode), model_server(mode) as model:
                job = self.store.create_job(self.payload(connection(model["base_url"])))
                result = wait_job(self.store, job["job_id"])
                generation = result["report"]["analysis"]["knowledge_enrichment"]["generation"]
                self.assertEqual(generation["status"], "FAILED")
                self.assertNotIn("summary", result["report"]["analysis"]["knowledge_enrichment"]["model_explanation"])

    def test_busy_job_is_not_queued_or_replayed(self):
        with model_server(delay=0.25) as model:
            payload = self.payload(connection(model["base_url"]))
            job = self.store.create_job(payload)
            with self.assertRaises(app.WorkbenchError) as error:
                self.store.create_job(payload)
            self.assertEqual(error.exception.code, "MODEL_BUSY")
            wait_job(self.store, job["job_id"])

    def test_unknown_job_and_changed_source_fail_closed(self):
        with self.assertRaises(app.WorkbenchError):
            self.store.read_job("../outside")
        with model_server() as model:
            job = self.store.create_job(self.payload(connection(model["base_url"])))
            wait_job(self.store, job["job_id"])
            path = self.store._job_root(job["job_id"]) / "output" / "external_diagnosis_input.json"
            bundle = json.loads(path.read_text("utf-8"))
            bundle["sources"][0]["text"] = "changed"
            path.write_text(json.dumps(bundle), "utf-8")
            with self.assertRaises(app.WorkbenchError):
                self.store.conversation(job["job_id"])

    def test_interrupted_job_resumes_with_same_model_identity_and_saved_input(self):
        from astrograph.integrations.external_diagnosis_analysis import _digest, _source_id
        job_id = "e" * 32
        root = self.store._job_root(job_id); (root / "output").mkdir(parents=True)
        text = "上游诊断报告：温度异常，需要核对。"
        source = {"source_id": "source-resume", "role": "diagnosis_results", "filename": "resume.txt",
                  "locator": "line:1", "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
        source["source_id"] = _source_id(job_id, source)
        bundle = {"schema_version": "astrograph-external-diagnosis-input.v1", "task_id": job_id,
                  "input_sha256": _digest([source]), "sources": [source], "character_count": len(text)}
        settings = connection()
        job = {"job_id": job_id, "task": "继续分析", "created_at": "2026-11-03T00:00:00Z",
               "status": "INTERRUPTED", "model": settings["model"], "mode": settings["mode"],
               "input_sha256": bundle["input_sha256"], "character_count": len(text),
               "execution_fingerprint": app._execution_fingerprint(settings)}
        (root / "job.json").write_text(json.dumps(job), "utf-8")
        (root / "output" / "external_diagnosis_input.json").write_text(json.dumps(bundle), "utf-8")
        release_model = threading.Event()
        class Backend:
            def generate_structured(self, *, agent_id, output_model, payload):
                # Keep the fixture model in flight until the RUNNING assertion.
                # Otherwise a fast worker may legitimately finish before resume returns.
                if not release_model.wait(5):
                    raise RuntimeError("test did not release the model fixture")
                sid = payload["sources"][0]["source_id"]
                return {"summary": "温度异常需要进一步核查。", "findings": [{"text": "报告记录温度异常。", "citation_ids": [sid]}],
                        "hypotheses": [], "information_gaps": [], "recommended_review_actions": [], "review_actions": [],
                        "common_findings": [], "report_disagreements": [], "complementary_evidence": [],
                        "event_alignment": [], "integrated_conclusion": ""}
        with mock.patch.object(app, "_backend", return_value=Backend()):
            try:
                resumed = self.store.resume_job(job_id, settings)
                self.assertEqual(resumed["status"], "RUNNING")
            finally:
                release_model.set()
            result = wait_job(self.store, job_id)
        self.assertEqual(result["job"]["status"], "SUCCEEDED")
        self.assertEqual(result["report"]["input_sha256"], bundle["input_sha256"])

    def test_resume_rejects_changed_model_identity(self):
        job_id = "f" * 32
        root = self.store._job_root(job_id); root.mkdir(parents=True)
        settings = connection()
        (root / "job.json").write_text(json.dumps({"job_id": job_id, "status": "INTERRUPTED",
            "execution_fingerprint": app._execution_fingerprint(settings)}), "utf-8")
        changed = {**settings, "model": "different-model"}
        with self.assertRaises(app.WorkbenchError) as error:
            self.store.resume_job(job_id, changed)
        self.assertEqual(error.exception.code, "RESUME_MODEL_MISMATCH")

    def test_restart_marks_inflight_job_interrupted(self):
        job_id = "a" * 32
        root = self.store._job_root(job_id)
        root.mkdir(parents=True)
        (root / "job.json").write_text(json.dumps({"job_id": job_id, "status": "RUNNING",
                                                  "task": "x", "created_at": "x"}), "utf-8")
        restored = app.Workbench(Path(self.temp.name))
        self.assertEqual(restored.get(job_id)["status"], "INTERRUPTED")

    def test_second_launcher_does_not_mark_active_job_interrupted(self):
        from astrograph.integrations.external_diagnosis_conversation import _exclusive_call
        job_id = "b" * 32
        root = self.store._job_root(job_id)
        root.mkdir(parents=True)
        (root / "job.json").write_text(json.dumps({"job_id": job_id, "status": "RUNNING",
                                                  "task": "x", "created_at": "x"}), "utf-8")
        with _exclusive_call(Path(self.temp.name) / ".workbench.lock"):
            with mock.patch.object(sys, "argv", ["workbench", "--data-dir", self.temp.name]):
                self.assertEqual(app.main(), 1)
        self.assertEqual(self.store.get(job_id)["status"], "RUNNING")

    def test_backup_restore_roundtrip_preserves_job_and_original_material(self):
        job_id = "c" * 32
        root = self.store._job_root(job_id)
        (root / "input_original").mkdir(parents=True)
        (root / "output").mkdir()
        original = "原始诊断结论：温度异常。".encode("utf-8")
        (root / "input_original" / "诊断.txt").write_bytes(original)
        (root / "job.json").write_text(json.dumps({"job_id": job_id, "status": "SUCCEEDED",
            "task": "温度异常分析", "created_at": "2026-11-03T00:00:00Z"}), "utf-8")
        (root / "output" / "report.json").write_text('{"immutable":true}', "utf-8")
        archive = self.store.create_backup()
        target = Path(self.temp.name) / "restored"
        restored = app.Workbench(target)
        result = restored.restore_backup(archive)
        self.assertEqual(result["restored_jobs"], 1)
        self.assertEqual(restored.get(job_id)["status"], "SUCCEEDED")
        self.assertEqual((restored._job_root(job_id) / "input_original" / "诊断.txt").read_bytes(), original)
        self.assertEqual((restored._job_root(job_id) / "output" / "report.json").read_text("utf-8"), '{"immutable":true}')

    def test_restore_rejects_path_traversal_and_does_not_modify_data(self):
        import io
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("backup.json", json.dumps({"schema_version": "astrograph-portable-backup.v1", "files": []}))
            bundle.writestr("../outside.txt", "bad")
        with self.assertRaises(app.WorkbenchError) as error:
            self.store.restore_backup(archive.getvalue())
        self.assertEqual(error.exception.code, "BACKUP_UNSAFE_ARCHIVE")
        self.assertEqual(list(self.store.jobs_dir.iterdir()), [])

    def test_http_backup_and_restore_require_session_token(self):
        root = self.store._job_root("d" * 32)
        root.mkdir(parents=True)
        (root / "job.json").write_text(json.dumps({"job_id": "d" * 32, "status": "INTERRUPTED"}), "utf-8")
        source_server = app.make_server(self.store, port=0)
        source_thread = threading.Thread(target=source_server.serve_forever, daemon=True)
        source_thread.start()
        target_store = app.Workbench(Path(self.temp.name) / "restore-target")
        target_server = app.make_server(target_store, port=0)
        target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
        target_thread.start()
        try:
            source_base = f"http://127.0.0.1:{source_server.server_port}"
            target_base = f"http://127.0.0.1:{target_server.server_port}"
            with urllib.request.urlopen(target_base + "/api/session") as response:
                token = json.loads(response.read())["token"]
            with urllib.request.urlopen(source_base + "/api/backup") as response:
                backup = response.read()
                self.assertEqual(response.headers.get_content_type(), "application/zip")
            request = urllib.request.Request(target_base + "/api/restore", data=backup, method="POST",
                headers={"Content-Type": "application/zip", "X-AstroGraph-Token": token})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(json.loads(response.read())["restored_jobs"], 1)
            self.assertEqual(target_store.get("d" * 32)["status"], "INTERRUPTED")
        finally:
            source_server.shutdown(); source_server.server_close(); source_thread.join(2)
            target_server.shutdown(); target_server.server_close(); target_thread.join(2)

    def test_local_transport_ignores_environment_proxy(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
                                          "NO_PROXY": "", "no_proxy": ""}):
            store = app.Workbench(Path(self.temp.name))
            with model_server() as model:
                job = store.create_job(self.payload(connection(model["base_url"])))
                result = wait_job(store, job["job_id"])
                self.assertEqual(result["report"]["analysis"]["knowledge_enrichment"]["generation"]["status"], "SUCCEEDED")

    def test_ipv6_local_model_never_sends_material_to_proxy(self):
        with model_server() as proxy, model_server(ipv6=True) as model:
            with mock.patch.dict(os.environ, {"HTTP_PROXY": proxy["base_url"].rsplit("/v1", 1)[0],
                                              "HTTPS_PROXY": proxy["base_url"].rsplit("/v1", 1)[0],
                                              "NO_PROXY": "", "no_proxy": ""}):
                store = app.Workbench(Path(self.temp.name))
                job = store.create_job(self.payload(connection(model["base_url"])))
                result = wait_job(store, job["job_id"])
                self.assertEqual(result["report"]["analysis"]["knowledge_enrichment"]["generation"]["status"], "SUCCEEDED")
                self.assertEqual(proxy["calls"], 0)
                self.assertEqual(model["calls"], 1)

    def test_http_same_origin_guards_and_real_roundtrip(self):
        server = app.make_server(self.store, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, body=None, headers=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(base + path, data=data, headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()
        try:
            code, session_raw = request("/api/session")
            self.assertEqual(code, 200)
            token = json.loads(session_raw)["token"]
            self.assertEqual(request("/api/health")[0], 200)
            self.assertEqual(request("/api/session", headers={"Host": "evil.example"})[0], 403)
            self.assertEqual(request("/api/jobs", self.payload(), {"Content-Type": "application/json"})[0], 403)
            valid_headers = {"Content-Type": "application/json", "X-AstroGraph-Token": token}
            self.assertEqual(request("/api/jobs", self.payload(), {**valid_headers, "Origin": "https://evil.example"})[0], 403)
            with model_server() as model:
                code, raw = request("/api/jobs", self.payload(connection(model["base_url"])), valid_headers)
                self.assertEqual(code, 202, raw)
                job = json.loads(raw)["job"]
                wait_job(self.store, job["job_id"])
                self.assertEqual(request(f"/api/jobs/{job['job_id']}/report.md")[0], 200)
            self.assertEqual(request("/../workbench.py")[0], 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
