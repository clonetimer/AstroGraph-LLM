"""Native Chrome HTTP acceptance for the P4 account and report workflow.

Set ASTROGRAPH_P4_BROWSER_MODEL=1 to include the configured local model call.
Optional screenshots and a small status receipt go to ASTROGRAPH_UI_EVIDENCE_DIR.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from astrograph.api import create_app
from astrograph.integrations.upload_report import UploadReportJobStore


ROOT = Path(__file__).resolve().parents[2]
CASE = next((ROOT / "examples" / "platform_acceptance_cases_v1").glob("TC01*"))
SETUP_SECRET = "P4_Browser_Isolated_Setup_20261107"


@pytest.fixture
def account_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "accounts"
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(data_root))
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "accounts")
    monkeypatch.setenv("ASTROGRAPH_ACCOUNT_SETUP_SECRET", SETUP_SECRET)
    # Run with the desktop thread executor by default; set the environment
    # variable to process to exercise the account deployment path as well.
    monkeypatch.setenv(
        "ASTROGRAPH_PLATFORM_JOB_EXECUTOR",
        os.environ.get("ASTROGRAPH_P4_BROWSER_EXECUTOR", "thread"),
    )
    if os.environ.get("ASTROGRAPH_P4_BROWSER_MODEL") == "1":
        monkeypatch.setenv("ASTROGRAPH_PLATFORM_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        monkeypatch.setenv("ASTROGRAPH_PLATFORM_LLM_MODEL", "qwen/qwen3.5-9b")
        monkeypatch.setenv("ASTROGRAPH_PLATFORM_LLM_PROVIDER", "local-openai-compatible")
        monkeypatch.setenv("ASTROGRAPH_PLATFORM_LLM_RESPONSE_FORMAT_STRATEGY", "json_schema")
        monkeypatch.setenv("ASTROGRAPH_PLATFORM_LLM_MAX_TOKENS", "1200")
        monkeypatch.setenv("ASTROGRAPH_PLATFORM_LLM_TIMEOUT_S", "240")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port,
                                           access_log=False, log_level="error"))
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline and thread.is_alive():
        time.sleep(.04)
    assert server.started
    yield f"http://127.0.0.1:{port}", data_root
    server.should_exit = True
    thread.join(timeout=8)


def test_account_upload_model_report_and_expert_review(account_server):
    origin, data_root = account_server
    evidence_dir = Path(os.environ["ASTROGRAPH_UI_EVIDENCE_DIR"]) if os.environ.get("ASTROGRAPH_UI_EVIDENCE_DIR") else None
    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)

    def screenshot(page, name: str, *, full_page: bool = True) -> None:
        if evidence_dir:
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(100)
            page.screenshot(path=str(evidence_dir / name), full_page=full_page)

    with sync_playwright() as playwright:
        launch = {"headless": True, "args": ["--no-sandbox"]}
        if os.environ.get("ASTROGRAPH_CHROMIUM_EXECUTABLE"):
            launch["executable_path"] = os.environ["ASTROGRAPH_CHROMIUM_EXECUTABLE"]
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, locale="zh-CN")
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(origin + "/platform", wait_until="networkidle")
        page.locator("#account-modal[open]").wait_for(timeout=15000)
        assert page.locator("#account-modal-title").inner_text() == "初始化系统管理员"
        page.locator("#account-name").fill("p4admin")
        page.locator("#account-display").fill("平台管理员")
        page.locator("#account-pass").fill("P4AdminPassword2026!")
        page.locator("#account-setup-key").fill(SETUP_SECRET)
        page.locator("#account-submit").click()
        page.wait_for_function("document.querySelector('#identity-label').textContent==='平台管理员'", timeout=20000)
        screenshot(page, "04_管理员登录.png")

        page.locator('[data-route="users"]').click()
        page.locator("#new-user-name").fill("p4expert")
        page.locator("#new-user-display").fill("领域专家")
        page.locator("#new-user-role").select_option("reviewer")
        page.locator("#new-user-password").fill("P4ExpertPassword2026!")
        page.locator('#users-create-form button[type="submit"]').click()
        page.wait_for_function("document.querySelector('#users-list').textContent.includes('p4expert')", timeout=15000)
        screenshot(page, "05_专家账号.png")

        page.locator('[data-route="upload"]').click()
        assert page.locator("#analysis-thinking-body").is_hidden()
        start_button_box = page.locator("#mission-start").bounding_box()
        assert start_button_box and start_button_box["y"] + start_button_box["height"] < 900
        screenshot(page, "06a_新建分析输入页.png")
        page.locator("#mission-task").fill(
            "请结合上传的诊断结果、运行日志和参考资料，说明反作用轮异常的现象、可能原因、证据缺口与下一步核查。"
        )
        for filename, element in (
            ("diagnosis_result.json", "mission-diagnosis"),
            ("operations.log", "mission-logs"),
            ("reference.md", "mission-references"),
        ):
            page.locator("#" + element).set_input_files(str(CASE / filename))
        page.wait_for_function("!document.querySelector('#mission-start').disabled", timeout=60000)
        assert "通过" in page.locator("#preflight-results").inner_text()
        assert page.locator("#analysis-thinking-body").is_visible()
        screenshot(page, "06_真实文件上传预检.png")
        page.locator("#mission-start").click()
        page.locator("#screen-report.active").wait_for(timeout=300000)
        page.locator("#action-evidence").wait_for(state="visible")
        assert page.locator("#screen-report h1").inner_text() == "故障分析报告"
        screenshot(page, "07_管理员查看分析报告.png")
        if os.environ.get("ASTROGRAPH_P4_BROWSER_MODEL") == "1":
            page.locator(".model-result-section").first.wait_for(state="visible")
            citation = page.locator("[data-model-citation]").first
            citation.wait_for(state="visible")
            citation_id = citation.get_attribute("data-model-citation")
            citation.click()
            page.locator("#evidence-detail[open]").wait_for()
            assert citation_id in page.locator("#evidence-source-meta").inner_text()
            assert page.locator("#evidence-source-quote").inner_text().strip()
            screenshot(page, "07b_模型引用来源.png", full_page=False)
            page.locator("#evidence-detail-close").click()
            uploaded = page.locator('[data-model-citation^="upload:"]').first
            if uploaded.count():
                uploaded.click()
                assert "不是当前卫星遥测" in page.locator(".evidence-source-boundary").inner_text()
                screenshot(page, "07c_上传资料引用.png", full_page=False)
                page.locator("#evidence-detail-close").click()
            assert not page.locator(".rule-report-details").evaluate("element => element.open")
            page.locator(".rule-report-details > summary").click()
            assert page.locator(".rule-report-details").evaluate("element => element.open")
            page.locator(".rule-report-details > summary").click()
        job_id = page.locator("#report-job-label").inner_text().strip()
        assert job_id.startswith("JOB-")
        page.goto(origin + "/platform/home", wait_until="domcontentloaded")
        page.wait_for_function("job => document.querySelector('#focus-job').textContent===job", arg=job_id, timeout=20000)
        assert page.evaluate("document.documentElement.scrollHeight <= window.innerHeight + 2")
        screenshot(page, "07a_任务总览.png")
        page.goto(origin + "/platform/mission#command", wait_until="networkidle")
        page.wait_for_function("document.querySelector('#identity-label').textContent==='平台管理员'", timeout=20000)

        page.locator("#identity-trigger").click()
        page.locator("#identity-logout").click()
        page.locator("#account-modal[open]").wait_for(timeout=15000)
        page.locator("#account-name").fill("p4expert")
        page.locator("#account-pass").fill("P4ExpertPassword2026!")
        page.locator("#account-submit").click()
        page.wait_for_function("document.querySelector('#identity-label').textContent==='领域审核专家'", timeout=20000)
        page.locator('[data-route="review"]').click()
        page.locator(f'[data-review-job="{job_id}"]').wait_for(timeout=30000)
        page.locator(f'[data-review-job="{job_id}"]').click()
        candidate = page.locator("[data-review-fault]").first
        candidate.wait_for(timeout=15000)
        candidate.locator('select[name="decision"]').select_option("EVIDENCE_INSUFFICIENT")
        candidate.locator('input[name="evidence_ref"]').fill("同工况遥测与正式规程待核验")
        candidate.locator('textarea[name="note"]').fill("先核对电流、轮速、温度与力矩指令的同步时窗，暂不确认机械根因。")
        candidate.locator('button[type="submit"]').click()
        page.wait_for_function("document.querySelector('#toast').textContent.includes('已记录复核意见')", timeout=20000)
        screenshot(page, "08_专家复核.png")
        assert not page_errors, page_errors
        browser.close()

    job = UploadReportJobStore(data_root).get(job_id)
    report = UploadReportJobStore(data_root).report(job_id)
    assert job["status"] == "SUCCEEDED"
    assert report["status"] == "SUCCEEDED"
    assert job["llm_call_status"] == report["analysis"]["knowledge_enrichment"]["generation"]["status"]
    if os.environ.get("ASTROGRAPH_P4_BROWSER_MODEL") == "1":
        assert job["llm_call_status"] == "SUCCEEDED"
    with httpx.Client(base_url=origin, trust_env=False, timeout=30) as client:
        token = client.post("/v1/platform/auth/login", json={
            "username": "p4expert", "password": "P4ExpertPassword2026!",
        }).json()["token"]
        detail = client.get(f"/v1/platform/material-knowledge/report-reviews/{job_id}",
                            headers={"Authorization": "Bearer " + token})
        assert detail.status_code == 200
        assert detail.json()["reviewed_count"] >= 1
    if evidence_dir:
        (evidence_dir / "workflow_receipt.json").write_text(json.dumps({
            "job_id": job_id,
            "report_status": report["status"],
            "llm_call_status": job["llm_call_status"],
            "expert_reviewed_count": detail.json()["reviewed_count"],
            "native_browser_http": True,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
