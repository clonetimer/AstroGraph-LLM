from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")
from playwright.sync_api import sync_playwright

from astrograph.api import create_app


def _chromium_executable() -> str | None:
    candidates = [
        os.environ.get("ASTROGRAPH_CHROMIUM_EXECUTABLE"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


def _launch(playwright):
    kwargs = {"headless": True, "args": ["--no-sandbox"]}
    executable = _chromium_executable()
    if executable:
        kwargs["executable_path"] = executable
    return playwright.chromium.launch(**kwargs)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture()
def live_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    port = _free_port()
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=3)
        pytest.fail("localhost AstroGraph platform did not start")
    yield f"http://127.0.0.1:{port}", tmp_path
    server.should_exit = True
    thread.join(timeout=10)


def _public_case_files(root: Path) -> list[str]:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["time_s", "eps.battery.soc"])
    writer.writeheader()
    writer.writerows([
        {"time_s": 0, "eps.battery.soc": 0.8},
        {"time_s": 1, "eps.battery.soc": 0.7},
    ])
    telemetry = buffer.getvalue().encode("utf-8")
    telemetry_path = root / "telemetry.csv"
    telemetry_path.write_bytes(telemetry)
    manifest = {
        "schema_version": "sat-sim.astrograph-public-case.v1",
        "case_id": "r33-1-browser-http-case",
        "telemetry": {
            "file": telemetry_path.name,
            "sha256": hashlib.sha256(telemetry).hexdigest(),
            "rows": 2,
            "fields": ["time_s", "eps.battery.soc"],
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return [str(manifest_path), str(telemetry_path)]


def test_real_http_upload_preflight_run_result_and_direct_knowledge(live_platform) -> None:
    base_url, tmp_path = live_platform
    files = _public_case_files(tmp_path)
    with sync_playwright() as p:
        browser = _launch(p)
        context = browser.new_context(viewport={"width": 1366, "height": 768}, locale="zh-CN")
        responses: dict[str, list[int]] = {}
        page = context.new_page()
        page.on("response", lambda response: responses.setdefault(response.url, []).append(response.status))
        page.goto(base_url + "/platform/legacy", wait_until="networkidle")
        page.locator('[data-view="new"]').click()
        page.locator("#file-input").set_input_files(files)
        page.locator("#preflight-btn").click()
        page.locator("#start-btn").wait_for(state="visible", timeout=30_000)
        page.wait_for_function("!document.querySelector('#start-btn').disabled", timeout=30_000)
        assert "blocked" not in (page.locator('[data-ops-step="2"]').get_attribute("class") or "")
        page.locator("#start-btn").click()
        try:
            page.locator("#ops-result-navigation").wait_for(state="visible", timeout=30_000)
        except Exception as exc:
            debug = page.evaluate(
                """() => ({hash:location.hash,toast:document.querySelector('#toast')?.innerText,
                progress:document.querySelector('#status-log')?.innerText,
                actionHidden:document.querySelector('#progress-actions')?.hidden,
                result:document.querySelector('#result-content')?.innerText})"""
            )
            raise AssertionError(f"real HTTP workflow did not reach result: {debug}") from exc
        assert page.locator('[data-ops-result-panel="conclusion"]').is_visible()
        conclusion = page.locator('[data-ops-result-panel="conclusion"]').inner_text()
        assert "SUCCEEDED" in conclusion
        assert page.locator('.ops-diagnosis-summary > div').nth(4).locator('strong').inner_text() == "未提供"
        assert not page.locator("#toast", has_text="Failed to fetch").is_visible()
        assert not page.locator("#toast", has_text="解析").is_visible()

        for endpoint in ("/v1/platform/overview", "/v1/platform/tools", "/v1/platform/agent-capabilities"):
            matching = [status for url, statuses in responses.items() if endpoint in url for status in statuses]
            assert matching and all(status == 200 for status in matching), (endpoint, matching)

        direct = context.new_page()
        direct_responses: dict[str, list[int]] = {}
        direct.on("response", lambda response: direct_responses.setdefault(response.url, []).append(response.status))
        direct.goto(base_url + "/platform#knowledge", wait_until="networkidle")
        direct.wait_for_function(
            "!document.querySelector('#knowledge-operations')?.innerText.includes('正在读取')",
            timeout=30_000,
        )
        knowledge = direct.locator("#knowledge-operations").inner_text()
        maintenance_statuses = [status for url, statuses in direct_responses.items() if url.endswith("/maintenance") for status in statuses]
        assert maintenance_statuses == [200]
        assert knowledge.strip()
        assert not direct.locator("#toast", has_text="读取失败").is_visible()
        browser.close()
