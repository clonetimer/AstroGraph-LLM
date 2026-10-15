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


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture()
def live_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
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
        "case_id": "r34-diagnostic-workspace-http-case",
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


def _authority_projection(payload: dict) -> dict:
    return {
        "diagnosis_disposition": payload.get("diagnosis_disposition", payload.get("diagnosis_status")),
        "release_status": payload.get("release_status"),
        "evidence_summary": payload.get("evidence_summary"),
    }


def test_real_http_result_workspace_graph_preserves_formal_authority(live_platform) -> None:
    base_url, tmp_path = live_platform
    with sync_playwright() as playwright:
        kwargs = {"headless": True, "args": ["--no-sandbox"]}
        executable = _chromium_executable()
        if executable:
            kwargs["executable_path"] = executable
        browser = playwright.chromium.launch(**kwargs)
        context = browser.new_context(viewport={"width": 1366, "height": 768}, locale="zh-CN")
        page = context.new_page()
        page.goto(base_url + "/platform/legacy", wait_until="networkidle")
        page.locator('[data-view="new"]').click()
        page.locator("#file-input").set_input_files(_public_case_files(tmp_path))
        page.locator("#preflight-btn").click()
        page.locator("#start-btn").wait_for(state="visible", timeout=30_000)
        page.wait_for_function("!document.querySelector('#start-btn').disabled", timeout=30_000)
        page.locator("#start-btn").click()
        page.locator("#diagnostic-workspace").wait_for(state="visible", timeout=30_000)

        job_id = page.locator("#current-job").inner_text().strip()
        before_response = context.request.get(f"{base_url}/v1/platform/jobs/{job_id}/maintenance")
        assert before_response.status == 200
        before = _authority_projection(before_response.json())

        page.locator('[data-dw-main="graph"]').click()
        assert "当前报告没有可交互图谱数据。" in page.locator("#dw-main").inner_text()
        assert page.locator("#diagnostic-workspace").is_visible()
        assert page.locator('input[name*="cypher" i], textarea[name*="cypher" i]').count() == 0
        assert page.locator('input[name*="threshold" i], input[name*="channel" i], input[name*="window" i]').count() == 0

        after_response = context.request.get(f"{base_url}/v1/platform/jobs/{job_id}/maintenance")
        assert after_response.status == 200
        after = _authority_projection(after_response.json())
        assert after == before
        browser.close()
