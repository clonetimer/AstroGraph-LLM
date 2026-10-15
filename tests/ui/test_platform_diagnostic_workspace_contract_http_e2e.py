from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")
from fastapi.responses import HTMLResponse
from playwright.sync_api import sync_playwright

from astrograph.api import create_app
from astrograph.integrations.maintenance_presentation import MAINTENANCE_SCHEMA_VERSION
from astrograph.integrations.upload_report import UploadReportJobStore
from tests.test_platform_diagnostic_workspace_contract_r34_1 import _graph, _inputs, _simulation_source


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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def live_workspace_contract(tmp_path: Path):
    # Seed the exact persisted artifacts consumed by the production store. The
    # endpoint must read, validate, and project them without a store test double.
    store = UploadReportJobStore(tmp_path / "platform")
    job = store.create("R34.2 real-store diagnostic workspace positive E2E")
    job_id = job["job_id"]
    report, maintenance, visualization = _inputs()
    report = dict(report, job_id=job_id)
    maintenance = dict(maintenance, schema_version=MAINTENANCE_SCHEMA_VERSION)
    maintenance["evidence_summary"] = dict(
        maintenance.get("evidence_summary") or {},
        supplemental_overlay_receipt_sha256=None,
    )

    output = store._job_root(job_id) / "output"
    _write_json(output / "report.json", report)
    _write_json(output / "maintenance.json", maintenance)
    _write_json(output / "visualization.json", visualization)
    job.update(
        status="SUCCEEDED",
        stage="COMPLETED",
        report_json="output/report.json",
        report_markdown=None,
    )
    store._save(job)
    retained = store.retain_diagnostic_workspace_sources(
        job_id,
        system_model_graph=_graph(),
        simulation_sources=(_simulation_source(),),
    )
    presentation = store.diagnostic_workspace(job_id)
    assert presentation["source_bundle_sha256"] == retained["bundle_sha256"]

    app = create_app()
    app.state.upload_report_store = store
    app.state.platform_state_repository = store.state_repository

    @app.get("/r34-2-positive", response_class=HTMLResponse)
    def positive_page():
        return f"""<!doctype html><html lang='zh-CN'><head>
        <link rel='stylesheet' href='/platform/assets/v0.21.1/platform_ui.css'></head>
        <body><main id='result-content'></main><script type='module'>
        import {{renderDiagnosticWorkspace,workspaceSnapshot}} from '/platform/assets/v0.21.1/platform_diagnostic_workspace.js';
        const projection=await fetch('/v1/platform/jobs/{job_id}/maintenance/diagnostic-workspace').then(r=>r.json());
        renderDiagnosticWorkspace({{diagnosticWorkspace:projection}});
        window.workspaceSnapshot=workspaceSnapshot;window.__projection=projection;window.__ready=true;
        </script></body></html>"""

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=3)
        pytest.fail("R34.2 localhost contract server did not start")
    yield f"http://127.0.0.1:{port}", job_id, presentation
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture()
def positive_page(live_workspace_contract):
    base_url, job_id, presentation = live_workspace_contract
    with sync_playwright() as playwright:
        kwargs = {"headless": True, "args": ["--no-sandbox"]}
        executable = _chromium_executable()
        if executable:
            kwargs["executable_path"] = executable
        browser = playwright.chromium.launch(**kwargs)
        context = browser.new_context(viewport={"width": 1366, "height": 768}, locale="zh-CN")
        page = context.new_page()
        page.goto(base_url + "/r34-2-positive", wait_until="networkidle")
        page.wait_for_function("window.__ready===true")
        yield page, context, base_url, job_id, presentation
        browser.close()


def test_real_graph_store_http_projection_renders_clickable_svg_without_authority_change(positive_page) -> None:
    page, context, base_url, job_id, presentation = positive_page
    before = presentation["authority_snapshot"]
    page.locator('[data-dw-main="graph"]').click()
    assert page.locator(".dw-node").count() >= 3
    page.locator('[data-dw-node="component:PUMP"]').click()
    assert page.evaluate("window.workspaceSnapshot().graphNodeId") == "component:PUMP"
    after = context.request.get(f"{base_url}/v1/platform/jobs/{job_id}/maintenance/diagnostic-workspace").json()
    assert after["source_bundle_status"] == "VERIFIED"
    assert after["authority_snapshot"] == before


def test_real_r31_store_http_projection_renders_numeric_overlay_without_authority_change(positive_page) -> None:
    page, context, base_url, job_id, presentation = positive_page
    before = presentation["authority_snapshot"]
    assert page.locator("#dw-simulation-select").count() == 1
    assert page.locator(".dw-overlay-chart path.observed").count() == 1
    assert page.locator(".dw-overlay-chart path.simulated").count() == 1
    page.locator("#dw-simulation-select").select_option("__NONE__")
    assert page.locator(".dw-overlay-chart path.simulated").count() == 0
    after = context.request.get(f"{base_url}/v1/platform/jobs/{job_id}/maintenance/diagnostic-workspace").json()
    assert after["authority_snapshot"] == before
    assert all(item["semantics"] == "SIMULATED_TRAJECTORY_NOT_PROBABILITY" for item in after["simulation_traces"])
