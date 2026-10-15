from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src/astrograph/resources"
HTML = (RESOURCES / "platform_ui.html").read_text(encoding="utf-8")
CSS = (RESOURCES / "platform_ui.css").read_text(encoding="utf-8")
CORE_JS = (RESOURCES / "platform_core.js").read_text(encoding="utf-8")
OPERATIONS_JS = (RESOURCES / "platform_operations.js").read_text(encoding="utf-8")
DIAGNOSTIC_WORKSPACE_JS = (RESOURCES / "platform_diagnostic_workspace.js").read_text(encoding="utf-8")
REPORT_JS = (RESOURCES / "platform_report.js").read_text(encoding="utf-8")
SYSTEM_JS = (RESOURCES / "platform_system.js").read_text(encoding="utf-8")
AGENT_JS = (RESOURCES / "platform_agent.js").read_text(encoding="utf-8")
DASHBOARD_JS = (RESOURCES / "platform_dashboard.js").read_text(encoding="utf-8")
JOB_JS = (RESOURCES / "platform_job.js").read_text(encoding="utf-8")
AGENT_LIVE_JS = (RESOURCES / "platform_agent_live.js").read_text(encoding="utf-8")
KNOWLEDGE_JS = (RESOURCES / "platform_knowledge.js").read_text(encoding="utf-8")
SHELL_JS = (RESOURCES / "platform_shell.js").read_text(encoding="utf-8")
GRAPH_WORKBENCH_JS = (RESOURCES / "platform_graph_workbench.js").read_text(encoding="utf-8")
ENGINE_SETTINGS_JS = (RESOURCES / "platform_engine_settings.js").read_text(encoding="utf-8")
MAIN_JS = (RESOURCES / "platform_ui.js").read_text(encoding="utf-8")

# Browser-behavior tests run from about:blank because navigation is restricted in this
# execution environment. Build a classic-script test bundle from the exact production
# modules by removing only their import/export declarations; production still ships ESM.
def _classic_test_bundle() -> str:
    core = CORE_JS.rsplit("\nexport {", 1)[0]
    operations = "\n".join(line for line in OPERATIONS_JS.splitlines() if not line.startswith("import "))
    operations = operations.rsplit("\nexport {", 1)[0]
    diagnostic_workspace = "\n".join(line for line in DIAGNOSTIC_WORKSPACE_JS.splitlines() if not line.startswith("import "))
    diagnostic_workspace = diagnostic_workspace.rsplit("\nexport {", 1)[0]
    report = "\n".join(line for line in REPORT_JS.splitlines() if not line.startswith("import "))
    report = report.rsplit("\nexport {", 1)[0]
    system = "\n".join(line for line in SYSTEM_JS.splitlines() if not line.startswith("import "))
    system = system.rsplit("\nexport {", 1)[0]
    agent = "\n".join(line for line in AGENT_JS.splitlines() if not line.startswith("import "))
    agent = agent.rsplit("\nexport {", 1)[0]
    dashboard = "\n".join(line for line in DASHBOARD_JS.splitlines() if not line.startswith("import "))
    dashboard = dashboard.rsplit("\nexport {", 1)[0]
    job = "\n".join(line for line in JOB_JS.splitlines() if not line.startswith("import "))
    job = job.rsplit("\nexport {", 1)[0]
    agent_live = "\n".join(line for line in AGENT_LIVE_JS.splitlines() if not line.startswith("import "))
    agent_live = agent_live.rsplit("\nexport {", 1)[0]
    knowledge = "\n".join(line for line in KNOWLEDGE_JS.splitlines() if not line.startswith("import "))
    knowledge = knowledge.rsplit("\nexport {", 1)[0]
    shell = "\n".join(line for line in SHELL_JS.splitlines() if not line.startswith("import "))
    shell = shell.rsplit("\nexport {", 1)[0]
    graph_workbench = "\n".join(line for line in GRAPH_WORKBENCH_JS.splitlines() if not line.startswith("import "))
    graph_workbench = graph_workbench.rsplit("\nexport {", 1)[0]
    engine_settings = "\n".join(line for line in ENGINE_SETTINGS_JS.splitlines() if not line.startswith("import "))
    engine_settings = engine_settings.rsplit("\nexport {", 1)[0]
    main = "\n".join(line for line in MAIN_JS.splitlines() if not line.startswith("import "))
    return "\n".join((core, operations, diagnostic_workspace, report, system, agent, dashboard, job, agent_live, knowledge, shell, graph_workbench, engine_settings, main))

JS = _classic_test_bundle()
OFFLINE_HTML = HTML.replace(
    '<link rel="stylesheet" href="/platform/assets/v0.21.1/platform_ui.css">', ''
).replace(
    '<script type="module" src="/platform/assets/v0.21.1/platform_ui.js"></script>', ''
)


def _chromium_executable() -> str | None:
    candidates = [
        os.environ.get("ASTROGRAPH_CHROMIUM_EXECUTABLE"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


def _launch(playwright):
    executable = _chromium_executable()
    kwargs = {"headless": True, "args": ["--no-sandbox"]}
    if executable:
        kwargs["executable_path"] = executable
    return playwright.chromium.launch(**kwargs)


def _overview() -> dict:
    return {
        "schema_version": "astrograph-platform-overview.v3",
        "recent_window_count": 3,
        "status_counts": {"SUCCEEDED": 1, "RUNNING": 1, "FAILED": 1},
        "active_job_count": 1,
        "work_mode_counts": {"normal": 2, "thinking": 1, "creative": 0},
        "artifact_kind_counts": {"PIPELINE_RUNTIME": 2},
        "diagnosis_disposition_counts": {"REVIEW_REQUIRED": 1},
        "release_status_counts": {"FORMAL_RELEASE_BLOCKED": 1},
        "top_fault_candidates": [{"candidate": "ADCS_RW_JAM", "count": 1}],
        "review_or_attention_count": 1,
        "daily_job_counts": [{"date": "2026-09-17", "count": 3}],
        "latest_completed_diagnosis": {
            "job_id": "JOB-AAAAAAAAAAAAAAAAAAAAAAAA",
            "candidate": "ADCS_RW_JAM",
            "diagnosis_disposition": "REVIEW_REQUIRED",
            "release_status": "FORMAL_RELEASE_BLOCKED",
            "requires_human_review": True,
            "artifact_kind": "PIPELINE_RUNTIME",
            "work_mode": "normal",
            "completed_at": "2026-09-17T08:00:00Z",
        },
        "latest_successful": None,
        "recent_jobs": [
            {
                "job_id": "JOB-AAAAAAAAAAAAAAAAAAAAAAAA", "status": "SUCCEEDED", "execution_status": "SUCCEEDED",
                "task": "ADCS 诊断", "candidate": "ADCS_RW_JAM", "diagnosis_disposition": "REVIEW_REQUIRED",
                "release_status": "FORMAL_RELEASE_BLOCKED", "requires_human_review": True,
                "artifact_kind": "PIPELINE_RUNTIME", "work_mode": "normal", "created_at": "2026-09-17T08:00:00Z",
            },
            {
                "job_id": "JOB-BBBBBBBBBBBBBBBBBBBBBBBB", "status": "RUNNING", "execution_status": "RUNNING",
                "task": "正在诊断", "candidate": None, "diagnosis_disposition": "NOT_AVAILABLE",
                "release_status": "NOT_AVAILABLE", "requires_human_review": False,
                "artifact_kind": None, "work_mode": "normal", "created_at": "2026-09-17T08:05:00Z",
            },
            {
                "job_id": "JOB-CCCCCCCCCCCCCCCCCCCCCCCC", "status": "FAILED", "execution_status": "FAILED",
                "task": "输入失败任务", "candidate": None, "diagnosis_disposition": "NOT_AVAILABLE",
                "release_status": "NOT_AVAILABLE", "requires_human_review": False,
                "artifact_kind": None, "work_mode": "normal", "created_at": "2026-09-17T08:10:00Z",
            },
        ],
        "generated_at": "2026-09-17T08:10:00Z",
    }


def _prepare(page) -> None:
    # The execution environment blocks browser navigation to localhost and arbitrary
    # hosts. set_content() is therefore used, but its about:blank origin cannot access
    # localStorage. Provide a test-host storage shim before loading the unchanged
    # production HTML so application initialization can proceed without modifying
    # production code.
    page.evaluate(
        """Object.defineProperty(window, 'localStorage', {
          configurable: true,
          value: {
            _data: {},
            getItem(k) { return Object.prototype.hasOwnProperty.call(this._data, k) ? this._data[k] : null; },
            setItem(k, v) { this._data[k] = String(v); },
            removeItem(k) { delete this._data[k]; },
            clear() { this._data = {}; }
          }
        });"""
    )
    page.set_content(OFFLINE_HTML, wait_until="domcontentloaded")
    page.add_style_tag(content=CSS)
    page.add_script_tag(content=JS)
    # The packaged page normally fills these through /v1 APIs. Browser tests inject
    # the same contracts directly because this execution environment blocks localhost
    # navigation at the browser policy layer.
    page.evaluate(
        """([overview]) => {
          state.overview = overview;
          state.capabilities = {knowledge_services:{local_session_retrieval:true}};
          state.llmSettings = {configured:false};
          state.agentCapabilities = {modes:[
            {id:'normal',display_name:'标准诊断',allowed_diagnose_tools:[]},
            {id:'thinking',display_name:'深入复核',allowed_diagnose_tools:[]},
            {id:'creative',display_name:'扩展排查',allowed_diagnose_tools:[]}
          ]};
          renderModeSelectors();
          renderDashboard();
        }""",
        [_overview()],
    )


def test_dashboard_browser_separates_execution_diagnosis_and_attention():
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        attention = page.locator("#dashboard-attention").inner_text()
        active = page.locator("#dashboard-active").inner_text()
        recent = page.locator("#dashboard-recent").inner_text()
        latest = page.locator("#dashboard-latest").inner_text()
        assert "反作用轮卡滞" in attention
        assert "执行失败" in attention
        assert "正在诊断" in active
        assert "执行：已完成" in recent
        assert "诊断：需人工复核" in recent
        assert "正式发布受阻" in latest
        browser.close()


def test_160_percent_font_at_1366_does_not_create_page_horizontal_overflow():
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        page.locator("#font-range").evaluate("el => {el.value='160'; el.dispatchEvent(new Event('input',{bubbles:true}))}")
        page.locator('[data-view="new"]').click()
        page.wait_for_timeout(50)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 1
        assert page.locator("#preflight-btn").is_visible()
        assert page.locator("#start-btn").is_visible()
        browser.close()


def test_r13_upstream_panel_keeps_manual_exchange_available_without_optional_providers():
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        page.evaluate(
            """() => {
              const host=document.createElement('div'); host.id='r13-upstream-host'; document.body.appendChild(host);
              host.innerHTML=upstreamRecomputeHtml({
                request:{request_id:'URR-TEST',request_sha256:'a'.repeat(64)},
                providers:{providers:[
                  {provider_id:'manual-import',display_name:'Manual request/response exchange',status:'MANUAL',execution_mode:'MANUAL_EXCHANGE',reason:'Built in'},
                  {provider_id:'sat-agent-platform',display_name:'Satellite Agent Platform',status:'UNAVAILABLE',execution_mode:'LOCAL_OPTIONAL',reason:'Optional provider not configured'},
                  {provider_id:'matlab-simulink',display_name:'MATLAB / Simulink',status:'UNAVAILABLE',execution_mode:'LOCAL_OPTIONAL',reason:'MATLAB optional'}
                ]}
              });
            }"""
        )
        text = page.locator("#r13-upstream-host").inner_text()
        assert "手动可用" in text
        assert "Satellite Agent Platform" in text
        assert "MATLAB / Simulink" in text
        assert "可选 / 未启用" in text
        assert page.locator("#download-upstream-request-r13").is_visible()
        assert page.locator("#import-upstream-response-r13").is_visible()
        browser.close()
