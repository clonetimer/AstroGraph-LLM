from __future__ import annotations

from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.upload_report import PLATFORM_UI_ASSET_VERSION


def test_platform_shell_references_versioned_external_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    client = TestClient(create_app())
    response = client.get("/platform/legacy")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache, max-age=0"
    assert f'/platform/assets/{PLATFORM_UI_ASSET_VERSION}/platform_ui.css' in response.text
    assert f'/platform/assets/{PLATFORM_UI_ASSET_VERSION}/platform_ui.js' in response.text
    assert '<script type="module"' in response.text
    assert "<style>" not in response.text
    assert "<script>" not in response.text


def test_versioned_platform_assets_have_immutable_cache_and_correct_types(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    client = TestClient(create_app())
    names = [
        "platform_ui.css", "platform_ui.js", "platform_core.js", "platform_report.js",
        "platform_system.js", "platform_agent.js", "platform_dashboard.js", "platform_job.js",
        "platform_agent_live.js", "platform_knowledge.js", "platform_shell.js", "platform_operations.js", "platform_graph_workbench.js", "platform_engine_settings.js",
    ]
    responses = {name: client.get(f"/platform/assets/{PLATFORM_UI_ASSET_VERSION}/{name}") for name in names}
    assert all(response.status_code == 200 for response in responses.values())
    assert responses["platform_ui.css"].headers["content-type"].startswith("text/css")
    for name, response in responses.items():
        if name.endswith(".js"):
            assert response.headers["content-type"].startswith("application/javascript")
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["x-content-type-options"] == "nosniff"
    assert "container-name:dashboard" in responses["platform_ui.css"].text
    assert "function renderDashboard" in responses["platform_dashboard.js"].text
    assert "export { $, $$, state" in responses["platform_core.js"].text
    assert "function renderReport" in responses["platform_report.js"].text
    assert "function loadLLMSettings" in responses["platform_system.js"].text
    assert "function loadAgentCapabilities" in responses["platform_agent.js"].text
    assert "function renderInputChecks" in responses["platform_job.js"].text
    assert "function renderAgentExecution" in responses["platform_agent_live.js"].text
    assert "function renderKnowledgeLibrary" in responses["platform_knowledge.js"].text
    assert "function initShell" in responses["platform_shell.js"].text
    assert "function operatorStatusViewModel" in responses["platform_operations.js"].text


def test_unknown_platform_asset_version_or_name_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    client = TestClient(create_app())
    assert client.get("/platform/assets/v0.0.0/platform_ui.js").status_code == 404
    assert client.get(f"/platform/assets/{PLATFORM_UI_ASSET_VERSION}/../../release.json").status_code == 404
    assert client.get(f"/platform/assets/{PLATFORM_UI_ASSET_VERSION}/unknown.js").status_code == 404
