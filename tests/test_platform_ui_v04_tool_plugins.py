from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from astrograph.api import create_app
from astrograph.orchestration.models import PermissionScope, WorkflowContext
from astrograph.tools.plugins import TrustedAgentToolPluginManager
from astrograph.tools.registry import ToolRegistry


def _plugin_source() -> str:
    return '''from astrograph.orchestration.models import PermissionScope, SideEffectLevel\nfrom astrograph.tools.plugins import AgentToolExport, AgentToolPlugin, AgentToolPluginManifest\nfrom astrograph.tools.registry import ToolSpec\n\ndef echo(payload):\n    return {"status":"SUCCEEDED","value":payload["value"]}\n\nclass Demo(AgentToolPlugin):\n    @property\n    def manifest(self):\n        return AgentToolPluginManifest(plugin_id="demo.echo", plugin_version="1.0", display_name="回显插件", review_status="APPROVED", source_refs=["review/demo-echo"])\n    def exports(self):\n        return [AgentToolExport(display_name="回显", purpose="隔离进程回显测试。", function_name="echo", spec=ToolSpec(tool_id="demo.echo.run", tool_version="1.0", input_schema={"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":False}, output_schema={"type":"object","properties":{"status":{"type":"string"},"value":{"type":"string"}},"required":["status","value"],"additionalProperties":False}, deterministic=True, side_effect_level=SideEffectLevel.NONE, required_permissions=[PermissionScope.READ_DATA], execution_mode="ISOLATED_PROCESS", timeout_s=10))]\nPLUGIN=Demo()\n'''


def test_platform_tool_catalog_exposes_purpose_schema_and_implementation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    client = TestClient(create_app())
    response = client.get("/v1/platform/tools")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "astrograph-platform-tool-catalog.v1"
    item = next(x for x in body["tools"] if x["tool_id"] == "diagnosis.fuse")
    assert item["purpose"]
    assert item["implementation"]["module"] == "astrograph.agents.mission.service"
    assert "input_schema" in item["spec"]
    assert item["platform_upload_bound"] is True


def test_trusted_agent_tool_plugin_hot_reload_registers_future_snapshot_and_executes_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reviewed-tools"
    root.mkdir()
    plugin_path = root / "demo_echo.py"
    plugin_path.write_text(_plugin_source(), encoding="utf-8")
    manager = TrustedAgentToolPluginManager(root, enabled=True)
    status = manager.reload()
    assert status["loaded_plugin_count"] == 1
    assert status["loaded_tool_count"] == 1

    registry = ToolRegistry()
    registered = manager.register_for_task(registry, task="diagnose", work_mode="normal")
    assert registered == ["demo.echo.run"]
    definition = registry.get("demo.echo.run")
    assert definition.spec.execution_mode == "ISOLATED_PROCESS"
    assert definition.worker_target is not None
    registry.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        registry.register(definition)

    result = registry.invoke(
        "demo.echo.run",
        {"value": "ok"},
        WorkflowContext(workflow_id="wf", run_id="run", permissions=[PermissionScope.READ_DATA]),
    )
    assert result == {"status": "SUCCEEDED", "value": "ok"}


def test_plugin_reload_api_is_admin_write_and_catalog_is_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "reviewed-tools"
    root.mkdir()
    (root / "demo_echo.py").write_text(_plugin_source(), encoding="utf-8")
    monkeypatch.setenv("ASTROGRAPH_AGENT_TOOL_PLUGINS_ENABLED", "1")
    monkeypatch.setenv("ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR", str(root))
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    client = TestClient(create_app())
    reloaded = client.post("/v1/platform/tool-plugins/reload")
    assert reloaded.status_code == 200
    assert reloaded.json()["loaded_plugin_count"] == 1
    catalog = client.get("/v1/platform/tools").json()
    plugin = next(x for x in catalog["tools"] if x["tool_id"] == "demo.echo.run")
    assert plugin["source"] == "trusted_plugin"
    assert plugin["spec"]["execution_mode"] == "ISOLATED_PROCESS"


def test_managed_plugin_install_update_rollback_and_uninstall(tmp_path: Path) -> None:
    root = tmp_path / "reviewed-tools"
    root.mkdir()
    manager = TrustedAgentToolPluginManager(root, enabled=True)

    installed = manager.install_source(filename="demo_echo.py", source=_plugin_source())
    assert installed["status"] == "SUCCEEDED"
    assert installed["plugin_id"] == "demo.echo"
    status = manager.status()
    plugin = next(item for item in status["plugins"] if item["plugin_id"] == "demo.echo")
    assert plugin["managed_install"] is True
    assert status["managed_install_policy"] == "ATOMIC_VALIDATE_THEN_COMMIT_WITH_ROLLBACK"

    updated_source = _plugin_source().replace('plugin_version="1.0"', 'plugin_version="1.1"')
    updated = manager.install_source(filename="demo_echo.py", source=updated_source)
    assert updated["plugin_version"] == "1.1"

    with pytest.raises(SyntaxError):
        manager.install_source(filename="demo_echo.py", source="def broken(:\n")
    status_after_failed_update = manager.status()
    retained = next(item for item in status_after_failed_update["plugins"] if item["plugin_id"] == "demo.echo")
    assert retained["plugin_version"] == "1.1"

    preview = manager.uninstall("demo.echo", force=False)
    assert preview["status"] == "PREVIEW"
    assert preview["running_registries_affected"] is False
    removed = manager.uninstall("demo.echo", force=True)
    assert removed["status"] == "SUCCEEDED"
    assert manager.status()["loaded_plugin_count"] == 0


def test_plugin_managed_install_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "reviewed-tools"
    root.mkdir()
    monkeypatch.setenv("ASTROGRAPH_AGENT_TOOL_PLUGINS_ENABLED", "1")
    monkeypatch.setenv("ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR", str(root))
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    client = TestClient(create_app())
    response = client.post(
        "/v1/platform/tool-plugins/install",
        json={"filename": "demo_echo.py", "source": _plugin_source()},
    )
    assert response.status_code == 200
    assert response.json()["plugin_id"] == "demo.echo"
    catalog = client.get("/v1/platform/tools").json()
    tool = next(item for item in catalog["tools"] if item["tool_id"] == "demo.echo.run")
    assert tool["plugin_managed_install"] is True
