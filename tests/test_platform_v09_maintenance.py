from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.platform_state import PlatformStateRepository
from astrograph.integrations.temporal_localization import localize_from_visualization
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.tools.neo4j_maintenance import get_fault_mechanism


def test_sqlite_control_plane_profiles_and_provider_roundtrip(tmp_path: Path) -> None:
    repo = PlatformStateRepository(tmp_path / "control.db")
    profile = repo.save_service_profile(
        service_kind="llm", name="local-lm", config={"enabled": False}, make_default=True
    )
    assert repo.default_service_profile("llm")["profile_id"] == profile["profile_id"]
    provider = repo.save_tool_provider(
        provider_id="mcp.neo4j", provider_type="mcp", name="Neo4j MCP",
        config={"transport": "streamable_http", "url": "http://127.0.0.1:9000/mcp", "read_only": True, "tool_allowlist": []},
    )
    assert provider["provider_type"] == "mcp"
    assert repo.get_tool_provider("mcp.neo4j")["config"]["read_only"] is True


def test_truth_safe_window_localization_uses_only_visualized_telemetry() -> None:
    points_a = []
    points_b = []
    for i in range(101):
        t = float(i)
        points_a.append({"t": t, "v": 0.0 if i < 55 else 12.0})
        points_b.append({"t": t, "v": 0.0 if i < 55 else -10.0})
    result = localize_from_visualization({
        "status": "AVAILABLE",
        "channels": [
            {"name": "a", "points": points_a},
            {"name": "b", "points": points_b},
        ],
        # A hostile/accidental truth field must have no effect on the localizer.
        "onset_time_s": 1.0,
        "truth": {"fault_onset_s": 1.0},
    }, diagnostic_fault_id="TEST_FAULT")
    assert result["truth_or_labels_used"] is False
    assert result["diagnostic_fault_id"] == "TEST_FAULT"
    assert result["first_detection_s"] is not None
    assert result["first_detection_s"] > 20.0
    assert result["first_detection_s"] != 1.0
    assert result["semantics"] == "END_STATE"


def test_neo4j_maintenance_tool_fails_closed_when_not_configured(monkeypatch) -> None:
    for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        monkeypatch.delenv(name, raising=False)
    result = get_fault_mechanism({"fault_id": "ADCS_RW_JAM"})
    assert result["status"] == "NOT_CONFIGURED"
    assert result["graph_query_executed"] is False


def test_llm_profile_api_persists_nonsecret_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "platform"))
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    client = TestClient(create_app())
    saved = client.post("/v1/platform/settings/llm/profiles", json={
        "name": "disabled-template",
        "make_default": True,
        "activate": True,
        "settings": {"enabled": False},
    })
    assert saved.status_code == 200, saved.text
    listed = client.get("/v1/platform/settings/llm/profiles")
    assert listed.status_code == 200
    profiles = listed.json()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["name"] == "disabled-template"
    assert "secret_ref" not in profiles[0]


def test_adcs_bundle_generates_maintenance_contract_and_temporal_localization(tmp_path: Path) -> None:
    source = Path("/mnt/data/ADCS_RW_JAM_UI_test_bundle.zip")
    if not source.is_file():
        return
    store = UploadReportJobStore(tmp_path / "platform")
    job = store.create("诊断反作用轮异常", "normal")
    store.put_file(job["job_id"], source.name, source.read_bytes())
    report = store.run(job["job_id"])
    assert report["status"] == "SUCCEEDED"
    maintenance = store.maintenance(job["job_id"])
    assert maintenance["headline"] == "反作用轮卡滞"
    assert maintenance["diagnosis_status"] == "REVIEW_REQUIRED"
    assert maintenance["temporal_localization"]["truth_or_labels_used"] is False
    assert maintenance["temporal_localization"]["first_detection_s"] is not None
    assert maintenance["mechanism_graph"]["graph_kind"] == "REFERENCE_MECHANISM"
    # Algorithm-internal scores are deliberately absent from the operator contract.
    assert "birrf" not in str(maintenance).lower()
    assert "model_score" not in maintenance


def test_sqlite_profile_update_preserves_existing_secret_reference(tmp_path: Path) -> None:
    repo = PlatformStateRepository(tmp_path / "control.db")
    first = repo.save_service_profile(
        service_kind="neo4j", name="local-kg", config={"enabled": True}, secret_ref="keyring:neo4j:abc"
    )
    second = repo.save_service_profile(
        service_kind="neo4j", name="local-kg", config={"enabled": True, "database": "neo4j"}, secret_ref=None
    )
    assert second["profile_id"] == first["profile_id"]
    assert second["secret_ref"] == "keyring:neo4j:abc"


def test_read_only_mcp_provider_is_snapshotted_into_task_registry(tmp_path: Path, monkeypatch) -> None:
    from astrograph.tools.providers import ProviderTool, ToolProviderManager
    from astrograph.tools.registry import ToolRegistry

    repo = PlatformStateRepository(tmp_path / "control.db")
    manager = ToolProviderManager(repo)

    class FakeProvider:
        provider_id = "demo"
        provider_type = "mcp"

        class Config:
            read_only = True
            timeout_s = 3.0
        config = Config()

        def list_tools(self):
            return [ProviderTool(
                provider_id="demo", provider_type="mcp", tool_id="lookup",
                display_name="查询", description="read only", input_schema={"type": "object"},
                output_schema={"type": "object"}, read_only=True, source="mcp",
            )]

        def invoke(self, tool_id, arguments):
            return {"status": "SUCCEEDED", "tool_id": tool_id, "arguments": arguments}

    monkeypatch.setattr(manager, "providers", lambda: [FakeProvider()])
    registry = ToolRegistry()
    ids, limits = manager.register_read_only_for_task(registry)
    assert limits == []
    assert ids == ["mcp.demo.lookup"]
    spec = registry.get("mcp.demo.lookup").spec
    assert spec.side_effect_level.value == "EXTERNAL_IO"
    assert spec.execution_mode == "TRUSTED_IN_PROCESS"


def test_mcp_provider_registration_api_persists_to_control_plane(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "platform"))
    monkeypatch.setenv("ASTROGRAPH_AUTH_MODE", "local")
    client = TestClient(create_app())
    saved = client.post("/v1/platform/tool-providers", json={
        "provider_id": "neo4j-local",
        "provider_type": "mcp",
        "name": "本机 Neo4j MCP",
        "url": "http://127.0.0.1:9000/mcp",
        "enabled": True,
        "read_only": True,
        "tool_allowlist": ["get_schema"],
        "timeout_s": 30,
    })
    assert saved.status_code == 200, saved.text
    listed = client.get("/v1/platform/tool-providers")
    assert listed.status_code == 200
    provider = listed.json()["providers"][0]
    assert provider["provider_id"] == "neo4j-local"
    assert provider["config"]["read_only"] is True
