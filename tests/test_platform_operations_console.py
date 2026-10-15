from __future__ import annotations

from pathlib import Path

from astrograph.integrations.upload_report import PLATFORM_UI_ASSET_VERSION, platform_asset


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "src/astrograph/resources"
HTML = (RESOURCES / "platform_ui.html").read_text(encoding="utf-8")
OPERATIONS = (RESOURCES / "platform_operations.js").read_text(encoding="utf-8")
REPORT = (RESOURCES / "platform_report.js").read_text(encoding="utf-8")


def test_r33_asset_version_and_operations_module_are_served() -> None:
    assert PLATFORM_UI_ASSET_VERSION == "v0.21.1"
    source, media_type = platform_asset("platform_operations.js")
    assert media_type.startswith("application/javascript")
    assert "operatorStatusViewModel" in source
    assert "/platform/assets/v0.21.1/platform_ui.js" in HTML


def test_operations_navigation_and_result_layers_are_present() -> None:
    for label in (
        "工作台",
        "新建诊断",
        "诊断任务",
        "诊断结果",
        "Agent 调查",
        "图谱与仿真",
        "大模型配置",
        "内置知识图谱",
    ):
        assert label in HTML
    for label in ("结论", "证据", "图谱与机理", "仿真", "Agent 调查", "技术详情"):
        assert label in OPERATIONS


def test_probe_display_mapping_and_minimal_request_contract() -> None:
    for probe_id, label in {
        "EVIDENCE_THRESHOLD_MARGIN": "阈值余量复核",
        "COMMAND_RESPONSE_RESIDUAL": "命令响应检查",
        "SIMULATION_COUNTERFACTUAL_REVIEW": "反事实仿真复核",
        "DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW": "泵机理仿真复核",
    }.items():
        assert probe_id in OPERATIONS
        assert label in OPERATIONS
    assert "return {probe_id:" in OPERATIONS
    assert "JSON.stringify(probeRequestBody(selected))" in REPORT


def test_browser_does_not_offer_arbitrary_probe_or_graph_execution_fields() -> None:
    forbidden_ids = (
        'id="probe-channel"',
        'id="probe-threshold"',
        'id="probe-window"',
        'id="probe-algorithm"',
        'id="cypher"',
        'id="simulator-function"',
    )
    for marker in forbidden_ids:
        assert marker not in HTML
    assert "不开放任意 Cypher" in HTML
    assert "不提供任意函数或参数执行" in OPERATIONS


def test_execution_diagnosis_and_release_are_independent_fields() -> None:
    assert "executionStatus:" in OPERATIONS
    assert "diagnosisDisposition:" in OPERATIONS
    assert "releaseStatus:" in OPERATIONS
    assert "job.status === \"SUCCEEDED\"" not in OPERATIONS
    assert "执行完成不会被转换为诊断成功" in OPERATIONS
    assert "rawReview===true?true:rawReview===false?false:null" in OPERATIONS
    assert "人工复核要求" in OPERATIONS
    assert "未提供" in OPERATIONS


def test_formal_evidence_unavailable_is_explicit() -> None:
    assert "正式证据摘要不可用" in OPERATIONS
    assert "波形、RAG、Agent 推理和图表不补造正式证据" in OPERATIONS
