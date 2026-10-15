from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "src" / "astrograph" / "resources"
HTML = RESOURCES / "platform_ui.html"
CSS = RESOURCES / "platform_ui.css"
JS = RESOURCES / "platform_ui.js"
CORE_JS = RESOURCES / "platform_core.js"
REPORT_JS = RESOURCES / "platform_report.js"
SYSTEM_JS = RESOURCES / "platform_system.js"
AGENT_JS = RESOURCES / "platform_agent.js"
DASHBOARD_JS = RESOURCES / "platform_dashboard.js"
JOB_JS = RESOURCES / "platform_job.js"
AGENT_LIVE_JS = RESOURCES / "platform_agent_live.js"
KNOWLEDGE_JS = RESOURCES / "platform_knowledge.js"
SHELL_JS = RESOURCES / "platform_shell.js"


def _text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in (HTML, CSS, CORE_JS, REPORT_JS, SYSTEM_JS, AGENT_JS, DASHBOARD_JS, JOB_JS, AGENT_LIVE_JS, KNOWLEDGE_JS, SHELL_JS, JS))


def test_maintenance_renderer_keeps_agent_receipt_helper_available() -> None:
    text = _text()
    assert "function agentExecutionHtml(agent)" in text
    assert "function safeUiSection(label,build)" in text
    assert "safeUiSection('Agent 执行回执'" in text


def test_dashboard_layout_has_font_scale_and_container_responsive_guards() -> None:
    text = _text()
    assert "container-name:dashboard" in text
    assert '@container dashboard (max-width:1120px)' in text
    assert 'document.documentElement.dataset.fontBand' in text
    assert 'html[data-font-band="xlarge"] .ops-grid' in text
    assert "#view-dashboard .ops-screen{flex:1" in text


def test_dashboard_runtime_cards_do_not_depend_on_fixed_content_heights_after_overrides() -> None:
    text = _text()
    css = CSS.read_text(encoding="utf-8")
    marker = css.index("/* v0.12 layout/runtime stability hardening */")
    overrides = css[marker:]
    assert ".ops-latest{height:auto" in overrides
    assert ".ops-mode-wrap{height:auto" in overrides
    assert ".ops-panel{overflow:visible" in overrides
    assert ".ops-grid{flex:1;grid-template-rows:minmax(255px,auto) minmax(235px,auto)" in overrides


def test_critical_maintenance_helpers_are_declared() -> None:
    text = _text()
    critical = {
        "agentExecutionHtml",
        "maintenanceTemporalHtml",
        "reviewActionsHtml",
        "findingsHtml",
        "mechanismHtml",
        "multidimHtml",
        "alternativesHtml",
        "diagnosticHistoryHtml",
        "bindMaintenanceStatInteractions",
        "activateMaintenanceTab",
        "evidenceInputContractHtml",
    }
    declared = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", text))
    assert critical <= declared


def test_operator_labels_hide_common_internal_status_codes_on_dashboard() -> None:
    text = _text()
    assert "REVIEW_REQUIRED:'需人工复核'" in text
    assert "PIPELINE_RUNTIME:'运行诊断工件'" in text
    assert "ADCS_RW_TORQUE_AUTHORITY_LOSS:'反作用轮扭矩能力下降'" in text


def test_maintenance_optional_interactions_are_isolated_after_core_render() -> None:
    text = _text()
    assert "maintenance enhancement failed" in text
    assert "诊断结果已加载；部分交互初始化失败" in text
    assert "维护工作台主数据加载失败" in text


def test_operator_channel_vocabulary_labels_common_adcs_channels() -> None:
    from astrograph.integrations.operator_vocabulary import channel_label_zh

    assert channel_label_zh("adcs.rw.measured_speed_rad_s_2") == "反作用轮 #3 转速"
    assert channel_label_zh("adcs.rw.motor_current_a_0") == "反作用轮 #1 电机电流"
    assert channel_label_zh("adcs.rw.speed_response_residual_norm_nm") == "反作用轮转速响应总残差"
    assert channel_label_zh("custom.signal") == "custom.signal"


def test_operator_ui_avoids_raw_codes_in_primary_dashboard_and_mode_tool_chips() -> None:
    text = _text()
    assert "排队 + 执行" in text
    assert "候选出现次数" in text
    assert "任务结果统计，不代表真实故障发生率" in text
    assert "const [,execText]=badgeFor(exec);const [,diagText]=badgeFor(disp)" in text
    assert "${artifactKindZh[vm.kind]||vm.kind||'未知诊断工件'}" in text
    # Compact task-mode tools use the operator display name, not the machine tool ID.
    assert "<strong>${esc(label)}</strong>${compact?'':" in text


def test_new_task_requires_backend_preflight_before_start() -> None:
    text = _text()
    assert 'id="preflight-btn"' in text
    assert 'id="preflight-result"' in text
    assert '/preflight`' in text
    assert "state.preflight?.can_start" in text
    assert "function isExecCandidate(name)" in text
    assert "是否可执行以后端预检为准" in text


def test_r9_maintenance_ui_exposes_machine_readable_input_contract_without_claiming_evidence_creation() -> None:
    text = _text()
    assert "正式证据输入合同" in text
    assert "复制输入合同 JSON" in text
    assert "data-copy-input-contract" in text
    assert "特征输出单位" in text
    assert "不会自行采集数据" in text or "不修改正式证据" in text


def test_r10_maintenance_ui_exposes_controlled_supplemental_input_actions_without_release_claims() -> None:
    text = _text()
    assert "复制补采模板 JSON" in text
    assert "导入补充输入 JSON" in text
    assert "data-copy-input-template" in text
    assert "data-import-supplemental-input" in text
    assert "admit-feature-input" in text
    assert "原始报告与发布状态未改变" in text
    assert "完整 telemetry/profile admission" in text
