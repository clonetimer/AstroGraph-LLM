from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src/astrograph/resources/platform_ui.html").read_text(encoding="utf-8")
RESOURCES = ROOT / "src/astrograph/resources"
JS = "\n".join((RESOURCES / name).read_text(encoding="utf-8") for name in ("platform_core.js", "platform_report.js", "platform_system.js", "platform_agent.js", "platform_dashboard.js", "platform_job.js", "platform_agent_live.js", "platform_knowledge.js", "platform_shell.js", "platform_ui.js"))
UI_TEXT = HTML + "\n" + JS


def test_dashboard_prioritizes_maintenance_actions_and_demotes_engineering_statistics():
    assert "当前需处理" in UI_TEXT
    assert 'id="dashboard-attention"' in UI_TEXT
    assert "执行中任务" in UI_TEXT
    assert 'id="dashboard-active"' in UI_TEXT
    assert "平台统计与工程监控" in UI_TEXT
    assert "候选出现次数" in UI_TEXT
    assert "任务结果统计，不代表真实故障发生率" in UI_TEXT
    assert "高频故障候选" not in UI_TEXT


def test_execution_and_diagnosis_states_are_visually_distinct_on_dashboard():
    assert "<span>执行完成</span>" in UI_TEXT
    assert "不等于已形成诊断" in UI_TEXT
    assert "执行：${esc(execText)} · 诊断：${esc(diagText)}" in UI_TEXT
    assert "正式发布受阻" in UI_TEXT


def test_agent_strategy_and_tool_details_are_advanced_not_primary_inputs():
    assert 'id="new-advanced-summary"' in UI_TEXT
    assert "modeZh[state.agentMode]||active?.display_name" in UI_TEXT
    assert "手动指定调查策略" in UI_TEXT
    assert "高级 · 当前策略允许的工具" in UI_TEXT
    assert "调查增强" in UI_TEXT
    assert "大模型调用</span>" not in UI_TEXT


def test_maintenance_result_exposes_diagnostic_contribution_without_calling_tool_use_gain():
    assert "调查贡献审计" in UI_TEXT
    assert "仅重算已有信息" in UI_TEXT
    assert "新原始观测" in UI_TEXT
    assert "正式新证据" in UI_TEXT
    assert "最近自动复核" in UI_TEXT
    assert "正式缺口仍未关闭" in UI_TEXT
    assert "shadow probe 不会自动关闭正式证据缺口" in UI_TEXT
