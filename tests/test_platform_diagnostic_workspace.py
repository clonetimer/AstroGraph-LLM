from pathlib import Path

from astrograph.integrations.upload_report import PLATFORM_UI_ASSET_VERSION, platform_asset


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "src/astrograph/resources"
WORKSPACE = (RESOURCES / "platform_diagnostic_workspace.js").read_text(encoding="utf-8")
OPERATIONS = (RESOURCES / "platform_operations.js").read_text(encoding="utf-8")
REPORT = (RESOURCES / "platform_report.js").read_text(encoding="utf-8")
UI = (RESOURCES / "platform_ui.js").read_text(encoding="utf-8")
HTML = (RESOURCES / "platform_ui.html").read_text(encoding="utf-8")


def test_r35_10_asset_version_and_workspace_module_are_served() -> None:
    assert PLATFORM_UI_ASSET_VERSION == "v0.21.1"
    source, media_type = platform_asset("platform_diagnostic_workspace.js")
    assert media_type.startswith("application/javascript")
    assert "renderDiagnosticWorkspace" in source
    assert "/platform/assets/v0.21.1/platform_ui.js" in HTML
    assert "/platform/assets/v0.21.1/platform_ui.css" in HTML


def test_workspace_is_hooked_to_the_production_projection_endpoint() -> None:
    assert "operationsHooks.renderDiagnosticWorkspace?.({report,maintenance,job,visualization,diagnosticWorkspace})" in OPERATIONS
    assert "maintenance/diagnostic-workspace" in REPORT
    assert "enhanceOperationsResult(report,m,jobState||state.job,viz,diagnosticWorkspace)" in REPORT
    assert "import { renderDiagnosticWorkspace } from './platform_diagnostic_workspace.js';" in UI
    assert "astrograph-diagnostic-workspace-presentation.v1" in WORKSPACE


def test_workspace_has_ephemeral_selection_and_fail_closed_empty_states() -> None:
    for field in ("candidateId", "componentId", "evidenceId", "graphNodeId", "observedTraceId", "simulationTraceId"):
        assert field in WORKSPACE
    assert "当前报告没有可交互图谱数据" in WORKSPACE
    assert "当前诊断没有可叠加的数值仿真轨迹" in WORKSPACE
    assert "当前遥测通道没有匹配的仿真轨迹" in WORKSPACE
    assert "正式证据摘要不可用" in WORKSPACE
    assert "simulation fit / distance ≠ fault probability" in WORKSPACE


def test_workspace_remains_read_only_and_does_not_execute_governed_actions() -> None:
    assert "fetch(" not in WORKSPACE
    assert "api(" not in WORKSPACE
    assert "localStorage" not in WORKSPACE
    assert "sessionStorage" not in WORKSPACE
    assert "executeProbe" not in WORKSPACE
    assert "runSimulation" not in WORKSPACE
    assert "type=\"number\"" not in WORKSPACE
    assert "点击步骤只导航到已有材料，不会自动执行 Probe" in WORKSPACE


def test_overlay_selector_depends_on_channel_compatible_traces() -> None:
    assert "const stateMessage=!simulated.length" in WORKSPACE
    assert "!compatible.length?'当前遥测通道没有匹配的仿真轨迹'" in WORKSPACE
    assert "${compatible.length?`<label>仿真轨迹" in WORKSPACE
    assert "trace.points||trace.samples" in WORKSPACE
    assert "fit_score" not in WORKSPACE
    assert "% probability" not in WORKSPACE.lower()


def test_r35_product_experience_stays_projection_only_and_ephemeral() -> None:
    assert "function workspaceAvailability" in WORKSPACE
    assert "工作区材料概览" in WORKSPACE
    assert "恢复默认上下文" in WORKSPACE
    assert "按 Esc 可恢复默认上下文" in WORKSPACE
    assert "当前图谱节点" in WORKSPACE
    assert "关联数量只来自当前投影内显式 edges" in WORKSPACE
    assert "时间范围" in WORKSPACE
    assert "浏览器不执行仿真" in WORKSPACE
    assert "aria-selected" in WORKSPACE
    assert "aria-pressed" in WORKSPACE


def test_r35_1_review_identity_and_handoff_are_projection_only() -> None:
    assert "审查身份与交接" in WORKSPACE
    assert "function handoffSummaryText" in WORKSPACE
    for field in ("presentation_sha256", "source_bundle_status", "source_bundle_sha256", "provenance_sha256", "source_trace_sha256"):
        assert field in WORKSPACE
    assert "TRANSIENT_ALTERNATIVE_BROWSE_ONLY" in WORKSPACE
    assert "FROZEN_REPORT_CANDIDATE" in WORKSPACE
    assert "复制交接摘要" in WORKSPACE
    assert "navigator.clipboard" in WORKSPACE
    assert "retained source bytes" in WORKSPACE


def test_r35_2_review_session_is_ephemeral_and_signal_only() -> None:
    assert "本页材料复核" in WORKSPACE
    assert "function reviewAttentionSignals" in WORKSPACE
    assert "function reviewSessionHtml" in WORKSPACE
    assert "TRANSIENT_CANDIDATE_BROWSE" in WORKSPACE
    assert "CONTRADICTING_EVIDENCE_PRESENT" in WORKSPACE
    assert "MISSING_EVIDENCE_DECLARED" in WORKSPACE
    assert "AGENT_STEP_INCOMPLETE" in WORKSPACE
    assert "勾选只保存在当前页面内存" in WORKSPACE
    assert "不是正式审批、签名或 release gate" in WORKSPACE
    assert "不生成新的故障判断、EvidenceFact、概率或资格结论" in WORKSPACE
    assert "Review session (ephemeral):" in WORKSPACE

def test_r35_3_review_closure_is_session_trace_only() -> None:
    assert "function reviewClosureState" in WORKSPACE
    assert "function reviewClosureHtml" in WORKSPACE
    for status in ("REVIEW_INCOMPLETE", "REVIEWED_WITH_ATTENTION", "REVIEWED_NO_EXPLICIT_ATTENTION"):
        assert status in WORKSPACE
    assert "SESSION_TRACE_ONLY_NOT_APPROVAL" in WORKSPACE
    assert "不是批准、签名、release gate" in WORKSPACE
    assert "Review closure status:" in WORKSPACE
    assert "Open review areas:" in WORKSPACE
    assert "Open attention signal codes:" in WORKSPACE



def test_r35_4_unresolved_item_navigation_is_read_only_and_does_not_resolve_state() -> None:
    assert "function reviewNavigationSpec" in WORKSPACE
    assert "function navigateReviewItem" in WORKSPACE
    assert "function reviewNavigationHtml" in WORKSPACE
    assert "data-dw-review-nav-area" in WORKSPACE
    assert "data-dw-review-nav-signal" in WORKSPACE
    assert "未关闭项定位" in WORKSPACE
    assert "不会自动标记已复核，也不会清除 attention signal" in WORKSPACE
    assert "CONTRADICTING_EVIDENCE_PRESENT" in WORKSPACE
    assert "AGENT_STEP_INCOMPLETE" in WORKSPACE


def test_r35_5_review_coverage_provenance_is_ephemeral_and_scope_only() -> None:
    assert "function reviewCoverageBasis" in WORKSPACE
    assert "function reviewCoverageDrift" in WORKSPACE
    assert "function reviewCoverageHtml" in WORKSPACE
    assert "BROWSER_SESSION_SCOPE_V1" in WORKSPACE
    assert "REVIEW_COVERAGE_CONTEXT_CHANGED" in WORKSPACE
    assert "Review coverage provenance (ephemeral):" in WORKSPACE
    assert "Coverage drift areas:" in WORKSPACE
    assert "MATERIAL_SCOPE_TRACE_ONLY_NOT_AUDIT_OR_APPROVAL" in WORKSPACE
    assert "不是审计签名、批准记录或材料充分性证明" in WORKSPACE


def test_r35_6_incoming_coverage_replay_is_browser_local_comparison_only() -> None:
    assert "BROWSER_REVIEW_COVERAGE_REPLAY_V1" in WORKSPACE
    assert "function reviewCoverageReplayManifest" in WORKSPACE
    assert "function parseReviewCoverageReplay" in WORKSPACE
    assert "function reviewCoverageReplayComparison" in WORKSPACE
    assert "MATCHED" in WORKSPACE
    assert "CHANGED" in WORKSPACE
    assert "NOT_REPRODUCED" in WORKSPACE
    assert "NOT_PROVIDED" in WORKSPACE
    assert "BEGIN_REVIEW_COVERAGE_REPLAY_V1" in WORKSPACE
    assert "Incoming scope comparison:" in WORKSPACE
    assert "INCOMING_SCOPE_COMPARISON_ONLY_NOT_AUDIT_OR_APPROVAL" in WORKSPACE
    assert "不会自动恢复 selection、不会标记已复核，也不会改变 closure 或 attention signal" in WORKSPACE
    assert "不是审计结论、差异处置或批准" in WORKSPACE


def test_r35_7_incoming_replay_delta_queue_is_guidance_only() -> None:
    assert "const REPLAY_DELTA_STATUSES = new Set(['CHANGED','NOT_REPRODUCED','NOT_PROVIDED'])" in WORKSPACE
    assert "function reviewReplayDeltaQueue" in WORKSPACE
    assert "function setReplayDeltaQueueRevisited" in WORKSPACE
    assert "function advanceReplayDeltaQueue" in WORKSPACE
    assert "接手重新查看队列" in WORKSPACE
    assert "REVISITED" in WORKSPACE
    assert "PENDING" in WORKSPACE
    assert "MATCHED 不进入队列" in WORKSPACE
    assert "INCOMING_REPLAY_REREVIEW_TRACE_ONLY_NOT_RESOLUTION_OR_APPROVAL" in WORKSPACE
    assert "不会把 CHANGED/NOT_REPRODUCED/NOT_PROVIDED 改成 MATCHED" in WORKSPACE
    assert "不表示差异已处置、批准或 release" in WORKSPACE



def test_r35_8_replay_basis_diff_is_exact_display_only_and_noncausal() -> None:
    assert "function replayBasisDiff" in WORKSPACE
    assert "function replayBasisDiffSnapshot" in WORKSPACE
    for status in ("SAME", "VALUE_CHANGED", "INCOMING_ONLY", "CURRENT_ONLY"):
        assert status in WORKSPACE
    assert "字段级 basis diff" in WORKSPACE
    assert "变化字段：" in WORKSPACE
    assert "Incoming changed basis fields:" in WORKSPACE
    assert "INCOMING_REPLAY_BASIS_DIFF_DISPLAY_ONLY_NOT_CAUSAL_EXPLANATION_OR_RESOLUTION" in WORKSPACE
    assert "不推断变化原因、不判断哪一侧正确，也不表示差异已处置" in WORKSPACE

def test_r35_9_incoming_rereview_convergence_is_summary_only() -> None:
    assert "BROWSER_INCOMING_REREVIEW_CONVERGENCE_V1" in WORKSPACE
    assert "function incomingReplayConvergenceState" in WORKSPACE
    assert "function incomingReplayConvergenceHtml" in WORKSPACE
    for status in ("ALL_MATCHED", "DELTAS_ALL_REVISITED", "REREVIEW_PENDING"):
        assert status in WORKSPACE
    assert "Incoming re-review convergence:" in WORKSPACE
    assert "Incoming re-review convergence counts:" in WORKSPACE
    assert "INCOMING_REREVIEW_CONVERGENCE_SUMMARY_ONLY_NOT_DISCREPANCY_RESOLUTION_OR_APPROVAL" in WORKSPACE
    assert "不是 discrepancy resolution、批准、签名或 release gate" in WORKSPACE



def test_r35_10_product_experience_stable_candidate_disambiguates_review_statuses() -> None:
    assert "R35_PRODUCT_EXPERIENCE_STABLE_CANDIDATE_READ_ONLY" in WORKSPACE
    assert "function reviewStatusOverviewState" in WORKSPACE
    assert "function reviewStatusOverviewHtml" in WORKSPACE
    assert "复核状态总览" in WORKSPACE
    assert "本页材料复核状态" in WORKSPACE
    assert "接手重新查看状态" in WORKSPACE
    assert "四组状态来自既有只读契约，彼此独立，不构成统一审批 gate" in WORKSPACE
    assert "Local page review status:" in WORKSPACE
    assert "Incoming re-review status:" in WORKSPACE
    assert "Review status boundary:" in WORKSPACE
    assert "接手复核收口摘要" not in WORKSPACE
    assert "会话复核收口状态" not in WORKSPACE
    # Keep the older handoff key for text-consumer compatibility while adding the explicit local alias.
    assert "Review closure status:" in WORKSPACE
