from __future__ import annotations

from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.integrations.temporal_localization import localize_from_visualization
from astrograph.integrations.multidimensional_diagnostics import assess_multidimensional_telemetry
from astrograph.integrations.operator_vocabulary import channel_label_zh
from astrograph.evidence import EpisodeEvidenceBundle, EvidenceFactStatus
from astrograph.investigation.diagnostic_probes import build_evidence_margin_probe_catalog
from astrograph.integrations.maintenance_input_admission import assess_supplemental_admission_support


MAINTENANCE_SCHEMA_VERSION = "astrograph-maintenance-diagnosis.v8"


FAULT_ZH = {
    "ADCS_RW_JAM": "反作用轮卡滞",
    "ADCS_RW_FRICTION_INCREASE": "反作用轮摩擦增大",
    "ADCS_RW_TORQUE_AUTHORITY_LOSS": "反作用轮扭矩能力下降",
    "NORMAL": "未检出已知故障",
}
CHANNEL_ZH = {
    "adcs.rw.measured_speed_rad_s_2": "反作用轮转速",
    "adcs.rw.motor_current_a_2": "反作用轮电机电流",
    "adcs.pointing_error_deg": "姿态指向误差",
    "adcs.rw.command_execution_residual_nm_2": "反作用轮指令执行残差",
    "adcs.rw.command_execution_residual_norm_nm": "反作用轮执行总残差",
    "adcs.rw.speed_response_residual_nm_2": "反作用轮转速响应残差",
}
MECH_ZH = {
    "rotor_lock": "轮体卡滞",
    "wheel_torque_path_lost": "扭矩执行路径受阻",
    "momentum_management_degraded": "动量管理能力下降",
    "pointing_error_response": "姿态指向误差增大",
    "causes": "导致",
    "reduces": "削弱",
    "can_increase": "可能增大",
}
MECH_CHANNELS = {
    "rotor_lock": ["adcs.rw.measured_speed_rad_s_2", "adcs.rw.motor_current_a_2"],
    "wheel_torque_path_lost": ["adcs.rw.command_execution_residual_nm_2", "adcs.rw.command_execution_residual_norm_nm"],
    "momentum_management_degraded": ["adcs.rw.speed_response_residual_nm_2", "adcs.rw.command_execution_residual_norm_nm"],
    "pointing_error_response": ["adcs.pointing_error_deg"],
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MaintenanceFinding(_Strict):
    finding_id: str
    title: str
    observation: str
    channel: str | None = None
    action_label: str | None = None
    evidence_kind: str = "TELEMETRY"


class ReviewAction(_Strict):
    action_id: str
    title: str
    detail: str = ""
    owner: str = "operator"
    priority: str = "MEDIUM"
    actionable: bool = True
    source_blocker: str | None = None


class MechanismGraph(_Strict):
    status: str
    graph_kind: str
    title: str
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    query_backed: bool = False
    contributes_to_diagnosis: bool = False
    boundary_zh: str = ""




class EvidenceSummaryItem(_Strict):
    evidence_definition_id: str
    evidence_fact_id: str | None = None
    label: str
    status: str
    role: str
    observed_value: float | None = None
    threshold_value: float | None = None
    unit: str | None = None
    feature_definition_id: str | None = None


class MaintenanceEvidenceSummary(_Strict):
    status: str
    fault_id: str | None = None
    critical_total: int = 0
    critical_satisfied: int = 0
    critical_missing: int = 0
    critical_contradicted: int = 0
    supporting_items: list[EvidenceSummaryItem] = Field(default_factory=list)
    missing_items: list[EvidenceSummaryItem] = Field(default_factory=list)
    contradicting_items: list[EvidenceSummaryItem] = Field(default_factory=list)
    summary_zh: str
    source_schema_version: str | None = None
    formal_evidence_only: bool = True
    evidence_source_mode: str = "ORIGINAL_FORMAL_BUNDLE"
    supplemental_overlay_receipt_sha256: str | None = None
    original_release_unchanged: bool = True
    boundary_zh: str = "仅统计正式 EpisodeEvidenceBundle 中的关键证据；不使用波形统计、文档命中或 LLM 解释伪造证据完整度。"


class InputAdmissionRequirement(_Strict):
    requirement_id: str
    status: str
    detail: str


class EvidenceInputContract(_Strict):
    contract_id: str
    status: str
    evidence_definition_id: str
    feature_definition_id: str | None = None
    feature_name: str | None = None
    current_fact_status: str
    source_field_template: str | None = None
    resolved_source_field: str | None = None
    aggregation: str | None = None
    required_semantics: list[str] = Field(default_factory=list)
    required_unit: str | None = None
    observed_unit: str | None = None
    threshold_unit: str | None = None
    unit_contract_scope: str = "FEATURE_RESULT_OUTPUT"
    window_scope: str | None = None
    required_window_start_s: float | None = None
    required_window_end_s: float | None = None
    observed_window_start_s: float | None = None
    observed_window_end_s: float | None = None
    window_source: str = "UNRESOLVED"
    paired_nominal_required: bool = False
    paired_nominal_run_id: str | None = None
    feature_algorithm_version: str | None = None
    observed_algorithm_version: str | None = None
    input_data_sha256: str | None = None
    sample_count: int | None = None
    quality_flags: list[str] = Field(default_factory=list)
    admission_requirements: list[InputAdmissionRequirement] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    supplemental_admission_status: str = "UNSUPPORTED"
    supplemental_admission_reason_codes: list[str] = Field(default_factory=list)
    supplemental_operator_id: str | None = None
    supplemental_operator_runtime_version: str | None = None
    supplemental_capsule_schema_version: str | None = None
    operator_action_zh: str
    boundary_zh: str = (
        "该输入合同只把正式 EvidenceDefinition/FeatureDefinition 翻译成可执行的数据准入要求。"
        "其中 required_unit 是 FeatureResult 输出单位，不冒充原始遥测通道单位；原始通道单位仍由 telemetry/profile admission 约束。"
        "它不会自行采集数据、不会绕过 telemetry admission，也不会因为字段齐全就自动生成正式 EvidenceFact。"
    )


class NextDiagnosticCheck(_Strict):
    status: str
    check_id: str | None = None
    title: str
    rationale: str
    evidence_definition_id: str | None = None
    feature_definition_id: str | None = None
    probe_id: str | None = None
    auto_executable: bool = False
    blocking_reason: str | None = None
    expected_information: str | None = None
    shadow_only: bool = True
    input_contract_id: str | None = None


class ProbeReviewSummary(_Strict):
    result_sha256: str | None = None
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    probe_id: str
    probe_kind: str | None = None
    evidence_definition_id: str | None = None
    status: str = "UNKNOWN"
    signed_margin: float | None = None
    normalized_signed_margin: float | None = None
    unit: str | None = None
    formal_status_after: str | None = None
    formal_gap_open_after: bool | None = None
    gap_closed_by_probe: bool = False
    linkage_status: str = "UNAVAILABLE"
    next_action_zh: str = "继续按正式证据协议处理。"
    boundary_zh: str = (
        "该记录是维护员触发的 shadow probe 复核结果。它不会修改 EpisodeEvidenceBundle，"
        "也不能因为 signed margin 与阈值一致就自动关闭正式证据缺口。"
    )


class DiagnosticContributionSummary(_Strict):
    status: str = "UNAVAILABLE"
    structural_status: str | None = None
    probe_execution_count: int = 0
    new_observation_count: int = 0
    new_raw_observation_count: int = 0
    formal_evidence_generated_count: int = 0
    resolved_gap_count: int = 0
    candidate_set_changed: bool = False
    decision_changed: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    reviewed_evidence_definition_ids: list[str] = Field(default_factory=list)
    reviewed_current_gap_definition_ids: list[str] = Field(default_factory=list)
    unresolved_gap_definition_ids: list[str] = Field(default_factory=list)
    unresolved_gap_count: int = 0
    gap_linkage_status: str = "UNAVAILABLE"
    gap_linkage_summary_zh: str = "当前没有可核对的调查贡献—正式证据缺口关联。"
    operator_summary_zh: str = "当前报告没有可展示的调查贡献审计。"
    boundary_zh: str = (
        "调查贡献审计用于区分‘执行了多步/工具’与‘真正增加了诊断信息’。"
        "shadow probe 对已准入数据的重算不计为新的原始观测，也不自动构成正式 EvidenceFact；"
        "‘复核了某条证据’不等于‘关闭了该证据缺口’。"
    )


class CandidateContextGuard(_Strict):
    status: str = "UNAVAILABLE"
    compact_reasoning_allowed: bool | None = None
    candidate_set_status: str | None = None
    escape_action: str | None = None
    escape_reason_codes: list[str] = Field(default_factory=list)
    preferred_candidate_fault_ids: list[str] = Field(default_factory=list)
    unselected_mandatory_fault_ids: list[str] = Field(default_factory=list)
    configured_top_k: int | None = None
    effective_top_k: int | None = None
    forced_candidate_count: int | None = None
    operator_summary_zh: str = "当前报告没有可展示的候选上下文保护回执。"
    boundary_zh: str = (
        "候选上下文保护只描述 LLM 收到的候选范围是否因运行时安全策略而扩展；"
        "它不提高任何故障的成立概率，也不构成正式诊断证据。"
    )

class MaintenancePresentation(_Strict):
    schema_version: str = MAINTENANCE_SCHEMA_VERSION
    job_id: str
    headline: str
    fault_id: str | None = None
    subsystem: str | None = None
    component: str | None = None
    diagnosis_status: str
    status_zh: str
    release_status_zh: str
    candidate_context_guard: CandidateContextGuard
    probe_review_history: list[ProbeReviewSummary] = Field(default_factory=list)
    diagnostic_contribution: DiagnosticContributionSummary
    evidence_summary: MaintenanceEvidenceSummary
    evidence_gap_input_contracts: list[EvidenceInputContract] = Field(default_factory=list)
    next_diagnostic_check: NextDiagnosticCheck
    temporal_localization: dict[str, Any]
    multidimensional_assessment: dict[str, Any] = Field(default_factory=dict)
    key_findings: list[MaintenanceFinding] = Field(default_factory=list)
    alternative_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    review_actions: list[ReviewAction] = Field(default_factory=list)
    mechanism_graph: MechanismGraph
    document_references: list[dict[str, Any]] = Field(default_factory=list)
    technical_trace_ref: str
    operator_summary: str


def _candidate(report: dict[str, Any]) -> tuple[str | None, str]:
    """Return the operator-facing candidate without inventing cross-artifact semantics.

    Pipeline/runtime diagnoses keep precedence.  Native OAG reports use a different
    shape (``analysis.oag_result.output.summary``), so their evidence-gated
    ``suspected_fault_id`` is projected explicitly.  OAG READY_FOR_EXPLANATION is
    review-only by contract and therefore maps to REVIEW_REQUIRED rather than a
    formal DIAGNOSED disposition.
    """
    analysis = report.get("analysis") or {}
    pipeline = analysis.get("pipeline_result") or {}
    diagnosis = analysis.get("diagnosis") or {}
    fusion = pipeline.get("fusion") or {}
    # ``pipeline.fault_id`` is the upstream experiment/artifact identity for the
    # SAT-SIM closure and may differ from the actual screening/fusion result.  The
    # maintenance title must follow the diagnosis candidate, never overwrite it with
    # the source artifact's requested fault label.  A nominal/top-tie fusion therefore
    # remains candidate-less instead of falling back to that upstream label.
    fault_id = fusion.get("top_fault_id") or diagnosis.get("selected_fault_id") or diagnosis.get("conclusion_fault_id")
    disposition = pipeline.get("disposition") or diagnosis.get("disposition")
    if fault_id or disposition:
        return (str(fault_id) if fault_id else None), str(disposition or "UNKNOWN")

    oag_result = analysis.get("oag_result") or {}
    oag_output = oag_result.get("output") or {}
    summary = oag_output.get("summary") or {}
    assessment = oag_output.get("assessment") or {}
    fault_id = summary.get("suspected_fault_id") or assessment.get("selected_fault_id")
    if not fault_id:
        decision_ids = list((summary.get("diagnostic_decision") or {}).get("candidate_fault_ids") or [])
        bounded_ids = list(assessment.get("bounded_candidate_fault_ids") or [])
        bounded = decision_ids or bounded_ids
        if len(bounded) == 1:
            # A singleton bounded hypothesis is useful for directing evidence
            # acquisition, but remains review-only and is never projected as a
            # formal diagnosis.
            fault_id = bounded[0]
    status = str(summary.get("status") or assessment.get("conclusion_status") or "UNKNOWN")
    disposition_map = {
        "READY_FOR_EXPLANATION": "REVIEW_REQUIRED",
        "SUPPORTED_CANDIDATE": "REVIEW_REQUIRED",
        "MODEL_HYPOTHESIS": "REVIEW_REQUIRED",
        "ALTERNATIVES": "REVIEW_REQUIRED",
        "AMBIGUOUS": "REVIEW_REQUIRED",
        "INSUFFICIENT_EVIDENCE": "ABSTAIN",
        "UNKNOWN": "ABSTAIN",
        "BLOCKED": "BLOCKED",
    }
    return (str(fault_id) if fault_id else None), disposition_map.get(status, status)


def _channel_observation(channel: dict[str, Any]) -> str:
    points = list(channel.get("points") or [])
    values = [float(p["v"]) for p in points if p.get("v") is not None]
    if len(values) < 6:
        return "已纳入关键波形复核。"
    n = max(2, len(values)//5)
    early = values[:n]
    late = values[-n:]
    e, l = fmean(early), fmean(late)
    span = max(values)-min(values)
    delta = l-e
    if span <= 1e-12:
        return "整段变化较小，作为对照通道保留。"
    ratio = abs(delta)/span
    if ratio >= 0.45:
        direction = "升高" if delta > 0 else "下降"
        return f"后段相对前段明显{direction}，建议结合时间轴核对变化起点。"
    # A simple oscillation cue for current-like signals.
    diffs = [values[i]-values[i-1] for i in range(1, len(values))]
    sign_changes = sum(1 for a,b in zip(diffs,diffs[1:]) if a*b < 0)
    if sign_changes >= max(4, len(diffs)//8):
        return "出现较明显的往复波动/振荡，建议与转速和姿态误差联动核对。"
    return "存在明显波形变化，建议在共享时间游标下与其他关键通道对照。"


def _key_findings(visualization: dict[str, Any], multidimensional: dict[str, Any] | None = None) -> list[MaintenanceFinding]:
    by_channel = {
        str(item.get("channel")): item
        for item in ((multidimensional or {}).get("channels") or [])
        if item.get("channel")
    }
    ranked = list(visualization.get("channels") or [])[:6]
    ranked.sort(
        key=lambda channel: max(
            [float(d.get("index") or 0.0) for d in (by_channel.get(str(channel.get("name") or ""), {}).get("dimensions") or [])]
            or [0.0]
        ),
        reverse=True,
    )
    findings: list[MaintenanceFinding] = []
    for i, channel in enumerate(ranked[:5], start=1):
        name = str(channel.get("name") or "")
        stat = by_channel.get(name) or {}
        observation = str(stat.get("operator_observation_zh") or _channel_observation(channel))
        findings.append(MaintenanceFinding(
            finding_id=f"FIND-{i:02d}", title=channel_label_zh(name, CHANNEL_ZH.get(name, name)), channel=name,
            observation=observation, action_label="定位曲线", evidence_kind="TELEMETRY_STATISTICAL_REVIEW",
        ))
    return findings


def _review_actions(blockers: list[str]) -> list[ReviewAction]:
    actions: list[ReviewAction] = []
    for item in blockers:
        raw = str(item)
        if raw.startswith("MISSING_REQUIRED_CHANNEL:"):
            channel = raw.split(":",1)[1]
            actions.append(ReviewAction(action_id=f"ACT-{len(actions)+1:02d}", title="确认缺失遥测通道", detail=f"诊断合同需要但当前工件未提供：{channel}", owner="operator", priority="HIGH", source_blocker=raw))
        elif raw == "NAMED_EXPERT_REVIEW_PENDING":
            actions.append(ReviewAction(action_id=f"ACT-{len(actions)+1:02d}", title="提交授权专家复核", detail="当前候选可以作为维护复核线索，但正式结论仍需命名专家确认。", owner="reviewer", priority="HIGH", source_blocker=raw))
        elif raw == "FORMAL_NEO4J_QUERY_NOT_EXECUTED":
            actions.append(ReviewAction(action_id=f"ACT-{len(actions)+1:02d}", title="可选：执行知识图谱复核", detail="本次未执行正式 Neo4j 只读查询；如本机知识图谱已配置，可补充机理路径证据。", owner="operator", priority="LOW", source_blocker=raw))
        elif raw.startswith("SAT_SIM_RUNTIME_STATUS:"):
            actions.append(ReviewAction(action_id=f"ACT-{len(actions)+1:02d}", title="确认仿真运行工件完整性", detail="当前工件仍带有仿真接口/运行状态限制，正式使用前需完成对应接口审查。", owner="engineer", priority="MEDIUM", source_blocker=raw))
    # Pure validation/governance blockers remain technical trace items, not operator chores.
    return actions[:8]


def _mechanism(report: dict[str, Any]) -> MechanismGraph:
    pipeline = ((report.get("analysis") or {}).get("pipeline_result") or {})
    kg = pipeline.get("kg_projection") or {}
    chain = pipeline.get("causal_chain") or kg.get("causal_chain") or []
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for row in chain:
        src, dst, rel = str(row.get("from") or ""), str(row.get("to") or ""), str(row.get("relation") or "")
        if not src or not dst:
            continue
        nodes[src] = {"id": src, "label": MECH_ZH.get(src, src), "kind": "mechanism", "related_channels": MECH_CHANNELS.get(src, [])}
        nodes[dst] = {"id": dst, "label": MECH_ZH.get(dst, dst), "kind": "mechanism", "related_channels": MECH_CHANNELS.get(dst, [])}
        edges.append({"from": src, "to": dst, "relation": rel, "label": MECH_ZH.get(rel, rel)})
    query_backed = bool((kg.get("forward_ranking") or {}).get("metadata", {}).get("graph_query_backed"))
    if not nodes:
        return MechanismGraph(status="UNAVAILABLE", graph_kind="NONE", title="故障机理", boundary_zh="当前报告没有可展示的机理路径。")
    return MechanismGraph(
        status="AVAILABLE", graph_kind=("FORMAL_KG_EVIDENCE" if query_backed else "REFERENCE_MECHANISM"),
        title=("本次证据图谱" if query_backed else "故障参考机理"), nodes=list(nodes.values()), edges=edges,
        query_backed=query_backed, contributes_to_diagnosis=bool(kg.get("contributes_to_fusion")),
        boundary_zh=("本次路径来自正式知识图谱只读查询。" if query_backed else "当前为候选故障对应的工程机理映射，用于理解与复核；不是本次正式图谱投票证据。"),
    )


def _alternatives(report: dict[str, Any], top_fault: str | None) -> list[dict[str, Any]]:
    pipeline = ((report.get("analysis") or {}).get("pipeline_result") or {})
    fusion = (pipeline.get("fusion") or {}).get("candidates") or []
    result = []
    for item in fusion:
        fid = str(item.get("fault_id") or "")
        if not fid or fid == top_fault or fid == "NORMAL":
            continue
        result.append({"fault_id": fid, "label": FAULT_ZH.get(fid, fid), "reason_zh": "作为相近故障候选保留，需结合关键波形与缺失证据进一步排除。"})
        if len(result) >= 3:
            break
    return result


def _documents(report: dict[str, Any]) -> list[dict[str, Any]]:
    knowledge = ((report.get("analysis") or {}).get("knowledge_enrichment") or {})
    hits = (knowledge.get("local_retrieval") or {}).get("hits") or []
    out = []
    for hit in hits[:8]:
        out.append({
            "filename": hit.get("filename"), "page": hit.get("page") or hit.get("page_number"),
            "quote": hit.get("quote") or hit.get("text") or "", "score": hit.get("score"),
            "diagnostic_vote_allowed": False,
        })
    return out



def extract_formal_evidence_bundle(report: dict[str, Any]) -> EpisodeEvidenceBundle | None:
    """Find only an explicitly serialized EpisodeEvidenceBundle in a report tree."""
    queue: list[Any] = [report]
    seen = 0
    while queue and seen < 20000:
        current = queue.pop(0)
        seen += 1
        if isinstance(current, dict):
            if current.get("schema_version") == "astrograph-episode-evidence-bundle.v1":
                try:
                    return EpisodeEvidenceBundle.model_validate(current)
                except Exception:
                    return None
            # Prefer semantically named keys, then bounded recursive traversal.
            for key in ("episode_evidence_bundle", "evidence_bundle"):
                value = current.get(key)
                if isinstance(value, dict):
                    queue.insert(0, value)
            for key, value in current.items():
                if key not in {"episode_evidence_bundle", "evidence_bundle"} and isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(item for item in current if isinstance(item, (dict, list)))
    return None


def _evidence_summary(
    report: dict[str, Any], fault_id: str | None, *,
    bundle_override: EpisodeEvidenceBundle | None = None,
    overlay_receipt_sha256: str | None = None,
) -> MaintenanceEvidenceSummary:
    bundle = bundle_override or extract_formal_evidence_bundle(report)
    if bundle is None:
        return MaintenanceEvidenceSummary(
            status="UNAVAILABLE", fault_id=fault_id,
            summary_zh="当前报告没有可验证的正式 EpisodeEvidenceBundle，不能用波形统计或模型解释代替证据完整度。",
        )
    if not fault_id:
        return MaintenanceEvidenceSummary(
            status="UNAVAILABLE", fault_id=None, source_schema_version=bundle.schema_version,
            summary_zh="当前尚未形成可对应到注册故障的正式候选，因此不计算故障级关键证据完整度。",
        )
    facts = {item.evidence_definition_id: item for item in bundle.evidence_facts}
    definitions = [
        item for item in bundle.evidence_definitions
        if item.target_fault_id == fault_id and item.critical
    ]
    supporting: list[EvidenceSummaryItem] = []
    missing: list[EvidenceSummaryItem] = []
    contradicting: list[EvidenceSummaryItem] = []
    applicable_total = 0
    for definition in definitions:
        fact = facts.get(definition.evidence_definition_id)
        status = fact.status if fact is not None else EvidenceFactStatus.MISSING_DATA
        if status == EvidenceFactStatus.NOT_APPLICABLE:
            continue
        applicable_total += 1
        item = EvidenceSummaryItem(
            evidence_definition_id=definition.evidence_definition_id,
            evidence_fact_id=(fact.evidence_fact_id if fact else None),
            label=definition.name, role=definition.role.value, status=status.value,
            observed_value=(fact.observed_value if fact else None),
            threshold_value=(fact.threshold_value if fact else float(definition.threshold)),
            unit=(fact.unit if fact else definition.threshold_unit),
            feature_definition_id=definition.feature_definition_id,
        )
        if status == EvidenceFactStatus.SATISFIED:
            supporting.append(item)
        elif status in {EvidenceFactStatus.NOT_SATISFIED, EvidenceFactStatus.CONTRADICTED}:
            contradicting.append(item)
        else:
            missing.append(item)
    if applicable_total == 0:
        summary = "该候选在当前正式证据协议中没有可展示的关键证据定义。"
    elif missing or contradicting:
        reasons=[]
        if missing: reasons.append(f"{len(missing)} 项关键证据缺失或不可判定")
        if contradicting: reasons.append(f"{len(contradicting)} 项关键证据存在反证")
        summary = f"关键证据 {len(supporting)}/{applicable_total} 满足；" + "，".join(reasons) + "。"
    else:
        summary = f"关键证据 {len(supporting)}/{applicable_total} 满足；当前正式证据摘要未发现关键缺口或反证。"
    overlay_applied = bundle_override is not None
    boundary = (
        "当前摘要基于服务器准入并确定性重算的维护补充证据 overlay；原始冻结报告和发布状态保持不变，"
        "该 overlay 必须重新进入完整生产流水线后才可成为 release evidence。"
        if overlay_applied else
        "仅统计正式 EpisodeEvidenceBundle 中的关键证据；不使用波形统计、文档命中或 LLM 解释伪造证据完整度。"
    )
    return MaintenanceEvidenceSummary(
        status="AVAILABLE", fault_id=fault_id, critical_total=applicable_total,
        critical_satisfied=len(supporting), critical_missing=len(missing),
        critical_contradicted=len(contradicting), supporting_items=supporting,
        missing_items=missing, contradicting_items=contradicting, summary_zh=summary,
        source_schema_version=bundle.schema_version, formal_evidence_only=(not overlay_applied),
        evidence_source_mode=("SUPPLEMENTAL_REANALYSIS_OVERLAY" if overlay_applied else "ORIGINAL_FORMAL_BUNDLE"),
        supplemental_overlay_receipt_sha256=overlay_receipt_sha256, original_release_unchanged=True,
        boundary_zh=boundary,
    )



def _evidence_gap_input_contracts(
    report: dict[str, Any], evidence: MaintenanceEvidenceSummary, *,
    bundle_override: EpisodeEvidenceBundle | None = None,
) -> list[EvidenceInputContract]:
    bundle = bundle_override or extract_formal_evidence_bundle(report)
    if bundle is None or evidence.status != "AVAILABLE":
        return []
    definitions = {item.evidence_definition_id: item for item in bundle.evidence_definitions}
    feature_definitions = {item.feature_definition_id: item for item in bundle.feature_definitions}
    facts = {item.evidence_definition_id: item for item in bundle.evidence_facts}
    feature_results_by_id = {item.feature_result_id: item for item in bundle.feature_results}
    feature_results_by_definition: dict[str, list[Any]] = {}
    for item in bundle.feature_results:
        feature_results_by_definition.setdefault(item.feature_definition_id, []).append(item)

    contracts: list[EvidenceInputContract] = []
    gap_items = [*evidence.missing_items, *evidence.contradicting_items]
    for gap in gap_items:
        definition = definitions.get(gap.evidence_definition_id)
        feature_definition = feature_definitions.get(gap.feature_definition_id or "")
        fact = facts.get(gap.evidence_definition_id)
        feature_result = feature_results_by_id.get(fact.feature_result_id) if fact is not None else None
        if feature_result is None and feature_definition is not None:
            candidates = feature_results_by_definition.get(feature_definition.feature_definition_id) or []
            feature_result = candidates[0] if len(candidates) == 1 else None

        requirements: list[InputAdmissionRequirement] = []
        blockers: list[str] = []
        if feature_definition is None:
            blockers.append("FEATURE_DEFINITION_UNAVAILABLE")
            requirements.append(InputAdmissionRequirement(
                requirement_id="FEATURE_DEFINITION_BOUND", status="UNSATISFIED",
                detail="当前正式证据包无法解析对应 FeatureDefinition。",
            ))
        else:
            requirements.append(InputAdmissionRequirement(
                requirement_id="FEATURE_DEFINITION_BOUND", status="SATISFIED",
                detail=f"{feature_definition.feature_definition_id} · {feature_definition.algorithm_version}",
            ))

        source_template = feature_definition.source_field_template if feature_definition is not None else None
        resolved_field = feature_result.resolved_field if feature_result is not None else None
        if resolved_field:
            requirements.append(InputAdmissionRequirement(
                requirement_id="SOURCE_FIELD_BOUND", status="SATISFIED", detail=resolved_field,
            ))
        else:
            requirements.append(InputAdmissionRequirement(
                requirement_id="SOURCE_FIELD_BOUND", status="UNSATISFIED",
                detail=(f"需要解析字段模板 {source_template}" if source_template else "所需字段模板当前不可解析"),
            ))
            blockers.append("SOURCE_FIELD_UNRESOLVED")

        required_unit = feature_definition.unit if feature_definition is not None else gap.unit
        observed_unit = feature_result.unit if feature_result is not None else None
        if observed_unit is None:
            unit_status = "UNSATISFIED"
            blockers.append("INPUT_UNIT_UNAVAILABLE")
            unit_detail = f"需要单位 {required_unit or '未声明'}"
        elif required_unit and observed_unit != required_unit:
            unit_status = "UNSATISFIED"
            blockers.append("INPUT_UNIT_MISMATCH")
            unit_detail = f"观测单位 {observed_unit}，合同要求 {required_unit}"
        else:
            unit_status = "SATISFIED"
            unit_detail = observed_unit or required_unit or "单位未声明"
        requirements.append(InputAdmissionRequirement(requirement_id="UNIT_MATCH", status=unit_status, detail=unit_detail))

        window_scope = feature_definition.window_scope if feature_definition is not None else None
        observed_window_start = feature_result.window_start_s if feature_result is not None else None
        observed_window_end = feature_result.window_end_s if feature_result is not None else None
        if window_scope == "episode":
            window_start = bundle.episode.start_s
            window_end = bundle.episode.end_s
            window_source = "EPISODE_CONTRACT"
            window_matches = (
                feature_result is not None
                and abs(float(feature_result.window_start_s) - float(window_start)) <= 1e-9
                and abs(float(feature_result.window_end_s) - float(window_end)) <= 1e-9
            )
            requirements.append(InputAdmissionRequirement(
                requirement_id="WINDOW_BOUND", status=("SATISFIED" if window_matches else "UNSATISFIED"),
                detail=(
                    f"episode 合同 {window_start:.3f}–{window_end:.3f} s；当前 FeatureResult "
                    f"{observed_window_start:.3f}–{observed_window_end:.3f} s"
                    if feature_result is not None else
                    f"需要覆盖 episode {window_start:.3f}–{window_end:.3f} s；当前尚未生成 FeatureResult。"
                ),
            ))
            if not window_matches:
                blockers.append("FEATURE_WINDOW_MISMATCH" if feature_result is not None else "FEATURE_RESULT_WINDOW_NOT_MATERIALIZED")
        elif window_scope == "full_run" and feature_result is not None:
            window_start = None
            window_end = None
            window_source = "FEATURE_RESULT_REQUIRES_RUN_CONTRACT_VERIFICATION"
            requirements.append(InputAdmissionRequirement(
                requirement_id="WINDOW_BOUND", status="UNVERIFIED",
                detail=f"FeatureResult 声明 full_run {feature_result.window_start_s:.3f}–{feature_result.window_end_s:.3f} s；需由受治理 run contract 复核完整覆盖。",
            ))
            blockers.append("FULL_RUN_WINDOW_CONTRACT_UNVERIFIED")
        else:
            window_start = None
            window_end = None
            window_source = "UNRESOLVED"
            requirements.append(InputAdmissionRequirement(
                requirement_id="WINDOW_BOUND", status="UNSATISFIED", detail="full_run 窗口需由受治理运行合同解析。",
            ))
            blockers.append("FULL_RUN_WINDOW_UNRESOLVED")

        if feature_result is not None and feature_result.input_data_sha256:
            requirements.append(InputAdmissionRequirement(
                requirement_id="SOURCE_HASH_BOUND", status="SATISFIED", detail=feature_result.input_data_sha256,
            ))
        else:
            requirements.append(InputAdmissionRequirement(
                requirement_id="SOURCE_HASH_BOUND", status="UNSATISFIED", detail="需要绑定输入数据 SHA256。",
            ))
            blockers.append("SOURCE_ARTIFACT_HASH_UNBOUND")

        sample_count = feature_result.sample_count if feature_result is not None else None
        if sample_count is not None and sample_count > 0:
            sample_status, sample_detail = "SATISFIED", f"sample_count={sample_count}"
        else:
            sample_status, sample_detail = "UNSATISFIED", "需要非空样本覆盖。"
            blockers.append("SAMPLE_COVERAGE_EMPTY")
        requirements.append(InputAdmissionRequirement(requirement_id="SAMPLE_COVERAGE", status=sample_status, detail=sample_detail))

        observed_algorithm = feature_result.algorithm_version if feature_result is not None else None
        required_algorithm = feature_definition.algorithm_version if feature_definition is not None else None
        if observed_algorithm is None:
            algo_status, algo_detail = "UNSATISFIED", f"需要算法版本 {required_algorithm or '未解析'}"
            blockers.append("FEATURE_ALGORITHM_VERSION_UNBOUND")
        elif required_algorithm and observed_algorithm != required_algorithm:
            algo_status, algo_detail = "UNSATISFIED", f"观测 {observed_algorithm}，合同要求 {required_algorithm}"
            blockers.append("FEATURE_ALGORITHM_VERSION_MISMATCH")
        else:
            algo_status, algo_detail = "SATISFIED", observed_algorithm or required_algorithm or "未声明"
        requirements.append(InputAdmissionRequirement(requirement_id="FEATURE_ALGORITHM_VERSION", status=algo_status, detail=algo_detail))

        aggregation = feature_definition.aggregation.value if feature_definition is not None else None
        paired_required = bool(aggregation and "PAIRED" in aggregation)
        paired_id = bundle.episode.paired_nominal_run_id or None
        if paired_required:
            pair_status = "SATISFIED" if paired_id else "UNSATISFIED"
            requirements.append(InputAdmissionRequirement(
                requirement_id="PAIRED_NOMINAL_BOUND", status=pair_status,
                detail=(paired_id or "该特征要求 paired nominal identity。"),
            ))
            if not paired_id:
                blockers.append("PAIRED_NOMINAL_IDENTITY_UNBOUND")

        value_available = feature_result is not None and feature_result.value is not None
        requirements.append(InputAdmissionRequirement(
            requirement_id="FEATURE_VALUE_AVAILABLE",
            status=("SATISFIED" if value_available else "UNSATISFIED"),
            detail=("已生成数值型 FeatureResult。" if value_available else "需要重新生成可判定的 FeatureResult 数值。"),
        ))
        if not value_available:
            blockers.append("FEATURE_VALUE_UNAVAILABLE")

        current_status = gap.status
        if current_status in {EvidenceFactStatus.NOT_SATISFIED.value, EvidenceFactStatus.CONTRADICTED.value} and not blockers:
            contract_status = "EXISTING_INPUT_READY_FOR_CONTRACT_REVIEW"
            action = "现有输入已具备可追溯特征结果；下一步应核对来源、单位、窗口和冻结阈值合同，而不是重复采集相同数据。"
        elif current_status in {EvidenceFactStatus.NOT_SATISFIED.value, EvidenceFactStatus.CONTRADICTED.value}:
            contract_status = "INPUT_CONTRACT_REPAIR_REQUIRED"
            action = "该反证同时存在输入合同缺口；先修复未满足的 admission requirement，再重新生成正式 EvidenceFact。"
        else:
            contract_status = "INPUT_REQUIRED"
            action = "按本合同补齐受治理输入，重新运行冻结 FeatureDefinition 算法并生成新的正式 EvidenceFact。"

        supplemental_support = assess_supplemental_admission_support(
            feature_definition=feature_definition,
            resolved_source_field=resolved_field,
            window_scope=window_scope,
        )

        contracts.append(EvidenceInputContract(
            contract_id=f"INPUT-{gap.evidence_definition_id}",
            status=contract_status,
            evidence_definition_id=gap.evidence_definition_id,
            feature_definition_id=(feature_definition.feature_definition_id if feature_definition is not None else gap.feature_definition_id),
            feature_name=(feature_definition.name if feature_definition is not None else None),
            current_fact_status=current_status,
            source_field_template=source_template, resolved_source_field=resolved_field,
            aggregation=aggregation,
            required_semantics=(list(feature_definition.required_semantics) if feature_definition is not None else []),
            required_unit=required_unit, observed_unit=observed_unit,
            threshold_unit=(definition.threshold_unit if definition is not None else gap.unit),
            window_scope=window_scope, required_window_start_s=window_start, required_window_end_s=window_end,
            observed_window_start_s=observed_window_start, observed_window_end_s=observed_window_end,
            window_source=window_source, paired_nominal_required=paired_required, paired_nominal_run_id=paired_id,
            feature_algorithm_version=required_algorithm, observed_algorithm_version=observed_algorithm,
            input_data_sha256=(feature_result.input_data_sha256 if feature_result is not None else None),
            sample_count=sample_count, quality_flags=(list(feature_result.quality_flags) if feature_result is not None else []),
            admission_requirements=requirements, blockers=sorted(set(blockers)),
            supplemental_admission_status=supplemental_support["status"],
            supplemental_admission_reason_codes=list(supplemental_support["reason_codes"]),
            supplemental_operator_id=supplemental_support.get("operator_id"),
            supplemental_operator_runtime_version=supplemental_support.get("operator_runtime_version"),
            supplemental_capsule_schema_version=supplemental_support.get("capsule_schema_version"),
            operator_action_zh=action,
        ))
    return contracts

def _next_diagnostic_check(
    report: dict[str, Any], evidence: MaintenanceEvidenceSummary, probe_reviews: list[ProbeReviewSummary] | None = None,
    input_contracts: list[EvidenceInputContract] | None = None,
) -> NextDiagnosticCheck:
    if evidence.status != "AVAILABLE":
        return NextDiagnosticCheck(
            status="UNAVAILABLE", title="正式证据复核暂不可生成",
            rationale="需要先取得可验证的 EpisodeEvidenceBundle，才能根据关键缺口生成诊断检查。",
            blocking_reason="FORMAL_EVIDENCE_BUNDLE_UNAVAILABLE",
        )
    target = evidence.missing_items[0] if evidence.missing_items else (evidence.contradicting_items[0] if evidence.contradicting_items else None)
    target_contract = next(
        (item for item in (input_contracts or []) if target is not None and item.evidence_definition_id == target.evidence_definition_id),
        None,
    )
    if target is None:
        return NextDiagnosticCheck(
            status="NOT_REQUIRED", title="当前没有未解决的关键证据项",
            rationale="正式关键证据摘要中没有缺失或反证项；是否继续深入检查应由其他治理门和人工复核决定。",
        )
    if target in evidence.missing_items:
        return NextDiagnosticCheck(
            status="ACTION_REQUIRED", check_id=f"CHECK-{target.evidence_definition_id}",
            title=f"补充关键证据：{target.label}",
            rationale="该关键证据当前缺失、无效或不可判定；继续重复计算已有数据不能关闭此缺口。",
            evidence_definition_id=target.evidence_definition_id, feature_definition_id=target.feature_definition_id,
            auto_executable=False, blocking_reason="MISSING_OR_INVALID_FORMAL_EVIDENCE",
            expected_information="补齐对应受治理特征输入后重新生成正式 EvidenceFact。",
            input_contract_id=(target_contract.contract_id if target_contract else None),
        )
    prior_reviews = [
        item for item in (probe_reviews or [])
        if item.evidence_definition_id == target.evidence_definition_id and item.status == "SUCCEEDED"
    ]
    if prior_reviews and any(item.formal_gap_open_after for item in prior_reviews):
        latest = prior_reviews[0]
        return NextDiagnosticCheck(
            status="ACTION_REQUIRED", check_id=f"CHECK-{target.evidence_definition_id}",
            title=f"反证已完成自动复核：{target.label}",
            rationale=(
                "相同 shadow probe 已执行且当前正式 EvidenceBundle 仍保留该反证。继续重复同一阈值复核不会关闭缺口；"
                "下一步应核对数据来源/单位/阈值合同，或补充新的受治理输入后重新生成正式 EvidenceFact。"
            ),
            evidence_definition_id=target.evidence_definition_id, feature_definition_id=target.feature_definition_id,
            auto_executable=False, blocking_reason="SHADOW_REVIEW_ALREADY_COMPLETED_FORMAL_GAP_STILL_OPEN",
            expected_information=(
                f"上次复核结果 {latest.result_sha256 or latest.probe_id} 已记录；需要新的正式证据状态，而不是重复 shadow 计算。"
            ),
            input_contract_id=(target_contract.contract_id if target_contract else None),
        )
    bundle = extract_formal_evidence_bundle(report)
    probe_id = None
    if bundle is not None:
        catalog = build_evidence_margin_probe_catalog(bundle)
        for probe in catalog.probes:
            if probe.evidence_definition_id == target.evidence_definition_id:
                probe_id = probe.probe_id
                break
    return NextDiagnosticCheck(
        status="REVIEW_AVAILABLE" if probe_id else "ACTION_REQUIRED",
        check_id=f"CHECK-{target.evidence_definition_id}",
        title=f"复核关键反证：{target.label}",
        rationale="该关键证据当前处于未满足/反证状态。自动复核只能重新计算已批准阈值边界距离，不会生成新传感器证据，也不会改变正式诊断。",
        evidence_definition_id=target.evidence_definition_id, feature_definition_id=target.feature_definition_id,
        probe_id=probe_id, auto_executable=bool(probe_id),
        blocking_reason=(None if probe_id else "NO_APPROVED_SHADOW_PROBE"),
        expected_information="量化现有观测距离冻结阈值的 signed margin，供人工核对。",
        input_contract_id=(target_contract.contract_id if target_contract else None),
    )


def _probe_review_history(
    probe_reviews: list[dict[str, Any]] | None, evidence: MaintenanceEvidenceSummary
) -> list[ProbeReviewSummary]:
    if not probe_reviews:
        return []
    evidence_by_id = {
        item.evidence_definition_id: item
        for item in [*evidence.supporting_items, *evidence.missing_items, *evidence.contradicting_items]
        if item.evidence_definition_id
    }
    gap_ids = {
        item.evidence_definition_id
        for item in [*evidence.missing_items, *evidence.contradicting_items]
        if item.evidence_definition_id
    }
    summaries: list[ProbeReviewSummary] = []
    for raw in probe_reviews:
        if not isinstance(raw, dict) or not raw.get("probe_id"):
            continue
        evidence_id = str(raw.get("evidence_definition_id") or "") or None
        item = evidence_by_id.get(evidence_id) if evidence_id else None
        gap_open = bool(evidence_id and evidence_id in gap_ids) if evidence.status == "AVAILABLE" else None
        if evidence.status != "AVAILABLE":
            linkage = "FORMAL_EVIDENCE_UNAVAILABLE"
            next_action = "先恢复可验证的正式 EpisodeEvidenceBundle，再判断该复核与正式证据缺口的关系。"
        elif item is None:
            linkage = "TARGET_NOT_IN_CURRENT_FORMAL_SUMMARY"
            next_action = "核对该 Probe 对应的 EvidenceDefinition 是否仍属于当前候选故障的正式证据协议。"
        elif gap_open:
            linkage = "FORMAL_GAP_STILL_OPEN"
            if item in evidence.missing_items:
                next_action = "补齐受治理输入并重新生成正式 EvidenceFact；重复 shadow probe 不能替代缺失数据。"
            else:
                next_action = "核对数据来源、单位和阈值合同；如需关闭反证，必须通过受治理输入重新生成正式 EvidenceFact。"
        else:
            linkage = "NO_CURRENT_FORMAL_GAP"
            next_action = "当前正式证据已不再把该项列为缺口；无需仅为提高置信感而重复执行相同 shadow probe。"
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        summaries.append(ProbeReviewSummary(
            result_sha256=(str(raw.get("result_sha256")) if raw.get("result_sha256") else None),
            reviewed_at=(str(raw.get("reviewed_at")) if raw.get("reviewed_at") else None),
            reviewed_by=(str(raw.get("reviewed_by")) if raw.get("reviewed_by") else None),
            probe_id=str(raw.get("probe_id")),
            probe_kind=(str(raw.get("kind")) if raw.get("kind") else None),
            evidence_definition_id=evidence_id,
            status=str(raw.get("status") or "UNKNOWN"),
            signed_margin=(float(metrics["signed_margin"]) if metrics.get("signed_margin") is not None else None),
            normalized_signed_margin=(float(metrics["normalized_signed_margin"]) if metrics.get("normalized_signed_margin") is not None else None),
            unit=(str(metrics.get("unit")) if metrics.get("unit") not in (None, "") else None),
            formal_status_after=(item.status if item is not None else None),
            formal_gap_open_after=gap_open,
            gap_closed_by_probe=False,
            linkage_status=linkage,
            next_action_zh=next_action,
        ))
    summaries.sort(key=lambda item: (item.reviewed_at or "", item.result_sha256 or ""), reverse=True)
    return summaries[:20]


def _diagnostic_contribution(
    report: dict[str, Any], evidence: MaintenanceEvidenceSummary
) -> DiagnosticContributionSummary:
    """Reconcile explicit controlled-trajectory contribution with current formal evidence gaps.

    Generic LLM/tool activity is deliberately ignored.  Probe targets are read only from
    recorded ``execute_diagnostic_probe`` observations.  A shadow recheck may be linked to
    a current contradiction, but it is never described as closing that gap unless the
    trajectory explicitly records formal evidence generation/gap reduction and the current
    formal bundle no longer contains the gap.
    """
    queue: list[Any] = [report]
    seen = 0
    snapshot: dict[str, Any] | None = None
    while queue and seen < 25000:
        current = queue.pop(0)
        seen += 1
        if isinstance(current, dict):
            contribution = current.get("contribution_audit")
            substance = current.get("substance_audit")
            schema = str(current.get("schema_version") or "")
            if isinstance(contribution, dict) and (
                isinstance(substance, dict) or schema.startswith("astrograph-llm-controlled-diagnostic-trajectory.")
            ):
                snapshot = {
                    "contribution": contribution,
                    "substance": substance or {},
                    "steps": current.get("steps") if isinstance(current.get("steps"), list) else [],
                }
                break
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(value for value in current if isinstance(value, (dict, list)))

    current_gap_ids = sorted({
        item.evidence_definition_id
        for item in [*evidence.missing_items, *evidence.contradicting_items]
        if item.evidence_definition_id
    }) if evidence.status == "AVAILABLE" else []
    if snapshot is None:
        return DiagnosticContributionSummary(
            unresolved_gap_definition_ids=current_gap_ids,
            unresolved_gap_count=len(current_gap_ids),
            gap_linkage_status=("NO_EXPLICIT_CONTRIBUTION_AUDIT" if current_gap_ids else "UNAVAILABLE"),
            gap_linkage_summary_zh=(
                f"当前仍有 {len(current_gap_ids)} 项正式关键证据缺口，但报告没有显式受控调查贡献审计，不能推断 LLM 是否复核过这些缺口。"
                if current_gap_ids else "当前没有显式受控调查贡献审计可与正式证据缺口核对。"
            ),
        )

    contribution = snapshot["contribution"]
    substance = snapshot["substance"]
    reviewed_ids: list[str] = []
    for step in snapshot["steps"]:
        if not isinstance(step, dict):
            continue
        proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        if proposal.get("capability_id") != "execute_diagnostic_probe":
            continue
        content = observation.get("content") if isinstance(observation.get("content"), dict) else {}
        evidence_id = content.get("evidence_definition_id")
        if evidence_id:
            reviewed_ids.append(str(evidence_id))
    reviewed_ids = sorted(set(reviewed_ids))
    reviewed_current_gap_ids = sorted(set(reviewed_ids) & set(current_gap_ids))

    status = str(contribution.get("status") or "UNAVAILABLE")
    resolved_gap_count = max(0, int(contribution.get("resolved_gap_count") or 0))
    formal_generated = max(0, int(contribution.get("formal_evidence_generated_count") or 0))
    summaries = {
        "NO_NEW_INFORMATION": "本次调查没有执行受控诊断 Probe，因此没有记录新的诊断观测。",
        "DERIVED_OBSERVATION_ONLY": "本次受控检查只重新计算了已经准入的数据；没有取得新的原始观测，也没有生成新的正式证据。",
        "NEW_INFORMATION_NO_DIAGNOSTIC_CHANGE": "本次调查记录了新的观测，但当前审计尚未证明这些观测改变了诊断候选或正式结论。",
        "EVIDENCE_GAP_REDUCED": "本次调查记录了证据缺口减少；是否影响正式结论仍受确定性证据编译和独立治理门控制。",
        "CANDIDATE_SET_REFINED": "本次调查改变了候选集合；该变化本身不等于正式故障成立。",
        "DECISION_CHANGED": "本次调查记录到诊断决策发生变化；正式发布仍取决于确定性证据编译和独立治理门。",
    }

    if evidence.status != "AVAILABLE":
        linkage_status = "FORMAL_EVIDENCE_UNAVAILABLE"
        linkage_summary = "当前没有可验证的正式 EvidenceBundle，不能判断 Probe 是否触及或关闭正式证据缺口。"
    elif not current_gap_ids:
        linkage_status = "NO_CURRENT_FORMAL_GAP"
        linkage_summary = "当前正式关键证据摘要没有未解决缺口；Probe 贡献不应被解释为额外提高故障概率。"
    elif reviewed_current_gap_ids and resolved_gap_count == 0 and formal_generated == 0:
        linkage_status = "REVIEWED_GAP_STILL_UNRESOLVED"
        linkage_summary = (
            f"本次受控 Probe 复核了 {len(reviewed_current_gap_ids)} 项当前正式证据缺口，但没有生成新的正式证据；"
            f"当前仍有 {len(current_gap_ids)} 项缺口未关闭。"
        )
    elif reviewed_ids and not reviewed_current_gap_ids:
        linkage_status = "REVIEWED_NON_GAP_EVIDENCE"
        linkage_summary = (
            f"本次 Probe 复核了 {len(reviewed_ids)} 项已有证据，但没有命中当前 {len(current_gap_ids)} 项正式关键证据缺口。"
        )
    elif resolved_gap_count > 0:
        linkage_status = "RECORDED_GAP_REDUCTION_REQUIRES_FORMAL_SNAPSHOT_CHECK"
        linkage_summary = (
            "调查审计记录了证据缺口减少；维护层仍以当前正式 EvidenceBundle 为准，不能仅凭贡献计数认定某个缺口已关闭。"
        )
    else:
        linkage_status = "NO_CURRENT_GAP_TARGET_RECORDED"
        linkage_summary = f"当前仍有 {len(current_gap_ids)} 项正式关键证据缺口，受控调查没有记录对这些缺口的 Probe 复核。"

    return DiagnosticContributionSummary(
        status=status,
        structural_status=(str(substance.get("status")) if substance.get("status") else None),
        probe_execution_count=max(0, int(contribution.get("probe_execution_count") or 0)),
        new_observation_count=max(0, int(contribution.get("new_observation_count") or 0)),
        new_raw_observation_count=max(0, int(contribution.get("new_raw_observation_count") or 0)),
        formal_evidence_generated_count=formal_generated,
        resolved_gap_count=resolved_gap_count,
        candidate_set_changed=bool(contribution.get("candidate_set_changed")),
        decision_changed=bool(contribution.get("decision_changed")),
        reason_codes=[str(item) for item in (contribution.get("reason_codes") or [])],
        reviewed_evidence_definition_ids=reviewed_ids,
        reviewed_current_gap_definition_ids=reviewed_current_gap_ids,
        unresolved_gap_definition_ids=current_gap_ids,
        unresolved_gap_count=len(current_gap_ids),
        gap_linkage_status=linkage_status,
        gap_linkage_summary_zh=linkage_summary,
        operator_summary_zh=summaries.get(status, "已记录调查贡献审计；请结合技术详情核对具体状态。"),
    )


def _candidate_context_guard(report: dict[str, Any]) -> CandidateContextGuard:
    """Extract the deterministic compact-reasoning escape receipt, if present.

    The receipt can occur in a reasoning stage artifact or in a structured backend call
    receipt nested under an Agent/runtime trace. Search is bounded and never infers a
    candidate escape from diagnosis outcome alone.
    """
    queue: list[Any] = [report]
    seen = 0
    snapshots: list[dict[str, Any]] = []
    while queue and seen < 25000:
        current = queue.pop(0)
        seen += 1
        if isinstance(current, dict):
            selection = current.get("candidate_selection")
            if isinstance(selection, dict):
                snapshots.append({
                    **selection,
                    "candidate_escape_applied": current.get("candidate_escape_applied"),
                })
            elif "compact_reasoning_allowed" in current and "candidate_set_status" in current:
                snapshots.append(dict(current))
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(value for value in current if isinstance(value, (dict, list)))
    if not snapshots:
        return CandidateContextGuard()
    # Prefer an actual escape event, then the most explicit non-compact snapshot.
    snapshot = next(
        (item for item in snapshots if item.get("candidate_escape_applied") is True),
        next((item for item in snapshots if item.get("compact_reasoning_allowed") is False), snapshots[0]),
    )
    compact_allowed = snapshot.get("compact_reasoning_allowed")
    escape = snapshot.get("candidate_escape_applied") is True or compact_allowed is False
    preferred = snapshot.get("preferred_candidate_fault_ids") or snapshot.get("selected_fault_ids") or []
    mandatory = snapshot.get("unselected_mandatory_fault_ids") or []
    reason_codes = snapshot.get("escape_reason_codes") or []
    if escape:
        summary = "候选上下文已自动扩展：系统检测到 Top-K 可能遗漏仍处于活跃/强制状态的候选，因此没有让大模型只看紧凑候选集。"
        if mandatory:
            summary += f" 本次额外保护了 {len(mandatory)} 个未进入原 Top-K 的强制候选。"
        status = "EXPANDED_CONTEXT"
    else:
        summary = "候选集合满足紧凑推理条件，本次未触发候选上下文扩展保护。"
        status = "COMPACT_CONTEXT"
    return CandidateContextGuard(
        status=status,
        compact_reasoning_allowed=(bool(compact_allowed) if compact_allowed is not None else None),
        candidate_set_status=(str(snapshot.get("candidate_set_status")) if snapshot.get("candidate_set_status") else None),
        escape_action=(str(snapshot.get("escape_action")) if snapshot.get("escape_action") else None),
        escape_reason_codes=[str(item) for item in reason_codes],
        preferred_candidate_fault_ids=[str(item) for item in preferred],
        unselected_mandatory_fault_ids=[str(item) for item in mandatory],
        configured_top_k=(int(snapshot["configured_top_k"]) if snapshot.get("configured_top_k") is not None else None),
        effective_top_k=(int(snapshot["effective_top_k"]) if snapshot.get("effective_top_k") is not None else None),
        forced_candidate_count=(int(snapshot["forced_candidate_count"]) if snapshot.get("forced_candidate_count") is not None else None),
        operator_summary_zh=summary,
    )


def build_maintenance_presentation(
    *, report: dict[str, Any], visualization: dict[str, Any], temporal: dict[str, Any] | None = None,
    multidimensional: dict[str, Any] | None = None, probe_reviews: list[dict[str, Any]] | None = None,
    supplemental_evidence_bundle: EpisodeEvidenceBundle | None = None,
    supplemental_overlay_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    fault_id, disposition = _candidate(report)
    temporal = temporal or localize_from_visualization(visualization, diagnostic_fault_id=fault_id)
    multidimensional = multidimensional or assess_multidimensional_telemetry(visualization, temporal=temporal)
    pipeline = ((report.get("analysis") or {}).get("pipeline_result") or {})
    blockers = [str(x) for x in (pipeline.get("blockers") or [])]
    status_zh = {"REVIEW_REQUIRED":"需要人工复核","DIAGNOSED":"已形成诊断","ABSTAIN":"证据不足，暂不下结论","BLOCKED":"诊断受阻"}.get(disposition, disposition or "未知")
    headline = FAULT_ZH.get(fault_id or "", fault_id or "暂未形成故障候选")
    subsystem = "姿态与轨道控制（ADCS）" if (fault_id or "").startswith("ADCS_") else None
    component = "反作用轮" if "RW_" in (fault_id or "") else None
    publishable = bool(report.get("formal_diagnosis_publishable") or pipeline.get("formal_diagnosis_publishable"))
    evidence_summary = _evidence_summary(
        report, fault_id, bundle_override=supplemental_evidence_bundle,
        overlay_receipt_sha256=supplemental_overlay_receipt_sha256,
    )
    probe_review_history = _probe_review_history(probe_reviews, evidence_summary)
    evidence_gap_input_contracts = _evidence_gap_input_contracts(
        report, evidence_summary, bundle_override=supplemental_evidence_bundle,
    )
    next_check = _next_diagnostic_check(report, evidence_summary, probe_review_history, evidence_gap_input_contracts)
    candidate_guard = _candidate_context_guard(report)
    diagnostic_contribution = _diagnostic_contribution(report, evidence_summary)
    operator_summary = f"当前判断为“{headline}”。"
    if disposition == "REVIEW_REQUIRED":
        operator_summary += " 该结果可用于维护排查，但仍需完成人工复核后才能作为正式结论。"
    if temporal.get("first_detection_s") is not None:
        operator_summary += f" 首次可观测异常检出约在 {float(temporal['first_detection_s']):.1f} s。"
    return MaintenancePresentation(
        job_id=str(report.get("job_id")), headline=headline, fault_id=fault_id, subsystem=subsystem, component=component,
        diagnosis_status=disposition, status_zh=status_zh,
        release_status_zh=("具备正式发布条件" if publishable else "当前仅用于维护复核，不作为正式发布结论"),
        candidate_context_guard=candidate_guard, probe_review_history=probe_review_history,
        diagnostic_contribution=diagnostic_contribution, evidence_summary=evidence_summary,
        evidence_gap_input_contracts=evidence_gap_input_contracts, next_diagnostic_check=next_check,
        temporal_localization=temporal, multidimensional_assessment=multidimensional, key_findings=_key_findings(visualization, multidimensional),
        alternative_hypotheses=_alternatives(report, fault_id), review_actions=_review_actions(blockers),
        mechanism_graph=_mechanism(report), document_references=_documents(report),
        technical_trace_ref=f"/v1/platform/jobs/{report.get('job_id')}/report",
        operator_summary=operator_summary,
    ).model_dump(mode="json")
