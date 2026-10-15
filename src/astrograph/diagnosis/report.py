from __future__ import annotations

from astrograph.diagnosis.conflicts import explain_diagnostic_conflicts
from astrograph.schemas import AnomalyEvidence, DiagnosisResult


def render_markdown(
    case_id: str,
    request_id: str,
    anomalies: list[AnomalyEvidence],
    result: DiagnosisResult,
) -> str:
    conclusion = result.conclusion_fault_name or result.status.value
    lines = [
        "# AstroGraph 确定性诊断报告",
        "",
        f"- 请求编号：`{request_id}`",
        f"- 案例：`{case_id}`",
        f"- 诊断状态：`{result.status.value}`",
        f"- 确定性结论：**{conclusion}**",
        f"- 最高调整得分：`{result.top_score:.6f}`",
        f"- 前两名分差：`{result.score_margin:.6f}`",
        f"- 决策策略版本：`{result.policy_version}`",
        f"- 知识版本：`{result.knowledge_version}`",
        f"- 适用边界：{result.claim_boundary}",
        "",
        "## 一、异常证据",
        "",
        "| 异常 | 信号 | 置信强度 | 首次检测时间/s | 类型 |",
        "|---|---|---:|---:|---|",
    ]
    for item in anomalies:
        observed = "" if item.observed_at_s is None else f"{item.observed_at_s:.3f}"
        lines.append(
            f"| {item.name} | `{item.signal}` | {item.confidence:.6f} | {observed} | {item.confidence_type} |"
        )

    lines.extend([
        "",
        "## 二、候选故障",
        "",
        "| 故障 | 最终得分 | 支持得分 | 总扣分 | 证据数 | 关键证据 | 时间约束 | 可裁决 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ])
    for candidate in result.candidates:
        lines.append(
            f"| {candidate.fault_name} | {candidate.score:.6f} | {candidate.support_score:.6f} | "
            f"{candidate.counter_evidence_penalty:.6f} | {candidate.evidence_count} | "
            f"{candidate.critical_evidence_satisfied} | {candidate.temporal_constraints_satisfied} | "
            f"{candidate.eligible_for_diagnosis} |"
        )

    lines.extend(["", "## 三、证据贡献", ""])
    if not result.candidates:
        lines.append("无匹配证据。")
    for candidate in result.candidates:
        lines.append(f"### {candidate.fault_name}")
        lines.append("")
        lines.append("| 异常 | 关系角色 | 证据等级 | 原始贡献 | 调整贡献 | 临时边 |")
        lines.append("|---|---|---|---:|---:|---|")
        for contribution in candidate.contributions:
            lines.append(
                f"| {contribution['anomaly_name']} | {contribution['edge_role']} | "
                f"{contribution['evidence_level']} | {contribution['raw_contribution']:.6f} | "
                f"{contribution['adjusted_contribution']:.6f} | {contribution['provisional']} |"
            )
        lines.append("")

        if candidate.missing_critical_evidence:
            lines.append("**缺失的关键证据：**")
            for item in candidate.missing_critical_evidence:
                lines.append(f"- `{item['requirement_id']}`：{item['rationale']}")
            lines.append("")
        if candidate.counter_evidence:
            lines.append("**反证扣分：**")
            for item in candidate.counter_evidence:
                lines.append(
                    f"- `{item['rule_id']}` / `{item['anomaly_id']}`：扣分 {item['penalty']:.6f}；{item['rationale']}"
                )
            lines.append("")
        if candidate.temporal_constraint_evaluations:
            lines.append("**Episode 时间约束：**")
            for item in candidate.temporal_constraint_evaluations:
                state = "满足" if item.get("satisfied") else "不满足"
                lines.append(
                    f"- `{item.get('constraint_id', '')}`：{state}；匹配 {item.get('matched_relation_count', 0)}/"
                    f"{item.get('required_match_count', 1)}；{item.get('rationale', '')}"
                )
            lines.append("")

    conflict = explain_diagnostic_conflicts(result)
    lines.extend(["## 四、证据冲突解释", ""])
    lines.append(f"- 冲突分类：`{conflict.classification.value}`")
    lines.append(f"- 摘要：{conflict.summary}")
    if conflict.competing_fault_ids:
        lines.append(f"- 竞争候选：{', '.join(conflict.competing_fault_ids)}")
    for item in conflict.candidate_explanations:
        if not item.explanation:
            continue
        lines.append(f"- **{item.fault_name}**：" + " ".join(item.explanation))

    lines.extend(["", "## 五、裁决说明", ""])
    lines.extend([f"- {reason}" for reason in result.reasons])
    lines.extend([
        "",
        "## 六、安全声明",
        "",
        "关键结论、得分、证据、冲突解释和版本字段均由确定性代码渲染；当前版本不调用 LLM 参与故障裁决。",
    ])
    return "\n".join(lines) + "\n"
