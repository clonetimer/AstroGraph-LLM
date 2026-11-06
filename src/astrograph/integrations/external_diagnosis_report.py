"""Presentation of an imported report and its separately identified LLM analysis."""
from __future__ import annotations

import html
from typing import Any


def imported_diagnosis_review(bundle: dict[str, Any]) -> dict[str, Any]:
    """Provide source receipts for existing report clients without evaluating faults."""
    sources = bundle["sources"]
    manifest = {
        (source["role"], source["filename"]): {
            key: source[key] for key in ("role", "filename", "sha256")
        } for source in sources
    }
    return {
        "schema_version": "astrograph-external-diagnosis-import.v1",
        "status": "INPUT_IMPORTED", "review_type": "EXTERNAL_DIAGNOSIS_ANALYSIS",
        "input_schema_version": "astrograph-result-log-review.v1",
        "input_sha256": bundle["input_sha256"],
        "source_manifest": list(manifest.values()),
        "evidence": [{"evidence_id": source["source_id"],
                      **{key: source[key] for key in ("role", "filename", "locator", "sha256")},
                      "excerpt": source["text"]} for source in sources],
        "candidates": [], "top_fault_id": None, "cause_hypotheses": [],
        "handling_measures": [], "verification_checks": [], "evidence_gaps": [],
        "conclusion": "外部诊断结果已导入；模型分析单独列示，原始诊断内容保持不变。",
        "limitations": ["本流程分析外部诊断文本，不计算或重新判定上游诊断结果。"],
        "evidence_policy": {"upstream_diagnosis_is_ground_truth": False,
                            "new_raw_telemetry_detection_executed": False,
                            "formal_diagnosis_publishable": False,
                            "automatic_control_authorized": False},
    }


def _text(value: Any) -> str:
    # Markdown downloads may later be rendered as HTML. Treat user/model markup
    # as plain text instead of allowing active links, HTML, or embedded images.
    text = html.escape(str(value), quote=False)
    for char in ("\\", "`", "*", "_", "[", "]", "#", "|", "~"):
        text = text.replace(char, "\\" + char)
    return text.replace("\r", "").replace("\n", "  \n")


def render_external_diagnosis_markdown(
    job: dict[str, Any], review: dict[str, Any], knowledge: dict[str, Any],
) -> str:
    model = knowledge["model_explanation"]
    generation = knowledge["generation"]
    lines = ["# 外部诊断报告与 LLM 分析", "",
             f"- 任务：{_text(job['job_id'])}",
             f"- 分析目标：{_text(job['task'])}",
             "- 输入模式：外部诊断文本；日志和参考资料为可选补充。",
             f"- 模型运行状态：`{generation['status']}`",
             f"- 引用检查：`{model['citation_reference_validation']}`", ""]
    if generation["status"] == "SUCCEEDED" and model["status"] == "AVAILABLE_REQUIRES_EXPERT_REVIEW":
        lines.extend(["## LLM 分析摘要", "", _text(model["summary"]), "",
                      "以下内容是模型对所提供材料的解读与假设；引用检查仅确认来源归属。", "",
                      "## 主张与证据核验", ""])
        governance = model.get("claim_governance") or {}
        verification = governance.get("verification") or {}
        verification_labels = {
            "COMPLETED_REQUIRES_EXPERT_REVIEW": "独立模型审阅已完成，仍需专家确认",
            "NOT_CONFIGURED": "未配置独立审阅模型",
            "NOT_RUN": "本次未执行独立审阅",
            "FAILED": "独立审阅未完成",
        }
        lines.extend([f"- 主张数量：`{len(governance.get('claims') or [])}`",
                      f"- 独立审阅：{verification_labels.get(verification.get('status'), verification.get('status', '未执行'))}",
                      f"- 审阅方法：`{verification.get('method', 'SOURCE_REFERENCE_ONLY')}`",
                      "- 边界：来源支持关系不等于现实事实已被证实；所有主张仍需专家复核。", ""])
        for claim in governance.get("claims") or []:
            evidence_ids = [item.get("source_id") for item in claim.get("evidence_refs") or []]
            opposing_ids = [item.get("source_id") for item in claim.get("opposing_evidence_refs") or []]
            lines.extend([f"### {_text(claim.get('claim_type', 'CLAIM'))} · `{_text(claim.get('claim_id', ''))}`",
                          "", _text(claim.get("text", "")),
                          f"- 认识状态：`{claim.get('epistemic_status', 'INFERRED')}`",
                          f"- 证据关系：`{claim.get('support_status', 'NOT_EVALUATED')}`",
                          f"- 专家审阅：`{claim.get('expert_review_status', 'PENDING')}`",
                          "- 支持来源：" + (", ".join(f"`{sid}`" for sid in evidence_ids) or "无"),
                          "- 反向来源：" + (", ".join(f"`{sid}`" for sid in opposing_ids) or "无"), ""])
        if verification.get("reason_code"):
            lines.append(f"- 审阅原因代码：`{verification['reason_code']}`")
        lines.extend(["", "## 对报告内容的解读", ""])
        for finding in model["findings"]:
            lines.extend([f"- {_text(finding['text'])}",
                          "  - 依据：" + ", ".join(f"`{sid}`" for sid in finding["citation_ids"])])
        for title, field in (("跨报告共识", "common_findings"), ("报告间分歧", "report_disagreements"),
                             ("互补证据", "complementary_evidence"), ("事件与时间对齐", "event_alignment")):
            items = model.get(field, [])
            if items:
                lines.extend(["", f"## {title}", ""])
                for item in items:
                    lines.extend([f"- {_text(item['text'])}",
                                  "  - 依据：" + ", ".join(f"`{sid}`" for sid in item["citation_ids"])])
        if model.get("integrated_conclusion"):
            lines.extend(["", "## 综合结论", "", _text(model["integrated_conclusion"]), ""])
        lines.extend(["", "## 原因假设", ""])
        if not model["hypotheses"]:
            lines.append("模型未提出可据材料进一步分析的原因假设。")
        for hypothesis in model["hypotheses"]:
            lines.extend([f"### {_text(hypothesis['title'])}", "", _text(hypothesis["analysis"]), "",
                          "依据：" + ", ".join(f"`{sid}`" for sid in hypothesis["citation_ids"]), ""])
            if hypothesis.get("opposing_citation_ids"):
                lines.extend(["反向依据：" + ", ".join(
                    f"`{sid}`" for sid in hypothesis["opposing_citation_ids"]), ""])
            for title, field in (("反向解释与限制", "counterpoints"),
                                 ("不确定性", "uncertainty"),
                                 ("区分性核查", "discriminating_check")):
                if hypothesis.get(field):
                    lines.extend([f"{title}：{_text(hypothesis[field])}", ""])
            if hypothesis.get("assumptions"):
                lines.extend(["前提假设：", "",
                              *[f"- {_text(item)}" for item in hypothesis["assumptions"]], ""])
        for title, field in (("缺失信息", "information_gaps"), ("建议核查", "recommended_review_actions")):
            lines.extend([f"## {title}", ""])
            lines.extend([f"- {_text(item)}" for item in model[field]] or ["模型未列出此项。"])
            lines.append("")
        if model.get("review_actions"):
            lines.extend(["## 结构化核查建议", ""])
            for action in model["review_actions"]:
                lines.extend([f"### {_text(action['check'])}", "",
                              f"核查目的：{_text(action['rationale'])}", "",
                              f"预期观察：{_text(action['expected_observations'])}", "",
                              "依据：" + ", ".join(f"`{sid}`" for sid in action["citation_ids"]), ""])
    else:
        message = {
            "NOT_CONFIGURED": "尚未配置可用的大模型；外部报告已导入，未生成模型分析。",
            "INPUT_TOO_LARGE": "输入超过本次模型分析的文本容量，未调用模型。请拆分材料后重试。",
            "FAILED": "本次模型调用或输出校验失败，未生成可展示的模型分析。",
        }.get(generation["status"], "本次没有可展示的模型分析。")
        lines.extend(["## 模型分析状态", "", message, ""])
        if generation.get("reason_code"):
            lines.extend([f"- 原因代码：`{generation['reason_code']}`", ""])
    lines.extend(["## 外部材料与引用定位", ""])
    roles = {"diagnosis_results": "外部诊断报告", "log_reports": "补充日志", "references": "参考资料"}
    for source in model["sources"]:
        lines.extend([f"### {_text(source['filename'])} · {_text(source['locator'])}", "",
                      f"- 引用编号：`{source['source_id']}`",
                      f"- 材料角色：{roles[source['role']]}",
                      f"- 原始文件 SHA-256：`{source['sha256']}`", "",
                      "> " + _text(source["text"]).replace("\n", "\n> "), ""])
    lines.extend(["## 分析边界", "", *[f"- {_text(item)}" for item in review["limitations"]],
                  "- 模型的解读、假设与建议需结合原始材料核对；不改写外部诊断，不执行设备操作。", ""])
    return "\n".join(lines)
