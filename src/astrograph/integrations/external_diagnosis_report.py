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
                      "## 对报告内容的解读", ""])
        for finding in model["findings"]:
            lines.extend([f"- {_text(finding['text'])}",
                          "  - 依据：" + ", ".join(f"`{sid}`" for sid in finding["citation_ids"])])
        lines.extend(["", "## 原因假设", ""])
        if not model["hypotheses"]:
            lines.append("模型未提出可据材料进一步分析的原因假设。")
        for hypothesis in model["hypotheses"]:
            lines.extend([f"### {_text(hypothesis['title'])}", "", _text(hypothesis["analysis"]), "",
                          "依据：" + ", ".join(f"`{sid}`" for sid in hypothesis["citation_ids"]), ""])
        for title, field in (("缺失信息", "information_gaps"), ("建议核查", "recommended_review_actions")):
            lines.extend([f"## {title}", ""])
            lines.extend([f"- {_text(item)}" for item in model[field]] or ["模型未列出此项。"])
            lines.append("")
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
