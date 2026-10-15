"""Material-first root-cause *review*, not telemetry-first fault detection.

The uploaded diagnosis is an upstream assertion. Logs are observations reported by
another system and references are unverified explanatory sources. None of them are
promoted to flight-qualified ground truth, BiRRF votes or control authorization.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.retrieval.structured_document import SectionFirstDocumentIngestor

RESULT_LOG_SCHEMA_VERSION = "astrograph-result-log-review.v1"
RESULT_LOG_ARTIFACT_KIND = "RESULT_LOG_REVIEW"
RESULT_SUFFIXES = frozenset({".json", ".csv", ".txt", ".md", ".docx", ".pdf"})
LOG_SUFFIXES = frozenset({".log", ".json", ".csv", ".txt", ".md", ".docx", ".pdf"})
REFERENCE_SUFFIXES = frozenset({".txt", ".md", ".docx", ".pdf"})
_MAX_TEXT_CHARS = 180_000
_MAX_CITATIONS_PER_CANDIDATE = 4


class ResultLogReviewManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-result-log-review.v1"]
    diagnosis_results: list[str] = Field(min_length=1, max_length=8)
    log_reports: list[str] = Field(min_length=1, max_length=12)
    references: list[str] = Field(min_length=1, max_length=16)
    equipment_scope: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,79}$")

    @model_validator(mode="after")
    def all_roles_are_explicit_and_disjoint(self) -> "ResultLogReviewManifest":
        roles = (
            ("diagnosis_results", self.diagnosis_results, RESULT_SUFFIXES),
            ("log_reports", self.log_reports, LOG_SUFFIXES),
            ("references", self.references, REFERENCE_SUFFIXES),
        )
        seen: set[str] = set()
        for role, names, suffixes in roles:
            for name in names:
                if not name or name in {".", ".."} or Path(name).name != name or re.search(r"[/\\\x00]", name):
                    raise ValueError(f"{role}: unsafe filename: {name!r}")
                if name == "analysis_input.json":
                    raise ValueError("analysis_input.json cannot be listed as evidence")
                if Path(name).suffix.lower() not in suffixes:
                    raise ValueError(f"{role}: unsupported file type: {name}")
                if name.casefold() in seen:
                    raise ValueError(f"a file must have exactly one role: {name}")
                seen.add(name.casefold())
        return self


@dataclass(frozen=True)
class SourceLine:
    role: str
    filename: str
    locator: str
    text: str
    sha256: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(expanded: Path, path: Path) -> tuple[ResultLogReviewManifest, dict[str, list[Path]]]:
    """Bind manifest filenames to one flat uploaded artifact namespace."""
    expanded = expanded.resolve()
    path = path.resolve()
    if path.parent != expanded or path.name != "analysis_input.json":
        raise ValueError("analysis_input.json must be at the top of the uploaded input set")
    manifest = ResultLogReviewManifest.model_validate_json(path.read_text(encoding="utf-8-sig"))
    bound: dict[str, list[Path]] = {}
    for role in ("diagnosis_results", "log_reports", "references"):
        bound[role] = []
        for name in getattr(manifest, role):
            target = (expanded / name).resolve(strict=True)
            if target.parent != expanded or not target.is_file() or target.is_symlink():
                raise ValueError(f"{role}: file missing or outside input root: {name}")
            bound[role].append(target)
    # An unbound file is not silently treated as evidence; the manifest determines role.
    actual = {p.name for p in expanded.iterdir() if p.is_file()}
    expected = {"analysis_input.json"} | {p.name for group in bound.values() for p in group}
    if actual != expected or any(p.is_dir() for p in expanded.iterdir()):
        raise ValueError(f"unlisted or nested inputs are forbidden: {sorted(actual - expected)}")
    return manifest, bound


def read_lines(path: Path, role: str) -> list[SourceLine]:
    """Use AstroGraph's existing DOCX/PDF structural parser; expose real locators."""
    suffix = path.suffix.lower()
    digest = _sha(path)
    if suffix in {".pdf", ".docx"}:
        parser = SectionFirstDocumentIngestor(maximum_section_units=18)
        if suffix == ".pdf":
            try:
                parsed, _audit = parser.ingest_pdf(path, source_id=digest[:16], knowledge_version="result-log-review-v1")
            except ValueError as exc:
                # Reference scans may pass preflight only as explicit untrusted
                # placeholders; their OCR text must NEVER become a fault cue.
                if (role != 'references' or 'no extractable text' not in str(exc)
                        or os.environ.get('ASTROGRAPH_MATERIAL_OCR_ENGINE','').lower()
                        not in {'tesseract','docling'}):
                    raise
                return [SourceLine(role,path.name,'pdf#scan:ocr-review-required',
                    '[SCANNED_REFERENCE_OCR_PENDING_INDEPENDENT_REVIEW]',digest)]
        else:
            parsed, _audit = parser.ingest_docx(path, source_id=digest[:16], knowledge_version="result-log-review-v1")
        content = [SourceLine(role, path.name, x.source_locator, x.text_representation, digest)
                   for x in parsed.elements if x.kind.value != "IMAGE" and x.text_representation.strip()]
        if sum(len(x.text) for x in content) > _MAX_TEXT_CHARS:
            raise ValueError(f"extracted text exceeds analysis limit: {path.name}")
    else:
        raw = path.read_text(encoding="utf-8-sig")
        if len(raw) > _MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds analysis limit: {path.name}")
        if suffix == ".json":
            value = json.loads(raw)
            if not isinstance(value, (dict, list)):
                raise ValueError(f"JSON evidence must be an object or list: {path.name}")
            # Validate JSON while keeping original physical line numbers for citations.
            json.dumps(value, allow_nan=False)
        elif suffix == ".csv":
            rows = list(csv.reader(io.StringIO(raw)))
            if len(rows) > 3000:
                raise ValueError(f"CSV row count exceeds analysis limit: {path.name}")
            content = [
                SourceLine(role, path.name, f"csv-row:{i}",
                           " | ".join(cell[:600] for cell in row[:80]), digest)
                for i, row in enumerate(rows, start=1) if any(cell.strip() for cell in row)
            ]
        if suffix != ".csv":
            content = [SourceLine(role, path.name, f"line:{i}", line.strip()[:2500], digest)
                       for i, line in enumerate(raw.splitlines(), start=1) if line.strip()]
    if not content:
        raise ValueError(f"no extractable text; scanned PDF requires reviewed transcription: {path.name}")
    return content


def validate_review_inputs(expanded: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, roles = load_manifest(expanded, manifest_path)
    descriptors: list[dict[str, Any]] = []
    for role, paths in roles.items():
        for path in paths:
            lines = read_lines(path, role)
            pending_ocr = (len(lines) == 1
                           and lines[0].text == '[SCANNED_REFERENCE_OCR_PENDING_INDEPENDENT_REVIEW]')
            descriptors.append({
                "role": role, "filename": path.name, "sha256": lines[0].sha256,
                "parsed_segment_count": 0 if pending_ocr else len(lines),
                "parser_status": ("OCR_PENDING_INDEPENDENT_REVIEW_NO_TRUSTED_TEXT"
                                  if pending_ocr else "TEXT_EXTRACTED"),
            })
    return {
        "detected_schema_versions": [manifest.schema_version],
        "files": descriptors,
        "role_counts": {k: len(v) for k, v in roles.items()},
        "executable_files": ["analysis_input.json", *manifest.diagnosis_results, *manifest.log_reports],
        "knowledge_files": list(manifest.references),
        "model_binding": {
            "status": "NOT_APPLICABLE_UPSTREAM_RESULT", "formal_binding_verified": False,
            "reason": "仅复核上游已产生的诊断结果，不再运行原始遥测检测模型。",
        },
    }


def _rules() -> list[dict[str, Any]]:
    data = json.loads(files("astrograph.resources").joinpath("result_log_fault_rules.json").read_text(encoding="utf-8"))
    if data.get("schema_version") != "astrograph-result-log-rule-catalog.v1":
        raise ValueError("invalid review rule catalog")
    return data["rules"]


def _matches(text: str, terms: list[str]) -> bool:
    haystack = text.casefold()
    return any(x.casefold() in haystack for x in terms if x)


def _candidate_mentions(line: SourceLine, rule: dict[str, Any]) -> bool:
    raw = line.text
    # Exclude obvious negative declarations / evaluation truth proxies from candidate extraction.
    if re.search(r"(ground[_ -]?truth|injected[_ -]?fault|true[_ -]?label|真实标签|标注真值)", raw, re.I):
        return False
    match = re.search(re.escape(rule["fault_id"]), raw, re.I)
    if match:
        pre = raw[max(0, match.start()-18):match.start()].casefold()
        return not any(x in pre for x in ("exclude", "excluded", "否认", "已排除", "排除", "非", "not "))
    for alias in rule["aliases"]:
        found = re.search(re.escape(alias), raw, re.I)
        if found and not any(x in raw[max(0, found.start()-12):found.start()].casefold()
                             for x in ("排除", "不是", "无", "no ", "not ", "exclude")):
            return True
    return False


# Preserve user-supplied fault candidates outside the limited engineering catalog.
# Their names are recorded as unverified assertions, never as inferred mechanisms.
_UPSTREAM_FAULT_CODE = re.compile(
    r"\b(?:ADCS|EPS|THERMAL|COMM|DATA|OBC|PAYLOAD|PROP|POWER|TTC|TTNC|GNC)"
    r"_[A-Z0-9]+(?:_[A-Z0-9]+)+\b"
)
_UPSTREAM_FAULT_NAME = re.compile(
    r"[\"']?(?:fault_name|fault_type|suspected_fault|故障名称|故障类型|诊断结论)[\"']?"
    r"\s*[:：=]\s*[\"']?([^\"',\n}\]]{2,70})", re.I,
)
_EXCLUDED_RESULT_LINE = re.compile(
    r"ground[_ -]?truth|injected[_ -]?fault|true[_ -]?label|真实标签|标注真值|已排除|exclude[d]?",
    re.I,
)


def _unmapped_upstream_candidates(
    result_lines: list[SourceLine], recognized: list[dict[str, Any]]
) -> list[tuple[str, str, SourceLine]]:
    known = {rule["fault_id"].casefold() for rule in recognized}
    already_named = {term.casefold() for rule in recognized for term in rule["aliases"]}
    seen: set[str] = set()
    result: list[tuple[str, str, SourceLine]] = []
    for line in result_lines:
        if _EXCLUDED_RESULT_LINE.search(line.text):
            continue
        for match in _UPSTREAM_FAULT_CODE.finditer(line.text):
            fault_id = match.group(0)
            if fault_id.casefold() in known or fault_id.casefold() in seen:
                continue
            before = line.text[max(0, match.start() - 24):match.start()]
            if re.search(r"(?i)\b(no|not|without|exclude)\s*$|(?:排除|不是|并非|否认|非)$", before):
                continue
            seen.add(fault_id.casefold())
            result.append((fault_id, fault_id, line))
        if not _UPSTREAM_FAULT_CODE.search(line.text):
            match = _UPSTREAM_FAULT_NAME.search(line.text)
            if match:
                value = match.group(1).strip().rstrip(" .。；; ")
                if (value.casefold() not in seen and value.casefold() not in already_named
                        and not re.fullmatch(r"normal|none|no.fault|healthy|ok|正常|无故障|未知", value, re.I)):
                    seen.add(value.casefold())
                    result.append((f"UPSTREAM_UNMAPPED_{len(result) + 1}", value, line))
    return result


def _source_receipt(line: SourceLine, ordinal: int, prefix: str) -> dict[str, Any]:
    return {
        "evidence_id": f"{prefix}-{ordinal:04d}", "role": line.role,
        "filename": line.filename, "locator": line.locator,
        "sha256": line.sha256, "excerpt": line.text[:420],
    }


def _pick_evidence(lines: list[SourceLine], predicate, prefix: str, limit: int) -> list[dict[str, Any]]:
    out = []
    for index, line in enumerate(lines, start=1):
        if predicate(line):
            out.append(_source_receipt(line, index, prefix))
            if len(out) >= limit:
                break
    return out


def analyze_result_log_review(expanded: Path, manifest_path: Path) -> dict[str, Any]:
    """Conservative, traceable review: candidate is sourced ONLY from prior result."""
    manifest, roles = load_manifest(expanded, manifest_path)
    sources: dict[str, list[SourceLine]] = {
        role: [line for path in paths for line in read_lines(path, role)]
        for role, paths in roles.items()
    }
    result_lines = sources["diagnosis_results"]
    log_lines = sources["log_reports"]
    reference_lines = sources["references"]
    rules = _rules()
    recognized = [r for r in rules if any(_candidate_mentions(line, r) for line in result_lines)]
    # Preserve order of first mention in upstream diagnosis, not arbitrary rule order.
    def mention_position(rule: dict[str, Any]) -> int:
        return next((i for i, line in enumerate(result_lines) if _candidate_mentions(line, rule)), 999999)
    recognized.sort(key=mention_position)
    candidates: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    gaps: list[str] = []
    for rank, rule in enumerate(recognized[:5], start=1):
        result_e = _pick_evidence(result_lines, lambda x: _candidate_mentions(x, rule), f"D{rank}", 2)
        log_e = _pick_evidence(log_lines, lambda x: _matches(x.text, rule["log_signals"]), f"L{rank}", _MAX_CITATIONS_PER_CANDIDATE)
        ref_e = _pick_evidence(reference_lines, lambda x: _matches(x.text, [rule["fault_id"], *rule["aliases"], *rule["reference_terms"]]), f"R{rank}", 3)
        evidence.extend([*result_e, *log_e, *ref_e])
        if not log_e:
            gaps.append(f"{rule['fault_id']}: 日志未检索到有区分度的关联现象；诊断候选未获日志交叉印证。")
        else:
            gaps.append(f"{rule['fault_id']}: 已找到相关日志术语，但尚未完成设备、时标与因果关系核验；不可据此确证根因。")
        if not ref_e:
            gaps.append(f"{rule['fault_id']}: 上传参考文献没有可关联的引文；规则机理不能冒充文献结论。")
        all_ids = [x["evidence_id"] for x in (*result_e, *log_e, *ref_e)]
        candidates.append({
            "rank": rank,
            "fault_id": rule["fault_id"], "fault_name": rule["label"],
            "status": "PROVISIONAL_LOG_CUE" if log_e else "PROVISIONAL_UNCORROBORATED",
            "diagnosis_source_ids": [x["evidence_id"] for x in result_e],
            "log_evidence_ids": [x["evidence_id"] for x in log_e],
            "reference_citation_ids": [x["evidence_id"] for x in ref_e],
            "cause_hypotheses": [{"text": rule["cause"], "status": "HYPOTHESIS_NOT_CONFIRMED", "basis_ids": all_ids}],
            "handling_measures": [
                {"action": action, "status": "RECOMMENDED_FOR_HUMAN_REVIEW",
                 "basis_ids": all_ids, "control_authorized": False}
                for action in rule["actions"]
            ],
            "verification_checks": rule["verification"],
        })
    remaining_slots = 5 - len(candidates)
    for fault_id, fault_name, source in _unmapped_upstream_candidates(result_lines, recognized)[:remaining_slots]:
        rank = len(candidates) + 1
        result_e = [_source_receipt(source, result_lines.index(source) + 1, f"D{rank}")]
        # Mentions of an unknown code in logs are links for examination, not
        # independent/time-aligned corroboration of the proposed root cause.
        search_terms = [fault_name] if fault_id.startswith("UPSTREAM_UNMAPPED_") else [fault_id]
        log_e = _pick_evidence(log_lines, lambda x: _matches(x.text, search_terms),
                               f"L{rank}", _MAX_CITATIONS_PER_CANDIDATE)
        ref_e = _pick_evidence(reference_lines, lambda x: _matches(x.text, search_terms), f"R{rank}", 3)
        evidence.extend([*result_e, *log_e, *ref_e])
        all_ids = [x["evidence_id"] for x in (*result_e, *log_e, *ref_e)]
        gaps.append(f"{fault_name}: 该上游候选不在已审规则目录内；缺少经专家确认的故障机理、判据和具体处置流程。")
        if not log_e:
            gaps.append(f"{fault_name}: 日志未找到同名线索；需核对诊断时间窗和告警原文。")
        if not ref_e:
            gaps.append(f"{fault_name}: 未找到明确提及该候选的参考段落。")
        candidates.append({
            "rank": rank, "fault_id": fault_id, "fault_name": fault_name,
            "status": "PROVISIONAL_UNCORROBORATED",
            "diagnosis_source_ids": [x["evidence_id"] for x in result_e],
            "log_evidence_ids": [x["evidence_id"] for x in log_e],
            "reference_citation_ids": [x["evidence_id"] for x in ref_e],
            "cause_hypotheses": [{
                "text": f"上游诊断报告提出‘{fault_name}’候选，但其具体物理根因尚无经过本系统验证的机理映射，不能据此断言故障原因。",
                "status": "UNKNOWN_MECHANISM_PENDING_REVIEW", "basis_ids": all_ids,
            }],
            "handling_measures": [
                {"action": "由领域专家核验原诊断算法的输出含义、证据阈值和对应设备配置，确认是否需要升级告警或制定处置预案。",
                 "status": "RECOMMENDED_FOR_HUMAN_REVIEW", "basis_ids": all_ids, "control_authorized": False},
                {"action": "核对事件时间窗及前后关键日志，按适用型号的故障处置手册选择经批准的操作，暂不生成自动控制指令。",
                 "status": "RECOMMENDED_FOR_HUMAN_REVIEW", "basis_ids": all_ids, "control_authorized": False},
            ],
            "verification_checks": [
                "取得上游诊断模型/规则版本及关键原始判据，并复核候选含义",
                "校核日志时间戳、设备标识、异常事件及参考手册的型号/版本适配性",
            ],
        })
    status = "REVIEW_REQUIRED" if candidates else "INSUFFICIENT_EVIDENCE"
    if not candidates:
        gaps.append("诊断结果未出现当前规则目录或通用上游故障字段中可识别的候选；系统不会仅凭日志推断新的已确认根因。")
    evidence = list({item["evidence_id"]: item for item in evidence}.values())
    source_manifest = [{"role": role, "filename": path.name, "sha256": _sha(path)}
                       for role, paths in roles.items() for path in paths]
    generic_log_e = _pick_evidence(
        log_lines,
        lambda x: bool(re.search(r"\b(error|failed|fault|critical|warning|timeout)\b|报警|故障|异常|告警|超时", x.text, re.I)),
        "LOG", 6,
    )
    if not candidates:
        evidence.extend(generic_log_e)
    return {
        "schema_version": "astrograph-result-log-root-cause-review.v1",
        "status": status,
        "input_schema_version": manifest.schema_version,
        "review_type": "UPSTREAM_RESULT_PLUS_LOG_PLUS_REFERENCE",
        "source_manifest": source_manifest,
        "equipment_scope": manifest.equipment_scope or "UNKNOWN",
        "candidates": candidates,
        "top_fault_id": candidates[0]["fault_id"] if candidates else None,
        "cause_hypotheses": [{"fault_id": c["fault_id"], **h} for c in candidates for h in c["cause_hypotheses"]],
        "handling_measures": [{"fault_id": c["fault_id"], **a} for c in candidates for a in c["handling_measures"]] if candidates else [
            {"fault_id": None, "action": "核对上游诊断结果的故障编码、关键告警与日志时序，并补充相关设备的运维手册。",
             "status": "RECOMMENDED_FOR_HUMAN_REVIEW", "basis_ids": [x["evidence_id"] for x in generic_log_e],
             "control_authorized": False},
        ],
        "verification_checks": [x for c in candidates for x in c["verification_checks"]] if candidates else [
            "确认诊断结果对应的设备、时间范围与告警版本", "核对日志采集完整性和参考文献适用型号/版本",
        ],
        "evidence": evidence[:80],
        "evidence_gaps": gaps,
        "evidence_policy": {
            "upstream_diagnosis_is_ground_truth": False,
            "log_is_verified_sensor_telemetry": False,
            "reference_is_observation": False,
            "new_raw_telemetry_detection_executed": False,
            "formal_diagnosis_publishable": False,
            "automatic_control_authorized": False,
            "rule_catalog_status": "ENGINEERING_HEURISTICS_REVIEW_REQUIRED",
            "log_matches_are_keyword_cues_only": True,
            "upstream_label_provenance": "UNVERIFIED_BY_THIS_PIPELINE",
        },
        "conclusion": (
            "已复核上游诊断候选、关联日志线索与参考材料，故障原因仍为待验证假设。"
            if candidates else "证据不足，未确认故障原因；给出补充证据与人工复核措施。"
        ),
        "limitations": [
            "报告是材料驱动的辅助研判，不执行基于原始遥测的检测、分类或置信度标定。",
            "工程规则是待专家审核的机理提示；文献匹配仅是文本关联，不代表全文核验。",
            "任何断电、切换、重构、指令或对在轨系统的操作均需专门批准。",
        ],
    }


def render_result_log_markdown(job: dict[str, Any], result: dict[str, Any], knowledge: dict[str, Any] | None = None, reasoning: dict[str, Any] | None = None) -> str:
    def safe(value: Any) -> str:
        return str(value or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    lines = [
        "# AstroGraph 诊断结果—日志—文献故障原因辅助分析报告", "",
        f"- 任务：`{job['job_id']}`", f"- 目标：{safe(job['task'])}",
        "- 输入模式：上游诊断结果＋运行日志＋参考文献（无须原始遥测）",
        f"- 研判状态：**{result['status']}**", "- 在轨控制/正式故障发布：**不授权**", "",
        "## 研判摘要", "", result["conclusion"], "",
        "## 故障候选与原因假设", "",
    ]
    if not result["candidates"]:
        lines.append("- 未识别可追溯的已知故障候选；不得从日志关键词推导已确认根因。")
    for candidate in result["candidates"]:
        lines.extend([f"### {candidate['rank']}. {candidate['fault_name']} (`{candidate['fault_id']}`)", "",
                      f"- 交叉印证：`{candidate['status']}`",
                      f"- 上游诊断依据：{', '.join(candidate['diagnosis_source_ids']) or '无'}",
                      f"- 日志证据：{', '.join(candidate['log_evidence_ids']) or '未找到'}",
                      f"- 参考引文：{', '.join(candidate['reference_citation_ids']) or '未找到'}"])
        for cause in candidate["cause_hypotheses"]:
            lines.append(f"- **可能原因（待验证）**：{safe(cause['text'])}")
        lines.extend(["", "**建议核查 / 处置措施（人工审核后）**", ""])
        lines.extend(f"- {safe(x['action'])}" for x in candidate["handling_measures"])
        lines.extend(["", "**验证方法**", ""])
        lines.extend(f"- {safe(x)}" for x in candidate["verification_checks"])
        lines.append("")
    if not result["candidates"]:
        lines.extend(["## 处置与验证建议", ""])
        lines.extend(f"- {safe(x['action'])}" for x in result["handling_measures"])
        lines.extend(f"- {safe(x)}" for x in result["verification_checks"])
    lines.extend(["", "## 逐条证据 / 引文定位", "",
                  "| 证据 ID | 角色 | 文件 | 定位 | 原文节选 |", "|---|---|---|---|---|"])
    for item in result["evidence"]:
        lines.append(f"| `{item['evidence_id']}` | {safe(item['role'])} | {safe(item['filename'])} | {safe(item['locator'])} | {safe(item['excerpt'])} |")
    lines.extend(["", "## 待补证据", ""])
    lines.extend(f"- {safe(x)}" for x in result["evidence_gaps"] or ["当前规则覆盖范围内未发现新增缺口；仍需专家复核。"])
    lines.extend(["", "## 输入文件 SHA-256 血缘", ""])
    lines.extend(f"- `{x['role']}` `{safe(x['filename'])}` sha256=`{x['sha256']}`" for x in result["source_manifest"])
    if reasoning is not None:
        lines.extend(["", "## RAG × 知识图谱 × 专家案例联合证据核查", "",
                      f"- 知识联查状态：`{reasoning.get('status', 'UNKNOWN')}`",
                      f"- 设备适用范围：`{safe(reasoning.get('equipment_scope'))}`",
                      f"- 图谱查询：`{(reasoning.get('graph_query') or {}).get('status', 'NOT_CONFIGURED')}`",
                      f"- 本地语义向量：`{(reasoning.get('semantic_backend') or {}).get('status', 'NOT_CONFIGURED')}`", ""])
        global_search=reasoning.get('global_community_retrieval') or {}
        community_index=reasoning.get('global_community_index') or {}
        lines.extend([f"- 已审知识社区数：{community_index.get('community_count',0)}；"
                      f"全局检索状态：`{safe(global_search.get('status'))}`",
                      "- 社区报告为已审文献关系的摘录索引，不是 Microsoft GraphRAG LLM 社区摘要或故障确诊。"])
        for pointer in (global_search.get('evidence_points') or [])[:8]:
            lines.append(f"- 全局关系 `{safe(pointer.get('edge_id'))}` "
                f"{safe(pointer.get('subject'))} -{safe(pointer.get('predicate'))}-> "
                f"{safe(pointer.get('object'))}；出处 {safe(pointer.get('filename'))} "
                f"{safe(','.join(pointer.get('source_locators') or []))}")
        for item in reasoning.get('candidates', []):
            lines.extend([f"### {safe(item['fault_name'])} (`{safe(item['fault_id'])}`)", "",
                          f"- 日志证据评估：`{item['evidence_assessment']['assessment']}`",
                          f"- 数字化概率：不提供；当前根因未独立确认", ""])
            log = item['log_evidence_matrix']
            for event in log['events']:
                lines.append(f"- 日志 `{safe(event['evidence_id'])}` [{safe(event['assessment'])}] "
                             f"{safe(event.get('timestamp') or '未解析时间')} "
                             f"`{safe(event['filename'])}:{safe(event['locator'])}`：{safe(event['excerpt'])}")
            refs = item['reference_retrieval']
            trace = refs.get('retrieval_trace') or {}
            lexicon = refs.get('terminology') or {}
            lines.extend(["", "**分层混合检索：文档 → 章节 → 来源片段（非因果确证）**", "",
                f"- 模式：`{safe(refs.get('retrieval_mode'))}`，状态：`{safe(refs.get('status'))}`",
                f"- 检索通道：{safe(', '.join(refs.get('methods') or []))}",
                f"- 文档数：{trace.get('document_index_count', 0)}；片段数：{trace.get('chunk_index_count', 0)}；"
                f"排除未核验表示片段：{trace.get('unsafe_representation_excluded_count', 0)}",
                f"- 术语词表：`{safe(lexicon.get('lexicon_version'))}`；"
                f"任务词表：`{safe((lexicon.get('mission_glossary') or {}).get('status','NOT_CONFIGURED'))}`",
                f"- 源库是否有未索引内容：`{trace.get('index_truncated', False)}`", "",
                "**按故障候选检索的参考片段（主题相关，不构成根因证明）**", ""])
            for hit in refs['hits']:
                lines.append(f"- `{safe(hit['knowledge_id'])}` `{safe(hit['filename'])}` "
                             f"[{safe(','.join(hit['source_locators']))}] sha256=`{hit['source_sha256']}` "
                             f"模态={safe(','.join(hit.get('modalities') or []))} "
                             f"检索通道={safe(','.join(hit['retrieval_channels']))}：{safe(hit['quote'][:380])}")
                if hit.get('table_header_context'):
                    lines.append(f"  - 表头定位（上下文，不属于上行连续引文）：{safe(hit['table_header_context'][:250])}")
            lines.extend(["", "**图谱预期证据关系（只读）**", ""])
            for path in item['graph_evidence_paths']:
                lines.append(f"- {safe(path['fault_id'])} -{safe(path['relation_type'])}-> "
                             f"{safe(path['evidence_definition_id'])}，对应日志指认："
                             f"{safe(','.join(path['matched_log_ids']) or '未核实')}；不是因果确证")
            reviewed_graph=item.get('approved_document_graph') or {}
            lines.extend(['',f"**已审文献图谱路径（图谱声明，非本次实测）：`{safe(reviewed_graph.get('status'))}`**",''])
            for relation in reviewed_graph.get('paths',[])[:16]:
                lines.append(f"- `{safe(relation['proposal_id'])}` "
                  f"{safe(relation['source']['name'])} -{safe(relation['relation_type'])}-> "
                  f"{safe(relation['target']['name'])} [{safe(relation['polarity'])}]，"
                  f"争议={safe(relation.get('conflict_status','NO_APPROVED_POLARITY_CONFLICT'))}，"
                  f"陈述限定={safe(relation.get('claim_modality','UNSPECIFIED'))}，"
                  f"文献 `{safe(relation['provenance']['filename'])}` "
                  f"sha256=`{relation['provenance']['source_sha256']}`，"
                  f"原文：{safe(relation['evidence_quote'][:220])}")
            memory = item['historical_confirmed_cases']
            lines.extend(["",f"**同型号已审核历史案例：`{memory['status']}`**", ""])
            for case in memory['hits']:
                lines.append(f"- `{safe(case['case_id'])}` 专家核验故障=`{safe(case['verified_fault_id'])}` "
                             f"独立来源=`{safe(case['truth_source_ref'])}` 适用范围=`{safe(case['equipment_scope'])}`；"
                             f"不是本次任务已确认原因；历史关系={safe(case.get('relation_status'))}")
            for action in item['review_action_options']:
                lines.append(f"- 历史已审措施（本次仍需批准）：{safe(action['action'])}")
            chain = item.get('multi_hop') or {}
            lines.extend(['', '**证据约束多跳推理（最多 3 跳）**',
                          f"- 根因回答闸门：`{safe(chain.get('decision','NOT_EVALUATED'))}`；" +
                          f"响应模式：`{safe(chain.get('response_mode','ABSTAIN_FROM_ROOT_CAUSE'))}`",
                          '- 图谱路径不等于因果确诊；未审核公共图谱不会升级为已审证据。'])
            for path in (chain.get('approved_paths') or [])[:12]:
                edges = path.get('hops') or []
                desc = '；'.join(f"{safe(h['source']['name'])} -{safe(h['relation_type'])}/{safe(h['direction'])}-> {safe(h['target']['name'])} " +
                    f"[{safe(h['edge_id'])}; {safe(','.join(h['provenance'].get('source_locators') or []))}]" for h in edges)
                lines.append(f"- 已审路径 `{safe(path['path_id'])}` ({path['hop_count']} 跳，"+
                             f"可用作初步假设={path.get('eligible_as_hypothesis_path',False)})：{desc}")
                for ev in (path.get('log_crosscheck') or {}).get('supporting',[])[:4]:
                    lines.append(f"  - 日志关联 `{safe(ev['evidence_id'])}` `{safe(ev['filename'])}:{safe(ev['locator'])}`；未独立核验")
                for ev in (path.get('log_crosscheck') or {}).get('contradicting',[])[:4]:
                    lines.append(f"  - 反证/恢复 `{safe(ev['evidence_id'])}` `{safe(ev['filename'])}:{safe(ev['locator'])}`")
            for path in (chain.get('public_reference_paths') or [])[:5]:
                lines.append(f"- **未审核公共知识** `{safe(path['path_id'])}` ({path['hop_count']} 跳)，"+
                             '仅供查阅，不得作为根因证明。')
            for gap in chain.get('knowledge_gap_codes') or []:
                lines.append(f'- 待补知识：`{safe(gap)}`')
            if not log['events']:
                lines.append('- 未检索到关联日志现象，需补齐设备和时间范围的证据。')
            lines.append("")
        answer_policy=reasoning.get('answer_policy') or {}
        lines.extend(['', '## 防幻觉回答政策',
            f"- 是否已确诊当前根因：`{answer_policy.get('current_root_cause_confirmed', False)}`",
            f"- 所有候选是否拒绝根因确诊：`{answer_policy.get('all_candidates_abstained', True)}`",
            '- 未经过专家确认的图谱关系、文献转述和日志关键词不能作为独立的根因证明；知识缺失明确拒答。'])
        gaps=reasoning.get('reasoning_gap_learning') or {}
        lines.append(f"- 知识缺口入待审队列：`{safe(gaps.get('status','NOT_STAGED'))}`")
        for gap in gaps.get('proposals',[]):
            lines.append(f"- 缺口 `{safe(gap.get('gap_id'))}`：{safe(', '.join(gap.get('knowledge_gap_codes') or []))}")
        learning = reasoning.get('case_learning') or {}
        lines.extend(["## 知识持续积累 / 审核状态", "",
                      f"- 自动保存待审核案例：`{learning.get('status', 'NOT_STAGED')}`",
                      f"- 活动图谱自动写入：**否**", ""])
        for item in learning.get('proposals', []):
            lines.append(f"- `{safe(item['case_id'])}`：`{safe(item['status'])}`（须专家独立确认，禁止自证）")
        documents = reasoning.get('document_learning') or {}
        lines.append(f"- 参考文献知识库提案：`{documents.get('status', 'NOT_STAGED')}`")
        for doc in documents.get('proposals', []):
            lines.append(f"- 文献 `{safe(doc['document_id'])}`：`{safe(doc['status'])}`（只在独立审核后进入可复用库）")
        build=reasoning.get('graph_construction') or {}
        lines.extend(['',f"### 文献驱动的 LLM 图谱构建：`{safe(build.get('status','DISABLED'))}`",''])
        lines.append(f"- 抽取片段={build.get('chunks_processed',0)}；有效候选关系={build.get('proposal_count',0)}；"
                     f"待审核入库={safe(build.get('staging_status','NOT_STAGED'))}")
        for relation in (build.get('candidate_relations') or [])[:12]:
            lines.append(f"- [未经审核] `{safe(relation['proposal_id'])}` "
              f"{safe(relation['source']['name'])} -{safe(relation['relation_type'])}-> "
              f"{safe(relation['target']['name'])} [{safe(relation['polarity'])}]，"
              f"来源={safe(relation['provenance']['filename'])} "
              f"定位={safe(','.join(relation['provenance']['source_locators']))}，"
              f"原文={safe(relation['evidence_quote'][:200])}")
    if knowledge is not None:
        lines.extend(["", "## 可选知识 / LLM 辅助解释（仅供复核）", ""])
        lines.append(f"- 检索与生成状态：`{knowledge.get('status', 'UNKNOWN')}` / `{(knowledge.get('generation') or {}).get('status', 'UNKNOWN')}`")
        model_explanation = knowledge.get("model_explanation") or {}
        if model_explanation.get("status") == "AVAILABLE_REQUIRES_EXPERT_REVIEW":
            lines.extend(["", "### 大模型综合处置结果（需专家复核）", ""])
            lines.append(f"- 综合判断：{safe(model_explanation.get('summary'))}")
            for claim in (model_explanation.get("claims") or [])[:12]:
                citations = ", ".join(str(item) for item in (claim.get("citation_ids") or []))
                lines.append(f"- `{safe(claim.get('kind'))}`：{safe(claim.get('text'))}；引用 `{safe(citations)}`")
            actions = model_explanation.get("recommended_review_actions") or []
            if actions:
                lines.extend(["", "#### 建议处置与核查", ""])
                lines.extend(f"- {safe(item)}" for item in actions[:12])
            lines.append("- 自动控制授权：**否**；涉及设备操作仍需人工审批。")
        explanation = ((knowledge.get("final_explanation") or {}).get("explanation") or {})
        if explanation:
            lines.append(f"- 规则与证据边界说明：{safe(explanation.get('summary'))}")
            for claim in (explanation.get("claims") or [])[:6]:
                lines.append(f"- `{safe(claim.get('kind'))}`：{safe(claim.get('text'))}；引用 {', '.join(claim.get('citation_ids') or [])}")
    lines.extend(["", "## 适用边界", ""])
    lines.extend(f"- {safe(x)}" for x in result["limitations"])
    return "\n".join(lines) + "\n"
