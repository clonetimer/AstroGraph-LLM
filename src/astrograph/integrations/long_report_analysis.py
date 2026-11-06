"""Bounded, source-preserving hierarchical analysis for long external reports."""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from astrograph.integrations.reviewed_context import validate_snapshot, CONTEXT_POLICY
from astrograph.integrations.stage_execution import for_stage, check_cancelled, AnalysisCancelled
from astrograph.integrations.execution_observability import event
from astrograph.integrations.claim_governance import build_claim_governance
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from astrograph.integrations.external_diagnosis_analysis import (
    ANALYSIS_SCHEMA, SYSTEM_POLICY, ExternalDiagnosisAnalysis, _canonical, _digest,
    _source_id, _validate_binding, run_external_diagnosis_analysis,
)

MAX_PORTABLE_SOURCE_CHARACTERS = 180_000
MAX_ANALYSIS_CHUNKS = 24
MAX_SYNTHESIS_CHARACTERS = 120_000
DEFAULT_CHUNK_CHARACTERS = 12_000
DEFAULT_OVERLAP_CHARACTERS = 600

@dataclass(frozen=True)
class AnalysisChunk:
    sequence: int
    source_id: str
    source_ids: tuple[str, ...]
    filename: str
    role: str
    locator: str
    sha256: str
    source_spans: tuple[dict[str, Any], ...]
    start_offset: int
    end_offset: int
    text: str
    text_hash: str


def _preferred_end(text: str, start: int, limit: int) -> int:
    if limit >= len(text):
        return len(text)
    floor = start + max(1, (limit - start) * 2 // 3)
    for marker in ("\n", "。", "！", "？", ".", "!", "?", "；", ";"):
        position = text.rfind(marker, floor, limit)
        if position >= floor:
            return position + 1
    return limit


def plan_analysis_chunks(sources: list[dict[str, Any]], *, max_chars: int = DEFAULT_CHUNK_CHARACTERS,
                         overlap_chars: int = DEFAULT_OVERLAP_CHARACTERS) -> list[AnalysisChunk]:
    if not 1024 <= max_chars <= 24_000 or not 0 <= overlap_chars < max_chars // 2:
        raise ValueError("invalid chunk limits")
    if sum(len(str(source.get("text", ""))) for source in sources) > MAX_PORTABLE_SOURCE_CHARACTERS:
        raise ValueError("INPUT_TOO_LARGE")
    # Pack adjacent physical lines/blocks from one file before chunking. This
    # prevents a TXT report with thousands of short lines becoming thousands
    # of model requests, while retaining every original line/page locator.
    groups: list[list[dict[str, Any]]] = []
    for source in sources:
        key = (source.get("role"), source.get("filename"), source.get("sha256"))
        if not groups or (groups[-1][0].get("role"), groups[-1][0].get("filename"), groups[-1][0].get("sha256")) != key:
            groups.append([])
        groups[-1].append(source)
    chunks: list[AnalysisChunk] = []
    for group in groups:
        combined_parts: list[str] = []
        spans: list[dict[str, Any]] = []
        cursor = 0
        for source in group:
            text = source.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("EMPTY_SOURCE")
            if combined_parts:
                combined_parts.append("\n")
                cursor += 1
            begin = cursor
            combined_parts.append(text)
            cursor += len(text)
            spans.append({"source_id": source["source_id"], "locator": source["locator"],
                          "start_offset": begin, "end_offset": cursor})
        combined = "".join(combined_parts)
        start = 0
        while start < len(combined):
            end = _preferred_end(combined, start, min(len(combined), start + max_chars))
            if end <= start:
                raise ValueError("CHUNK_PROGRESS_FAILURE")
            piece = combined[start:end]
            overlap_spans = [span for span in spans if span["end_offset"] > start and span["start_offset"] < end]
            if not overlap_spans:
                raise ValueError("CHUNK_SOURCE_MAPPING_FAILED")
            first, last = overlap_spans[0], overlap_spans[-1]
            if first["source_id"] == last["source_id"]:
                locator = f"{first['locator']}#chars:{max(0, start-first['start_offset'])}-{min(first['end_offset'],end)-first['start_offset']}"
            else:
                locator = f"source-span:{first['locator']}..{last['locator']}"
            chunks.append(AnalysisChunk(
                sequence=len(chunks) + 1, source_id=first["source_id"],
                source_ids=tuple(dict.fromkeys(span["source_id"] for span in overlap_spans)),
                filename=group[0]["filename"], role=group[0]["role"], locator=locator,
                sha256=group[0]["sha256"], source_spans=tuple(overlap_spans),
                start_offset=start, end_offset=end, text=piece,
                text_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest()))
            if end >= len(combined):
                break
            start = max(start + 1, end - overlap_chars)
    if len(chunks) > MAX_ANALYSIS_CHUNKS:
        raise ValueError("TOO_MANY_ANALYSIS_CHUNKS")
    return chunks



def coalesce_sources(task_id: str, sources: list[dict[str, Any]], *, max_chars: int = 10_000) -> list[dict[str, Any]]:
    """Coalesce physical lines/blocks into source records with locator ranges."""
    output: list[dict[str, Any]] = []
    groups: list[list[dict[str, Any]]] = []
    for source in sources:
        key = (source.get("role"), source.get("filename"), source.get("sha256"))
        if not groups or (groups[-1][0].get("role"), groups[-1][0].get("filename"), groups[-1][0].get("sha256")) != key:
            groups.append([])
        groups[-1].append(source)
    for group in groups:
        current: list[dict[str, Any]] = []
        size = 0
        for source in group:
            text = source["text"]
            if current and size + len(text) + 1 > max_chars:
                output.append(_coalesced_record(task_id, current))
                current, size = [], 0
            if len(text) > max_chars:
                if current:
                    output.append(_coalesced_record(task_id, current))
                    current, size = [], 0
                start = 0
                while start < len(text):
                    end = min(len(text), start + max_chars)
                    part = {**source, "text": text[start:end],
                            "locator": f"{source['locator']}#chars:{start}-{end}"}
                    output.append(_coalesced_record(task_id, [part]))
                    start = end
            else:
                current.append(source)
                size += len(text) + (1 if len(current) > 1 else 0)
        if current:
            output.append(_coalesced_record(task_id, current))
    return output


def _coalesced_record(task_id: str, group: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = group[0], group[-1]
    locator = first["locator"] if len(group) == 1 else f"source-span:{first['locator']}..{last['locator']}"
    record = {"role": first["role"], "filename": first["filename"], "locator": locator,
              "sha256": first["sha256"], "text": "\n".join(item["text"] for item in group)}
    record["source_id"] = _source_id(task_id, record)
    return record

def _failure(input_bundle: dict, code: str, status: str = "FAILED", sources: list | None = None) -> dict:
    explanation = {"schema_version": ANALYSIS_SCHEMA, "status": status, "source": "LLM_EXTERNAL_DIAGNOSIS",
                   "input_sha256": input_bundle["input_sha256"], "citation_reference_validation": "NOT_EVALUATED",
                   "reason_code": code, "sources": sources or input_bundle["sources"],
                   "automatic_control_authorized": False}
    return {"generation": {"status": status, "reason_code": code}, "model_explanation": explanation}


def run_hierarchical_analysis(*, task_id: str, task: str, input_bundle: dict[str, Any],
                              llm_backend: Any, output_root: Path,
                              progress: Callable[[dict[str, Any]], None] | None = None,
                              execution_fingerprint: str = "", context_snapshot=None) -> dict[str, Any]:
    try:
        input_bundle = _validate_binding(task_id, input_bundle)
    except Exception:
        return _failure(input_bundle, "INVALID_INPUT_BINDING")
    if context_snapshot is not None:
        context_snapshot = validate_snapshot(context_snapshot)
    check_cancelled()
    try:
        chunks = plan_analysis_chunks(input_bundle["sources"])
    except ValueError as exc:
        code = str(exc)
        return _failure(input_bundle, code, "INPUT_TOO_LARGE" if code in {"INPUT_TOO_LARGE", "TOO_MANY_ANALYSIS_CHUNKS"} else "FAILED")

    chunk_records: list[dict[str, Any]] = []
    for chunk in chunks:
        record = {"role": chunk.role, "filename": chunk.filename,
                  "locator": chunk.locator, "sha256": chunk.sha256, "text": chunk.text,
                  "source_spans": list(chunk.source_spans)}
        record["source_id"] = _source_id(task_id, record)
        chunk_records.append(record)

    task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()
    chunk_fingerprint = _digest([{"source_id": record["source_id"],
                                  "text_sha256": hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
                                  "locator": record["locator"]} for record in chunk_records])
    checkpoint_path = Path(output_root) / "checkpoint.json"
    checkpoint = {"schema_version": "astrograph-long-report-checkpoint.v1",
                  "input_sha256": input_bundle["input_sha256"], "task_sha256": task_hash,
                  "chunk_fingerprint": chunk_fingerprint, "execution_fingerprint": execution_fingerprint,
                  "chunk_count": len(chunks), "completed": [],
                  "context_sha256": context_snapshot["sha256"] if context_snapshot else None}
    try:
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if all(saved.get(key) == checkpoint[key] for key in
               ("schema_version", "input_sha256", "task_sha256", "chunk_fingerprint",
                "execution_fingerprint", "chunk_count", "context_sha256")) and isinstance(saved.get("completed"), list):
            valid = []
            for item in saved["completed"]:
                sequence = item.get("sequence") if isinstance(item, dict) else None
                if (type(sequence) is int and 1 <= sequence <= len(chunks)
                        and item.get("source_id") == chunk_records[sequence - 1]["source_id"]
                        and isinstance(item.get("summary"), dict)
                        and all(isinstance(item["summary"].get(field), expected)
                                for field, expected in (("summary", str), ("findings", list),
                                                        ("hypotheses", list), ("information_gaps", list)))):
                    valid.append(item)
            checkpoint["completed"] = valid
    except (OSError, ValueError, TypeError):
        pass

    def save_checkpoint() -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        temporary.write_text(_canonical(checkpoint) + "\n", encoding="utf-8")
        temporary.replace(checkpoint_path)

    summaries_by_sequence = {item["sequence"]: item["summary"] for item in checkpoint["completed"]}
    for index, chunk in enumerate(chunks, start=1):
        check_cancelled()
        record = chunk_records[index - 1]
        if index in summaries_by_sequence:
            event("CHUNK_RESTORED", index=index, total=len(chunks))
            if progress:
                progress({"stage": "CHUNK_RESTORED", "completed": len(summaries_by_sequence), "total": len(chunks)})
            continue
        subtask = f"{task_id}-chunk-{index:03d}"
        subrecord = {key: record[key] for key in ("role", "filename", "locator", "sha256", "text")}
        subrecord["source_id"] = _source_id(subtask, subrecord)
        subbundle = {"schema_version": input_bundle["schema_version"], "task_id": subtask,
                     "input_sha256": _digest([subrecord]), "sources": [subrecord], "character_count": len(chunk.text)}
        event("CHUNK_ANALYSIS", index=index, total=len(chunks))
        result = run_external_diagnosis_analysis(
            task_id=subtask,
            task=f"{task}\n仅分析第 {index}/{len(chunks)} 个连续原文片段，保留局部不确定性，不推断片段之外的信息。",
            input_bundle=subbundle, llm_backend=for_stage(llm_backend, "chunk"),
            output_root=Path(output_root) / "chunks" / f"{index:03d}", context_snapshot=context_snapshot)
        if result["generation"]["status"] != "SUCCEEDED":
            failed = _failure(input_bundle, "CHUNK_ANALYSIS_FAILED", sources=[*input_bundle["sources"], *chunk_records])
            failed["generation"]["failed_chunk"] = index
            failed["generation"]["chunk_reason_code"] = result["generation"].get("reason_code", "UNKNOWN")
            failed["model_explanation"]["chunk_reason_code"] = result["generation"].get("reason_code", "UNKNOWN")
            return failed
        exp = result["model_explanation"]
        summary = {"source_id": record["source_id"], "filename": chunk.filename,
                   "locator": record["locator"], "summary": exp["summary"],
                   "findings": exp["findings"], "hypotheses": exp["hypotheses"],
                   "information_gaps": exp["information_gaps"]}
        summaries_by_sequence[index] = summary
        checkpoint["completed"] = [{"sequence": seq, "source_id": chunk_records[seq - 1]["source_id"],
                                     "summary": summaries_by_sequence[seq]}
                                    for seq in sorted(summaries_by_sequence)]
        save_checkpoint()
        if progress:
            progress({"stage": "CHUNK_ANALYSIS", "completed": len(summaries_by_sequence), "total": len(chunks)})
    chunk_summaries = [summaries_by_sequence[index] for index in range(1, len(chunks) + 1)]

    # Synthesis sees validated per-chunk analysis summaries, while citations point
    # to exact character spans in the immutable original files.
    event("SYNTHESIS", total=len(chunks))
    synthesis_sources = []
    for item in chunk_summaries:
        source = next(source for source in chunk_records if source["source_id"] == item["source_id"])
        summary_text = _canonical({k: v for k, v in item.items() if k != "source_id"})
        synthesis_sources.append({**source, "text": summary_text})
    if sum(len(source["text"]) for source in synthesis_sources) > MAX_SYNTHESIS_CHARACTERS:
        return _failure(input_bundle, "SYNTHESIS_CONTEXT_TOO_LARGE", "INPUT_TOO_LARGE")
    payload = {
        "task_id": task_id,
        "task": (f"综合以下 {len(chunk_summaries)} 个连续片段的分析，识别共识、冲突、互补证据、可能的时间顺序和未解决问题。"
                 "不要把不同时间或不同事件合并。每条结论必须引用 source_id。用户目标：" + task),
        "input_sha256": input_bundle["input_sha256"],
        "sources": synthesis_sources,
        "system_policy": SYSTEM_POLICY,
    }
    if context_snapshot is not None:
        payload["reviewed_context"] = deepcopy(context_snapshot)
        payload["system_policy"] += " " + CONTEXT_POLICY
    check_cancelled()
    try:
        raw = for_stage(llm_backend, "synthesis").generate_structured(agent_id="external-diagnosis-analysis",
                                              output_model=ExternalDiagnosisAnalysis, payload=payload)
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="python")
        analysis = ExternalDiagnosisAnalysis.model_validate(raw)
        chunk_to_original = {record["source_id"]: chunk.source_ids
                             for record, chunk in zip(chunk_records, chunks)}
        allowed_chunks = set(chunk_to_original)
        citations = [item.citation_ids for item in [*analysis.findings, *analysis.hypotheses, *analysis.review_actions,
                                                       *analysis.common_findings, *analysis.report_disagreements,
                                                       *analysis.complementary_evidence, *analysis.event_alignment]]
        citations.extend(item.opposing_citation_ids for item in analysis.hypotheses)
        if any(not set(ids).issubset(allowed_chunks) for ids in citations):
            return _failure(input_bundle, "CITATION_REFERENCE_INVALID")
        data = analysis.model_dump(mode="python")
        for field in ("findings", "hypotheses", "review_actions", "common_findings",
                      "report_disagreements", "complementary_evidence", "event_alignment"):
            for item in data[field]:
                item["citation_ids"] = list(dict.fromkeys(
                    source_id for chunk_id in item["citation_ids"] for source_id in chunk_to_original[chunk_id]))
        for item in data["hypotheses"]:
            item["opposing_citation_ids"] = list(dict.fromkeys(
                source_id for chunk_id in item["opposing_citation_ids"] for source_id in chunk_to_original[chunk_id]))
        analysis = ExternalDiagnosisAnalysis.model_validate(data)
    except AnalysisCancelled:
        raise
    except Exception:
        return _failure(input_bundle, "SYNTHESIS_FAILED", sources=[*input_bundle["sources"], *chunk_records])

    analysis_backend = for_stage(llm_backend, "synthesis")
    verification_backend = (for_stage(llm_backend, "verification")
                            if callable(getattr(llm_backend, "has_stage", None))
                            and llm_backend.has_stage("verification") else None)
    claim_governance = build_claim_governance(
        task_id=task_id, input_sha256=input_bundle["input_sha256"], analysis=analysis,
        sources=input_bundle["sources"], analysis_backend=analysis_backend,
        verification_backend=verification_backend,
        analysis_model_id=(getattr(llm_backend, "model_id", "") or getattr(analysis_backend, "model_id", "")),
    )
    explanation = {
        "schema_version": ANALYSIS_SCHEMA, "status": "AVAILABLE_REQUIRES_EXPERT_REVIEW",
        "source": "LLM_EXTERNAL_DIAGNOSIS", "input_sha256": input_bundle["input_sha256"],
        "citation_reference_validation": "PASS", "semantic_hallucination_assessment": "NOT_EVALUATED",
        "analysis_verification": "UNVERIFIED_MODEL_ANALYSIS",
        "sources": input_bundle["sources"],
        "automatic_control_authorized": False,
        "claim_governance": claim_governance,
        **analysis.model_dump(mode="json"),
        "processing": {"mode": "HIERARCHICAL_CHUNK_SYNTHESIS", "chunk_count": len(chunks),
                       "chunk_characters": DEFAULT_CHUNK_CHARACTERS, "overlap_characters": DEFAULT_OVERLAP_CHARACTERS,
                       "source_coverage": "COMPLETE_WITH_OVERLAP",
                       "chunk_map": [{"chunk_source_id": record["source_id"], "citation_source_ids": list(chunk.source_ids),
                                      "locator": record["locator"], "text_sha256": chunk.text_hash,
                                      "start_offset": chunk.start_offset, "end_offset": chunk.end_offset}
                                     for record, chunk in zip(chunk_records, chunks)]},
    }
    if context_snapshot is not None:
        explanation["reviewed_context"] = deepcopy(context_snapshot)
    check_cancelled()
    result = {"generation": {"status": "SUCCEEDED", "citation_reference_validation": "PASS"},
              "model_explanation": explanation}
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "external_diagnosis_analysis.json").write_text(_canonical(result) + "\n", encoding="utf-8")
    if progress:
        progress({"stage": "SYNTHESIS", "completed": 1, "total": 1})
    return result
