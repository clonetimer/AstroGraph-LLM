"""Traceable machine candidates for early-design versus current-code drift."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .structured_document import SectionFirstDocumentIngestor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_DRIFT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "drift_id": "DRIFT-DOCX-NATIVE-001",
        "source_locator": "word/document.xml#paragraph:21",
        "required_terms": ["Word", "先转换为PDF", "markdown"],
        "topic": "document ingestion",
        "early_design_summary": "Word material is first converted to PDF and then normalized to Markdown.",
        "disposition": "SUPERSEDED",
        "current_summary": "AstroGraph parses DOCX OOXML natively and preserves headings, tables, images and formulas; PDF is a separate representation.",
        "code_paths": ["src/astrograph/retrieval/structured_document.py"],
    },
    {
        "drift_id": "DRIFT-BASE-MODEL-002",
        "source_locator": "word/document.xml#paragraph:15",
        "required_terms": ["Qwen2.5-3B"],
        "topic": "base model selection",
        "early_design_summary": "The design names Qwen2.5-3B-InstructB as the base model.",
        "disposition": "VALIDATION_REQUIRED",
        "current_summary": "The local vLLM backend discovers or receives a model ID; no flight base model is frozen by this source contract.",
        "code_paths": ["src/astrograph/agents/knowledge_curation.py"],
    },
    {
        "drift_id": "DRIFT-SELF-INSTRUCT-003",
        "source_locator": "word/document.xml#paragraph:39",
        "required_terms": ["Self-Instruct", "种子池"],
        "topic": "generated knowledge feedback",
        "early_design_summary": "Reviewed generated samples are added back into a seed pool for iterative expansion.",
        "disposition": "SUPERSEDED_BY_GOVERNANCE",
        "current_summary": "Generated extraction remains a draft candidate and cannot publish or modify Active KG without critic and human gates.",
        "code_paths": [
            "src/astrograph/agents/knowledge_curation.py",
            "src/astrograph/knowledge/construction.py",
        ],
    },
    {
        "drift_id": "DRIFT-FLIGHT-RUNTIME-004",
        "source_locator": "word/document.xml#paragraph:60",
        "required_terms": ["拟采用", "IxRT", "IGIE"],
        "topic": "flight inference runtime",
        "early_design_summary": "IxRT or IGIE is proposed for a specific accelerator.",
        "disposition": "FUTURE_ASSUMPTION_VALIDATION_REQUIRED",
        "current_summary": "Current code supplies an advisory local reasoning backend contract; it is not flight-hardware qualification.",
        "code_paths": [
            "src/astrograph/orchestration/realtime_reasoning.py",
            "src/astrograph/agents/knowledge_curation.py",
        ],
    },
    {
        "drift_id": "DRIFT-STRUCTURED-OUTPUT-005",
        "source_locator": "word/document.xml#paragraph:66",
        "required_terms": ["文本文件形式存储"],
        "topic": "diagnostic output",
        "early_design_summary": "LLM diagnosis description and attribution are stored as text files.",
        "disposition": "SUPERSEDED",
        "current_summary": "AstroGraph persists typed stage artifacts, evidence references, conflicts and decision records in addition to optional narrative output.",
        "code_paths": [
            "src/astrograph/orchestration/online_stage_artifacts.py",
            "src/astrograph/tools/online.py",
        ],
    },
    {
        "drift_id": "DRIFT-RAG-OAG-006",
        "source_locator": "word/document.xml#paragraph:47",
        "required_terms": ["实体提取", "关系识别", "多跳查询"],
        "topic": "graph-aware retrieval",
        "early_design_summary": "Retrieval uses entity/relation parsing, reciprocal ranking and multi-hop graph paths.",
        "disposition": "INHERITED_WITH_STRONGER_GOVERNANCE",
        "current_summary": "Pre-reasoning retrieval keeps ontology paths and source chunks separate, applies access review before scoring, and does not promote retrieval hits to facts.",
        "code_paths": [
            "src/astrograph/retrieval/service.py",
            "src/astrograph/retrieval/document_governance.py",
        ],
    },
)


def build_design_drift_candidates(
    *, source_path: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    repo_root = Path(repo_root).resolve()
    elements, _, audit = SectionFirstDocumentIngestor().parse_docx(source_path)
    by_locator = {item.source_locator: item for item in elements}
    candidates: list[dict[str, Any]] = []
    code_hashes: dict[str, str] = {}
    for spec in _DRIFT_SPECS:
        locator = spec["source_locator"]
        element = by_locator.get(locator)
        if element is None:
            raise ValueError(f"design drift source locator is missing: {locator}")
        missing_terms = [term for term in spec["required_terms"] if term not in element.text_representation]
        if missing_terms:
            raise ValueError(f"design drift anchor changed at {locator}: {missing_terms}")
        evidence: list[dict[str, str]] = []
        for relative in spec["code_paths"]:
            path = (repo_root / relative).resolve()
            if not path.is_relative_to(repo_root) or not path.is_file():
                raise ValueError(f"design drift code evidence is missing: {relative}")
            digest = _sha256(path)
            code_hashes[relative] = digest
            evidence.append({"path": relative, "sha256": digest})
        candidates.append({
            key: value for key, value in spec.items() if key not in {"required_terms", "code_paths"}
        } | {
            "source_element_id": element.element_id,
            "source_section_path": element.section_path,
            "source_span_sha256": hashlib.sha256(element.text_representation.encode()).hexdigest(),
            "required_anchor_terms": spec["required_terms"],
            "current_code_evidence": evidence,
            "review_state": "MACHINE_CANDIDATE_HUMAN_REVIEW_REQUIRED",
            "active_kg_modified": False,
            "runtime_modified": False,
        })
    body = {
        "schema_version": "astrograph-design-drift-candidates.v1",
        "status": "MACHINE_CANDIDATES_HUMAN_REVIEW_REQUIRED",
        "source_document_sha256": audit.source_sha256,
        "current_code_sha256": dict(sorted(code_hashes.items())),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "publication_allowed": False,
        "active_kg_modified": False,
        "runtime_modified": False,
        "claim_boundary": (
            "Traceable drift candidates only. Code presence supports a current implementation "
            "contract but does not prove deployment, performance, flight qualification or human approval."
        ),
    }
    return {**body, "artifact_sha256": _canonical_sha(body)}


__all__ = ["build_design_drift_candidates"]
