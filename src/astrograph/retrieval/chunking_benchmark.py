"""Pre-registered machine source-span comparison for document chunking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .document_corpus_builder import sha256_file
from .document_governance import (
    DocumentAccessContext,
    DocumentAccessMode,
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)
from .evaluation import RetrievalBenchmarkCase, evaluate_retrieval_cases
from .models import (
    DiagnosticRetrievalQuery,
    GovernedTextChunk,
    PreReasoningRetrievalConfig,
    PreReasoningRetrievalRequest,
    RetrievalStatus,
)
from .structured_document import SectionFirstDocumentIngestor


def _fixed_character_chunks(
    text: str,
    *,
    source_sha256: str,
    descriptor: GovernedDocumentDescriptor,
    chunk_characters: int,
) -> list[GovernedTextChunk]:
    chunks: list[GovernedTextChunk] = []
    for index, start in enumerate(range(0, len(text), chunk_characters), 1):
        end = min(len(text), start + chunk_characters)
        content = text[start:end]
        if not content.strip():
            continue
        chunks.append(
            GovernedTextChunk(
                chunk_id=f"fixed-char:{index:04d}",
                source_id="fault-handbook-fixed-char",
                source_uri="corpus://design/fault_design_fmea_test_handbook.md",
                source_sha256=source_sha256,
                chunk_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                knowledge_version="document-chunking-benchmark-r001",
                text=content,
                section_id=None,
                section_path=[],
                modalities=["TEXT"],
                source_locators=[f"chars:{start}-{end}"],
                document_descriptor=descriptor,
            )
        )
    return chunks


def _first_description(row: dict[str, object]) -> str | None:
    summaries = row.get("generator_condition_summaries", [])
    if not isinstance(summaries, list):
        return None
    for item in summaries:
        if isinstance(item, dict) and str(item.get("description", "")).strip():
            return str(item["description"]).strip()
    return None


def _relevant_by_exact_span(
    chunks: list[GovernedTextChunk], description: str, *, source_text: str
) -> list[str]:
    occurrences: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = source_text.find(description, cursor)
        if start < 0:
            break
        occurrences.append((start, start + len(description)))
        cursor = start + 1
    relevant: list[str] = []
    for chunk in chunks:
        if description in chunk.text:
            relevant.append(chunk.chunk_id)
            continue
        for locator in chunk.source_locators:
            if not locator.startswith("chars:"):
                continue
            start_text, end_text = locator.removeprefix("chars:").split("-", 1)
            chunk_start, chunk_end = int(start_text), int(end_text)
            if any(chunk_start < end and start < chunk_end for start, end in occurrences):
                relevant.append(chunk.chunk_id)
                break
    return relevant


def _requests(
    *,
    facts: list[dict[str, object]],
    chunks: list[GovernedTextChunk],
    retrieval_config: PreReasoningRetrievalConfig,
    top_k: int,
    system_id: str,
    source_text: str,
) -> tuple[list[RetrievalBenchmarkCase], list[str]]:
    cases: list[RetrievalBenchmarkCase] = []
    excluded: list[str] = []
    for row in facts:
        fault_id = str(row["fault_class_id"])
        description = _first_description(row)
        if not description:
            excluded.append(f"{fault_id}:NO_DESCRIPTION")
            continue
        relevant = _relevant_by_exact_span(
            chunks, description, source_text=source_text
        )
        if not relevant:
            excluded.append(f"{fault_id}:SOURCE_SPAN_NOT_IN_CHUNK")
            continue
        query = DiagnosticRetrievalQuery(
            query_id=f"{system_id}:{fault_id}",
            episode_id="machine-chunking-benchmark",
            text=description,
            top_k=top_k,
            document_access_context=DocumentAccessContext(
                mode=DocumentAccessMode.CORPUS_BUILD
            ),
        )
        cases.append(
            RetrievalBenchmarkCase(
                case_id=f"{system_id}:{fault_id}",
                request=PreReasoningRetrievalRequest(
                    query=query,
                    chunks=chunks,
                    config=retrieval_config,
                ),
                relevant_chunk_ids=relevant,
                expected_status=RetrievalStatus.SUFFICIENT,
            )
        )
    return cases, excluded


def run_chunking_machine_benchmark(
    *,
    preregistration_path: Path,
    handbook_markdown_path: Path,
    canonical_facts_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    prereg = json.loads(preregistration_path.read_text(encoding="utf-8"))
    source_sha = sha256_file(handbook_markdown_path)
    facts_sha = sha256_file(canonical_facts_path)
    expected = prereg["source_artifacts"]
    if source_sha != expected["handbook_markdown_sha256"]:
        raise ValueError("handbook hash differs from preregistration")
    if facts_sha != expected["canonical_facts_sha256"]:
        raise ValueError("canonical facts hash differs from preregistration")
    text = handbook_markdown_path.read_text(encoding="utf-8")
    facts = json.loads(canonical_facts_path.read_text(encoding="utf-8"))
    if len(facts) != prereg["case_construction"]["expected_case_count"]:
        raise ValueError("fault case count differs from preregistration")
    descriptor = GovernedDocumentDescriptor(
        document_id="astrograph:synthetic-fault-handbook-markdown:r001",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="corpus://design/fault_design_fmea_test_handbook.md",
        source_sha256=source_sha,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        synthetic_from_simulation=True,
        runtime_vote_allowed=False,
    )
    section_chunks = SectionFirstDocumentIngestor(
        maximum_section_units=prereg["systems"]["section_aware"]["maximum_section_units"]
    ).ingest_markdown(
        text,
        source_id="fault-handbook-section-aware",
        source_uri=descriptor.source_uri,
        knowledge_version="document-chunking-benchmark-r001",
        asset_root=handbook_markdown_path.parent,
        document_descriptor=descriptor,
        source_sha256=source_sha,
    ).chunks
    fixed_chunks = _fixed_character_chunks(
        text,
        source_sha256=source_sha,
        descriptor=descriptor,
        chunk_characters=prereg["systems"]["fixed_character"]["chunk_characters"],
    )
    retrieval = prereg["retrieval"]
    config = PreReasoningRetrievalConfig(
        bm25_weight=retrieval["bm25_weight"],
        term_coverage_weight=retrieval["term_coverage_weight"],
        dense_weight=retrieval["dense_weight"],
        minimum_local_score=retrieval["minimum_local_score"],
        require_local_and_ontology=retrieval["require_local_and_ontology"],
    )
    section_cases, section_excluded = _requests(
        facts=facts,
        chunks=section_chunks,
        retrieval_config=config,
        top_k=retrieval["top_k"],
        system_id="section-aware",
        source_text=text,
    )
    fixed_cases, fixed_excluded = _requests(
        facts=facts,
        chunks=fixed_chunks,
        retrieval_config=config,
        top_k=retrieval["top_k"],
        system_id="fixed-character",
        source_text=text,
    )
    if section_excluded or fixed_excluded:
        raise ValueError(
            f"benchmark case construction incomplete: {section_excluded + fixed_excluded}"
        )
    section_report = evaluate_retrieval_cases(section_cases)
    fixed_report = evaluate_retrieval_cases(fixed_cases)
    raw = {
        "schema_version": "astrograph-document-chunking-machine-results.v1",
        "preregistration_sha256": sha256_file(preregistration_path),
        "section_aware": section_report.model_dump(mode="json"),
        "fixed_character": fixed_report.model_dump(mode="json"),
    }
    metrics = (
        "mean_recall_at_k",
        "mean_precision_at_k",
        "mean_ndcg_at_k",
        "mean_reciprocal_rank",
        "citation_hash_integrity",
    )
    comparison = {
        metric: {
            "section_aware": getattr(section_report, metric),
            "fixed_character": getattr(fixed_report, metric),
            "delta": getattr(section_report, metric) - getattr(fixed_report, metric),
        }
        for metric in metrics
    }
    receipt = {
        "schema_version": "astrograph-document-chunking-machine-receipt.v1",
        "status": "COMPLETED_MACHINE_SOURCE_SPAN_ONLY",
        "preregistration_sha256": sha256_file(preregistration_path),
        "handbook_markdown_sha256": source_sha,
        "canonical_facts_sha256": facts_sha,
        "case_count": len(section_cases),
        "section_chunk_count": len(section_chunks),
        "fixed_character_chunk_count": len(fixed_chunks),
        "comparison": comparison,
        "section_report_sha256": section_report.report_sha256,
        "fixed_report_sha256": fixed_report.report_sha256,
        "promotion_decision": "NONE_MACHINE_BENCHMARK_NOT_HUMAN_GOLD",
        "claim_boundary": prereg["claim_boundary"],
    }
    return raw, receipt


__all__ = ["run_chunking_machine_benchmark"]
