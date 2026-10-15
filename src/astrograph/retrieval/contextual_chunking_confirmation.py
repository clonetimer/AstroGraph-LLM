"""Prospective confirmation of section-path-aware lexical retrieval.

The source, queries, systems and promotion thresholds are frozen before the
first retrieval execution.  Results are exposed through blinded workbooks and
only named human judgements can establish relevance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from astrograph.filesystem import publish_staged_directory
from .chunking_gold_review import _bootstrap_lower_bound, _system_metrics
from .document_governance import DocumentAccessContext, DocumentAccessMode
from .models import (
    DiagnosticRetrievalQuery,
    GovernedTextChunk,
    PreReasoningRetrievalConfig,
    PreReasoningRetrievalRequest,
)
from .service import PreReasoningRetrievalService


SYSTEMS = ("section_context_aware", "fixed_character")
FIELDS = [
    "review_item_id", "query_id", "query_text", "blinded_arm_id", "rank",
    "candidate_display_id", "chunk_sha256", "source_sha256", "section_path_json",
    "source_locators_json", "chunk_text", "relevance_grade",
    "citation_locator_valid", "reviewer", "review_comment",
]
HUMAN_FIELDS = {
    "relevance_grade", "citation_locator_valid", "reviewer", "review_comment"
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_frozen_inputs(preregistration_path: Path, ingestion_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = json.loads(preregistration_path.read_text(encoding="utf-8"))
    ingestion = json.loads(ingestion_path.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version")
        != "astrograph-contextual-chunking-confirmation-preregistration.v1"
        or prereg.get("status") != "FROZEN_BEFORE_EXECUTION"
    ):
        raise ValueError("contextual chunking confirmation is not prospectively frozen")
    source = prereg.get("source", {})
    descriptor = ingestion.get("document_descriptor", {})
    if (
        source.get("retrieval_ingestion_file_sha256") != _sha(ingestion_path)
        or source.get("ingestion_sha256") != ingestion.get("ingestion_sha256")
        or source.get("source_document_sha256") != ingestion.get("source_sha256")
        or source.get("source_document_sha256") != descriptor.get("source_sha256")
        or source.get("source_is_independent_of_r004_handbook") is not True
    ):
        raise ValueError("contextual chunking confirmation source identity changed")
    queries = prereg.get("query_construction", {}).get("queries", [])
    if (
        prereg.get("query_construction", {}).get("status")
        != "FROZEN_WITH_SOURCE_BEFORE_RETRIEVAL_EXECUTION"
        or prereg.get("query_construction", {}).get("fault_label_or_condition_id_used")
        is not False
        or prereg.get("query_construction", {}).get("query_count") != len(queries)
        or len({item.get("query_id") for item in queries}) != len(queries)
        or not queries
    ):
        raise ValueError("contextual chunking confirmation queries are invalid")
    if set(prereg.get("systems", {})) != set(SYSTEMS):
        raise ValueError("contextual chunking confirmation system set changed")
    promotion = prereg.get("promotion", {})
    required = {
        "minimum_section_recall_at_k", "minimum_section_ndcg_at_k",
        "minimum_section_mrr", "minimum_citation_locator_validity",
        "minimum_ndcg_delta", "bootstrap_confidence", "bootstrap_iterations",
        "bootstrap_seed",
    }
    if set(promotion) != required or float(promotion["minimum_ndcg_delta"]) <= 0:
        raise ValueError("contextual chunking confirmation promotion contract is invalid")
    return prereg, ingestion


def _eligible_section_chunks(ingestion: dict[str, Any]) -> list[GovernedTextChunk]:
    chunks = [GovernedTextChunk.model_validate(item) for item in ingestion.get("chunks", [])]
    return [
        item for item in chunks
        if item.lifecycle_status == "ACTIVE"
        and item.review_status == "HUMAN_REVIEWED"
        and item.content_safety_status == "PASS"
    ]


def _fixed_chunks(
    ingestion: dict[str, Any], section_chunks: list[GovernedTextChunk], *, size: int
) -> list[GovernedTextChunk]:
    if size < 100:
        raise ValueError("fixed-character confirmation chunks must be at least 100 characters")
    allowed_locators = {
        locator for chunk in section_chunks for locator in chunk.source_locators
    }
    pieces: list[tuple[str, str]] = []
    prior_path: list[str] = []
    for element in ingestion.get("elements", []):
        locator = str(element.get("source_locator", ""))
        text = str(element.get("text_representation", "")).strip()
        path = [str(value) for value in element.get("section_path", [])]
        if not locator or locator not in allowed_locators or not text:
            continue
        common = 0
        while common < min(len(prior_path), len(path)) and prior_path[common] == path[common]:
            common += 1
        for depth, heading in enumerate(path[common:], start=common + 1):
            pieces.append(("#" * min(depth, 6) + " " + heading + "\n\n", locator))
        pieces.append((text + "\n\n", locator))
        prior_path = path
    linear = ""
    spans: list[tuple[int, int, str]] = []
    for text, locator in pieces:
        start = len(linear)
        linear += text
        spans.append((start, len(linear), locator))
    descriptor = section_chunks[0].document_descriptor if section_chunks else None
    chunks: list[GovernedTextChunk] = []
    source_sha = str(ingestion["source_sha256"])
    for index, start in enumerate(range(0, len(linear), size), 1):
        end = min(len(linear), start + size)
        text = linear[start:end]
        if not text.strip():
            continue
        locators = list(dict.fromkeys(
            locator for span_start, span_end, locator in spans
            if start < span_end and span_start < end
        ))
        chunks.append(
            GovernedTextChunk(
                chunk_id=f"early-design-fixed-char:{index:04d}",
                source_id="early-design-fixed-character-confirmation",
                source_uri=str(ingestion["source_uri"]),
                source_sha256=source_sha,
                chunk_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                knowledge_version="contextual-chunking-confirmation-r001",
                text=text,
                section_path=[],
                source_locators=locators,
                document_descriptor=descriptor,
            )
        )
    return chunks


def run_contextual_chunking_confirmation(
    *, preregistration_path: str | Path, ingestion_path: str | Path
) -> dict[str, Any]:
    preregistration_path = Path(preregistration_path).resolve()
    ingestion_path = Path(ingestion_path).resolve()
    prereg, ingestion = _load_frozen_inputs(preregistration_path, ingestion_path)
    section_chunks = _eligible_section_chunks(ingestion)
    if not section_chunks:
        raise ValueError("contextual chunking confirmation has no eligible section chunks")
    fixed_size = int(prereg["systems"]["fixed_character"]["chunk_characters"])
    fixed_chunks = _fixed_chunks(ingestion, section_chunks, size=fixed_size)
    chunks_by_system = {
        "section_context_aware": section_chunks,
        "fixed_character": fixed_chunks,
    }
    retrieval = prereg["retrieval"]
    system_results: dict[str, list[dict[str, Any]]] = {}
    for system in SYSTEMS:
        config = PreReasoningRetrievalConfig(
            bm25_weight=retrieval["bm25_weight"],
            term_coverage_weight=retrieval["term_coverage_weight"],
            dense_weight=retrieval["dense_weight"],
            minimum_local_score=retrieval["minimum_local_score"],
            require_local_and_ontology=retrieval["require_local_and_ontology"],
            include_section_path_in_lexical_index=(system == "section_context_aware"),
        )
        rows: list[dict[str, Any]] = []
        for query in prereg["query_construction"]["queries"]:
            package = PreReasoningRetrievalService().retrieve(
                PreReasoningRetrievalRequest(
                    query=DiagnosticRetrievalQuery(
                        query_id=str(query["query_id"]),
                        episode_id="early-design-context-confirmation",
                        text=str(query["text"]),
                        top_k=int(retrieval["top_k"]),
                        document_access_context=DocumentAccessContext(
                            mode=DocumentAccessMode.CORPUS_BUILD
                        ),
                    ),
                    chunks=chunks_by_system[system],
                    config=config,
                )
            )
            lookup = {item.chunk_id: item for item in chunks_by_system[system]}
            rows.append(
                {
                    "query_id": str(query["query_id"]),
                    "query_text": str(query["text"]),
                    "status": package.status.value,
                    "hits": [
                        {
                            "rank": hit.rank,
                            "chunk_id": hit.chunk_id,
                            "chunk_sha256": hit.chunk_sha256,
                            "source_sha256": hit.source_sha256,
                            "section_path": hit.section_path,
                            "source_locators": hit.source_locators,
                            "chunk_text": lookup[hit.chunk_id].text,
                        }
                        for hit in package.local_hits
                    ],
                }
            )
        system_results[system] = rows
    body = {
        "schema_version": "astrograph-contextual-chunking-confirmation-results.v1",
        "benchmark_id": prereg["benchmark_id"],
        "status": "MACHINE_RESULTS_FROZEN_HUMAN_GOLD_REQUIRED",
        "preregistration_file_sha256": _sha(preregistration_path),
        "ingestion_file_sha256": _sha(ingestion_path),
        "source_document_sha256": ingestion["source_sha256"],
        "query_count": len(prereg["query_construction"]["queries"]),
        "system_chunk_counts": {
            name: len(chunks_by_system[name]) for name in SYSTEMS
        },
        "systems": system_results,
        "human_relevance_used": False,
        "promotion_decision": "NONE_HUMAN_GOLD_REQUIRED",
        "claim_boundary": prereg["claim_boundary"],
    }
    return {**body, "results_sha256": _canonical_sha(body)}


def _machine_rows(
    preregistration_path: Path, ingestion_path: Path, results_path: Path
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, dict[str, str]]]:
    prereg, _ = _load_frozen_inputs(preregistration_path, ingestion_path)
    supplied = json.loads(results_path.read_text(encoding="utf-8"))
    recomputed = run_contextual_chunking_confirmation(
        preregistration_path=preregistration_path, ingestion_path=ingestion_path
    )
    if supplied != recomputed:
        raise ValueError("contextual chunking machine results differ from source reconstruction")
    arm_ids = {
        system: "ARM-" + _canonical_sha(
            {"benchmark_id": prereg["benchmark_id"], "system": system}
        )[:12].upper()
        for system in SYSTEMS
    }
    rows: list[dict[str, str]] = []
    mapping: dict[str, dict[str, str]] = {}
    for system in SYSTEMS:
        for case in supplied["systems"][system]:
            for hit in case["hits"]:
                review_id = "CTX-GOLD-" + _canonical_sha(
                    {
                        "benchmark_id": prereg["benchmark_id"],
                        "query_id": case["query_id"],
                        "system": system,
                        "chunk_sha256": hit["chunk_sha256"],
                    }
                )[:20].upper()
                row = {
                    "review_item_id": review_id,
                    "query_id": case["query_id"],
                    "query_text": case["query_text"],
                    "blinded_arm_id": arm_ids[system],
                    "rank": str(hit["rank"]),
                    "candidate_display_id": "CAND-" + _canonical_sha(
                        {"review_id": review_id, "chunk": hit["chunk_sha256"]}
                    )[:16].upper(),
                    "chunk_sha256": hit["chunk_sha256"],
                    "source_sha256": hit["source_sha256"],
                    "section_path_json": json.dumps(
                        hit["section_path"], ensure_ascii=False, separators=(",", ":")
                    ),
                    "source_locators_json": json.dumps(
                        hit["source_locators"], ensure_ascii=False, separators=(",", ":")
                    ),
                    "chunk_text": hit["chunk_text"],
                    "relevance_grade": "",
                    "citation_locator_valid": "",
                    "reviewer": "",
                    "review_comment": "",
                }
                if review_id in mapping:
                    raise ValueError("contextual chunking review identity collision")
                rows.append(row)
                mapping[review_id] = {"system_id": system, "chunk_id": hit["chunk_id"]}
    return sorted(rows, key=lambda row: (row["query_id"], row["blinded_arm_id"], int(row["rank"]))), prereg, mapping


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != FIELDS:
            raise ValueError("contextual chunking gold columns changed")
        return list(reader)


def build_contextual_chunking_gold_workbook(
    *, preregistration_path: str | Path, ingestion_path: str | Path,
    results_path: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    preregistration_path, ingestion_path, results_path = (
        Path(preregistration_path).resolve(), Path(ingestion_path).resolve(),
        Path(results_path).resolve(),
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite contextual chunking gold workbook")
    rows, prereg, _ = _machine_rows(preregistration_path, ingestion_path, results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.txn-", dir=output.parent))
    try:
        baseline = staging / "retrieval_gold_review_baseline.csv"
        working = staging / "retrieval_gold_review_working_copy.csv"
        _write_csv(baseline, rows)
        shutil.copyfile(baseline, working)
        body = {
            "schema_version": "astrograph-contextual-chunking-gold-workbook.v1",
            "status": "READY_FOR_BLINDED_NAMED_HUMAN_REVIEW",
            "benchmark_id": prereg["benchmark_id"],
            "input_file_sha256": {
                "preregistration": _sha(preregistration_path),
                "ingestion": _sha(ingestion_path),
                "machine_results": _sha(results_path),
            },
            "query_count": prereg["query_construction"]["query_count"],
            "system_count": len(SYSTEMS),
            "review_item_count": len(rows),
            "baseline_file_sha256": _sha(baseline),
            "working_copy_initial_file_sha256": _sha(working),
            "system_identity_blinded": True,
            "human_decisions_written": False,
            "promotion_decision": "NONE_HUMAN_REVIEW_PENDING",
            "active_graph_modified": False,
            "claim_boundary": prereg["claim_boundary"],
        }
        result = {**body, "receipt_sha256": _canonical_sha(body)}
        (staging / "receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging, output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _per_query_ndcg(rows: list[dict[str, str]], system: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["system_id"] == system:
            grouped.setdefault(row["query_id"], []).append(row)
    result: dict[str, float] = {}
    for query_id, query_rows in grouped.items():
        grades = [int(row["relevance_grade"]) for row in sorted(query_rows, key=lambda x: int(x["rank"]))]
        dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
        ideal = sorted(grades, reverse=True)
        idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
        result[query_id] = dcg / idcg if idcg else 0.0
    return result


def freeze_contextual_chunking_gold(
    *, preregistration_path: str | Path, ingestion_path: str | Path,
    results_path: str | Path, workbook_root: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    preregistration_path, ingestion_path, results_path = (
        Path(preregistration_path).resolve(), Path(ingestion_path).resolve(),
        Path(results_path).resolve(),
    )
    root, output = Path(workbook_root).resolve(), Path(output_path).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite contextual chunking gold freeze")
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if _canonical_sha({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != receipt.get("receipt_sha256"):
        raise ValueError("contextual chunking workbook semantic hash mismatch")
    expected, prereg, mapping = _machine_rows(preregistration_path, ingestion_path, results_path)
    expected_hashes = {
        "preregistration": _sha(preregistration_path), "ingestion": _sha(ingestion_path),
        "machine_results": _sha(results_path),
    }
    if receipt.get("input_file_sha256") != expected_hashes:
        raise ValueError("contextual chunking workbook inputs changed")
    baseline = _read_csv(root / "retrieval_gold_review_baseline.csv")
    edited = _read_csv(root / "retrieval_gold_review_working_copy.csv")
    if _sha(root / "retrieval_gold_review_baseline.csv") != receipt.get("baseline_file_sha256") or baseline != expected:
        raise ValueError("contextual chunking baseline changed")
    by_id = {row["review_item_id"]: row for row in baseline}
    if len(by_id) != len(baseline) or {row["review_item_id"] for row in edited} != set(by_id):
        raise ValueError("contextual chunking review identity set changed")
    blockers: list[str] = []
    reviewers: set[str] = set()
    for row in edited:
        original = by_id[row["review_item_id"]]
        for field in FIELDS:
            if field not in HUMAN_FIELDS and row[field] != original[field]:
                raise ValueError(f"contextual chunking machine field changed: {row['review_item_id']}:{field}")
        if row["relevance_grade"].strip() not in {"0", "1", "2"}:
            blockers.append(f"{row['review_item_id']}:RELEVANCE_GRADE_REQUIRED")
        if row["citation_locator_valid"].strip().upper() not in {"PASS", "FAIL"}:
            blockers.append(f"{row['review_item_id']}:CITATION_LOCATOR_DECISION_REQUIRED")
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append(f"{row['review_item_id']}:REVIEWER_COMMENT_REQUIRED")
        if row["reviewer"].strip():
            reviewers.add(row["reviewer"].strip())
    analysis_rows = [{**row, **mapping[row["review_item_id"]]} for row in edited]
    metrics = None
    comparison = None
    status = "BLOCKED"
    decision = "NONE_HUMAN_REVIEW_INCOMPLETE"
    if not blockers:
        metrics = {system: _system_metrics(analysis_rows, system)[0] for system in SYSTEMS}
        ndcg = {system: _per_query_ndcg(analysis_rows, system) for system in SYSTEMS}
        query_ids = sorted(ndcg["section_context_aware"])
        if set(query_ids) != set(ndcg["fixed_character"]):
            raise ValueError("contextual chunking systems do not cover the same queries")
        deltas = [
            ndcg["section_context_aware"][query_id] - ndcg["fixed_character"][query_id]
            for query_id in query_ids
        ]
        promotion = prereg["promotion"]
        comparison = {
            "mean_ndcg_delta": sum(deltas) / len(deltas),
            "paired_bootstrap_lower_bound": _bootstrap_lower_bound(
                deltas, confidence=float(promotion["bootstrap_confidence"]),
                iterations=int(promotion["bootstrap_iterations"]),
                seed=int(promotion["bootstrap_seed"]),
            ),
            "paired_bootstrap_confidence": float(promotion["bootstrap_confidence"]),
        }
        section = metrics["section_context_aware"]
        promote = (
            section["mean_recall_at_k"] >= float(promotion["minimum_section_recall_at_k"])
            and section["mean_ndcg_at_k"] >= float(promotion["minimum_section_ndcg_at_k"])
            and section["mean_reciprocal_rank"] >= float(promotion["minimum_section_mrr"])
            and section["citation_locator_validity"] >= float(promotion["minimum_citation_locator_validity"])
            and comparison["mean_ndcg_delta"] >= float(promotion["minimum_ndcg_delta"])
            and comparison["paired_bootstrap_lower_bound"] > 0.0
        )
        status = "COMPLETE"
        decision = (
            "PROMOTE_SECTION_CONTEXT_AWARE_HUMAN_GOLD"
            if promote else "NO_PROMOTION_HUMAN_GOLD_THRESHOLDS_NOT_MET"
        )
    counts = Counter(row["relevance_grade"].strip() for row in edited if row["relevance_grade"].strip())
    body = {
        "schema_version": "astrograph-contextual-chunking-human-gold-freeze.v1",
        "status": status,
        "benchmark_id": prereg["benchmark_id"],
        "workbook_receipt_file_sha256": _sha(root / "receipt.json"),
        "working_copy_file_sha256": _sha(root / "retrieval_gold_review_working_copy.csv"),
        "preregistration_file_sha256": _sha(preregistration_path),
        "machine_results_file_sha256": _sha(results_path),
        "query_count": prereg["query_construction"]["query_count"],
        "review_item_count": len(edited),
        "reviewers": sorted(reviewers),
        "decision_counts": dict(sorted(counts.items())),
        "blockers": blockers,
        "human_gold_complete": not blockers,
        "metrics": metrics,
        "comparison": comparison,
        "promotion_decision": decision,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": prereg["claim_boundary"],
    }
    result = {**body, "freeze_sha256": _canonical_sha(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = [
    "run_contextual_chunking_confirmation",
    "build_contextual_chunking_gold_workbook",
    "freeze_contextual_chunking_gold",
]
