"""Source-reconstructed human gold for the document chunking comparison.

The earlier benchmark deliberately used exact generator-description spans as a
machine proxy.  This module exposes the frozen top-k outputs of both chunking
systems for named human judgement, then recomputes query-level retrieval
metrics without treating that proxy as relevance gold.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from astrograph.filesystem import publish_staged_directory

from .chunking_benchmark import (
    _first_description,
    _fixed_character_chunks,
)
from .document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)
from .evaluation import RetrievalBenchmarkReport
from .structured_document import SectionFirstDocumentIngestor


FIELDS = [
    "review_item_id",
    "query_id",
    "query_text",
    "blinded_arm_id",
    "rank",
    "candidate_display_id",
    "chunk_sha256",
    "source_sha256",
    "section_path_json",
    "source_locators_json",
    "chunk_text",
    "relevance_grade",
    "citation_locator_valid",
    "reviewer",
    "review_comment",
]
HUMAN_FIELDS = {
    "relevance_grade",
    "citation_locator_valid",
    "reviewer",
    "review_comment",
}
SYSTEMS = ("section_aware", "fixed_character")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != FIELDS:
            raise ValueError("chunking gold review columns changed")
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if (
        protocol.get("schema_version")
        != "astrograph-document-chunking-human-gold-protocol.v1"
        or protocol.get("status") != "FROZEN_BEFORE_HUMAN_LABELING"
    ):
        raise ValueError("unsupported or unfrozen chunking human-gold protocol")
    if protocol.get("judgement", {}).get("relevance_grades") != {
        "0": "IRRELEVANT",
        "1": "PARTIAL_SUPPORT",
        "2": "DIRECT_SUPPORT",
    }:
        raise ValueError("chunking human-gold relevance scale changed")
    promotion = protocol.get("promotion", {})
    required = {
        "minimum_section_recall_at_k",
        "minimum_section_ndcg_at_k",
        "minimum_section_mrr",
        "minimum_citation_locator_validity",
        "minimum_ndcg_delta",
        "bootstrap_confidence",
        "bootstrap_iterations",
        "bootstrap_seed",
    }
    if set(promotion) != required:
        raise ValueError("chunking human-gold promotion contract is incomplete")
    for key in (
        "minimum_section_recall_at_k",
        "minimum_section_ndcg_at_k",
        "minimum_section_mrr",
        "minimum_citation_locator_validity",
        "bootstrap_confidence",
    ):
        if not 0.0 <= float(promotion[key]) <= 1.0:
            raise ValueError(f"invalid human-gold threshold: {key}")
    if float(promotion["minimum_ndcg_delta"]) <= 0.0:
        raise ValueError("minimum_ndcg_delta must be positive")
    if int(promotion["bootstrap_iterations"]) < 1_000:
        raise ValueError("human-gold bootstrap_iterations must be at least 1000")


def _load_inputs(
    *,
    preregistration_path: Path,
    machine_results_path: Path,
    handbook_path: Path,
    canonical_facts_path: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], str]:
    prereg = json.loads(preregistration_path.read_text(encoding="utf-8"))
    results = json.loads(machine_results_path.read_text(encoding="utf-8"))
    facts = json.loads(canonical_facts_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    source_sha = _sha(handbook_path)
    if prereg.get("source_artifacts") != {
        "handbook_markdown_sha256": source_sha,
        "canonical_facts_sha256": _sha(canonical_facts_path),
    }:
        raise ValueError("chunking human-gold sources differ from machine preregistration")
    if (
        results.get("schema_version") != "astrograph-document-chunking-machine-results.v1"
        or results.get("preregistration_sha256") != _sha(preregistration_path)
        or protocol.get("machine_preregistration_sha256") != _sha(preregistration_path)
        or protocol.get("machine_results_sha256") != _sha(machine_results_path)
        or protocol.get("expected_query_count") != len(facts)
    ):
        raise ValueError("chunking human-gold protocol is stale or machine results differ")
    for system in SYSTEMS:
        RetrievalBenchmarkReport.model_validate(results[system])
    return prereg, results, facts, protocol, source_sha


def _machine_rows(
    *,
    preregistration_path: Path,
    machine_results_path: Path,
    handbook_path: Path,
    canonical_facts_path: Path,
    protocol_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, dict[str, str]]]:
    prereg, results, facts, protocol, source_sha = _load_inputs(
        preregistration_path=preregistration_path,
        machine_results_path=machine_results_path,
        handbook_path=handbook_path,
        canonical_facts_path=canonical_facts_path,
        protocol_path=protocol_path,
    )
    text = handbook_path.read_text(encoding="utf-8")
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
        asset_root=handbook_path.parent,
        document_descriptor=descriptor,
        source_sha256=source_sha,
    ).chunks
    fixed_chunks = _fixed_character_chunks(
        text,
        source_sha256=source_sha,
        descriptor=descriptor,
        chunk_characters=prereg["systems"]["fixed_character"]["chunk_characters"],
    )
    chunks = {
        "section_aware": {item.chunk_id: item for item in section_chunks},
        "fixed_character": {item.chunk_id: item for item in fixed_chunks},
    }
    facts_by_id = {str(item["fault_class_id"]): item for item in facts}
    if len(facts_by_id) != len(facts):
        raise ValueError("chunking human-gold query identities are not unique")
    rows: list[dict[str, str]] = []
    protected_mapping: dict[str, dict[str, str]] = {}
    protocol_id = str(protocol.get("protocol_id", "chunking-human-gold"))
    arm_ids = {
        system: "ARM-" + _canonical_sha({"protocol_id": protocol_id, "system": system})[:12].upper()
        for system in SYSTEMS
    }
    if len(set(arm_ids.values())) != len(SYSTEMS):
        raise ValueError("blinded system arm identities collided")
    seen_query_systems: set[tuple[str, str]] = set()
    for system in SYSTEMS:
        report = results[system]
        if report.get("case_count") != len(facts):
            raise ValueError("machine report query count differs from frozen facts")
        for case in report["case_results"]:
            prefix = "section-aware:" if system == "section_aware" else "fixed-character:"
            case_id = str(case["case_id"])
            if not case_id.startswith(prefix):
                raise ValueError("machine result case/system identity mismatch")
            query_id = case_id.removeprefix(prefix)
            fact = facts_by_id.get(query_id)
            query_text = _first_description(fact or {})
            if fact is None or not query_text:
                raise ValueError(f"machine result has no frozen query source: {case_id}")
            key = (query_id, system)
            if key in seen_query_systems:
                raise ValueError("duplicate query/system result")
            seen_query_systems.add(key)
            retrieved = list(case["retrieved_chunk_ids"])
            if len(retrieved) > int(prereg["retrieval"]["top_k"]):
                raise ValueError("machine result exceeds preregistered top_k")
            for rank, chunk_id in enumerate(retrieved, start=1):
                chunk = chunks[system].get(str(chunk_id))
                if chunk is None:
                    raise ValueError(f"machine result references unknown chunk: {chunk_id}")
                review_id = "RET-GOLD-" + _canonical_sha(
                    {"query_id": query_id, "system_id": system, "chunk_id": chunk.chunk_id}
                )[:20].upper()
                candidate_display_id = "CAND-" + _canonical_sha(
                    {"review_item_id": review_id, "chunk_sha256": chunk.chunk_sha256}
                )[:16].upper()
                rows.append(
                    {
                        "review_item_id": review_id,
                        "query_id": query_id,
                        "query_text": query_text,
                        "blinded_arm_id": arm_ids[system],
                        "rank": str(rank),
                        "candidate_display_id": candidate_display_id,
                        "chunk_sha256": chunk.chunk_sha256,
                        "source_sha256": source_sha,
                        "section_path_json": json.dumps(
                            chunk.section_path, ensure_ascii=False, separators=(",", ":")
                        ),
                        "source_locators_json": json.dumps(
                            chunk.source_locators, ensure_ascii=False, separators=(",", ":")
                        ),
                        "chunk_text": chunk.text,
                        "relevance_grade": "",
                        "citation_locator_valid": "",
                        "reviewer": "",
                        "review_comment": "",
                    }
                )
                protected_mapping[review_id] = {
                    "system_id": system,
                    "chunk_id": chunk.chunk_id,
                }
    expected_pairs = {(query_id, system) for query_id in facts_by_id for system in SYSTEMS}
    if seen_query_systems != expected_pairs:
        raise ValueError("machine results do not cover the frozen query/system product")
    if len({row["review_item_id"] for row in rows}) != len(rows):
        raise ValueError("chunking gold review item identities collided")
    return (
        sorted(
            rows,
            key=lambda row: (row["query_id"], row["blinded_arm_id"], int(row["rank"])),
        ),
        protocol,
        protected_mapping,
    )


def build_chunking_gold_review_workbook(
    *,
    preregistration_path: str | Path,
    machine_results_path: str | Path,
    handbook_path: str | Path,
    canonical_facts_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a locked baseline plus an editable named-human review copy."""

    inputs = [Path(value).resolve() for value in (
        preregistration_path,
        machine_results_path,
        handbook_path,
        canonical_facts_path,
        protocol_path,
    )]
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite chunking gold review workbook")
    rows, protocol, _ = _machine_rows(
        preregistration_path=inputs[0],
        machine_results_path=inputs[1],
        handbook_path=inputs[2],
        canonical_facts_path=inputs[3],
        protocol_path=inputs[4],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        baseline = staging / "retrieval_gold_review_baseline.csv"
        working = staging / "retrieval_gold_review_working_copy.csv"
        _write_csv(baseline, rows)
        shutil.copyfile(baseline, working)
        body = {
            "schema_version": "astrograph-document-chunking-human-gold-workbook.v1",
            "status": "READY_FOR_NAMED_HUMAN_RELEVANCE_REVIEW",
            "input_file_sha256": {
                name: _sha(path)
                for name, path in zip(
                    (
                        "machine_preregistration",
                        "machine_results",
                        "handbook_markdown",
                        "canonical_facts",
                        "human_gold_protocol",
                    ),
                    inputs,
                    strict=True,
                )
            },
            "query_count": int(protocol["expected_query_count"]),
            "system_count": len(SYSTEMS),
            "review_item_count": len(rows),
            "baseline_file_sha256": _sha(baseline),
            "working_copy_initial_file_sha256": _sha(working),
            "machine_proxy_is_human_gold": False,
            "system_identity_blinded_in_working_copy": True,
            "human_decisions_written": False,
            "human_gold_complete": False,
            "promotion_decision": "NONE_HUMAN_REVIEW_PENDING",
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "The exact-span proxy remains only in the separately frozen machine benchmark "
                "and is withheld from reviewers. Human relevance, citation validity, metrics "
                "and any chunking promotion are established only by the source-reconstructed "
                "named-review freeze."
            ),
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


def _system_metrics(rows: list[dict[str, str]], system: str) -> tuple[dict[str, float], list[float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["system_id"] == system:
            grouped.setdefault(row["query_id"], []).append(row)
    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_checks: list[float] = []
    for query_rows in grouped.values():
        ranked = sorted(query_rows, key=lambda row: int(row["rank"]))
        grades = [int(row["relevance_grade"]) for row in ranked]
        recalls.append(float(any(grade == 2 for grade in grades)))
        precisions.append(sum(grade > 0 for grade in grades) / len(grades))
        dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
        ideal = sorted(grades, reverse=True)
        idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        reciprocal_ranks.append(
            next((1.0 / rank for rank, grade in enumerate(grades, 1) if grade == 2), 0.0)
        )
        citation_checks.extend(
            1.0 if row["citation_locator_valid"].strip().upper() == "PASS" else 0.0
            for row in ranked
        )
    if not grouped:
        raise ValueError(f"human gold has no rows for system: {system}")
    metrics = {
        "mean_recall_at_k": sum(recalls) / len(recalls),
        "mean_precision_at_k": sum(precisions) / len(precisions),
        "mean_ndcg_at_k": sum(ndcgs) / len(ndcgs),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "citation_locator_validity": sum(citation_checks) / len(citation_checks),
    }
    return metrics, ndcgs


def _bootstrap_lower_bound(
    deltas: list[float], *, confidence: float, iterations: int, seed: int
) -> float:
    if not deltas:
        raise ValueError("paired bootstrap requires query deltas")
    generator = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(sum(generator.choice(deltas) for _ in deltas) / len(deltas))
    samples.sort()
    lower_tail = (1.0 - confidence) / 2.0
    index = min(len(samples) - 1, max(0, math.floor(lower_tail * len(samples))))
    return samples[index]


def freeze_chunking_gold_review(
    *,
    preregistration_path: str | Path,
    machine_results_path: str | Path,
    handbook_path: str | Path,
    canonical_facts_path: str | Path,
    protocol_path: str | Path,
    workbook_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze named judgements and recompute human retrieval metrics fail-closed."""

    inputs = [Path(value).resolve() for value in (
        preregistration_path,
        machine_results_path,
        handbook_path,
        canonical_facts_path,
        protocol_path,
    )]
    root = Path(workbook_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite chunking human-gold freeze")
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _canonical_sha(receipt_body) != receipt.get("receipt_sha256"):
        raise ValueError("chunking gold workbook semantic hash mismatch")
    expected_input_hashes = {
        name: _sha(path)
        for name, path in zip(
            (
                "machine_preregistration",
                "machine_results",
                "handbook_markdown",
                "canonical_facts",
                "human_gold_protocol",
            ),
            inputs,
            strict=True,
        )
    }
    if receipt.get("input_file_sha256") != expected_input_hashes:
        raise ValueError("chunking gold workbook inputs changed")
    expected, protocol, protected_mapping = _machine_rows(
        preregistration_path=inputs[0],
        machine_results_path=inputs[1],
        handbook_path=inputs[2],
        canonical_facts_path=inputs[3],
        protocol_path=inputs[4],
    )
    baseline_path = root / "retrieval_gold_review_baseline.csv"
    working_path = root / "retrieval_gold_review_working_copy.csv"
    if receipt.get("baseline_file_sha256") != _sha(baseline_path):
        raise ValueError("chunking gold baseline changed")
    baseline = _read_csv(baseline_path)
    edited = _read_csv(working_path)
    if baseline != expected:
        raise ValueError("chunking gold baseline differs from source reconstruction")
    by_id = {row["review_item_id"]: row for row in baseline}
    if len(by_id) != len(baseline) or len(edited) != len(by_id) or {
        row["review_item_id"] for row in edited
    } != set(by_id):
        raise ValueError("chunking gold review identity set changed")
    for row in edited:
        original = by_id[row["review_item_id"]]
        for field in FIELDS:
            if field not in HUMAN_FIELDS and row[field] != original[field]:
                raise ValueError(
                    f"chunking gold machine field changed: {row['review_item_id']}:{field}"
                )

    blockers: list[str] = []
    reviewers: set[str] = set()
    for row in edited:
        item_id = row["review_item_id"]
        if row["relevance_grade"].strip() not in {"0", "1", "2"}:
            blockers.append(f"{item_id}:RELEVANCE_GRADE_REQUIRED")
        if row["citation_locator_valid"].strip().upper() not in {"PASS", "FAIL"}:
            blockers.append(f"{item_id}:CITATION_LOCATOR_DECISION_REQUIRED")
        reviewer = row["reviewer"].strip()
        if not reviewer or not row["review_comment"].strip():
            blockers.append(f"{item_id}:REVIEWER_COMMENT_REQUIRED")
        if reviewer:
            reviewers.add(reviewer)

    analysis_rows = [
        {**row, **protected_mapping[row["review_item_id"]]}
        for row in edited
    ]
    metrics: dict[str, dict[str, float]] | None = None
    comparison: dict[str, float] | None = None
    promotion_decision = "NONE_HUMAN_REVIEW_INCOMPLETE"
    status = "BLOCKED"
    if not blockers:
        system_metrics: dict[str, dict[str, float]] = {}
        for system in SYSTEMS:
            system_metrics[system], _ = _system_metrics(analysis_rows, system)
        query_order = sorted({row["query_id"] for row in analysis_rows})
        per_query_ndcg: dict[str, dict[str, float]] = {system: {} for system in SYSTEMS}
        for system in SYSTEMS:
            grouped: dict[str, list[dict[str, str]]] = {}
            for row in analysis_rows:
                if row["system_id"] == system:
                    grouped.setdefault(row["query_id"], []).append(row)
            for query_id, query_rows in grouped.items():
                grades = [
                    int(row["relevance_grade"])
                    for row in sorted(query_rows, key=lambda row: int(row["rank"]))
                ]
                dcg = sum(
                    (2**grade - 1) / math.log2(rank + 1)
                    for rank, grade in enumerate(grades, 1)
                )
                ideal = sorted(grades, reverse=True)
                idcg = sum(
                    (2**grade - 1) / math.log2(rank + 1)
                    for rank, grade in enumerate(ideal, 1)
                )
                per_query_ndcg[system][query_id] = dcg / idcg if idcg else 0.0
        if any(set(per_query_ndcg[system]) != set(query_order) for system in SYSTEMS):
            raise ValueError("human gold systems do not cover the same queries")
        deltas = [
            per_query_ndcg["section_aware"][query_id]
            - per_query_ndcg["fixed_character"][query_id]
            for query_id in query_order
        ]
        promotion = protocol["promotion"]
        ndcg_delta = sum(deltas) / len(deltas)
        lower = _bootstrap_lower_bound(
            deltas,
            confidence=float(promotion["bootstrap_confidence"]),
            iterations=int(promotion["bootstrap_iterations"]),
            seed=int(promotion["bootstrap_seed"]),
        )
        comparison = {
            "mean_ndcg_delta": ndcg_delta,
            "paired_bootstrap_lower_bound": lower,
            "paired_bootstrap_confidence": float(promotion["bootstrap_confidence"]),
        }
        section = system_metrics["section_aware"]
        passed = (
            section["mean_recall_at_k"]
            >= float(promotion["minimum_section_recall_at_k"])
            and section["mean_ndcg_at_k"]
            >= float(promotion["minimum_section_ndcg_at_k"])
            and section["mean_reciprocal_rank"]
            >= float(promotion["minimum_section_mrr"])
            and section["citation_locator_validity"]
            >= float(promotion["minimum_citation_locator_validity"])
            and ndcg_delta >= float(promotion["minimum_ndcg_delta"])
            and lower > 0.0
        )
        metrics = system_metrics
        status = "COMPLETE"
        promotion_decision = (
            "PROMOTE_SECTION_AWARE_HUMAN_GOLD"
            if passed
            else "NO_PROMOTION_HUMAN_GOLD_THRESHOLDS_NOT_MET"
        )

    frozen_judgements = [] if blockers else [
        {
            "review_item_id": row["review_item_id"],
            "query_id": row["query_id"],
            "system_id": row["system_id"],
            "rank": int(row["rank"]),
            "chunk_id": row["chunk_id"],
            "chunk_sha256": row["chunk_sha256"],
            "relevance_grade": int(row["relevance_grade"]),
            "citation_locator_valid": row["citation_locator_valid"].strip().upper(),
            "reviewer": row["reviewer"].strip(),
            "review_comment": row["review_comment"].strip(),
        }
        for row in analysis_rows
    ]
    body = {
        "schema_version": "astrograph-document-chunking-human-gold-freeze.v1",
        "status": status,
        "workbook_receipt_file_sha256": _sha(receipt_path),
        "working_copy_file_sha256": _sha(working_path),
        "protocol_file_sha256": _sha(inputs[4]),
        "query_count": int(protocol["expected_query_count"]),
        "review_item_count": len(edited),
        "reviewers": sorted(reviewers),
        "blockers": blockers,
        "human_gold_complete": status == "COMPLETE",
        "metrics": metrics,
        "comparison": comparison,
        "promotion_decision": promotion_decision,
        "frozen_judgements": frozen_judgements,
        "machine_proxy_used_as_gold": False,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "COMPLETE freezes named relevance and citation judgements for this synthetic handbook "
            "comparison. Promotion requires all preregistered absolute and paired-superiority gates; "
            "it does not establish real-document, diagnosis, GraphRAG or Active-KG gain."
        ),
    }
    result = {**body, "freeze_sha256": _canonical_sha(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = [
    "build_chunking_gold_review_workbook",
    "freeze_chunking_gold_review",
]
