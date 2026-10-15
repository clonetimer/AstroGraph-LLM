import csv
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.retrieval.contextual_chunking_confirmation import (
    build_contextual_chunking_gold_workbook,
    freeze_contextual_chunking_gold,
    run_contextual_chunking_confirmation,
)
from astrograph.retrieval.document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)
from astrograph.retrieval.models import GovernedTextChunk


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_sha = "a" * 64
    descriptor = GovernedDocumentDescriptor(
        document_id="fixture:context-doc",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="fixture://context-doc",
        source_sha256=source_sha,
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
    )
    chunks = []
    elements = []
    for index, (heading, body) in enumerate(
        [("Alpha subsystem", "shared procedure details one"),
         ("Beta subsystem", "shared procedure details two")], 1
    ):
        locator = f"fixture#paragraph:{index}"
        chunks.append(
            GovernedTextChunk(
                chunk_id=f"section:{index}", source_id="fixture", source_uri="fixture://context-doc",
                source_sha256=source_sha, chunk_sha256=_sha_bytes(body.encode()),
                knowledge_version="fixture-v1", text=body, section_path=[heading],
                source_locators=[locator], document_descriptor=descriptor,
            ).model_dump(mode="json")
        )
        elements.append({
            "element_id": f"element:{index}", "kind": "TEXT", "section_path": [heading],
            "text_representation": body, "source_locator": locator,
            "representation_reviewed": True,
        })
    ingestion = {
        "source_id": "fixture", "source_uri": "fixture://context-doc",
        "source_sha256": source_sha, "ingestion_sha256": "b" * 64,
        "document_descriptor": descriptor.model_dump(mode="json"),
        "elements": elements, "chunks": chunks,
    }
    ingestion_path = tmp_path / "ingestion.json"
    ingestion_path.write_text(json.dumps(ingestion), encoding="utf-8")
    prereg = {
        "schema_version": "astrograph-contextual-chunking-confirmation-preregistration.v1",
        "benchmark_id": "fixture-context-r001", "status": "FROZEN_BEFORE_EXECUTION",
        "source": {
            "retrieval_ingestion_file_sha256": _sha_bytes(ingestion_path.read_bytes()),
            "ingestion_sha256": "b" * 64, "source_document_sha256": source_sha,
            "source_is_independent_of_r004_handbook": True,
        },
        "query_construction": {
            "status": "FROZEN_WITH_SOURCE_BEFORE_RETRIEVAL_EXECUTION",
            "fault_label_or_condition_id_used": False, "query_count": 2,
            "queries": [
                {"query_id": "q-alpha", "text": "alpha subsystem procedure"},
                {"query_id": "q-beta", "text": "beta subsystem procedure"},
            ],
        },
        "systems": {
            "section_context_aware": {"include_section_path_in_lexical_index": True},
            "fixed_character": {"chunk_characters": 100, "overlap_characters": 0},
        },
        "retrieval": {
            "top_k": 2, "bm25_weight": 0.7, "term_coverage_weight": 0.3,
            "dense_weight": 0.0, "minimum_local_score": 0.0001,
            "require_local_and_ontology": False,
        },
        "promotion": {
            "minimum_section_recall_at_k": 0.9, "minimum_section_ndcg_at_k": 0.85,
            "minimum_section_mrr": 0.9, "minimum_citation_locator_validity": 1.0,
            "minimum_ndcg_delta": 0.05, "bootstrap_confidence": 0.95,
            "bootstrap_iterations": 1000, "bootstrap_seed": 7,
        },
        "claim_boundary": "fixture only",
    }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    return prereg_path, ingestion_path


def test_confirmation_is_deterministic_and_blank_review_is_blocked(tmp_path: Path) -> None:
    prereg, ingestion = _fixture(tmp_path)
    first = run_contextual_chunking_confirmation(
        preregistration_path=prereg, ingestion_path=ingestion
    )
    second = run_contextual_chunking_confirmation(
        preregistration_path=prereg, ingestion_path=ingestion
    )
    assert first == second
    assert first["human_relevance_used"] is False
    results = tmp_path / "results.json"
    results.write_text(json.dumps(first), encoding="utf-8")
    workbook = tmp_path / "workbook"
    receipt = build_contextual_chunking_gold_workbook(
        preregistration_path=prereg, ingestion_path=ingestion,
        results_path=results, output_dir=workbook,
    )
    assert receipt["system_identity_blinded"] is True
    frozen = freeze_contextual_chunking_gold(
        preregistration_path=prereg, ingestion_path=ingestion,
        results_path=results, workbook_root=workbook,
        output_path=tmp_path / "blank-freeze.json",
    )
    assert frozen["status"] == "BLOCKED"
    assert frozen["human_gold_complete"] is False


def test_named_blinded_review_can_promote_and_machine_columns_are_locked(tmp_path: Path) -> None:
    prereg, ingestion = _fixture(tmp_path)
    results = tmp_path / "results.json"
    results.write_text(json.dumps(run_contextual_chunking_confirmation(
        preregistration_path=prereg, ingestion_path=ingestion
    )), encoding="utf-8")
    workbook = tmp_path / "workbook"
    build_contextual_chunking_gold_workbook(
        preregistration_path=prereg, ingestion_path=ingestion,
        results_path=results, output_dir=workbook,
    )
    working = workbook / "retrieval_gold_review_working_copy.csv"
    with working.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames or []), list(reader)
    for row in rows:
        is_section = json.loads(row["section_path_json"]) != []
        row["relevance_grade"] = "2" if is_section and row["rank"] == "1" else "0"
        row["citation_locator_valid"] = "PASS"
        row["reviewer"] = "fixture-reviewer"
        row["review_comment"] = "reviewed against fixture source"
    with working.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    frozen = freeze_contextual_chunking_gold(
        preregistration_path=prereg, ingestion_path=ingestion,
        results_path=results, workbook_root=workbook,
        output_path=tmp_path / "freeze.json",
    )
    assert frozen["status"] == "COMPLETE"
    assert frozen["promotion_decision"] == "PROMOTE_SECTION_CONTEXT_AWARE_HUMAN_GOLD"

    rows[0]["query_text"] = "tampered"
    with working.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="machine field changed"):
        freeze_contextual_chunking_gold(
            preregistration_path=prereg, ingestion_path=ingestion,
            results_path=results, workbook_root=workbook,
            output_path=tmp_path / "tampered-freeze.json",
        )
