from __future__ import annotations

import hashlib
import csv
import json
import sys
from pathlib import Path

import pytest

from astrograph.cli import main
from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_candidate_entrypoint import (
    build_document_candidate_package,
)
from astrograph.knowledge.document_candidate_correction import (
    CORRECTION_FIELDS,
    apply_document_candidate_correction,
    prepare_document_candidate_correction,
)
from astrograph.knowledge.document_graph_review import (
    build_single_document_graph_review_handoff,
)
from astrograph.knowledge.document_graph_review_decisions import (
    freeze_document_graph_review_decisions,
)
from astrograph.knowledge.document_ingestion import (
    ingest_governed_document_to_directory,
)
from astrograph.knowledge.construction import GovernedDocumentProcessingBundle
from astrograph.knowledge.construction import knowledge_source_profile_set_sha256
from astrograph.retrieval.document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)


def _source_and_descriptor(tmp_path: Path):
    source = tmp_path / "design.md"
    source.write_text(
        "# Wheel design\nCurrent may rise under friction.\n"
        "## Verification\nCompare current and speed residuals.\n",
        encoding="utf-8",
    )
    descriptor = GovernedDocumentDescriptor(
        document_id="document:test:wheel-design",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri="test://wheel-design.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        synthetic_from_simulation=True,
    )
    return source, descriptor


def _ingestion_package(tmp_path: Path) -> tuple[Path, GovernedDocumentProcessingBundle]:
    source, descriptor = _source_and_descriptor(tmp_path)
    output = tmp_path / "dual-use"
    ingest_governed_document_to_directory(
        descriptor=descriptor,
        source_path=source,
        output_dir=output,
        independence_group="synthetic:test",
        license_id="INTERNAL-RESEARCH",
    )
    return output, GovernedDocumentProcessingBundle.model_validate_json(
        (output / "processing_bundle.json").read_text(encoding="utf-8")
    )


def _refreeze(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "refreeze.json"
    path.write_text(
        json.dumps({
            "temporary_frozen": True,
            "publication_allowed": False,
            "refreeze_sha256": "5" * 64,
        }),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _proposal(bundle: GovernedDocumentProcessingBundle) -> KnowledgeExtractionProposal:
    fragment = bundle.knowledge_staging.source_profile.fragments[0]
    return KnowledgeExtractionProposal(
        source_profile_sha256=knowledge_source_profile_set_sha256(
            [bundle.knowledge_staging.source_profile]
        ),
        nodes=[{
            "node_id": "candidate:wheel-friction",
            "node_type": "FailureMechanism",
            "canonical_name": "Wheel friction",
            "description": "Review-required mechanism candidate.",
            "source_fragment_ids": [fragment.fragment_id],
            "confidence": 0.9,
        }],
        proposer_id="frozen-proposal-fixture",
        proposer_version="1",
    )


def test_stable_document_ingestion_writes_complete_dual_use_package(tmp_path: Path) -> None:
    source, descriptor = _source_and_descriptor(tmp_path)
    output = tmp_path / "dual-use"
    receipt = ingest_governed_document_to_directory(
        descriptor=descriptor,
        source_path=source,
        output_dir=output,
        independence_group="synthetic:test",
        license_id="INTERNAL-RESEARCH",
    )
    assert receipt.status == "STAGED_REVIEW_REQUIRED"
    assert receipt.crosswalk_count == receipt.retrieval_chunk_count == 2
    assert receipt.knowledge_section_count == 4
    assert receipt.publication_allowed is False
    assert receipt.runtime_vote_allowed is False
    bundle = GovernedDocumentProcessingBundle.model_validate_json(
        (output / "processing_bundle.json").read_text(encoding="utf-8")
    )
    assert bundle.processing_sha256 == receipt.processing_sha256
    assert bundle.content_sha256 == receipt.content_sha256
    for filename, expected in receipt.artifacts.items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert json.loads((output / "receipt.json").read_text(encoding="utf-8"))["document_id"] == (
        descriptor.document_id
    )


def test_document_ingestion_fails_closed_before_partial_output(tmp_path: Path) -> None:
    source, descriptor = _source_and_descriptor(tmp_path)
    output = tmp_path / "dual-use"
    with pytest.raises(ValueError, match="source SHA-256"):
        ingest_governed_document_to_directory(
            descriptor=descriptor.model_copy(update={"source_sha256": "f" * 64}),
            source_path=source,
            output_dir=output,
            independence_group="synthetic:test",
            license_id="INTERNAL-RESEARCH",
        )
    assert not output.exists()
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        ingest_governed_document_to_directory(
            descriptor=descriptor,
            source_path=source,
            output_dir=output,
            independence_group="synthetic:test",
            license_id="INTERNAL-RESEARCH",
        )


def test_process_document_cli_exposes_dual_use_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, descriptor = _source_and_descriptor(tmp_path)
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    output = tmp_path / "cli-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "astrograph",
            "process-document",
            "--source",
            str(source),
            "--descriptor",
            str(descriptor_path),
            "--output",
            str(output),
            "--independence-group",
            "synthetic:test",
            "--license-id",
            "INTERNAL-RESEARCH",
        ],
    )
    assert main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "STAGED_REVIEW_REQUIRED"
    assert (output / "retrieval_ingestion.json").is_file()
    assert (output / "knowledge_staging.json").is_file()


def test_candidate_stage_consumes_frozen_ingestion_and_proposal(tmp_path: Path) -> None:
    package, bundle = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)
    output = tmp_path / "candidate"
    receipt = build_document_candidate_package(
        ingestion_package_dir=package,
        output_dir=output,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_file_sha,
        graph_id="graph:test:document-candidate",
        proposal=_proposal(bundle),
    )
    assert receipt.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    assert receipt.candidate_graph_built is True
    assert receipt.candidate_node_count == 1
    assert receipt.publication_allowed is False
    assert receipt.runtime_vote_allowed is False
    assert receipt.active_graph_modified is False
    assert set(receipt.artifacts) >= {
        "raw_extraction_proposal.json", "construction_bundle.json",
        "candidate_graph.json", "quality_report.json",
    }
    for filename, expected in receipt.artifacts.items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert json.loads((output / "receipt.json").read_text(encoding="utf-8"))["status"] == receipt.status


def test_candidate_stage_retains_backend_exception_as_blocked_package(tmp_path: Path) -> None:
    package, _ = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)

    class BrokenBackend:
        model_id = "broken"
        model_version = "1"
        response_trace = [{
            "prompt_fragment_ids": ["fragment:test:failed"],
            "finish_reason": "length",
            "raw_content": "{\"truncated\":",
        }]

        def generate_structured(self, **kwargs):
            raise RuntimeError("controlled extraction failure")

    output = tmp_path / "failed-candidate"
    receipt = build_document_candidate_package(
        ingestion_package_dir=package,
        output_dir=output,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_file_sha,
        graph_id="graph:test:failed-document-candidate",
        backend=BrokenBackend(),
    )
    assert receipt.status == "BLOCKED_BY_EXCEPTION"
    assert receipt.exception_type == "RuntimeError"
    assert receipt.proposer_id == "broken"
    assert receipt.blocker_codes == ["CANDIDATE_BUILD_EXCEPTION"]
    assert receipt.generation_trace_count == 1
    assert receipt.exception_fragment_ids == ["fragment:test:failed"]
    assert receipt.raw_response_trace_sha256 == receipt.artifacts["raw_generation_trace.json"]
    assert (output / "raw_generation_trace.json").is_file()
    assert (output / "receipt.json").is_file()
    assert not (output / "candidate_graph.json").exists()


def test_blocked_candidate_items_are_labeled_for_inspection_not_as_empty(
    tmp_path: Path,
) -> None:
    package, bundle = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)
    proposal_payload = _proposal(bundle).model_dump(mode="json")
    proposal_payload["relations"] = [{
        "edge_id": "candidate-edge:invalid-self-shape",
        "source_id": "candidate:wheel-friction",
        "relation_type": "HAS_FUNCTIONAL_ROLE",
        "target_id": "candidate:wheel-friction",
        "source_fragment_ids": [
            bundle.knowledge_staging.source_profile.fragments[0].fragment_id
        ],
        "confidence": 0.9,
        "rationale": "Deliberately invalid relation shape fixture.",
    }]
    proposal = KnowledgeExtractionProposal.model_validate(proposal_payload)
    candidate = tmp_path / "blocked-candidate"
    receipt = build_document_candidate_package(
        ingestion_package_dir=package,
        output_dir=candidate,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_file_sha,
        graph_id="graph:test:blocked-with-items",
        proposal=proposal,
    )
    assert receipt.status == "BLOCKED"
    handoff = build_single_document_graph_review_handoff(
        candidate_package_root=candidate,
        output_dir=tmp_path / "blocked-review",
    )
    assert handoff["status"] == "BLOCKED_CANDIDATES_FOR_INSPECTION"
    assert handoff["review_item_count"] == 2


def test_named_candidate_correction_rebuilds_through_all_machine_gates(
    tmp_path: Path,
) -> None:
    package, bundle = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)
    proposal_payload = _proposal(bundle).model_dump(mode="json")
    proposal_payload["relations"] = [{
        "edge_id": "candidate-edge:invalid-self-shape",
        "source_id": "candidate:wheel-friction",
        "relation_type": "HAS_FUNCTIONAL_ROLE",
        "target_id": "candidate:wheel-friction",
        "source_fragment_ids": [
            bundle.knowledge_staging.source_profile.fragments[0].fragment_id
        ],
        "confidence": 0.9,
        "rationale": "Deliberately invalid relation shape fixture.",
    }]
    blocked = tmp_path / "blocked-candidate"
    first = build_document_candidate_package(
        ingestion_package_dir=package,
        output_dir=blocked,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_file_sha,
        graph_id="graph:test:blocked-before-correction",
        proposal=KnowledgeExtractionProposal.model_validate(proposal_payload),
    )
    assert first.status == "BLOCKED"
    handoff_root = tmp_path / "correction-handoff"
    handoff = prepare_document_candidate_correction(
        candidate_package_root=blocked, output_dir=handoff_root
    )
    assert handoff["correction_item_count"] == 2
    sheet = handoff_root / "human_correction_decisions.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["action"] = "DROP" if row["item_kind"] == "RELATION" else "KEEP"
        row["reviewer"] = "domain-reviewer"
        row["review_comment"] = "Drop the invalid shape; retain the grounded node for review."
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["candidate_sha256"] = "f" * 64
    with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CORRECTION_FIELDS)
        writer.writeheader()
        writer.writerows(tampered_rows)
    with pytest.raises(ValueError, match="machine fields changed"):
        apply_document_candidate_correction(
            ingestion_package_dir=package,
            candidate_package_root=blocked,
            correction_sheet_path=sheet,
            review_refreeze_path=refreeze,
            expected_review_refreeze_file_sha256=refreeze_file_sha,
            graph_id="graph:test:tampered-correction",
            output_dir=tmp_path / "tampered-correction",
        )
    assert not (tmp_path / "tampered-correction").exists()
    with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CORRECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    corrected_root = tmp_path / "corrected-package"
    result = apply_document_candidate_correction(
        ingestion_package_dir=package,
        candidate_package_root=blocked,
        correction_sheet_path=sheet,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_file_sha,
        graph_id="graph:test:corrected-candidate",
        output_dir=corrected_root,
    )
    assert result["status"] == "CORRECTED_CANDIDATE_READY_FOR_HUMAN_REVIEW"
    rebuilt = json.loads((corrected_root / "candidate/receipt.json").read_text(encoding="utf-8"))
    assert rebuilt["status"] == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    assert rebuilt["proposal_node_count"] == 1
    assert rebuilt["proposal_relation_count"] == 0
    assert result["publication_allowed"] is False
    assert result["active_graph_modified"] is False


def test_candidate_stage_rejects_changed_ingestion_and_refreeze(tmp_path: Path) -> None:
    package, bundle = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)
    with pytest.raises(ValueError, match="refreeze file SHA-256 mismatch"):
        build_document_candidate_package(
            ingestion_package_dir=package,
            output_dir=tmp_path / "wrong-refreeze-candidate",
            review_refreeze_path=refreeze,
            expected_review_refreeze_file_sha256="f" * 64,
            graph_id="graph:test:wrong-refreeze",
            proposal=_proposal(bundle),
        )
    assert not (tmp_path / "wrong-refreeze-candidate").exists()
    (package / "retrieval_ingestion.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        build_document_candidate_package(
            ingestion_package_dir=package,
            output_dir=tmp_path / "candidate",
            review_refreeze_path=refreeze,
            expected_review_refreeze_file_sha256=refreeze_file_sha,
            graph_id="graph:test:document-candidate",
            proposal=_proposal(bundle),
        )


def test_candidate_cli_accepts_a_frozen_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package, bundle = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(_proposal(bundle).model_dump_json(indent=2), encoding="utf-8")
    output = tmp_path / "cli-candidate"
    monkeypatch.setattr(sys, "argv", [
        "astrograph", "build-document-candidate-graph",
        "--input-package", str(package),
        "--output", str(output),
        "--review-refreeze", str(refreeze),
        "--expected-review-refreeze-sha256", refreeze_file_sha,
        "--graph-id", "graph:test:cli-candidate",
        "--proposal", str(proposal_path),
    ])
    assert main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    assert printed["candidate_graph_built"] is True


def test_single_candidate_review_handoff_is_blank_and_freezes_fail_closed(
    tmp_path: Path,
) -> None:
    package, bundle = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)
    candidate = tmp_path / "candidate"
    build_document_candidate_package(
        ingestion_package_dir=package,
        output_dir=candidate,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_file_sha,
        graph_id="graph:test:review-candidate",
        proposal=_proposal(bundle),
    )
    review = tmp_path / "review"
    handoff = build_single_document_graph_review_handoff(
        candidate_package_root=candidate, output_dir=review
    )
    assert handoff["status"] == "READY_FOR_HUMAN_REVIEW"
    assert handoff["review_item_count"] == 1
    sheet = review / "human_review_decisions.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["decision"] == rows[0]["reviewer"] == ""
    frozen = freeze_document_graph_review_decisions(
        review_sheet_path=sheet,
        graph_roots=[candidate],
        output_path=tmp_path / "blank-freeze.json",
    )
    assert frozen["status"] == "BLOCKED"
    assert frozen["approved_items"] == []

    fields = list(rows[0])
    rows[0].update({
        "decision": "APPROVE",
        "approved_canonical_id": "mechanism:wheel-friction",
        "reviewer": "human:test",
        "review_comment": "source and schema checked",
    })
    with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    approved = freeze_document_graph_review_decisions(
        review_sheet_path=sheet,
        graph_roots=[candidate],
        output_path=tmp_path / "approved-freeze.json",
    )
    assert approved["status"] == "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL"
    assert len(approved["approved_items"]) == 1
    assert approved["publication_allowed"] is False


def test_blocked_empty_candidate_produces_closed_empty_handoff(tmp_path: Path) -> None:
    package, bundle = _ingestion_package(tmp_path)
    refreeze, refreeze_file_sha = _refreeze(tmp_path)
    candidate = tmp_path / "empty-candidate"
    empty = _proposal(bundle).model_copy(update={"nodes": []})
    result = build_document_candidate_package(
        ingestion_package_dir=package,
        output_dir=candidate,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_file_sha,
        graph_id="graph:test:empty-candidate",
        proposal=empty,
    )
    assert result.status == "BLOCKED"
    review = tmp_path / "empty-review"
    handoff = build_single_document_graph_review_handoff(
        candidate_package_root=candidate, output_dir=review
    )
    assert handoff["status"] == "NO_CANDIDATES_BLOCKED"
    assert handoff["review_item_count"] == 0
    frozen = freeze_document_graph_review_decisions(
        review_sheet_path=review / "human_review_decisions.csv",
        graph_roots=[candidate],
        output_path=tmp_path / "empty-freeze.json",
    )
    assert frozen["status"] == "BLOCKED"
    assert frozen["blockers"] == [
        {"code": "EMPTY_REVIEW_SHEET", "review_item_id": "*"}
    ]
    assert frozen["knowledge_patch_proposal_allowed"] is False
