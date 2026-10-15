"""Atomic manifest entry point composed from the stable single-document stages."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from astrograph.filesystem import publish_staged_directory
from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionBackend,
    KnowledgeExtractionProposal,
)
from astrograph.retrieval.document_governance import GovernedDocumentDescriptor

from .document_candidate_entrypoint import build_document_candidate_package
from .document_ingestion import ingest_governed_document_to_directory


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def build_document_pack_candidate_packages(
    *,
    manifest_path: str | Path,
    corpus_root: str | Path,
    output_root: str | Path,
    review_refreeze_path: str | Path,
    expected_review_refreeze_file_sha256: str,
    backend_factory: Callable[[], KnowledgeExtractionBackend] | None = None,
    frozen_proposal_root: str | Path | None = None,
    selected_fault_ids: list[str] | None = None,
    maximum_documents: int | None = None,
    section_batched: bool = True,
) -> dict[str, object]:
    """Build content-addressed ingestion and candidate packages for a governed manifest."""

    if (backend_factory is None) == (frozen_proposal_root is None):
        raise ValueError("provide exactly one of backend_factory or frozen_proposal_root")
    manifest_path = Path(manifest_path).resolve()
    corpus_root = Path(corpus_root).resolve()
    output_root = Path(output_root).resolve()
    review_refreeze_path = Path(review_refreeze_path).resolve()
    proposal_root = Path(frozen_proposal_root).resolve() if frozen_proposal_root else None
    if output_root.exists():
        raise ValueError("refusing to overwrite document pack candidate output")
    if _sha(review_refreeze_path) != expected_review_refreeze_file_sha256:
        raise ValueError("document pack review refreeze file SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "astrograph-fault-confusion-document-pack.v1":
        raise ValueError("unsupported document pack manifest schema")
    rows = list(manifest.get("documents", []))
    if selected_fault_ids is not None:
        if len(selected_fault_ids) != len(set(selected_fault_ids)):
            raise ValueError("selected document pack fault IDs must be unique")
        selected = set(selected_fault_ids)
        rows = [row for row in rows if row.get("fault_class_id") in selected]
        found = {str(row["fault_class_id"]) for row in rows}
        if found != selected:
            raise ValueError(
                "requested fault IDs are absent from document pack: "
                + repr(sorted(selected - found))
            )
    if maximum_documents is not None:
        if maximum_documents < 1:
            raise ValueError("maximum_documents must be positive")
        rows = rows[:maximum_documents]
    if not rows:
        raise ValueError("document pack contains no selected documents")

    validated: list[tuple[dict[str, object], Path, GovernedDocumentDescriptor]] = []
    keys: set[tuple[str, str]] = set()
    for row in rows:
        dataset_id, fault_id = str(row["dataset_id"]), str(row["fault_class_id"])
        key = (dataset_id, fault_id)
        if key in keys:
            raise ValueError(f"duplicate document pack dataset/fault identity: {key}")
        keys.add(key)
        source = (corpus_root / str(row["path"])).resolve()
        if not source.is_relative_to(corpus_root) or not source.is_file():
            raise ValueError(f"document path is missing or escapes corpus root: {row['path']}")
        if _sha(source) != row["source_sha256"]:
            raise ValueError(f"document source hash mismatch: {row['path']}")
        descriptor = GovernedDocumentDescriptor.model_validate(row["descriptor"])
        if descriptor.source_sha256 != row["source_sha256"]:
            raise ValueError(f"descriptor source hash mismatch: {row['path']}")
        if proposal_root is not None:
            frozen = proposal_root / dataset_id / fault_id / "raw_extraction_proposal.json"
            if not frozen.is_file():
                raise ValueError(f"frozen document proposal is missing: {dataset_id}/{fault_id}")
            KnowledgeExtractionProposal.model_validate_json(
                frozen.read_text(encoding="utf-8-sig")
            )
        validated.append((row, source, descriptor))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.astrograph-txn-", dir=output_root.parent)
    )
    results: list[dict[str, object]] = []
    try:
        for row, source, descriptor in validated:
            dataset_id, fault_id = str(row["dataset_id"]), str(row["fault_class_id"])
            document_root = staging / dataset_id / fault_id
            ingestion_root = document_root / "ingestion"
            candidate_root = document_root / "candidate"
            ingestion = ingest_governed_document_to_directory(
                descriptor=descriptor,
                source_path=source,
                output_dir=ingestion_root,
                independence_group=(
                    descriptor.provenance_group_id
                    or "document-pack-unresolved-provenance"
                ),
                license_id="project-generated-internal-research",
                knowledge_version="document-pack-stable-v1",
            )
            proposal = None
            backend = None
            if proposal_root is not None:
                proposal_path = proposal_root / dataset_id / fault_id / "raw_extraction_proposal.json"
                proposal = KnowledgeExtractionProposal.model_validate_json(
                    proposal_path.read_text(encoding="utf-8-sig")
                )
            else:
                assert backend_factory is not None
                backend = backend_factory()
            candidate = build_document_candidate_package(
                ingestion_package_dir=ingestion_root,
                output_dir=candidate_root,
                review_refreeze_path=review_refreeze_path,
                expected_review_refreeze_file_sha256=expected_review_refreeze_file_sha256,
                graph_id=f"graph:document:{dataset_id}:{fault_id}:stable-r001",
                backend=backend,
                proposal=proposal,
                section_batched=section_batched,
            )
            results.append({
                "dataset_id": dataset_id,
                "fault_class_id": fault_id,
                "document_id": descriptor.document_id,
                "source_sha256": descriptor.source_sha256,
                "ingestion_receipt_file_sha256": _sha(ingestion_root / "receipt.json"),
                "processing_sha256": ingestion.processing_sha256,
                "crosswalk_complete": ingestion.crosswalk_complete,
                "candidate_receipt_file_sha256": _sha(candidate_root / "receipt.json"),
                "candidate_status": candidate.status,
                "proposal_node_count": candidate.proposal_node_count,
                "proposal_relation_count": candidate.proposal_relation_count,
                "unresolved_claim_count": candidate.unresolved_claim_count,
                "generation_trace_count": candidate.generation_trace_count,
                "candidate_graph_built": candidate.candidate_graph_built,
                "candidate_node_count": candidate.candidate_node_count,
                "candidate_edge_count": candidate.candidate_edge_count,
                "blocker_codes": candidate.blocker_codes,
                "publication_allowed": False,
                "runtime_vote_allowed": False,
                "active_graph_modified": False,
            })
        results.sort(key=lambda item: (str(item["dataset_id"]), str(item["fault_class_id"])))
        status_counts: dict[str, int] = {}
        for result in results:
            status = str(result["candidate_status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        body = {
            "schema_version": "astrograph-stable-document-pack-candidate-receipt.v1",
            "status": (
                "COMPLETED_WITH_BLOCKED_DOCUMENTS"
                if any(item["candidate_status"] != "CANDIDATE_READY_FOR_HUMAN_REVIEW" for item in results)
                else "ALL_CANDIDATES_READY_FOR_HUMAN_REVIEW"
            ),
            "source_manifest_file_sha256": _sha(manifest_path),
            "source_content_release_sha256": manifest.get("content_release_sha256"),
            "review_refreeze_file_sha256": expected_review_refreeze_file_sha256,
            "selected_document_count": len(results),
            "status_counts": dict(sorted(status_counts.items())),
            "all_crosswalks_complete": all(bool(item["crosswalk_complete"]) for item in results),
            "results": results,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Manifest-scale composition of stable document ingestion and candidate stages. "
                "Candidate readiness is not human approval, publication or diagnostic evidence."
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha(body)}
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging, output_root)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["build_document_pack_candidate_packages"]
