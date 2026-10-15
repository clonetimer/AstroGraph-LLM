"""Materialize human-checked rewrite candidates for review-only RAG/KG processing."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from astrograph.filesystem import publish_staged_directory
from .document_governance import GovernedDocumentDescriptor, DocumentReviewState


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_sha(value: object) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _validate_semantic_receipt(payload: dict[str, Any], field: str) -> None:
    body = {key: value for key, value in payload.items() if key != field}
    if _canonical_sha(body) != payload.get(field):
        raise ValueError(f"{field} semantic hash mismatch")


def materialize_reviewed_rewrite_dual_use_pack(
    *,
    source_manifest_path: str | Path,
    rewrite_pack_root: str | Path,
    review_freeze_path: str | Path,
    output_root: str | Path,
    release_id: str,
) -> dict[str, Any]:
    """Build a manifest consumable by both retrieval and candidate-KG stages.

    Rewrite review establishes only that the paraphrase preserves its canonical
    source.  The derived descriptors therefore remain ``REVIEW_REQUIRED`` and
    cannot grant runtime retrieval, knowledge publication, or diagnostic votes.
    """

    source_manifest_path = Path(source_manifest_path).resolve()
    rewrite_root = Path(rewrite_pack_root).resolve()
    review_freeze_path = Path(review_freeze_path).resolve()
    output = Path(output_root).resolve()
    release_id = release_id.strip()
    if not release_id or any(char in release_id for char in ("/", "\\", "\x00")):
        raise ValueError("release_id must be a non-path identifier")
    if output.exists():
        raise ValueError("refusing to overwrite reviewed rewrite dual-use pack")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_body = {
        key: value for key, value in source_manifest.items() if key != "manifest_sha256"
    }
    if _canonical_sha(source_body) != source_manifest.get("manifest_sha256"):
        raise ValueError("source manifest semantic hash mismatch")

    rewrite_receipt_path = rewrite_root / "receipt.json"
    rewrite_receipt = json.loads(rewrite_receipt_path.read_text(encoding="utf-8"))
    _validate_semantic_receipt(rewrite_receipt, "receipt_sha256")
    if (
        rewrite_receipt.get("source_manifest_file_sha256") != _sha(source_manifest_path)
        or rewrite_receipt.get("source_manifest_sha256")
        != source_manifest.get("manifest_sha256")
    ):
        raise ValueError("rewrite pack is not bound to the supplied source manifest")
    if rewrite_receipt.get("status") != "HUMAN_REVIEW_REQUIRED":
        raise ValueError("rewrite pack is not eligible for human-reviewed handoff")

    review_freeze = json.loads(review_freeze_path.read_text(encoding="utf-8"))
    _validate_semantic_receipt(review_freeze, "freeze_sha256")
    if review_freeze.get("rewrite_pack_receipt_file_sha256") != _sha(
        rewrite_receipt_path
    ):
        raise ValueError("rewrite review freeze targets a different rewrite pack")
    if review_freeze.get("review_basis_sha256") != rewrite_receipt.get(
        "review_basis_sha256"
    ):
        raise ValueError("rewrite review freeze targets a different review basis")
    if review_freeze.get("status") != "READY_FOR_CORPUS_RELEASE_CONSIDERATION":
        raise ValueError("rewrite review is not ready for corpus release consideration")
    if review_freeze.get("unsupported_claim_rate") != 0.0:
        raise ValueError("reviewed rewrite handoff requires zero unsupported claims")
    decisions = list(review_freeze.get("decisions", []))
    if (
        len(decisions) != rewrite_receipt.get("block_count")
        or not decisions
        or any(item.get("decision") != "SUPPORTED" for item in decisions)
    ):
        raise ValueError("reviewed rewrite handoff requires every block to be SUPPORTED")

    source_documents = {
        (str(item["dataset_id"]), str(item["fault_class_id"])): item
        for item in source_manifest.get("documents", [])
    }
    rewrite_results = {
        (str(item["dataset_id"]), str(item["fault_class_id"])): item
        for item in rewrite_receipt.get("results", [])
    }
    if (
        len(source_documents) != len(source_manifest.get("documents", []))
        or len(rewrite_results) != len(rewrite_receipt.get("results", []))
        or set(source_documents) != set(rewrite_results)
    ):
        raise ValueError("source manifest and rewrite pack document identities differ")

    # Validate every input before creating a transaction directory.
    validated: list[tuple[dict[str, Any], dict[str, Any], Path, Path]] = []
    for key in sorted(source_documents):
        source_row = source_documents[key]
        result = rewrite_results[key]
        if (
            result.get("status") != "CANDIDATE_READY_FOR_HUMAN_REVIEW"
            or result.get("source_sha256") != source_row.get("source_sha256")
        ):
            raise ValueError(f"rewrite candidate is not eligible for handoff: {key}")
        candidate = (rewrite_root / str(result["candidate_path"])).resolve()
        if (
            not candidate.is_relative_to(rewrite_root)
            or not candidate.is_file()
            or _sha(candidate) != result.get("candidate_sha256")
        ):
            raise ValueError(f"rewrite candidate is missing or changed: {key}")
        relative = Path(str(source_row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source manifest document path escapes corpus: {key}")
        descriptor = GovernedDocumentDescriptor.model_validate(source_row["descriptor"])
        if descriptor.source_sha256 != source_row.get("source_sha256"):
            raise ValueError(f"source descriptor hash differs from manifest: {key}")
        validated.append((source_row, result, candidate, relative))

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent)
    )
    try:
        derived_documents: list[dict[str, Any]] = []
        artifact_hashes: dict[str, str] = {}
        for source_row, result, candidate, relative in validated:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, target)
            candidate_sha = _sha(target)
            descriptor = GovernedDocumentDescriptor.model_validate(source_row["descriptor"])
            template_version = descriptor.template_version or "unspecified"
            derived_descriptor = descriptor.model_copy(
                update={
                    "document_id": (
                        f"{descriptor.document_id}:reviewed-rewrite:{release_id}"
                    ),
                    "source_uri": f"corpus://{release_id}/{relative.as_posix()}",
                    "source_sha256": candidate_sha,
                    # Semantic-equivalence review is not factual knowledge review.
                    "review_state": DocumentReviewState.REVIEW_REQUIRED,
                    "template_version": (
                        f"{template_version}+controlled-rewrite:{release_id}"
                    ),
                    "runtime_vote_allowed": False,
                }
            )
            preserved = {
                key: value
                for key, value in source_row.items()
                if key
                not in {
                    "path",
                    "source_sha256",
                    "descriptor",
                    "retrieval_ingestion_sha256",
                    "knowledge_staging_sha256",
                    "processing_sha256",
                    "parsed_element_count",
                    "retrieval_chunk_count",
                    "knowledge_section_count",
                    "knowledge_extraction_task_count",
                    "crosswalk_count",
                    "crosswalk_complete",
                    "candidate_kg_staged",
                    "candidate_graph_built",
                    "publication_allowed",
                    "runtime_vote_allowed",
                }
            }
            derived_documents.append(
                {
                    **preserved,
                    "path": relative.as_posix(),
                    "source_sha256": candidate_sha,
                    "canonical_source_sha256": source_row["source_sha256"],
                    "descriptor": derived_descriptor.model_dump(mode="json"),
                    "rewrite_candidate_sha256": result["candidate_sha256"],
                    "rewrite_review_freeze_sha256": review_freeze["freeze_sha256"],
                    "candidate_kg_staged": False,
                    "candidate_graph_built": False,
                    "publication_allowed": False,
                    "runtime_vote_allowed": False,
                }
            )
            artifact_hashes[relative.as_posix()] = candidate_sha

        manifest_body = {
            "schema_version": source_manifest["schema_version"],
            "release_id": release_id,
            "source_manifest_file_sha256": _sha(source_manifest_path),
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "rewrite_pack_receipt_file_sha256": _sha(rewrite_receipt_path),
            "rewrite_review_freeze_file_sha256": _sha(review_freeze_path),
            "rewrite_review_freeze_sha256": review_freeze["freeze_sha256"],
            "document_count": len(derived_documents),
            "documents": derived_documents,
            "all_documents_dual_processed": False,
            "candidate_graph_built": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "review_state": "KNOWLEDGE_REVIEW_REQUIRED",
            "claim_boundary": (
                "Human review established paraphrase equivalence only. Documents may be "
                "processed into RAG assets and candidate-KG staging, but their claims remain "
                "unapproved and cannot enter Active KG or runtime diagnostic voting."
            ),
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": _canonical_sha(manifest_body),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        receipt_body = {
            "schema_version": "astrograph-reviewed-rewrite-dual-use-handoff.v1",
            "status": "READY_FOR_REVIEW_ONLY_DUAL_PROCESSING",
            "release_id": release_id,
            "document_count": len(derived_documents),
            "manifest_file_sha256": _sha(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "rag_corpus_build_allowed": True,
            "kg_staging_allowed": True,
            "runtime_retrieval_allowed": False,
            "knowledge_claims_human_reviewed": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": manifest_body["claim_boundary"],
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": _canonical_sha(receipt_body),
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_staged_directory(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["materialize_reviewed_rewrite_dual_use_pack"]
