"""Batch a governed document pack through scoped LLM extraction and DRAFT KG gates."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from astrograph.agents.knowledge_curation import KnowledgeExtractionBackend
from astrograph.retrieval.document_governance import GovernedDocumentDescriptor

from .construction import process_governed_document, write_construction_bundle
from .document_graph import build_document_candidate_graph, propose_document_knowledge


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def build_document_pack_candidate_graphs(
    *,
    manifest_path: str | Path,
    corpus_root: str | Path,
    output_root: str | Path,
    backend: KnowledgeExtractionBackend,
    excluded_review_refreeze_sha256: str,
    max_workers: int = 1,
    maximum_documents: int | None = None,
    selected_fault_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Persist every proposal, blocker and candidate graph from a document pack."""

    manifest_path = Path(manifest_path).resolve()
    corpus_root = Path(corpus_root).resolve()
    output_root = Path(output_root).resolve()
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "astrograph-fault-confusion-document-pack.v1":
        raise ValueError("unsupported document pack manifest schema")
    rows = list(manifest.get("documents", []))
    if selected_fault_ids is not None:
        requested = set(selected_fault_ids)
        rows = [row for row in rows if row["fault_class_id"] in requested]
        found = {row["fault_class_id"] for row in rows}
        if found != requested:
            raise ValueError(f"requested fault IDs are absent from document pack: {sorted(requested - found)}")
    if maximum_documents is not None:
        if maximum_documents < 1:
            raise ValueError("maximum_documents must be positive")
        rows = rows[:maximum_documents]
    if not rows:
        raise ValueError("document pack contains no selected documents")

    # Validate the full selected input set before creating an output directory.
    for row in rows:
        path = (corpus_root / row["path"]).resolve()
        if not path.is_relative_to(corpus_root) or not path.is_file():
            raise ValueError(f"document path is missing or escapes corpus root: {row['path']}")
        if _sha(path) != row["source_sha256"]:
            raise ValueError(f"document source hash mismatch: {row['path']}")
        descriptor = GovernedDocumentDescriptor.model_validate(row["descriptor"])
        if descriptor.source_sha256 != row["source_sha256"]:
            raise ValueError(f"descriptor source hash mismatch: {row['path']}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty document candidate graph directory")
    output_root.mkdir(parents=True, exist_ok=True)

    def run(row: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(row["dataset_id"])
        fault_id = str(row["fault_class_id"])
        source = (corpus_root / row["path"]).resolve()
        destination = output_root / dataset_id / fault_id
        descriptor = GovernedDocumentDescriptor.model_validate(row["descriptor"])
        processed = process_governed_document(
            descriptor=descriptor,
            source_path=source,
            independence_group=(
                descriptor.provenance_group_id
                or "document-pack-unresolved-provenance"
            ),
            license_id="project-generated",
            knowledge_version="document-candidate-graph-r001",
        )
        destination.mkdir(parents=True, exist_ok=True)
        try:
            raw_proposal = propose_document_knowledge(
                processed=processed,
                backend=backend,
            )
            raw_proposal_path = destination / "raw_extraction_proposal.json"
            raw_proposal_path.write_text(
                raw_proposal.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            bundle = build_document_candidate_graph(
                processed=processed,
                proposal=raw_proposal,
                excluded_review_refreeze_sha256=excluded_review_refreeze_sha256,
                graph_id=f"graph:document:{dataset_id}:{fault_id}:r001",
            )
            paths = write_construction_bundle(bundle, output_dir=destination)
            blockers = sorted(
                {
                    finding.code
                    for finding in bundle.candidate_validation.findings
                    if finding.severity == "BLOCKER"
                }
            )
            warnings = sorted(
                {
                    finding.code
                    for finding in bundle.candidate_validation.findings
                    if finding.severity == "WARNING"
                }
            )
            return {
                "dataset_id": dataset_id,
                "fault_class_id": fault_id,
                "source_sha256": row["source_sha256"],
                "status": bundle.status,
                "proposal_node_count": len(bundle.proposal.nodes),
                "proposal_relation_count": len(bundle.proposal.relations),
                "unresolved_claim_count": len(bundle.proposal.unresolved_claims),
                "candidate_graph_built": bundle.candidate_graph is not None,
                "blocker_codes": blockers,
                "warning_codes": warnings,
                "bundle_sha256": bundle.bundle_sha256,
                "artifact_sha256": {
                    name: _sha(path) for name, path in sorted(paths.items())
                },
                "raw_extraction_proposal_sha256": _sha(raw_proposal_path),
                "publication_allowed": bundle.publication_allowed,
                "active_graph_modified": bundle.active_graph_modified,
            }
        except Exception as exc:
            failure = {
                "schema_version": "astrograph-document-graph-extraction-failure.v1",
                "dataset_id": dataset_id,
                "fault_class_id": fault_id,
                "source_sha256": row["source_sha256"],
                "status": "BLOCKED_BY_EXCEPTION",
                "exception_type": type(exc).__name__,
                "detail": str(exc)[:2000],
                "candidate_graph_built": False,
                "publication_allowed": False,
                "active_graph_modified": False,
            }
            raw_path = destination / "raw_extraction_proposal.json"
            if raw_path.is_file():
                failure["raw_extraction_proposal_sha256"] = _sha(raw_path)
            failure_path = destination / "failure.json"
            failure_path.write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return {**failure, "failure_sha256": _sha(failure_path)}

    results: list[dict[str, Any]] = []
    if max_workers == 1:
        results = [run(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run, row): row for row in rows}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: (row["dataset_id"], row["fault_class_id"]))
    ready = sum(row["status"] == "CANDIDATE_READY_FOR_HUMAN_REVIEW" for row in results)
    blocked = len(results) - ready
    receipt_body = {
        "schema_version": "astrograph-document-candidate-graph-pack-receipt.v1",
        "source_manifest_sha256": _sha(manifest_path),
        "source_content_release_sha256": manifest["content_release_sha256"],
        "excluded_review_refreeze_sha256": excluded_review_refreeze_sha256,
        "backend_model_id": backend.model_id,
        "backend_model_version": backend.model_version,
        "selected_document_count": len(results),
        "candidate_ready_count": ready,
        "blocked_count": blocked,
        "results": results,
        "publication_allowed": False,
        "active_graph_modified": False,
        "runtime_vote_allowed": False,
        "claim_boundary": (
            "Schema-constrained LLM extraction and DRAFT candidate-graph artifacts only. "
            "READY means machine gates passed, not that a claim is true or human-approved. "
            "Blocked outputs are preserved and no result can publish or vote at runtime."
        ),
    }
    receipt = {**receipt_body, "receipt_sha256": _canonical_sha(receipt_body)}
    (output_root / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = ["build_document_pack_candidate_graphs"]
