"""Atomic second-stage entry point for governed document candidate graphs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.filesystem import publish_staged_directory
from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionBackend,
    KnowledgeExtractionProposal,
)

from .construction import (
    GovernedDocumentProcessingBundle,
    write_construction_bundle,
)
from .document_graph import (
    build_staged_document_candidate_graph,
    propose_staged_document_knowledge,
)
from .document_ingestion import GovernedDocumentIngestionReceipt


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DocumentCandidateBuildReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-document-candidate-build-receipt.v1"] = (
        "astrograph-document-candidate-build-receipt.v1"
    )
    status: Literal[
        "BLOCKED_BY_EXCEPTION", "BLOCKED", "CANDIDATE_READY_FOR_HUMAN_REVIEW"
    ]
    document_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    processing_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_refreeze_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    excluded_review_refreeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposer_id: str | None = None
    proposer_version: str | None = None
    section_batched: bool
    raw_proposal_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_response_trace_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    generation_trace_count: int = Field(default=0, ge=0)
    exception_fragment_ids: list[str] = Field(default_factory=list)
    construction_bundle_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    construction_content_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    proposal_node_count: int | None = Field(default=None, ge=0)
    proposal_relation_count: int | None = Field(default=None, ge=0)
    unresolved_claim_count: int | None = Field(default=None, ge=0)
    candidate_graph_built: bool = False
    candidate_node_count: int = Field(default=0, ge=0)
    candidate_edge_count: int = Field(default=0, ge=0)
    blocker_codes: list[str] = Field(default_factory=list)
    exception_type: str | None = None
    exception_detail: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    publication_allowed: Literal[False] = False
    runtime_vote_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    claim_boundary: str = (
        "This package is a review-only document candidate graph. Even a ready candidate cannot "
        "publish knowledge, alter Active KG, or contribute a runtime diagnostic vote."
    )

    @field_validator("artifacts")
    @classmethod
    def content_addressed_artifacts(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
            for digest in value.values()
        ):
            raise ValueError("candidate graph artifact hashes must be lowercase SHA-256")
        return value


def _load_ingestion_package(root: Path) -> tuple[
    GovernedDocumentIngestionReceipt, GovernedDocumentProcessingBundle
]:
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("document ingestion package has no receipt")
    receipt = GovernedDocumentIngestionReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    for filename, expected in receipt.artifacts.items():
        path = root / filename
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"document ingestion artifact is missing or changed: {filename}")
    bundle = GovernedDocumentProcessingBundle.model_validate_json(
        (root / "processing_bundle.json").read_text(encoding="utf-8")
    )
    if (
        bundle.processing_sha256 != receipt.processing_sha256
        or bundle.content_sha256 != receipt.content_sha256
        or bundle.knowledge_staging.document_descriptor.document_id != receipt.document_id
    ):
        raise ValueError("document ingestion receipt and processing bundle differ")
    return receipt, bundle


def _load_refreeze(path: Path, expected_file_sha256: str) -> str:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError("review refreeze file SHA-256 mismatch")
    payload = json.loads(raw.decode("utf-8"))
    refreeze = str(payload.get("refreeze_sha256", ""))
    if len(refreeze) != 64 or any(char not in "0123456789abcdef" for char in refreeze):
        raise ValueError("review refreeze has no valid semantic SHA-256")
    if not payload.get("temporary_frozen", False) or payload.get("publication_allowed", True):
        raise ValueError("review refreeze does not retain the publication freeze")
    return refreeze


def _persist_backend_response_trace(
    staging_dir: Path, backend: KnowledgeExtractionBackend | None
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    trace = getattr(backend, "response_trace", None)
    if not isinstance(trace, list) or not trace:
        return {}, []
    normalized = [item for item in trace if isinstance(item, dict)]
    if not normalized:
        return {}, []
    path = staging_dir / "raw_generation_trace.json"
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {path.name: _sha256_file(path)}, normalized


def build_document_candidate_package(
    *,
    ingestion_package_dir: str | Path,
    output_dir: str | Path,
    review_refreeze_path: str | Path,
    expected_review_refreeze_file_sha256: str,
    graph_id: str,
    backend: KnowledgeExtractionBackend | None = None,
    proposal: KnowledgeExtractionProposal | None = None,
    section_batched: bool = True,
) -> DocumentCandidateBuildReceipt:
    """Extract, govern, compile and atomically persist a review-only candidate graph."""

    if (backend is None) == (proposal is None):
        raise ValueError("provide exactly one of extraction backend or frozen proposal")
    ingestion_root = Path(ingestion_package_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("document candidate output directory already exists")
    _, processed = _load_ingestion_package(ingestion_root)
    refreeze_path = Path(review_refreeze_path).resolve()
    refreeze = _load_refreeze(refreeze_path, expected_review_refreeze_file_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent)
    )
    raw_path = staging_dir / "raw_extraction_proposal.json"
    try:
        if proposal is None:
            assert backend is not None
            proposal = propose_staged_document_knowledge(
                staging=processed.knowledge_staging,
                backend=backend,
                section_batched=section_batched,
            )
        trace_artifacts, response_trace = _persist_backend_response_trace(staging_dir, backend)
        raw_path.write_text(proposal.model_dump_json(indent=2) + "\n", encoding="utf-8")
        construction = build_staged_document_candidate_graph(
            staging=processed.knowledge_staging,
            proposal=proposal,
            excluded_review_refreeze_sha256=refreeze,
            graph_id=graph_id,
        )
        paths = write_construction_bundle(construction, output_dir=staging_dir)
        artifacts = {
            "raw_extraction_proposal.json": _sha256_file(raw_path),
            **trace_artifacts,
            **{
                path.name: _sha256_file(path)
                for path in paths.values()
            },
        }
        graph = construction.candidate_graph
        receipt = DocumentCandidateBuildReceipt(
            status=construction.status,
            document_id=processed.knowledge_staging.document_descriptor.document_id,
            graph_id=graph_id,
            processing_sha256=processed.processing_sha256,
            processing_content_sha256=processed.content_sha256 or "",
            review_refreeze_file_sha256=expected_review_refreeze_file_sha256,
            excluded_review_refreeze_sha256=refreeze,
            proposer_id=proposal.proposer_id,
            proposer_version=proposal.proposer_version,
            section_batched=section_batched if backend is not None else False,
            raw_proposal_sha256=artifacts["raw_extraction_proposal.json"],
            raw_response_trace_sha256=artifacts.get("raw_generation_trace.json"),
            generation_trace_count=len(response_trace),
            construction_bundle_sha256=construction.bundle_sha256,
            construction_content_sha256=construction.content_sha256,
            proposal_node_count=len(construction.proposal.nodes),
            proposal_relation_count=len(construction.proposal.relations),
            unresolved_claim_count=len(construction.proposal.unresolved_claims),
            candidate_graph_built=graph is not None,
            candidate_node_count=len(graph.nodes) if graph else 0,
            candidate_edge_count=len(graph.edges) if graph else 0,
            blocker_codes=sorted(
                {
                    item.code
                    for item in construction.candidate_validation.findings
                    if item.severity == "BLOCKER"
                }
            ),
            artifacts=dict(sorted(artifacts.items())),
        )
    except Exception as exc:
        trace_artifacts, response_trace = _persist_backend_response_trace(staging_dir, backend)
        artifacts = {
            **trace_artifacts,
            **(
            {"raw_extraction_proposal.json": _sha256_file(raw_path)}
            if raw_path.is_file()
            else {}
            ),
        }
        exception_fragment_ids = (
            list(response_trace[-1].get("prompt_fragment_ids", []))
            if response_trace
            else []
        )
        receipt = DocumentCandidateBuildReceipt(
            status="BLOCKED_BY_EXCEPTION",
            document_id=processed.knowledge_staging.document_descriptor.document_id,
            graph_id=graph_id,
            processing_sha256=processed.processing_sha256,
            processing_content_sha256=processed.content_sha256 or "",
            review_refreeze_file_sha256=expected_review_refreeze_file_sha256,
            excluded_review_refreeze_sha256=refreeze,
            proposer_id=(
                getattr(proposal, "proposer_id", None)
                or getattr(backend, "model_id", None)
            ),
            proposer_version=(
                getattr(proposal, "proposer_version", None)
                or getattr(backend, "model_version", None)
            ),
            section_batched=section_batched if backend is not None else False,
            raw_proposal_sha256=artifacts.get("raw_extraction_proposal.json"),
            raw_response_trace_sha256=artifacts.get("raw_generation_trace.json"),
            generation_trace_count=len(response_trace),
            exception_fragment_ids=exception_fragment_ids,
            exception_type=type(exc).__name__,
            exception_detail=str(exc)[:2000],
            blocker_codes=["CANDIDATE_BUILD_EXCEPTION"],
            artifacts=artifacts,
        )
    try:
        (staging_dir / "receipt.json").write_text(
            receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging_dir, output)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return receipt


__all__ = ["DocumentCandidateBuildReceipt", "build_document_candidate_package"]
