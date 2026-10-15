"""Stable, atomic entry point for governed dual-use document ingestion."""
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
from astrograph.retrieval.document_governance import GovernedDocumentDescriptor

from .construction import GovernedDocumentProcessingBundle, process_governed_document


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GovernedDocumentIngestionReceipt(BaseModel):
    """Portable receipt for one RAG/OAG plus candidate-KG preprocessing run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-governed-document-ingestion-receipt.v1"] = (
        "astrograph-governed-document-ingestion-receipt.v1"
    )
    status: Literal["STAGED_REVIEW_REQUIRED"] = "STAGED_REVIEW_REQUIRED"
    document_id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_chunk_count: int = Field(ge=1)
    knowledge_section_count: int = Field(ge=1)
    crosswalk_count: int = Field(ge=1)
    crosswalk_complete: Literal[True] = True
    candidate_graph_built: Literal[False] = False
    publication_allowed: Literal[False] = False
    runtime_vote_allowed: Literal[False] = False
    artifacts: dict[str, str]
    claim_boundary: str = (
        "The source was parsed once into trace-linked retrieval and candidate-KG staging assets. "
        "No graph claim was approved, published, or admitted as runtime diagnostic evidence."
    )

    @field_validator("artifacts")
    @classmethod
    def exact_content_addressed_artifacts(cls, value: dict[str, str]) -> dict[str, str]:
        expected = {
            "processing_bundle.json",
            "retrieval_ingestion.json",
            "knowledge_staging.json",
            "document_processing_crosswalk.json",
        }
        if set(value) != expected:
            raise ValueError("document ingestion receipt must bind the complete artifact set")
        if any(
            len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
            for digest in value.values()
        ):
            raise ValueError("document ingestion artifact hashes must be lowercase SHA-256")
        return value


def ingest_governed_document_to_directory(
    *,
    descriptor: GovernedDocumentDescriptor,
    source_path: str | Path,
    output_dir: str | Path,
    independence_group: str,
    license_id: str,
    knowledge_version: str = "document-dual-use-v1",
) -> GovernedDocumentIngestionReceipt:
    """Create a complete dual-use package through a directory-level atomic commit."""

    source = Path(source_path).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise ValueError("governed document source is missing")
    if output.exists():
        raise ValueError("governed document output directory already exists")
    if descriptor.source_sha256 != _sha256_file(source):
        raise ValueError("document descriptor source SHA-256 does not match source bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = process_governed_document(
        descriptor=descriptor,
        source_path=source,
        independence_group=independence_group,
        license_id=license_id,
        knowledge_version=knowledge_version,
    )
    if bundle.content_sha256 is None:
        raise ValueError("governed document processor did not emit a portable content hash")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        payloads: dict[str, Any] = {
            "processing_bundle.json": bundle.model_dump(mode="json"),
            "retrieval_ingestion.json": bundle.retrieval_ingestion.model_dump(mode="json"),
            "knowledge_staging.json": bundle.knowledge_staging.model_dump(mode="json"),
            "document_processing_crosswalk.json": [
                item.model_dump(mode="json") for item in bundle.crosswalk
            ],
        }
        for filename, payload in payloads.items():
            (staging / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        artifacts = {
            filename: _sha256_file(staging / filename) for filename in sorted(payloads)
        }
        receipt = GovernedDocumentIngestionReceipt(
            document_id=descriptor.document_id,
            source_filename=source.name,
            source_sha256=descriptor.source_sha256,
            processing_sha256=bundle.processing_sha256,
            content_sha256=bundle.content_sha256,
            retrieval_chunk_count=len(bundle.retrieval_ingestion.chunks),
            knowledge_section_count=len(bundle.knowledge_staging.source_tree.sections),
            crosswalk_count=len(bundle.crosswalk),
            artifacts=artifacts,
        )
        (staging / "receipt.json").write_text(
            receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_document_descriptor(path: str | Path) -> GovernedDocumentDescriptor:
    return GovernedDocumentDescriptor.model_validate_json(
        Path(path).read_text(encoding="utf-8-sig")
    )


__all__ = [
    "GovernedDocumentIngestionReceipt",
    "ingest_governed_document_to_directory",
    "load_document_descriptor",
]
