"""Executable relocation audit for governed document-to-graph identities."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from astrograph.retrieval.document_governance import GovernedDocumentDescriptor

from .construction import (
    knowledge_source_profile_set_sha256,
    process_governed_document,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def audit_document_identity_relocation(
    *,
    source_path: str | Path,
    descriptor: GovernedDocumentDescriptor,
    independence_group: str,
    license_id: str,
) -> dict[str, Any]:
    """Copy one real document twice and verify portable candidate-KG identities."""

    source_path = Path(source_path).resolve()
    if not source_path.is_file() or _sha(source_path) != descriptor.source_sha256:
        raise ValueError("relocation audit source is missing or differs from descriptor")
    with tempfile.TemporaryDirectory(prefix="astrograph-document-relocation-") as temp:
        root = Path(temp)
        copies = [root / "machine-a" / source_path.name, root / "machine-b" / "docs" / source_path.name]
        processed = []
        for copy in copies:
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, copy)
            local_descriptor = descriptor.model_copy(update={"source_uri": str(copy)})
            processed.append(process_governed_document(
                descriptor=local_descriptor,
                source_path=copy,
                independence_group=independence_group,
                license_id=license_id,
                knowledge_version="document-relocation-audit-v1",
            ))
        left, right = processed
        left_staging, right_staging = left.knowledge_staging, right.knowledge_staging
        comparisons = {
            "source_file_sha256_equal": (
                left_staging.source_profile.source_sha256
                == right_staging.source_profile.source_sha256
            ),
            "profile_content_identity_equal": (
                knowledge_source_profile_set_sha256([left_staging.source_profile])
                == knowledge_source_profile_set_sha256([right_staging.source_profile])
            ),
            "source_tree_sha256_equal": (
                left_staging.source_tree.tree_sha256
                == right_staging.source_tree.tree_sha256
            ),
            "staging_sha256_equal": (
                left_staging.staging_sha256 == right_staging.staging_sha256
            ),
            "processing_content_sha256_equal": (
                left.content_sha256 is not None
                and left.content_sha256 == right.content_sha256
            ),
            "processing_artifact_sha256_different": (
                left.processing_sha256 != right.processing_sha256
            ),
            "source_paths_different": (
                left_staging.source_profile.source_path
                != right_staging.source_profile.source_path
            ),
            "native_ingestion_sha256_different": (
                left.retrieval_ingestion.ingestion_sha256
                != right.retrieval_ingestion.ingestion_sha256
            ),
        }
        portable_keys = [
            "source_file_sha256_equal",
            "profile_content_identity_equal",
            "source_tree_sha256_equal",
            "staging_sha256_equal",
            "processing_content_sha256_equal",
            "source_paths_different",
        ]
        status = "PASS" if all(comparisons[key] for key in portable_keys) else "BLOCKED"
        body = {
            "schema_version": "astrograph-document-identity-relocation-audit.v1",
            "status": status,
            "source_document_id": descriptor.document_id,
            "source_sha256": descriptor.source_sha256,
            "comparison_count": 2,
            "comparisons": comparisons,
            "portable_identity": {
                "profile_content_sha256": knowledge_source_profile_set_sha256(
                    [left_staging.source_profile]
                ),
                "source_tree_sha256": left_staging.source_tree.tree_sha256,
                "staging_sha256": left_staging.staging_sha256,
                "processing_content_sha256": left.content_sha256,
            },
            "local_locator_policy": (
                "source_path, source_uri and native retrieval ingestion SHA remain machine-local "
                "audit locators; they are excluded from candidate-KG content identity."
            ),
            "publication_allowed": False,
            "runtime_vote_allowed": False,
        }
        return {**body, "audit_sha256": _canonical_sha(body)}


__all__ = ["audit_document_identity_relocation"]
