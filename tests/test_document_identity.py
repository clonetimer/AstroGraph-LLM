import hashlib

from astrograph.knowledge.document_identity import audit_document_identity_relocation
from astrograph.retrieval.document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)


def test_relocation_audit_separates_content_identity_from_local_locator(tmp_path) -> None:
    source = tmp_path / "manual.md"
    source.write_text("# Manual\n\n## Evidence\n\nObserved current rise.\n", encoding="utf-8")
    descriptor = GovernedDocumentDescriptor(
        document_id="doc:relocation:test",
        document_type=DocumentType.DESIGN_PROPOSAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state=DocumentReviewState.REVIEW_REQUIRED,
        authority=DocumentAuthority.NON_NORMATIVE,
        runtime_vote_allowed=False,
    )
    audit = audit_document_identity_relocation(
        source_path=source,
        descriptor=descriptor,
        independence_group="fixture",
        license_id="test",
    )
    assert audit["status"] == "PASS"
    assert all(audit["comparisons"][key] for key in (
        "source_file_sha256_equal", "profile_content_identity_equal",
        "source_tree_sha256_equal", "staging_sha256_equal", "source_paths_different",
        "processing_content_sha256_equal",
    ))
    assert audit["comparisons"]["native_ingestion_sha256_different"] is True
    assert audit["comparisons"]["processing_artifact_sha256_different"] is True
