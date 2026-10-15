"""Deterministic review-gate fixtures for ambiguous document structure."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.knowledge.construction import process_governed_document

from .document_governance import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)
from .document_pack_builder import _write_pdf
from .multimodal_contract_builder import _write_stable_docx


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path, *, document_id: str) -> GovernedDocumentDescriptor:
    return GovernedDocumentDescriptor(
        document_id=document_id,
        document_type=DocumentType.DESIGN_MANUAL,
        visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
        source_uri=f"corpus://structure-stress/{path.name}",
        source_sha256=_sha(path),
        # The source document itself is reviewed.  The test therefore isolates
        # representation-level gates instead of relying on a document-level block.
        review_state=DocumentReviewState.HUMAN_REVIEWED,
        authority=DocumentAuthority.ENGINEERING_REFERENCE,
        synthetic_from_simulation=False,
        runtime_vote_allowed=False,
    )


def _unsafe_summary(processed: Any) -> dict[str, Any]:
    elements = processed.retrieval_ingestion.elements
    unsafe = [item for item in elements if not item.representation_reviewed]
    unsafe_locators = {item.source_locator for item in unsafe}
    unsafe_chunks = [
        chunk
        for chunk in processed.retrieval_ingestion.chunks
        if unsafe_locators.intersection(chunk.source_locators)
    ]
    staging_text = "\n".join(
        fragment.text for fragment in processed.knowledge_staging.source_profile.fragments
    )
    return {
        "element_count": len(elements),
        "unsafe_element_count": len(unsafe),
        "unsafe_element_kinds": sorted({item.kind.value for item in unsafe}),
        "unsafe_source_locators": sorted(unsafe_locators),
        "unsafe_chunk_count": len(unsafe_chunks),
        "all_unsafe_chunks_review_gated": all(
            chunk.review_status == "MACHINE_VALIDATED"
            and chunk.content_safety_status == "REVIEW_REQUIRED"
            for chunk in unsafe_chunks
        ),
        "kg_withheld_marker_count": staging_text.count(
            "representation withheld pending human review"
        ),
        "raw_unsafe_representation_absent_from_kg_staging": all(
            item.text_representation not in staging_text for item in unsafe
        ),
        "crosswalk_complete": len(processed.crosswalk) == len(elements),
        "publication_allowed": processed.publication_allowed,
        "runtime_vote_allowed": processed.runtime_vote_allowed,
    }


def build_document_structure_stress_pack(*, output_root: str | Path) -> dict[str, Any]:
    """Build missing-caption and cross-page-table review-gate evidence."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite nonempty structure stress directory")
    output.mkdir(parents=True, exist_ok=True)

    missing_caption = output / "missing_caption_image.docx"
    _write_stable_docx(missing_caption, image_caption=None)
    missing_processed = process_governed_document(
        descriptor=_descriptor(
            missing_caption,
            document_id="astrograph:structure-stress:missing-caption:r001",
        ),
        source_path=missing_caption,
        independence_group="astrograph-structure-stress-r001",
        license_id="project-generated",
    )

    cross_page = output / "cross_page_table.pdf"
    table_lines = ["Channel    Unit    Status"] + [
        f"RW_CURRENT_ROW_{index:03d}    A    candidate" for index in range(1, 96)
    ]
    _write_pdf(cross_page, "\n".join(table_lines))
    cross_processed = process_governed_document(
        descriptor=_descriptor(
            cross_page,
            document_id="astrograph:structure-stress:cross-page-table:r001",
        ),
        source_path=cross_page,
        independence_group="astrograph-structure-stress-r001",
        license_id="project-generated",
    )

    missing_summary = _unsafe_summary(missing_processed)
    cross_summary = _unsafe_summary(cross_processed)
    if not (
        missing_summary["all_unsafe_chunks_review_gated"]
        and missing_summary["raw_unsafe_representation_absent_from_kg_staging"]
        and cross_summary["all_unsafe_chunks_review_gated"]
        and cross_summary["raw_unsafe_representation_absent_from_kg_staging"]
    ):
        raise ValueError("ambiguous structure did not fail closed")

    review_path = output / "human_review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "review_id", "document_id", "issue_kind", "source_locators",
                "decision", "reviewer", "comment",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "review_id": "STRUCTURE-REVIEW-MISSING-CAPTION-R001",
            "document_id": "astrograph:structure-stress:missing-caption:r001",
            "issue_kind": "IMAGE_REPRESENTATION_REVIEW_REQUIRED",
            "source_locators": json.dumps(
                missing_summary["unsafe_source_locators"], ensure_ascii=False
            ),
            "decision": "", "reviewer": "", "comment": "",
        })
        writer.writerow({
            "review_id": "STRUCTURE-REVIEW-CROSS-PAGE-TABLE-R001",
            "document_id": "astrograph:structure-stress:cross-page-table:r001",
            "issue_kind": "PDF_TABLE_STRUCTURE_REVIEW_REQUIRED",
            "source_locators": json.dumps(
                cross_summary["unsafe_source_locators"], ensure_ascii=False
            ),
            "decision": "", "reviewer": "", "comment": "",
        })

    body = {
        "schema_version": "astrograph-document-structure-stress-receipt.v1",
        "status": "MACHINE_VALIDATED_HUMAN_REVIEW_REQUIRED",
        "fixtures": {
            "missing_caption_image": {
                "source_sha256": _sha(missing_caption), **missing_summary,
            },
            "cross_page_table": {
                "source_sha256": _sha(cross_page), **cross_summary,
            },
        },
        "human_review_sha256": _sha(review_path),
        "ambiguous_representation_rag_admission_allowed": False,
        "ambiguous_representation_kg_extraction_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "The raw representations are retained for review, but ambiguous image/table "
            "representations are not retrievable and their content is withheld from KG staging. "
            "The fixture does not establish OCR, caption or cross-page table reconstruction quality."
        ),
    }
    receipt_path = output / "build_receipt.json"
    receipt_path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return body


__all__ = ["build_document_structure_stress_pack"]
