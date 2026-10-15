import csv
import hashlib
from pathlib import Path

from astrograph.knowledge import process_governed_document
from astrograph.retrieval import (
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    SectionFirstDocumentIngestor,
)
from astrograph.retrieval.document_pack_builder import build_fault_document_pack


def test_fault_pack_builds_real_formats_and_stages_docx_for_candidate_kg(tmp_path):
    coverage = tmp_path / "coverage.csv"
    fields = [
        "dataset_id",
        "dataset_fault_class_id",
        "physical_condition_ids",
        "generator_condition_summaries",
        "runtime_fault_id",
        "proposed_runtime_fault_id",
        "component_id",
        "mechanism_id",
        "expected_effect_pattern_ids",
        "evidence_definition_ids",
        "coverage_status",
        "review_status",
    ]
    with coverage.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset_id": "fixture.v1",
                "dataset_fault_class_id": "wheel_friction",
                "physical_condition_ids": '["friction"]',
                "generator_condition_summaries": '[{"description":"Wheel drag.","parameter_names":["coefficient"]}]',
                "runtime_fault_id": "WHEEL_FRICTION",
                "proposed_runtime_fault_id": "WHEEL_FRICTION",
                "component_id": "wheel",
                "mechanism_id": "mechanism:friction",
                "expected_effect_pattern_ids": '["effect:torque-loss"]',
                "evidence_definition_ids": '["ED-1"]',
                "coverage_status": "EXECUTABLE",
                "review_status": "HUMAN_REVIEW_REQUIRED",
            }
        )
    generator = tmp_path / "generator.py"
    generator.write_text("CONDITIONS = []\n", encoding="utf-8")
    output = tmp_path / "pack"

    receipt = build_fault_document_pack(
        coverage_matrix_path=coverage,
        output_root=output,
        generator_paths={"fixture": generator},
    )

    assert receipt["status"] == "DRAFT_REVIEW_REQUIRED"
    assert receipt["fault_class_count"] == 1
    assert receipt["nominal_reference_count"] == 1
    assert receipt["format_coverage"] == [
        "MARKDOWN", "DOCX", "PDF", "CSV", "JSON", "SVG", "LATEX"
    ]
    assert receipt["kg_staging"]["publication_allowed"] is False
    assert receipt["kg_staging"]["runtime_vote_allowed"] is False
    assert (output / "fault_design_fmea_test_handbook.pdf").read_bytes().startswith(b"%PDF-1.4")
    assert (output / "nominal_operation_reference.docx").is_file()
    assert (output / "nominal_operation_reference.pdf").read_bytes().startswith(b"%PDF-1.4")
    elements, _, audit = SectionFirstDocumentIngestor().parse_docx(
        output / "fault_design_fmea_test_handbook.docx"
    )
    assert audit.heading_count == 5
    assert audit.table_count == 1
    assert any(item.kind.value == "TABLE" for item in elements)

    pdf_elements, pdf_warnings, pdf_audit = SectionFirstDocumentIngestor().parse_pdf_layout(
        output / "fault_design_fmea_test_handbook.pdf"
    )
    assert pdf_audit.page_count >= 1
    assert pdf_audit.text_page_count == pdf_audit.page_count
    assert pdf_audit.text_block_count == len(pdf_elements)
    assert all(item.source_locator.startswith("pdf#page:") for item in pdf_elements)
    assert pdf_warnings == pdf_audit.warnings

    pdf_path = output / "fault_design_fmea_test_handbook.pdf"
    processed_pdf = process_governed_document(
        descriptor=GovernedDocumentDescriptor(
            document_id="design:pdf:fixture",
            document_type=DocumentType.DESIGN_PROPOSAL,
            visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
            source_uri=str(pdf_path),
            source_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            review_state=DocumentReviewState.REVIEW_REQUIRED,
            authority=DocumentAuthority.NON_NORMATIVE,
            runtime_vote_allowed=False,
        ),
        source_path=pdf_path,
        independence_group="fixture-pdf-origin",
        license_id="test-fixture",
    )
    assert processed_pdf.knowledge_staging.source_profile.extraction_method == (
        "astrograph-pdf-layout-section-v1"
    )
    assert len(processed_pdf.crosswalk) == len(processed_pdf.retrieval_ingestion.elements)

    second = build_fault_document_pack(
        coverage_matrix_path=coverage,
        output_root=tmp_path / "pack-second",
        generator_paths={"fixture": generator},
    )
    assert second["artifacts"] == receipt["artifacts"]
