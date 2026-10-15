from pathlib import Path

from astrograph.retrieval.document_structure_stress import (
    build_document_structure_stress_pack,
)


def test_ambiguous_document_structures_are_review_gated_and_kg_redacted(
    tmp_path: Path,
) -> None:
    first = build_document_structure_stress_pack(output_root=tmp_path / "first")
    second = build_document_structure_stress_pack(output_root=tmp_path / "second")
    assert first == second
    assert first["status"] == "MACHINE_VALIDATED_HUMAN_REVIEW_REQUIRED"
    missing = first["fixtures"]["missing_caption_image"]
    cross_page = first["fixtures"]["cross_page_table"]
    assert missing["unsafe_element_kinds"] == ["FORMULA", "IMAGE"]
    assert missing["unsafe_element_count"] == 2
    assert cross_page["unsafe_element_kinds"] == ["TABLE"]
    assert cross_page["unsafe_element_count"] == 2
    assert cross_page["unsafe_chunk_count"] == 2
    for item in (missing, cross_page):
        assert item["all_unsafe_chunks_review_gated"] is True
        assert item["raw_unsafe_representation_absent_from_kg_staging"] is True
        assert item["kg_withheld_marker_count"] == item["unsafe_element_count"]
        assert item["crosswalk_complete"] is True
        assert item["publication_allowed"] is False
        assert item["runtime_vote_allowed"] is False
    assert first["ambiguous_representation_rag_admission_allowed"] is False
    assert first["ambiguous_representation_kg_extraction_allowed"] is False
    assert first["active_graph_modified"] is False
    rows = (tmp_path / "first/human_review.csv").read_text(encoding="utf-8")
    assert "IMAGE_REPRESENTATION_REVIEW_REQUIRED" in rows
    assert "PDF_TABLE_STRUCTURE_REVIEW_REQUIRED" in rows
