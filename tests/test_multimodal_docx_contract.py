from pathlib import Path

from astrograph.retrieval.multimodal_contract_builder import build_multimodal_docx_contract


def test_multimodal_docx_contract_is_deterministic_and_non_publishing(tmp_path: Path) -> None:
    first = build_multimodal_docx_contract(output_root=tmp_path / "first")
    second = build_multimodal_docx_contract(output_root=tmp_path / "second")
    assert first == second
    assert first["status"] == "MACHINE_VALIDATED_HUMAN_GOLD_PENDING"
    assert first["modality_counts"] == {
        "TEXT": 1,
        "TABLE": 1,
        "FORMULA": 1,
        "IMAGE": 1,
    }
    assert first["crosswalk_complete"] is True
    assert first["formula_text_preserved"] is True
    assert first["table_cells_preserved_as_rows"] is True
    assert first["image_hash_preserved"] is True
    assert first["human_gold_complete"] is False
    assert first["publication_allowed"] is False
    assert first["runtime_vote_allowed"] is False
