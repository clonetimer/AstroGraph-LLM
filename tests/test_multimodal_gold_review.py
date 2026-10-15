import csv
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.retrieval.multimodal_gold_review import (
    build_multimodal_gold_review_workbook,
    freeze_multimodal_gold_review,
)


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "fixture.docx"
    source.write_bytes(b"docx-fixture")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate = _write(tmp_path / "candidate.json", {
        "review_state": "MACHINE_GOLD_CANDIDATE_HUMAN_REVIEW_REQUIRED",
        "expected_formula_text": "z=x", "parsed_formula_match": True,
        "expected_table_text": "|a|b|", "parsed_table_match": True,
        "expected_image_caption": "flow", "expected_image_sha256": "a" * 64,
        "parsed_image_match": True,
    })
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    receipt = _write(tmp_path / "build.json", {
        "status": "MACHINE_VALIDATED_HUMAN_GOLD_PENDING",
        "source_sha256": source_sha,
        "machine_gold_candidate_sha256": candidate_sha,
        "publication_allowed": False, "runtime_vote_allowed": False,
    })
    template = _write(tmp_path / "template.json", {
        "review_state": "UNREVIEWED_TEMPLATE", "source_sha256": source_sha,
        "decisions": [
            {"item_id": item_id, "question": f"Review {item_id}"}
            for item_id in ("FORMULA-001", "TABLE-001", "IMAGE-001", "EPISTEMIC-001")
        ],
    })
    return source, receipt, candidate, template


def _edit(path: Path, update) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or [])
    for row in rows:
        update(row)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def test_multimodal_gold_review_blocks_blank_and_freezes_named_approval(tmp_path: Path) -> None:
    source, receipt, candidate, template = _inputs(tmp_path)
    workbook = tmp_path / "workbook"
    built = build_multimodal_gold_review_workbook(
        source_path=source, build_receipt_path=receipt,
        machine_candidate_path=candidate, review_template_path=template,
        output_dir=workbook,
    )
    assert built["review_item_count"] == 4
    assert built["human_decisions_written"] is False
    assert (workbook / "multimodal_review_baseline.csv").read_bytes() == (
        workbook / "multimodal_review_working_copy.csv"
    ).read_bytes()
    blank = freeze_multimodal_gold_review(
        source_path=source, build_receipt_path=receipt,
        machine_candidate_path=candidate, review_template_path=template,
        workbook_root=workbook, output_path=tmp_path / "blank.json",
    )
    assert blank["status"] == "BLOCKED"
    assert blank["human_gold_complete"] is False

    _edit(workbook / "multimodal_review_working_copy.csv", lambda row: row.update({
        "decision": "APPROVE", "reviewer": "multimodal-reviewer",
        "review_comment": "Compared with the rendered DOCX fixture.",
    }))
    complete = freeze_multimodal_gold_review(
        source_path=source, build_receipt_path=receipt,
        machine_candidate_path=candidate, review_template_path=template,
        workbook_root=workbook, output_path=tmp_path / "complete.json",
    )
    assert complete["status"] == "COMPLETE"
    assert complete["human_gold_complete"] is True
    assert complete["formula_gold_complete"] is True
    assert complete["publication_allowed"] is False


def test_multimodal_gold_review_rejects_machine_field_tampering(tmp_path: Path) -> None:
    source, receipt, candidate, template = _inputs(tmp_path)
    workbook = tmp_path / "workbook"
    build_multimodal_gold_review_workbook(
        source_path=source, build_receipt_path=receipt,
        machine_candidate_path=candidate, review_template_path=template,
        output_dir=workbook,
    )
    _edit(
        workbook / "multimodal_review_working_copy.csv",
        lambda row: row.update({"question": "forged"}),
    )
    with pytest.raises(ValueError, match="machine field changed"):
        freeze_multimodal_gold_review(
            source_path=source, build_receipt_path=receipt,
            machine_candidate_path=candidate, review_template_path=template,
            workbook_root=workbook, output_path=tmp_path / "tampered.json",
        )
    assert not (tmp_path / "tampered.json").exists()
