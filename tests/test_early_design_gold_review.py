import csv
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.retrieval.early_design_gold_review import (
    build_early_design_gold_review_workbook,
    freeze_early_design_gold_review,
)


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_sha = hashlib.sha256(b"design-doc").hexdigest()
    code = tmp_path / "src" / "example.py"
    code.parent.mkdir(parents=True)
    code.write_text("VALUE = 1\n", encoding="utf-8")
    code_sha = hashlib.sha256(code.read_bytes()).hexdigest()
    inventory = _write(tmp_path / "inventory.json", {
        "descriptor": {"source_sha256": source_sha},
        "structure_matches_expected": True,
        "headings": [{"text": "Design", "level": 1}],
        "expected_structure": {"table_count": 1, "image_count": 1},
        "actual_structure": {"table_count": 1, "image_count": 1},
    })
    epistemic = _write(tmp_path / "epistemic.json", {
        "status": "MACHINE_CANDIDATES_REVIEW_REQUIRED",
        "source_sha256": source_sha,
        "candidate_count": 2,
        "candidates": [
            {
                "candidate_id": f"span:{index}", "element_id": f"element:{index}",
                "source_locator": f"paragraph:{index}",
                "source_span_sha256": hashlib.sha256(f"span-{index}".encode()).hexdigest(),
                "excerpt": f"design statement {index}",
                "suggested_status": "DESIGN_PROPOSAL",
                "trigger_codes": ["DOCUMENT_LEVEL_PROPOSAL_DEFAULT"],
            }
            for index in range(2)
        ],
    })
    drift = _write(tmp_path / "drift.json", {
        "status": "MACHINE_CANDIDATES_HUMAN_REVIEW_REQUIRED",
        "source_document_sha256": source_sha,
        "current_code_sha256": {"src/example.py": code_sha},
        "candidate_count": 1,
        "candidates": [{
            "drift_id": "DRIFT-1", "source_locator": "paragraph:1",
            "source_span_sha256": hashlib.sha256(b"span-1").hexdigest(),
            "topic": "runtime", "disposition": "SUPERSEDED",
            "early_design_summary": "old", "current_summary": "new",
            "current_code_evidence": [{"path": "src/example.py", "sha256": "a" * 64}],
        }],
    })
    template = _write(tmp_path / "template.json", {
        "status": "UNREVIEWED_TEMPLATE", "source_sha256": source_sha,
        "tables": [{
            "source_locator": "word/document.xml#table:1", "semantic_role": "flow",
            "cell_structure_decision": None, "cell_corrections": [],
        }],
        "images": [{
            "asset_uri": "docx://word/media/image1.png", "asset_sha256": "b" * 64,
            "default_epistemic_status": "ILLUSTRATIVE", "region_caption_links": [],
        }],
    })
    return inventory, epistemic, drift, template


def _edit(path: Path, update) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or [])
    for row in rows:
        update(row)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_early_design_gold_workbook_blocks_blank_and_freezes_complete_review(
    tmp_path: Path,
) -> None:
    inventory, epistemic, drift, template = _inputs(tmp_path)
    workbook = tmp_path / "workbook"
    receipt = build_early_design_gold_review_workbook(
        inventory_path=inventory,
        epistemic_candidates_path=epistemic,
        drift_candidates_path=drift,
        review_template_path=template,
        project_root=tmp_path,
        output_dir=workbook,
    )
    assert receipt["structure_item_count"] == 3
    assert receipt["epistemic_item_count"] == 2
    assert receipt["drift_item_count"] == 1
    assert receipt["human_decisions_written"] is False
    for stem in ("structure", "epistemic", "drift"):
        assert (workbook / f"{stem}_review_baseline.csv").read_bytes() == (
            workbook / f"{stem}_review_working_copy.csv"
        ).read_bytes()

    blank = freeze_early_design_gold_review(
        workbook_root=workbook,
        output_path=tmp_path / "blank-freeze.json",
    )
    assert blank["status"] == "BLOCKED"
    assert blank["human_gold_complete"] is False

    _edit(workbook / "structure_review_working_copy.csv", lambda row: row.update({
        "decision": "ACCEPT", "reviewer": "docx-reviewer",
        "review_comment": "Compared with the rendered source.",
    }))
    _edit(workbook / "epistemic_review_working_copy.csv", lambda row: row.update({
        "decision": "ACCEPT", "approved_epistemic_status": row["suggested_status"],
        "applicability": "non-normative early design document",
        "reviewer": "claim-reviewer", "review_comment": "Checked source span and modality.",
    }))
    _edit(workbook / "drift_review_working_copy.csv", lambda row: row.update({
        "decision": "ACCEPT_DISPOSITION", "approved_disposition": row["machine_disposition"],
        "reviewer": "implementation-reviewer", "review_comment": "Compared with current code evidence.",
    }))
    complete = freeze_early_design_gold_review(
        workbook_root=workbook,
        output_path=tmp_path / "complete-freeze.json",
    )
    assert complete["status"] == "COMPLETE"
    assert complete["human_gold_complete"] is True
    assert complete["docx_heading_table_image_gold_complete"] is True
    assert complete["publication_allowed"] is False


def test_early_design_gold_freeze_rejects_machine_field_tampering(tmp_path: Path) -> None:
    inventory, epistemic, drift, template = _inputs(tmp_path)
    workbook = tmp_path / "workbook"
    build_early_design_gold_review_workbook(
        inventory_path=inventory, epistemic_candidates_path=epistemic,
        drift_candidates_path=drift, review_template_path=template,
        project_root=tmp_path, output_dir=workbook,
    )
    _edit(
        workbook / "epistemic_review_working_copy.csv",
        lambda row: row.update({"excerpt": "forged"}),
    )
    with pytest.raises(ValueError, match="machine field changed"):
        freeze_early_design_gold_review(
            workbook_root=workbook,
            output_path=tmp_path / "tampered-freeze.json",
        )
    assert not (tmp_path / "tampered-freeze.json").exists()
