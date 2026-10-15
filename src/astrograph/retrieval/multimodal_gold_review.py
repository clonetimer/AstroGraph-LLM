"""Named human-gold review for the deterministic DOCX multimodal contract."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from astrograph.filesystem import publish_staged_directory


FIELDS = [
    "review_item_id", "question", "source_sha256", "machine_candidate_sha256",
    "machine_evidence_json", "decision", "correction_json", "reviewer", "review_comment",
]
HUMAN_FIELDS = {"decision", "correction_json", "reviewer", "review_comment"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != FIELDS:
            raise ValueError("multimodal gold review columns changed")
        return list(reader)


def _machine_rows(
    *, source_path: Path, receipt_path: Path, candidate_path: Path, template_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    source_sha = _sha(source_path)
    if (
        receipt.get("status") != "MACHINE_VALIDATED_HUMAN_GOLD_PENDING"
        or receipt.get("source_sha256") != source_sha
        or receipt.get("machine_gold_candidate_sha256") != _sha(candidate_path)
        or template.get("review_state") != "UNREVIEWED_TEMPLATE"
        or template.get("source_sha256") != source_sha
        or candidate.get("review_state") != "MACHINE_GOLD_CANDIDATE_HUMAN_REVIEW_REQUIRED"
    ):
        raise ValueError("multimodal gold inputs are stale, inconsistent or already reviewed")
    evidence = {
        "FORMULA-001": {
            "expected_formula_text": candidate.get("expected_formula_text"),
            "parsed_formula_match": candidate.get("parsed_formula_match"),
        },
        "TABLE-001": {
            "expected_table_text": candidate.get("expected_table_text"),
            "parsed_table_match": candidate.get("parsed_table_match"),
        },
        "IMAGE-001": {
            "expected_image_caption": candidate.get("expected_image_caption"),
            "expected_image_sha256": candidate.get("expected_image_sha256"),
            "parsed_image_match": candidate.get("parsed_image_match"),
        },
        "EPISTEMIC-001": {
            "unreviewed_nontext_withheld": True,
            "publication_allowed": receipt.get("publication_allowed"),
            "runtime_vote_allowed": receipt.get("runtime_vote_allowed"),
        },
    }
    template_items = template.get("decisions", [])
    if {str(item.get("item_id")) for item in template_items} != set(evidence):
        raise ValueError("multimodal gold template item set differs from the contract")
    candidate_sha = _sha(candidate_path)
    rows = [{
        "review_item_id": str(item["item_id"]),
        "question": str(item["question"]),
        "source_sha256": source_sha,
        "machine_candidate_sha256": candidate_sha,
        "machine_evidence_json": json.dumps(evidence[str(item["item_id"])], ensure_ascii=False, separators=(",", ":")),
        "decision": "", "correction_json": "", "reviewer": "", "review_comment": "",
    } for item in template_items]
    return rows, receipt


def build_multimodal_gold_review_workbook(
    *, source_path: str | Path, build_receipt_path: str | Path,
    machine_candidate_path: str | Path, review_template_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create an immutable baseline plus editable copy for four modalities/policies."""

    source = Path(source_path).resolve()
    receipt_path = Path(build_receipt_path).resolve()
    candidate_path = Path(machine_candidate_path).resolve()
    template_path = Path(review_template_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite multimodal gold review workbook")
    rows, _ = _machine_rows(
        source_path=source, receipt_path=receipt_path,
        candidate_path=candidate_path, template_path=template_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        baseline = staging / "multimodal_review_baseline.csv"
        working = staging / "multimodal_review_working_copy.csv"
        with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
        shutil.copyfile(baseline, working)
        body = {
            "schema_version": "astrograph-multimodal-gold-review-workbook.v1",
            "status": "READY_FOR_NAMED_HUMAN_GOLD_REVIEW",
            "source_file_sha256": _sha(source),
            "build_receipt_file_sha256": _sha(receipt_path),
            "machine_candidate_file_sha256": _sha(candidate_path),
            "review_template_file_sha256": _sha(template_path),
            "review_item_count": len(rows),
            "baseline_file_sha256": _sha(baseline),
            "working_copy_initial_file_sha256": _sha(working),
            "human_decisions_written": False,
            "human_gold_complete": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "The workbook exposes formula, table, image and epistemic-policy machine evidence "
                "for named review. Machine matches do not constitute human gold."
            ),
        }
        result = {**body, "receipt_sha256": _canonical_sha(body)}
        (staging / "receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging, output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def freeze_multimodal_gold_review(
    *, source_path: str | Path, build_receipt_path: str | Path,
    machine_candidate_path: str | Path, review_template_path: str | Path,
    workbook_root: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Reconstruct machine fields and freeze complete named multimodal judgments."""

    source = Path(source_path).resolve()
    receipt_path = Path(build_receipt_path).resolve()
    candidate_path = Path(machine_candidate_path).resolve()
    template_path = Path(review_template_path).resolve()
    root = Path(workbook_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite multimodal gold review freeze")
    workbook_receipt_path = root / "receipt.json"
    workbook_receipt = json.loads(workbook_receipt_path.read_text(encoding="utf-8"))
    workbook_body = {key: value for key, value in workbook_receipt.items() if key != "receipt_sha256"}
    if _canonical_sha(workbook_body) != workbook_receipt.get("receipt_sha256"):
        raise ValueError("multimodal gold workbook semantic hash mismatch")
    baseline_path = root / "multimodal_review_baseline.csv"
    working_path = root / "multimodal_review_working_copy.csv"
    if (
        workbook_receipt.get("source_file_sha256") != _sha(source)
        or workbook_receipt.get("build_receipt_file_sha256") != _sha(receipt_path)
        or workbook_receipt.get("machine_candidate_file_sha256") != _sha(candidate_path)
        or workbook_receipt.get("review_template_file_sha256") != _sha(template_path)
        or workbook_receipt.get("baseline_file_sha256") != _sha(baseline_path)
    ):
        raise ValueError("multimodal gold workbook is not bound to current inputs")
    expected, _ = _machine_rows(
        source_path=source, receipt_path=receipt_path,
        candidate_path=candidate_path, template_path=template_path,
    )
    baseline = _read(baseline_path)
    edited = _read(working_path)
    if baseline != expected:
        raise ValueError("multimodal gold baseline differs from source reconstruction")
    base = {row["review_item_id"]: row for row in baseline}
    if len(edited) != len(base) or {row["review_item_id"] for row in edited} != set(base):
        raise ValueError("multimodal gold review identity set changed")
    for row in edited:
        for field in FIELDS:
            if field not in HUMAN_FIELDS and row[field] != base[row["review_item_id"]][field]:
                raise ValueError(f"multimodal gold machine field changed: {row['review_item_id']}:{field}")

    blockers: list[str] = []
    rejected: list[str] = []
    decision_counts: dict[str, int] = {}
    for row in edited:
        item_id = row["review_item_id"]
        decision = row["decision"].strip().upper()
        if decision not in {"APPROVE", "CORRECT", "REJECT"}:
            blockers.append(f"{item_id}:DECISION_REQUIRED")
        else:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append(f"{item_id}:REVIEWER_COMMENT_REQUIRED")
        correction = row["correction_json"].strip()
        if decision == "CORRECT" and not correction:
            blockers.append(f"{item_id}:CORRECTION_REQUIRED")
        if decision != "CORRECT" and correction:
            blockers.append(f"{item_id}:CORRECTION_NOT_ALLOWED")
        if correction:
            try:
                json.loads(correction)
            except json.JSONDecodeError:
                blockers.append(f"{item_id}:INVALID_CORRECTION_JSON")
        if decision == "REJECT":
            rejected.append(item_id)
    status = "BLOCKED" if blockers else ("COMPLETE_PARSER_FAILURE" if rejected else "COMPLETE")
    body = {
        "schema_version": "astrograph-multimodal-gold-review-freeze.v1",
        "status": status,
        "source_sha256": _sha(source),
        "workbook_receipt_file_sha256": _sha(workbook_receipt_path),
        "working_copy_file_sha256": _sha(working_path),
        "decision_counts": dict(sorted(decision_counts.items())),
        "rejected_item_ids": rejected,
        "blockers": blockers,
        "human_gold_complete": status == "COMPLETE",
        "formula_gold_complete": status == "COMPLETE",
        "table_gold_complete": status == "COMPLETE",
        "image_gold_complete": status == "COMPLETE",
        "epistemic_policy_gold_complete": status == "COMPLETE",
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "COMPLETE establishes human gold for this deterministic parser fixture only. It is "
            "not OCR gold, document-claim approval, KG publication or runtime evidence admission."
        ),
    }
    result = {**body, "freeze_sha256": _canonical_sha(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["build_multimodal_gold_review_workbook", "freeze_multimodal_gold_review"]
