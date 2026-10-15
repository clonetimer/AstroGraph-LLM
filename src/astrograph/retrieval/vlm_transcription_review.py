"""Human review handoff and freeze for VLM scan transcription candidates."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .structured_document import SectionFirstDocumentIngestor


REVIEW_FIELDS = [
    "review_item_id", "source_pdf_sha256", "scan_image_sha256", "source_page",
    "candidate_sha256", "candidate_text_sha256", "candidate_text",
    "machine_reference_text_sha256", "machine_reference_text",
    "decision", "corrected_text", "reviewer", "review_comment",
]
HUMAN_FIELDS = {"decision", "corrected_text", "reviewer", "review_comment"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _machine_row(*, transcription_root: Path, source_pdf_path: Path) -> dict[str, str]:
    receipt = json.loads((transcription_root / "receipt.json").read_text(encoding="utf-8"))
    candidate_path = transcription_root / "transcription_candidate.json"
    if _sha(candidate_path) != receipt["candidate_file_sha256"]:
        raise ValueError("VLM transcription candidate hash mismatch")
    if _sha(source_pdf_path) != receipt["source_pdf_sha256"]:
        raise ValueError("VLM review source PDF hash mismatch")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    transcription = "\n".join(candidate["transcription"]["lines"])
    elements, _, _ = SectionFirstDocumentIngestor().parse_pdf_layout(source_pdf_path)
    prefix = f"pdf#page:{int(receipt['source_page'])}:"
    reference = "\n".join(
        item.text_representation for item in elements if item.source_locator.startswith(prefix)
    )
    if not reference:
        raise ValueError("VLM review machine reference is empty")
    review_id = "VLM-REVIEW-" + _canonical_sha({
        "candidate_sha256": candidate["candidate_sha256"],
        "source_pdf_sha256": receipt["source_pdf_sha256"],
        "source_page": receipt["source_page"],
    })[:20].upper()
    return {
        "review_item_id": review_id,
        "source_pdf_sha256": receipt["source_pdf_sha256"],
        "scan_image_sha256": receipt["scan_image_sha256"],
        "source_page": str(receipt["source_page"]),
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_text_sha256": _sha_text(transcription),
        "candidate_text": transcription,
        "machine_reference_text_sha256": _sha_text(reference),
        "machine_reference_text": reference,
        "decision": "", "corrected_text": "", "reviewer": "", "review_comment": "",
    }


def build_vlm_transcription_review_handoff(
    *, transcription_root: str | Path, source_pdf_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    transcription_root, source_pdf_path = Path(transcription_root).resolve(), Path(source_pdf_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty VLM review handoff")
    output_root.mkdir(parents=True, exist_ok=True)
    row = _machine_row(transcription_root=transcription_root, source_pdf_path=source_pdf_path)
    sheet = output_root / "human_review.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS); writer.writeheader(); writer.writerow(row)
    body = {
        "schema_version": "astrograph-vlm-transcription-review-handoff.v1",
        "status": "UNREVIEWED",
        "review_item_id": row["review_item_id"],
        "transcription_receipt_sha256": _sha(transcription_root / "receipt.json"),
        "review_sheet_sha256": _sha(sheet),
        "human_decision_count": 0,
        "rag_admission_allowed": False,
        "kg_staging_allowed": False,
        "runtime_vote_allowed": False,
        "claim_boundary": (
            "Blank VLM transcription review handoff. Machine-reference similarity cannot replace "
            "a named human decision and no representation is admitted automatically."
        ),
    }
    result = {**body, "handoff_sha256": _canonical_sha(body)}
    (output_root / "handoff_receipt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def freeze_vlm_transcription_review(
    *, transcription_root: str | Path, source_pdf_path: str | Path,
    review_sheet_path: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    transcription_root, source_pdf_path = Path(transcription_root).resolve(), Path(source_pdf_path).resolve()
    review_sheet_path, output_root = Path(review_sheet_path).resolve(), Path(output_root).resolve()
    with review_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != REVIEW_FIELDS:
            raise ValueError("VLM review sheet columns changed")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError("VLM review sheet must contain exactly one row")
    actual = rows[0]
    expected = _machine_row(transcription_root=transcription_root, source_pdf_path=source_pdf_path)
    changed = [field for field in REVIEW_FIELDS if field not in HUMAN_FIELDS and actual[field] != expected[field]]
    blockers: list[str] = []
    if changed:
        blockers.append("MACHINE_FIELDS_CHANGED:" + ";".join(changed))
    decision = actual["decision"].strip().upper()
    if decision not in {"APPROVE_AS_IS", "CORRECT_AND_APPROVE", "REJECT"}:
        blockers.append("VALID_DECISION_REQUIRED")
    if not actual["reviewer"].strip() or not actual["review_comment"].strip():
        blockers.append("NAMED_REVIEWER_AND_COMMENT_REQUIRED")
    corrected = actual["corrected_text"].strip()
    if decision == "CORRECT_AND_APPROVE" and not corrected:
        blockers.append("CORRECTED_TEXT_REQUIRED")
    if decision != "CORRECT_AND_APPROVE" and corrected:
        blockers.append("CORRECTED_TEXT_NOT_ALLOWED_FOR_DECISION")
    approved = not blockers and decision in {"APPROVE_AS_IS", "CORRECT_AND_APPROVE"}
    reviewed_text = corrected if decision == "CORRECT_AND_APPROVE" else actual["candidate_text"]
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty VLM review freeze")
    output_root.mkdir(parents=True, exist_ok=True)
    representation_path = None
    if approved:
        representation_path = output_root / "reviewed_transcription.txt"
        representation_path.write_text(reviewed_text.rstrip("\n") + "\n", encoding="utf-8")
    body = {
        "schema_version": "astrograph-vlm-transcription-review-freeze.v1",
        "status": "APPROVED_REPRESENTATION" if approved else "REJECTED" if not blockers else "BLOCKED",
        "review_item_id": expected["review_item_id"],
        "review_sheet_sha256": _sha(review_sheet_path),
        "decision": decision or None,
        "reviewer": actual["reviewer"].strip() or None,
        "review_comment": actual["review_comment"].strip() or None,
        "source_pdf_sha256": expected["source_pdf_sha256"],
        "scan_image_sha256": expected["scan_image_sha256"],
        "candidate_sha256": expected["candidate_sha256"],
        "reviewed_text_sha256": _sha(representation_path) if representation_path else None,
        "blockers": blockers,
        "representation_reviewed": approved,
        "rag_admission_allowed": approved,
        "kg_staging_allowed": approved,
        "runtime_vote_allowed": False,
        "publication_allowed": False,
        "claim_boundary": (
            "Human-reviewed transcription representation only. Approval permits a separate governed "
            "document ingestion step; it does not itself index RAG, stage KG, publish or vote."
        ),
    }
    result = {**body, "freeze_sha256": _canonical_sha(body)}
    (output_root / "freeze_receipt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["build_vlm_transcription_review_handoff", "freeze_vlm_transcription_review"]
