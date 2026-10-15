"""Fail-closed human-gold handoff for AstroGraph's first real design DOCX."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from astrograph.filesystem import publish_staged_directory


STRUCTURE_FIELDS = [
    "review_item_id", "item_kind", "source_locator", "source_sha256",
    "machine_summary_json", "decision", "correction_json", "reviewer", "review_comment",
]
EPISTEMIC_FIELDS = [
    "candidate_id", "element_id", "source_locator", "source_span_sha256", "excerpt",
    "suggested_status", "trigger_codes_json", "decision", "approved_epistemic_status",
    "applicability", "reviewer", "review_comment",
]
DRIFT_FIELDS = [
    "drift_id", "source_locator", "source_span_sha256", "topic", "machine_disposition",
    "early_design_summary", "current_summary", "current_code_evidence_json", "decision",
    "approved_disposition", "reviewer", "review_comment",
]
STRUCTURE_HUMAN_FIELDS = {"decision", "correction_json", "reviewer", "review_comment"}
EPISTEMIC_HUMAN_FIELDS = {
    "decision", "approved_epistemic_status", "applicability", "reviewer", "review_comment"
}
DRIFT_HUMAN_FIELDS = {"decision", "approved_disposition", "reviewer", "review_comment"}
ALLOWED_EPISTEMIC_STATUSES = {
    "SOURCE_FACT", "DESIGN_PROPOSAL", "ENGINEERING_ESTIMATE", "MODEL_INFERENCE",
    "ONTOLOGY_ASSOCIATION", "ILLUSTRATIVE",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != fields:
            raise ValueError(f"early design gold columns changed: {path.name}")
        return list(reader)


def _load_inputs(
    inventory_path: Path,
    epistemic_candidates_path: Path,
    drift_candidates_path: Path,
    review_template_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    epistemic = json.loads(epistemic_candidates_path.read_text(encoding="utf-8"))
    drift = json.loads(drift_candidates_path.read_text(encoding="utf-8"))
    template = json.loads(review_template_path.read_text(encoding="utf-8"))
    source_sha = inventory.get("descriptor", {}).get("source_sha256")
    if not source_sha or any(
        item.get(key) != source_sha
        for item, key in (
            (epistemic, "source_sha256"),
            (drift, "source_document_sha256"),
            (template, "source_sha256"),
        )
    ):
        raise ValueError("early design gold sources do not share one source SHA-256")
    if inventory.get("structure_matches_expected") is not True:
        raise ValueError("early design machine inventory does not match its frozen structure")
    if epistemic.get("status") != "MACHINE_CANDIDATES_REVIEW_REQUIRED":
        raise ValueError("early design epistemic candidates have an unexpected state")
    if drift.get("status") != "MACHINE_CANDIDATES_HUMAN_REVIEW_REQUIRED":
        raise ValueError("early design drift candidates have an unexpected state")
    if template.get("status") != "UNREVIEWED_TEMPLATE":
        raise ValueError("early design gold input must be the unreviewed template")
    return inventory, epistemic, drift, template


def _rows(
    inventory: dict[str, Any],
    epistemic: dict[str, Any],
    drift: dict[str, Any],
    template: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    source_sha = str(template["source_sha256"])
    structure: list[dict[str, str]] = [{
        "review_item_id": "HEADING-HIERARCHY",
        "item_kind": "HEADING_HIERARCHY",
        "source_locator": "word/document.xml",
        "source_sha256": source_sha,
        "machine_summary_json": json.dumps({
            "headings": inventory.get("headings", []),
            "expected_structure": inventory.get("expected_structure", {}),
            "actual_structure": inventory.get("actual_structure", {}),
        }, ensure_ascii=False, separators=(",", ":")),
        "decision": "", "correction_json": "", "reviewer": "", "review_comment": "",
    }]
    for index, table in enumerate(template.get("tables", []), start=1):
        structure.append({
            "review_item_id": f"TABLE-{index:03d}", "item_kind": "TABLE",
            "source_locator": str(table["source_locator"]), "source_sha256": source_sha,
            "machine_summary_json": json.dumps({
                "semantic_role": table.get("semantic_role"),
                "cell_structure_decision": table.get("cell_structure_decision"),
                "cell_corrections": table.get("cell_corrections", []),
            }, ensure_ascii=False, separators=(",", ":")),
            "decision": "", "correction_json": "", "reviewer": "", "review_comment": "",
        })
    for index, image in enumerate(template.get("images", []), start=1):
        structure.append({
            "review_item_id": f"IMAGE-{index:03d}", "item_kind": "IMAGE",
            "source_locator": str(image["asset_uri"]), "source_sha256": source_sha,
            "machine_summary_json": json.dumps({
                "asset_sha256": image.get("asset_sha256"),
                "default_epistemic_status": image.get("default_epistemic_status"),
                "region_caption_links": image.get("region_caption_links", []),
            }, ensure_ascii=False, separators=(",", ":")),
            "decision": "", "correction_json": "", "reviewer": "", "review_comment": "",
        })
    epistemic_rows = [{
        "candidate_id": str(item["candidate_id"]),
        "element_id": str(item["element_id"]),
        "source_locator": str(item["source_locator"]),
        "source_span_sha256": str(item["source_span_sha256"]),
        "excerpt": str(item["excerpt"]),
        "suggested_status": str(item["suggested_status"]),
        "trigger_codes_json": json.dumps(item.get("trigger_codes", []), ensure_ascii=False, separators=(",", ":")),
        "decision": "", "approved_epistemic_status": "", "applicability": "",
        "reviewer": "", "review_comment": "",
    } for item in epistemic.get("candidates", [])]
    drift_rows = [{
        "drift_id": str(item["drift_id"]),
        "source_locator": str(item["source_locator"]),
        "source_span_sha256": str(item["source_span_sha256"]),
        "topic": str(item["topic"]),
        "machine_disposition": str(item["disposition"]),
        "early_design_summary": str(item["early_design_summary"]),
        "current_summary": str(item["current_summary"]),
        "current_code_evidence_json": json.dumps(item.get("current_code_evidence", []), ensure_ascii=False, separators=(",", ":")),
        "decision": "", "approved_disposition": "", "reviewer": "", "review_comment": "",
    } for item in drift.get("candidates", [])]
    if not structure or len(epistemic_rows) != epistemic.get("candidate_count") or len(drift_rows) != drift.get("candidate_count"):
        raise ValueError("early design gold source counts are incomplete")
    return structure, epistemic_rows, drift_rows


def build_early_design_gold_review_workbook(
    *, inventory_path: str | Path, epistemic_candidates_path: str | Path,
    drift_candidates_path: str | Path, review_template_path: str | Path,
    project_root: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    """Create immutable baselines and editable copies for the first DOCX gold task."""

    paths = [Path(value).resolve() for value in (
        inventory_path, epistemic_candidates_path, drift_candidates_path, review_template_path
    )]
    project_root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite early design gold review workbook")
    inventory, epistemic, drift, template = _load_inputs(*paths)
    code_hashes = drift.get("current_code_sha256")
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise ValueError("early design drift candidates do not bind current code hashes")
    for relative, expected in code_hashes.items():
        path = (project_root / str(relative)).resolve()
        if not path.is_relative_to(project_root) or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"early design drift code evidence is stale or missing: {relative}")
    structure, spans, drifts = _rows(inventory, epistemic, drift, template)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        artifacts: dict[str, str] = {}
        for stem, fields, rows in (
            ("structure", STRUCTURE_FIELDS, structure),
            ("epistemic", EPISTEMIC_FIELDS, spans),
            ("drift", DRIFT_FIELDS, drifts),
        ):
            baseline = staging / f"{stem}_review_baseline.csv"
            working = staging / f"{stem}_review_working_copy.csv"
            _write_csv(baseline, fields, rows)
            shutil.copyfile(baseline, working)
            artifacts[baseline.name] = _sha(baseline)
            artifacts[working.name] = _sha(working)
        body = {
            "schema_version": "astrograph-early-design-gold-review-workbook.v1",
            "status": "READY_FOR_NAMED_HUMAN_GOLD_REVIEW",
            "source_sha256": template["source_sha256"],
            "project_root_code_evidence_sha256": dict(sorted(code_hashes.items())),
            "input_file_sha256": {path.name: _sha(path) for path in paths},
            "structure_item_count": len(structure),
            "epistemic_item_count": len(spans),
            "drift_item_count": len(drifts),
            "artifacts": dict(sorted(artifacts.items())),
            "human_decisions_written": False,
            "human_gold_complete": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Machine fields are review aids for the first real DOCX. Only named human "
                "decisions compiled by the freeze step can establish parsing and epistemic gold."
            ),
        }
        result = {**body, "receipt_sha256": _canonical_sha(body)}
        (staging / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        publish_staged_directory(staging, output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_edited(
    baseline: list[dict[str, str]], edited: list[dict[str, str]],
    *, identity: str, fields: list[str], human_fields: set[str], label: str,
) -> None:
    base = {row[identity]: row for row in baseline}
    if len(base) != len(baseline) or len(edited) != len(baseline) or {row[identity] for row in edited} != set(base):
        raise ValueError(f"{label} review identity set changed")
    for row in edited:
        for field in fields:
            if field not in human_fields and row[field] != base[row[identity]][field]:
                raise ValueError(f"{label} machine field changed: {row[identity]}:{field}")


def freeze_early_design_gold_review(
    *, workbook_root: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Freeze complete named decisions or persist a content-addressed BLOCKED result."""

    root = Path(workbook_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite early design gold review freeze")
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _canonical_sha(body) != receipt.get("receipt_sha256"):
        raise ValueError("early design gold workbook semantic hash mismatch")
    specifications = (
        ("structure", STRUCTURE_FIELDS, STRUCTURE_HUMAN_FIELDS, "review_item_id"),
        ("epistemic", EPISTEMIC_FIELDS, EPISTEMIC_HUMAN_FIELDS, "candidate_id"),
        ("drift", DRIFT_FIELDS, DRIFT_HUMAN_FIELDS, "drift_id"),
    )
    loaded: dict[str, list[dict[str, str]]] = {}
    edited_hashes: dict[str, str] = {}
    for stem, fields, human_fields, identity in specifications:
        baseline_path = root / f"{stem}_review_baseline.csv"
        working_path = root / f"{stem}_review_working_copy.csv"
        if receipt.get("artifacts", {}).get(baseline_path.name) != _sha(baseline_path):
            raise ValueError(f"early design gold baseline changed: {stem}")
        baseline = _read_csv(baseline_path, fields)
        edited = _read_csv(working_path, fields)
        _validate_edited(baseline, edited, identity=identity, fields=fields, human_fields=human_fields, label=stem)
        loaded[stem] = edited
        edited_hashes[working_path.name] = _sha(working_path)

    blockers: list[str] = []
    parser_failures: list[str] = []
    for row in loaded["structure"]:
        decision = row["decision"].strip().upper()
        item_id = row["review_item_id"]
        if decision not in {"ACCEPT", "CORRECT", "REJECT"}:
            blockers.append(f"{item_id}:DECISION_REQUIRED")
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append(f"{item_id}:REVIEWER_COMMENT_REQUIRED")
        correction = row["correction_json"].strip()
        if decision == "CORRECT" and not correction:
            blockers.append(f"{item_id}:CORRECTION_REQUIRED")
        if correction:
            try:
                json.loads(correction)
            except json.JSONDecodeError:
                blockers.append(f"{item_id}:INVALID_CORRECTION_JSON")
        if decision == "REJECT":
            parser_failures.append(item_id)

    epistemic_counts: Counter[str] = Counter()
    for row in loaded["epistemic"]:
        decision = row["decision"].strip().upper()
        item_id = row["candidate_id"]
        if decision not in {"ACCEPT", "CORRECT", "REJECT"}:
            blockers.append(f"{item_id}:DECISION_REQUIRED")
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append(f"{item_id}:REVIEWER_COMMENT_REQUIRED")
        approved = row["approved_epistemic_status"].strip().upper()
        if decision in {"ACCEPT", "CORRECT"}:
            if approved not in ALLOWED_EPISTEMIC_STATUSES:
                blockers.append(f"{item_id}:APPROVED_STATUS_REQUIRED")
            elif decision == "ACCEPT" and approved != row["suggested_status"]:
                blockers.append(f"{item_id}:ACCEPT_STATUS_DIFFERS_FROM_SUGGESTION")
            if not row["applicability"].strip():
                blockers.append(f"{item_id}:APPLICABILITY_REQUIRED")
        elif approved or row["applicability"].strip():
            blockers.append(f"{item_id}:REJECT_HAS_APPROVED_CONTENT")
        if approved in ALLOWED_EPISTEMIC_STATUSES:
            epistemic_counts[approved] += 1

    drift_counts: Counter[str] = Counter()
    for row in loaded["drift"]:
        decision = row["decision"].strip().upper()
        item_id = row["drift_id"]
        if decision not in {"ACCEPT_DISPOSITION", "REVISE_DISPOSITION", "REJECT_CANDIDATE"}:
            blockers.append(f"{item_id}:DECISION_REQUIRED")
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append(f"{item_id}:REVIEWER_COMMENT_REQUIRED")
        approved = row["approved_disposition"].strip()
        if decision == "ACCEPT_DISPOSITION" and approved != row["machine_disposition"]:
            blockers.append(f"{item_id}:ACCEPT_DISPOSITION_MISMATCH")
        if decision == "REVISE_DISPOSITION" and not approved:
            blockers.append(f"{item_id}:REVISED_DISPOSITION_REQUIRED")
        if decision == "REJECT_CANDIDATE" and approved:
            blockers.append(f"{item_id}:REJECT_HAS_DISPOSITION")
        if decision:
            drift_counts[decision] += 1

    status = "BLOCKED" if blockers else ("COMPLETE_PARSER_FAILURE" if parser_failures else "COMPLETE")
    freeze_body = {
        "schema_version": "astrograph-early-design-human-gold-freeze.v1",
        "status": status,
        "source_sha256": receipt["source_sha256"],
        "workbook_receipt_file_sha256": _sha(receipt_path),
        "edited_file_sha256": dict(sorted(edited_hashes.items())),
        "structure_item_count": len(loaded["structure"]),
        "epistemic_item_count": len(loaded["epistemic"]),
        "drift_item_count": len(loaded["drift"]),
        "epistemic_status_counts": dict(sorted(epistemic_counts.items())),
        "drift_decision_counts": dict(sorted(drift_counts.items())),
        "parser_failure_item_ids": parser_failures,
        "blockers": blockers,
        "human_gold_complete": status == "COMPLETE",
        "docx_heading_table_image_gold_complete": status == "COMPLETE",
        "claim_annotations_complete": status == "COMPLETE",
        "design_drift_review_complete": status == "COMPLETE",
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "COMPLETE establishes named human gold for this non-normative DOCX only. It does not "
            "approve document claims as operational facts, publish KG knowledge or allow runtime voting."
        ),
    }
    result = {**freeze_body, "freeze_sha256": _canonical_sha(freeze_body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = [
    "build_early_design_gold_review_workbook",
    "freeze_early_design_gold_review",
]
