"""Source-bound human correction loop for blocked document graph candidates."""
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

from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionProposal,
    KnowledgeNodeCandidate,
    KnowledgeRelationCandidate,
)

from .document_candidate_entrypoint import (
    DocumentCandidateBuildReceipt,
    build_document_candidate_package,
)


CORRECTION_ACTIONS = {"KEEP", "DROP", "REPLACE"}
CORRECTION_FIELDS = [
    "correction_item_id",
    "item_kind",
    "candidate_id",
    "candidate_sha256",
    "candidate_json",
    "item_blocker_codes",
    "bundle_sha256",
    "action",
    "replacement_candidate_json",
    "reviewer",
    "review_comment",
]

CORRECTION_SUGGESTION_FIELDS = [
    "correction_item_id",
    "item_kind",
    "candidate_id",
    "machine_suggestion",
    "suggested_action",
    "reason_codes",
    "item_blocker_codes",
    "candidate_receipt_file_sha256",
    "construction_bundle_sha256",
]

CORRECTION_HUMAN_FIELDS = [
    "action",
    "replacement_candidate_json",
    "reviewer",
    "review_comment",
]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_candidate_package(
    root: Path,
) -> tuple[DocumentCandidateBuildReceipt, dict[str, Any], KnowledgeExtractionProposal]:
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("document candidate package has no receipt")
    receipt = DocumentCandidateBuildReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    for filename, expected in receipt.artifacts.items():
        artifact = root / filename
        if not artifact.is_file() or _sha_file(artifact) != expected:
            raise ValueError(f"document candidate artifact is missing or changed: {filename}")
    if receipt.status != "BLOCKED":
        raise ValueError("only a governed BLOCKED candidate package can enter correction")
    bundle_path = root / "construction_bundle.json"
    if not bundle_path.is_file():
        raise ValueError("blocked candidate package has no construction bundle")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("bundle_sha256") != receipt.construction_bundle_sha256:
        raise ValueError("candidate receipt and construction bundle identities differ")
    proposal = KnowledgeExtractionProposal.model_validate(bundle["proposal"])
    return receipt, bundle, proposal


def _candidate_rows(bundle: dict[str, Any], proposal: KnowledgeExtractionProposal) -> list[dict[str, str]]:
    blockers_by_subject: dict[str, set[str]] = {}
    for finding in bundle["candidate_validation"]["findings"]:
        if finding.get("severity") == "BLOCKER" and finding.get("subject_id"):
            blockers_by_subject.setdefault(str(finding["subject_id"]), set()).add(
                str(finding["code"])
            )
    output: list[dict[str, str]] = []
    items: list[tuple[str, Any]] = [
        *[("NODE", item) for item in proposal.nodes],
        *[("RELATION", item) for item in proposal.relations],
    ]
    for kind, candidate in items:
        candidate_id = candidate.node_id if kind == "NODE" else candidate.edge_id
        payload = candidate.model_dump(mode="json")
        output.append({
            "correction_item_id": "DOC-KG-CORRECTION-" + _canonical_sha({
                "bundle_sha256": bundle["bundle_sha256"],
                "item_kind": kind,
                "candidate_id": candidate_id,
            })[:20].upper(),
            "item_kind": kind,
            "candidate_id": candidate_id,
            "candidate_sha256": _canonical_sha(payload),
            "candidate_json": _canonical_json(payload),
            "item_blocker_codes": ";".join(sorted(blockers_by_subject.get(candidate_id, set()))),
            "bundle_sha256": str(bundle["bundle_sha256"]),
            "action": "",
            "replacement_candidate_json": "",
            "reviewer": "",
            "review_comment": "",
        })
    return sorted(output, key=lambda row: (row["item_kind"], row["candidate_id"]))


def prepare_document_candidate_correction(
    *, candidate_package_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Create an immutable-machine-field correction handoff for one blocked package."""

    root = Path(candidate_package_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite document candidate correction handoff")
    receipt, bundle, proposal = _load_candidate_package(root)
    rows = _candidate_rows(bundle, proposal)
    if not rows:
        raise ValueError("blocked candidate package has no candidates to correct")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        sheet = staging / "human_correction_decisions.csv"
        with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CORRECTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        body = {
            "schema_version": "astrograph-document-candidate-correction-handoff.v1",
            "status": "HUMAN_CORRECTION_REQUIRED",
            "candidate_receipt_file_sha256": _sha_file(root / "receipt.json"),
            "construction_bundle_sha256": bundle["bundle_sha256"],
            "correction_item_count": len(rows),
            "blocked_item_count": sum(bool(row["item_blocker_codes"]) for row in rows),
            "correction_sheet_sha256": _sha_file(sheet),
            "allowed_actions": sorted(CORRECTION_ACTIONS),
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Human correction handoff for a blocked machine proposal. Completing it can only "
                "request a fresh machine validation; it cannot approve or publish knowledge."
            ),
        }
        result = {**body, "handoff_sha256": _canonical_sha(body)}
        (staging / "correction_handoff_receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging, output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_document_candidate_correction_suggestions(
    *,
    candidate_package_root: str | Path,
    correction_handoff_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build non-binding critic-derived advice beside a blank correction sheet.

    Only a relation rejected by the deterministic V3 shape allowlist receives a
    concrete DROP suggestion.  Semantic, confidence and provenance blockers are
    intentionally left for named human adjudication.
    """

    candidate_root = Path(candidate_package_root).resolve()
    handoff_root = Path(correction_handoff_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite document candidate correction suggestions")
    receipt, bundle, proposal = _load_candidate_package(candidate_root)
    handoff_path = handoff_root / "correction_handoff_receipt.json"
    sheet_path = handoff_root / "human_correction_decisions.csv"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff_body = {key: value for key, value in handoff.items() if key != "handoff_sha256"}
    if _canonical_sha(handoff_body) != handoff.get("handoff_sha256"):
        raise ValueError("correction handoff semantic hash mismatch")
    if (
        handoff.get("candidate_receipt_file_sha256")
        != _sha_file(candidate_root / "receipt.json")
        or handoff.get("construction_bundle_sha256") != bundle["bundle_sha256"]
        or handoff.get("correction_sheet_sha256") != _sha_file(sheet_path)
    ):
        raise ValueError("correction handoff is not bound to the candidate package")
    rows = _read_correction_rows(sheet_path)
    expected = _candidate_rows(bundle, proposal)
    if rows != expected:
        raise ValueError("correction sheet differs from the source-reconstructed handoff")
    if any(
        row["action"].strip()
        or row["replacement_candidate_json"].strip()
        or row["reviewer"].strip()
        or row["review_comment"].strip()
        for row in rows
    ):
        raise ValueError("correction suggestions require a blank human correction sheet")

    suggestions: list[dict[str, str]] = []
    for row in rows:
        blockers = {item for item in row["item_blocker_codes"].split(";") if item}
        if row["item_kind"] == "RELATION" and "V3_RELATION_SHAPE_NOT_ALLOWED" in blockers:
            status = "DROP_DETERMINISTICALLY_INVALID_RELATION_SHAPE"
            action = "DROP"
            reasons = ["V3_RELATION_SHAPE_NOT_ALLOWED", "REVALIDATION_REQUIRED_AFTER_HUMAN_DECISION"]
        elif blockers:
            status = "HUMAN_REPLACE_OR_DROP_REVIEW_REQUIRED"
            action = ""
            reasons = sorted(blockers) + ["NO_SAFE_AUTOMATIC_SEMANTIC_CORRECTION"]
        else:
            status = "HUMAN_KEEP_OR_DROP_REVIEW_REQUIRED"
            action = ""
            reasons = ["ITEM_NOT_DIRECTLY_BLOCKED", "BATCH_CONTEXT_STILL_REQUIRES_HUMAN_REVIEW"]
        suggestions.append(
            {
                "correction_item_id": row["correction_item_id"],
                "item_kind": row["item_kind"],
                "candidate_id": row["candidate_id"],
                "machine_suggestion": status,
                "suggested_action": action,
                "reason_codes": ";".join(reasons),
                "item_blocker_codes": row["item_blocker_codes"],
                "candidate_receipt_file_sha256": _sha_file(candidate_root / "receipt.json"),
                "construction_bundle_sha256": str(bundle["bundle_sha256"]),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        suggestion_path = staging / "machine_correction_suggestions.csv"
        with suggestion_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CORRECTION_SUGGESTION_FIELDS)
            writer.writeheader()
            writer.writerows(suggestions)
        counts: dict[str, int] = {}
        for row in suggestions:
            counts[row["machine_suggestion"]] = counts.get(row["machine_suggestion"], 0) + 1
        body = {
            "schema_version": "astrograph-document-candidate-correction-suggestions.v1",
            "status": "MACHINE_SUGGESTIONS_READY_FOR_HUMAN_USE",
            "candidate_receipt_file_sha256": _sha_file(candidate_root / "receipt.json"),
            "correction_handoff_receipt_file_sha256": _sha_file(handoff_path),
            "correction_sheet_file_sha256": _sha_file(sheet_path),
            "construction_bundle_sha256": bundle["bundle_sha256"],
            "suggestion_count": len(suggestions),
            "suggestion_counts": dict(sorted(counts.items())),
            "suggestion_sheet_file_sha256": _sha_file(suggestion_path),
            "human_decisions_written": False,
            "corrected_candidate_built": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Critic-derived correction suggestions are non-binding. Only deterministic V3 "
                "shape violations receive a DROP suggestion; a named reviewer must still decide "
                "every row before a fresh candidate validation can run."
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


def build_document_candidate_correction_workbook(
    *,
    candidate_package_root: str | Path,
    correction_handoff_root: str | Path,
    suggestion_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge a locked correction handoff and advice into a fillable workbook."""

    candidate_root = Path(candidate_package_root).resolve()
    handoff_root = Path(correction_handoff_root).resolve()
    suggestion_root = Path(suggestion_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite document candidate correction workbook")
    _, bundle, proposal = _load_candidate_package(candidate_root)
    handoff_path = handoff_root / "correction_handoff_receipt.json"
    correction_sheet_path = handoff_root / "human_correction_decisions.csv"
    suggestion_receipt_path = suggestion_root / "receipt.json"
    suggestion_sheet_path = suggestion_root / "machine_correction_suggestions.csv"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff_body = {key: value for key, value in handoff.items() if key != "handoff_sha256"}
    if _canonical_sha(handoff_body) != handoff.get("handoff_sha256"):
        raise ValueError("correction handoff semantic hash mismatch")
    suggestion_receipt = json.loads(suggestion_receipt_path.read_text(encoding="utf-8"))
    suggestion_body = {
        key: value for key, value in suggestion_receipt.items() if key != "receipt_sha256"
    }
    if _canonical_sha(suggestion_body) != suggestion_receipt.get("receipt_sha256"):
        raise ValueError("correction suggestion semantic hash mismatch")
    if (
        handoff.get("candidate_receipt_file_sha256")
        != _sha_file(candidate_root / "receipt.json")
        or handoff.get("construction_bundle_sha256") != bundle["bundle_sha256"]
        or handoff.get("correction_sheet_sha256") != _sha_file(correction_sheet_path)
        or suggestion_receipt.get("candidate_receipt_file_sha256")
        != _sha_file(candidate_root / "receipt.json")
        or suggestion_receipt.get("correction_handoff_receipt_file_sha256")
        != _sha_file(handoff_path)
        or suggestion_receipt.get("correction_sheet_file_sha256")
        != _sha_file(correction_sheet_path)
        or suggestion_receipt.get("suggestion_sheet_file_sha256")
        != _sha_file(suggestion_sheet_path)
    ):
        raise ValueError("correction workbook inputs are not content-bound")
    rows = _read_correction_rows(correction_sheet_path)
    if rows != _candidate_rows(bundle, proposal):
        raise ValueError("correction sheet differs from source-reconstructed candidates")
    if any(any(row[field].strip() for field in CORRECTION_HUMAN_FIELDS) for row in rows):
        raise ValueError("correction workbook requires a blank human correction sheet")
    with suggestion_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != CORRECTION_SUGGESTION_FIELDS:
            raise ValueError("correction suggestion sheet columns changed")
        suggestion_rows = list(reader)
    suggestion_by_id = {row["correction_item_id"]: row for row in suggestion_rows}
    if len(suggestion_by_id) != len(suggestion_rows) or set(suggestion_by_id) != {
        row["correction_item_id"] for row in rows
    }:
        raise ValueError("correction suggestion and handoff identities differ")

    machine_fields = CORRECTION_FIELDS[:-4]
    workbook_rows: list[dict[str, str]] = []
    for row in rows:
        suggestion = suggestion_by_id[row["correction_item_id"]]
        for field in ("item_kind", "candidate_id", "item_blocker_codes", "bundle_sha256"):
            suggestion_field = (
                "construction_bundle_sha256" if field == "bundle_sha256" else field
            )
            if suggestion[suggestion_field] != row[field]:
                raise ValueError(f"correction suggestion differs from handoff: {field}")
        status = suggestion["machine_suggestion"]
        if status == "DROP_DETERMINISTICALLY_INVALID_RELATION_SHAPE":
            priority = "P0_DETERMINISTIC_RELATION_SHAPE_FAILURE"
        elif status == "HUMAN_REPLACE_OR_DROP_REVIEW_REQUIRED":
            priority = "P1_SEMANTIC_BLOCKER_REVIEW"
        else:
            priority = "P2_BATCH_CONTEXT_REVIEW"
        workbook_rows.append(
            {
                "review_priority": priority,
                **{field: row[field] for field in machine_fields},
                "machine_suggestion": status,
                "suggested_action": suggestion["suggested_action"],
                "suggestion_reason_codes": suggestion["reason_codes"],
                **{field: "" for field in CORRECTION_HUMAN_FIELDS},
            }
        )
    workbook_rows.sort(
        key=lambda row: (row["review_priority"], row["item_kind"], row["candidate_id"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        baseline = staging / "human_correction_priority_queue.csv"
        fields = list(workbook_rows[0])
        with baseline.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(workbook_rows)
        working = staging / "human_correction_working_copy.csv"
        shutil.copyfile(baseline, working)
        body = {
            "schema_version": "astrograph-document-candidate-correction-workbook.v1",
            "status": "READY_FOR_NAMED_HUMAN_CORRECTION",
            "candidate_receipt_file_sha256": _sha_file(candidate_root / "receipt.json"),
            "correction_handoff_receipt_file_sha256": _sha_file(handoff_path),
            "correction_suggestion_receipt_file_sha256": _sha_file(suggestion_receipt_path),
            "correction_item_count": len(workbook_rows),
            "priority_counts": dict(
                sorted(Counter(row["review_priority"] for row in workbook_rows).items())
            ),
            "priority_queue_file_sha256": _sha_file(baseline),
            "working_copy_initial_file_sha256": _sha_file(working),
            "human_decisions_written": False,
            "corrected_candidate_built": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "The workbook only combines immutable candidate fields and critic advice for "
                "named human correction. It grants no correction, review, publication or runtime authority."
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


def compile_document_candidate_correction_workbook(
    *,
    candidate_package_root: str | Path,
    workbook_root: str | Path,
    workbook_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate edits and restore the canonical correction-sheet schema and order."""

    candidate_root = Path(candidate_package_root).resolve()
    workbook_root = Path(workbook_root).resolve()
    workbook_path = Path(workbook_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite compiled correction workbook")
    receipt_path = workbook_root / "receipt.json"
    baseline_path = workbook_root / "human_correction_priority_queue.csv"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _canonical_sha(receipt_body) != receipt.get("receipt_sha256"):
        raise ValueError("correction workbook semantic hash mismatch")
    if (
        receipt.get("candidate_receipt_file_sha256")
        != _sha_file(candidate_root / "receipt.json")
        or receipt.get("priority_queue_file_sha256") != _sha_file(baseline_path)
    ):
        raise ValueError("correction workbook is not bound to candidate package")
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        baseline_rows = list(reader)
    with workbook_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != fields:
            raise ValueError("correction workbook columns changed")
        edited_rows = list(reader)
    if len(edited_rows) != len(baseline_rows):
        raise ValueError("correction workbook row count changed")
    baseline_by_id = {row["correction_item_id"]: row for row in baseline_rows}
    if len(baseline_by_id) != len(baseline_rows):
        raise ValueError("correction workbook baseline has duplicate identities")
    editable = set(CORRECTION_HUMAN_FIELDS)
    for row in edited_rows:
        baseline = baseline_by_id.get(row["correction_item_id"])
        if baseline is None:
            raise ValueError("correction workbook contains an unknown identity")
        for field in fields:
            if field not in editable and row[field] != baseline[field]:
                raise ValueError(f"correction workbook machine field changed: {field}")
    if {row["correction_item_id"] for row in edited_rows} != set(baseline_by_id):
        raise ValueError("correction workbook identity set changed")

    blockers: list[str] = []
    canonical_rows: list[dict[str, str]] = []
    for row in sorted(edited_rows, key=lambda item: (item["item_kind"], item["candidate_id"])):
        action = row["action"].strip().upper()
        replacement = row["replacement_candidate_json"].strip()
        reviewer = row["reviewer"].strip()
        comment = row["review_comment"].strip()
        item_id = row["correction_item_id"]
        if action not in CORRECTION_ACTIONS:
            blockers.append(f"{item_id}:MISSING_OR_INVALID_ACTION")
        if not reviewer:
            blockers.append(f"{item_id}:MISSING_REVIEWER")
        if not comment:
            blockers.append(f"{item_id}:MISSING_REVIEW_COMMENT")
        if action == "REPLACE":
            if not replacement:
                blockers.append(f"{item_id}:MISSING_REPLACEMENT")
            else:
                try:
                    raw = json.loads(replacement)
                    if row["item_kind"] == "NODE":
                        candidate = KnowledgeNodeCandidate.model_validate(raw)
                        candidate_id = candidate.node_id
                    else:
                        candidate = KnowledgeRelationCandidate.model_validate(raw)
                        candidate_id = candidate.edge_id
                    if candidate_id != row["candidate_id"]:
                        blockers.append(f"{item_id}:REPLACEMENT_ID_CHANGED")
                except (ValueError, json.JSONDecodeError):
                    blockers.append(f"{item_id}:INVALID_REPLACEMENT_JSON")
        elif replacement:
            blockers.append(f"{item_id}:UNEXPECTED_REPLACEMENT")
        canonical_rows.append({field: row[field] for field in CORRECTION_FIELDS})
        canonical_rows[-1]["action"] = action
        canonical_rows[-1]["reviewer"] = reviewer
        canonical_rows[-1]["review_comment"] = comment
        canonical_rows[-1]["replacement_candidate_json"] = replacement

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        canonical_path = staging / "human_correction_decisions.csv"
        with canonical_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CORRECTION_FIELDS)
            writer.writeheader()
            writer.writerows(canonical_rows)
        body = {
            "schema_version": "astrograph-document-candidate-correction-workbook-compile.v1",
            "status": "READY_FOR_CORRECTION_REVALIDATION" if not blockers else "BLOCKED",
            "candidate_receipt_file_sha256": _sha_file(candidate_root / "receipt.json"),
            "workbook_receipt_file_sha256": _sha_file(receipt_path),
            "workbook_file_sha256": _sha_file(workbook_path),
            "canonical_correction_sheet_file_sha256": _sha_file(canonical_path),
            "correction_item_count": len(canonical_rows),
            "blockers": blockers,
            "corrected_candidate_built": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Compilation validates and restores a canonical correction sheet only. A separate "
                "apply step must re-run every candidate governance gate."
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


def _read_correction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != CORRECTION_FIELDS:
            raise ValueError("document candidate correction sheet columns changed")
        return list(reader)


def apply_document_candidate_correction(
    *,
    ingestion_package_dir: str | Path,
    candidate_package_root: str | Path,
    correction_sheet_path: str | Path,
    review_refreeze_path: str | Path,
    expected_review_refreeze_file_sha256: str,
    graph_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Apply named corrections and re-run every candidate governance gate in a new package."""

    candidate_root = Path(candidate_package_root).resolve()
    sheet_path = Path(correction_sheet_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite corrected document candidate package")
    original_receipt, bundle, original = _load_candidate_package(candidate_root)
    expected_rows = {
        row["correction_item_id"]: row for row in _candidate_rows(bundle, original)
    }
    rows = _read_correction_rows(sheet_path)
    if len(rows) != len(expected_rows) or len({row["correction_item_id"] for row in rows}) != len(rows):
        raise ValueError("correction sheet item set is incomplete or duplicated")

    corrected_nodes: list[KnowledgeNodeCandidate] = []
    corrected_relations: list[KnowledgeRelationCandidate] = []
    decisions: list[dict[str, str]] = []
    for row in rows:
        expected = expected_rows.get(row["correction_item_id"])
        if expected is None:
            raise ValueError("correction sheet contains an unknown item")
        machine_fields = CORRECTION_FIELDS[:7]
        changed = [field for field in machine_fields if row.get(field, "") != expected[field]]
        if changed:
            raise ValueError("document candidate correction machine fields changed: " + ";".join(changed))
        action = row["action"].strip().upper()
        reviewer = row["reviewer"].strip()
        comment = row["review_comment"].strip()
        if action not in CORRECTION_ACTIONS:
            raise ValueError("every correction item requires KEEP, DROP or REPLACE")
        if not reviewer or not comment:
            raise ValueError("every correction item requires a named reviewer and comment")
        replacement_text = row["replacement_candidate_json"].strip()
        if action == "REPLACE":
            if not replacement_text:
                raise ValueError("REPLACE requires replacement_candidate_json")
            raw = json.loads(replacement_text)
        else:
            if replacement_text:
                raise ValueError("KEEP and DROP cannot provide replacement_candidate_json")
            raw = json.loads(row["candidate_json"])
        if action != "DROP":
            if row["item_kind"] == "NODE":
                candidate = KnowledgeNodeCandidate.model_validate(raw)
                if candidate.node_id != row["candidate_id"]:
                    raise ValueError("replacement node must preserve candidate_id")
                corrected_nodes.append(candidate)
            else:
                candidate = KnowledgeRelationCandidate.model_validate(raw)
                if candidate.edge_id != row["candidate_id"]:
                    raise ValueError("replacement relation must preserve candidate_id")
                corrected_relations.append(candidate)
        decisions.append({
            "correction_item_id": row["correction_item_id"],
            "action": action,
            "reviewer": reviewer,
            "review_comment": comment,
            "replacement_sha256": _canonical_sha(raw) if action == "REPLACE" else "",
        })

    corrected = KnowledgeExtractionProposal(
        source_profile_sha256=original.source_profile_sha256,
        nodes=corrected_nodes,
        relations=corrected_relations,
        unresolved_claims=original.unresolved_claims,
        proposer_mode="DETERMINISTIC_DRAFT",
        proposer_id="human-corrected-document-candidate",
        proposer_version="1",
        executable_content_present=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent))
    try:
        candidate_output = staging / "candidate"
        rebuilt = build_document_candidate_package(
            ingestion_package_dir=ingestion_package_dir,
            output_dir=candidate_output,
            review_refreeze_path=review_refreeze_path,
            expected_review_refreeze_file_sha256=expected_review_refreeze_file_sha256,
            graph_id=graph_id,
            proposal=corrected,
        )
        frozen_sheet = staging / "human_correction_decisions.csv"
        shutil.copy2(sheet_path, frozen_sheet)
        decision_body = {
            "schema_version": "astrograph-document-candidate-correction-freeze.v1",
            "source_candidate_receipt_file_sha256": _sha_file(candidate_root / "receipt.json"),
            "source_construction_bundle_sha256": bundle["bundle_sha256"],
            "correction_sheet_sha256": _sha_file(frozen_sheet),
            "decisions": sorted(decisions, key=lambda item: item["correction_item_id"]),
            "corrected_proposal_sha256": _canonical_sha(corrected.model_dump(mode="json")),
            "rebuilt_candidate_receipt_file_sha256": _sha_file(candidate_output / "receipt.json"),
            "rebuilt_candidate_status": rebuilt.status,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
        }
        result = {
            **decision_body,
            "status": (
                "CORRECTED_CANDIDATE_READY_FOR_HUMAN_REVIEW"
                if rebuilt.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
                else "CORRECTION_REVALIDATION_BLOCKED"
            ),
            "correction_freeze_sha256": _canonical_sha(decision_body),
            "claim_boundary": (
                "Named corrections were compiled only into a fresh review candidate and re-ran "
                "all machine gates. Even READY requires the normal human review and Patch workflow."
            ),
        }
        (staging / "correction_freeze_receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        publish_staged_directory(staging, output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CORRECTION_ACTIONS",
    "CORRECTION_FIELDS",
    "CORRECTION_SUGGESTION_FIELDS",
    "apply_document_candidate_correction",
    "build_document_candidate_correction_suggestions",
    "build_document_candidate_correction_workbook",
    "compile_document_candidate_correction_workbook",
    "prepare_document_candidate_correction",
]
