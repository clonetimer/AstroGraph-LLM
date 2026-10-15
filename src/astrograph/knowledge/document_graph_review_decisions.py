"""Validate completed document-KG review sheets into a non-publishing subset."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal

from .document_graph_review import REVIEW_FIELDS
from .document_candidate_entrypoint import DocumentCandidateBuildReceipt


DECISIONS = {"APPROVE", "REJECT", "DEFER"}
HUMAN_FIELDS = {"decision", "approved_canonical_id", "reviewer", "review_comment"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _bundle_index(
    roots: list[Path],
) -> dict[str, tuple[Path, dict[str, Any], str, str]]:
    index: dict[str, tuple[Path, dict[str, Any], str, str]] = {}
    for root in roots:
        for path in sorted(root.rglob("construction_bundle.json")):
            bundle = json.loads(path.read_text(encoding="utf-8"))
            digest = str(bundle.get("bundle_sha256", ""))
            if not digest:
                continue
            relative = path.relative_to(root)
            if len(relative.parts) >= 4 and relative.parts[2:] == (
                "candidate", "construction_bundle.json"
            ):
                dataset_id, fault_class_id = relative.parts[0], relative.parts[1]
                candidate_root = path.parent
                receipt_path = candidate_root / "receipt.json"
                candidate_receipt = DocumentCandidateBuildReceipt.model_validate_json(
                    receipt_path.read_text(encoding="utf-8")
                )
                for filename, expected in candidate_receipt.artifacts.items():
                    artifact = candidate_root / filename
                    if not artifact.is_file() or _sha(artifact) != expected:
                        raise ValueError(
                            f"stable pack candidate artifact is missing or changed: {filename}"
                        )
                if candidate_receipt.construction_bundle_sha256 != digest:
                    raise ValueError("stable pack candidate receipt and bundle differ")
            elif len(relative.parts) < 3:
                if relative.parts != ("construction_bundle.json",):
                    raise ValueError(f"construction bundle has no dataset/fault path: {path}")
                receipt_path = root / "receipt.json"
                candidate_receipt = DocumentCandidateBuildReceipt.model_validate_json(
                    receipt_path.read_text(encoding="utf-8")
                )
                for filename, expected in candidate_receipt.artifacts.items():
                    artifact = root / filename
                    if not artifact.is_file() or _sha(artifact) != expected:
                        raise ValueError(
                            f"single document candidate artifact is missing or changed: {filename}"
                        )
                if candidate_receipt.construction_bundle_sha256 != digest:
                    raise ValueError("single document candidate receipt and bundle differ")
                dataset_id = "document-scope"
                fault_class_id = candidate_receipt.document_id
            else:
                dataset_id, fault_class_id = relative.parts[0], relative.parts[1]
            if digest in index and index[digest][0] != path:
                raise ValueError(f"duplicate construction bundle identity: {digest}")
            index[digest] = (path, bundle, dataset_id, fault_class_id)
    return index


def _expected_machine_fields(
    row: dict[str, str], candidate: Any
) -> dict[str, str]:
    properties = candidate.properties
    common = {
        "item_kind": row["item_kind"],
        "source_document_id": str(properties.get("source_document_id", "")),
        "source_document_sha256": str(properties.get("source_document_sha256", "")),
        "candidate_id": candidate.node_id if row["item_kind"] == "NODE" else candidate.edge_id,
        "source_fragment_ids": ";".join(candidate.source_fragment_ids),
        "confidence": format(float(candidate.confidence), ".6g"),
        "ambiguous": str(bool(candidate.ambiguous)).lower(),
    }
    if row["item_kind"] == "NODE":
        common.update({
            "node_type": candidate.node_type.value,
            "canonical_name": candidate.canonical_name,
            "relation_type": "",
            "source_candidate_id": "",
            "target_candidate_id": "",
            "description_or_rationale": candidate.description,
        })
    else:
        common.update({
            "node_type": "",
            "canonical_name": "",
            "relation_type": candidate.relation_type.value,
            "source_candidate_id": candidate.source_id,
            "target_candidate_id": candidate.target_id,
            "description_or_rationale": candidate.rationale,
        })
    return common


def freeze_document_graph_review_decisions(
    *,
    review_sheet_path: str | Path,
    graph_roots: list[str | Path],
    output_path: str | Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Freeze human decisions after reconstructing every machine-owned field."""

    review_sheet_path = Path(review_sheet_path).resolve()
    roots = [Path(root).resolve() for root in graph_roots]
    output_path = Path(output_path).resolve()
    fields, rows = _read_rows(review_sheet_path)
    if fields != REVIEW_FIELDS:
        raise ValueError("document graph review sheet columns changed")
    if not rows:
        body = {
            "schema_version": "astrograph-document-graph-review-freeze.v1",
            "status": "BLOCKED",
            "review_sheet_sha256": _sha(review_sheet_path),
            "review_item_count": 0,
            "decision_counts": {value: 0 for value in sorted(DECISIONS)},
            "approved_items": [],
            "blockers": [{"code": "EMPTY_REVIEW_SHEET", "review_item_id": "*"}],
            "knowledge_patch_proposal_allowed": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "An empty source-bound review sheet is a blocked negative result, never an "
                "implicit approval or permission to propose a KnowledgePatch."
            ),
        }
        receipt = {**body, "freeze_sha256": _canonical_sha(body)}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return receipt
    bundles = _bundle_index(roots)
    blockers: list[dict[str, str]] = []
    approved: list[dict[str, Any]] = []
    decisions: dict[str, str] = {}
    seen_review_ids: set[str] = set()
    approved_node_ids: set[tuple[str, str]] = set()

    resolved: list[tuple[dict[str, str], Any, Path]] = []
    for row in rows:
        review_id = row["review_item_id"]
        if review_id in seen_review_ids:
            blockers.append({"code": "DUPLICATE_REVIEW_ITEM", "review_item_id": review_id})
            continue
        seen_review_ids.add(review_id)
        decision = row["decision"].strip().upper()
        decisions[review_id] = decision
        if decision not in DECISIONS:
            blockers.append({
                "code": "MISSING_OR_INVALID_DECISION" if not decision else "INVALID_DECISION",
                "review_item_id": review_id,
            })
            continue
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append({"code": "REVIEWER_AND_COMMENT_REQUIRED", "review_item_id": review_id})
        bundle_entry = bundles.get(row["bundle_sha256"])
        if bundle_entry is None:
            blockers.append({"code": "SOURCE_BUNDLE_NOT_FOUND", "review_item_id": review_id})
            continue
        bundle_path, bundle, dataset_id, fault_class_id = bundle_entry
        proposal = KnowledgeExtractionProposal.model_validate(bundle["proposal"])
        candidates = {
            **{item.node_id: item for item in proposal.nodes},
            **{item.edge_id: item for item in proposal.relations},
        }
        candidate = candidates.get(row["candidate_id"])
        if candidate is None:
            blockers.append({"code": "CANDIDATE_NOT_FOUND", "review_item_id": review_id})
            continue
        actual_kind = "NODE" if hasattr(candidate, "node_id") else "RELATION"
        if row["item_kind"] != actual_kind:
            blockers.append({"code": "MACHINE_FIELD_TAMPERED", "review_item_id": review_id,
                             "detail": "item_kind"})
            continue
        expected = _expected_machine_fields(row, candidate)
        item_kind = row["item_kind"]
        expected_review_id = "DOC-KG-REVIEW-" + _canonical_sha({
            "dataset_id": dataset_id,
            "fault_id": fault_class_id,
            "kind": item_kind,
            "id": row["candidate_id"],
        })[:20].upper()
        blocker_codes = sorted({
            finding["code"]
            for finding in bundle["candidate_validation"]["findings"]
            if finding["severity"] == "BLOCKER"
        })
        expected.update({
            "review_item_id": expected_review_id,
            "dataset_id": dataset_id,
            "fault_class_id": fault_class_id,
            "batch_status": str(bundle["status"]),
            "blocker_codes": ";".join(blocker_codes),
            "bundle_sha256": str(bundle["bundle_sha256"]),
        })
        changed = sorted(key for key, value in expected.items() if row.get(key, "") != value)
        if changed:
            blockers.append({
                "code": "MACHINE_FIELD_TAMPERED",
                "review_item_id": review_id,
                "detail": ";".join(changed),
            })
            continue
        if decision == "APPROVE" and row["batch_status"] != "CANDIDATE_READY_FOR_HUMAN_REVIEW":
            blockers.append({"code": "BLOCKED_BATCH_CANNOT_BE_APPROVED", "review_item_id": review_id})
        if row["item_kind"] == "NODE" and decision == "APPROVE":
            if not row["approved_canonical_id"].strip():
                blockers.append({"code": "APPROVED_NODE_CANONICAL_ID_REQUIRED", "review_item_id": review_id})
            approved_node_ids.add((row["bundle_sha256"], row["candidate_id"]))
        if decision != "APPROVE" and row["approved_canonical_id"].strip():
            blockers.append({
                "code": "NON_APPROVAL_HAS_CANONICAL_ID",
                "review_item_id": review_id,
            })
        resolved.append((row, candidate, bundle_path))

    for row, candidate, bundle_path in resolved:
        if decisions.get(row["review_item_id"]) != "APPROVE":
            continue
        if row["item_kind"] == "RELATION":
            for endpoint in (candidate.source_id, candidate.target_id):
                if endpoint.startswith("candidate:") and (
                    row["bundle_sha256"], endpoint
                ) not in approved_node_ids:
                    blockers.append({
                        "code": "APPROVED_RELATION_ENDPOINT_NOT_APPROVED",
                        "review_item_id": row["review_item_id"],
                        "detail": endpoint,
                    })
        approved.append({
            "review_item_id": row["review_item_id"],
            "item_kind": row["item_kind"],
            "approved_canonical_id": row["approved_canonical_id"].strip() or None,
            "reviewer": row["reviewer"].strip(),
            "review_comment": row["review_comment"].strip(),
            "bundle_sha256": row["bundle_sha256"],
            "source_bundle_path": str(bundle_path),
            "candidate": candidate.model_dump(mode="json"),
        })

    if require_complete and any(not value for value in decisions.values()):
        blockers.append({"code": "INCOMPLETE_REVIEW", "review_item_id": "*"})
    status = (
        "BLOCKED"
        if blockers
        else "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL"
        if approved
        else "CLOSED_WITHOUT_APPROVALS"
    )
    body = {
        "schema_version": "astrograph-document-graph-review-freeze.v1",
        "status": status,
        "review_sheet_sha256": _sha(review_sheet_path),
        "review_item_count": len(rows),
        "decision_counts": {
            value: sum(decision == value for decision in decisions.values())
            for value in sorted(DECISIONS)
        },
        "approved_items": approved if not blockers else [],
        "blockers": blockers,
        "knowledge_patch_proposal_allowed": status == "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL",
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "A source-reconstructed freeze of human decisions. READY permits only a separate "
            "KnowledgePatch proposal; replay, governance approval and atomic publication remain required."
        ),
    }
    receipt = {**body, "freeze_sha256": _canonical_sha(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = ["freeze_document_graph_review_decisions"]
