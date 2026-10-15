"""Create a blank, source-bound human review handoff for document KG candidates."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .document_candidate_entrypoint import DocumentCandidateBuildReceipt


REVIEW_FIELDS = [
    "review_item_id",
    "item_kind",
    "dataset_id",
    "fault_class_id",
    "source_document_id",
    "source_document_sha256",
    "candidate_id",
    "node_type",
    "canonical_name",
    "relation_type",
    "source_candidate_id",
    "target_candidate_id",
    "description_or_rationale",
    "source_fragment_ids",
    "confidence",
    "ambiguous",
    "batch_status",
    "blocker_codes",
    "bundle_sha256",
    "decision",
    "approved_canonical_id",
    "reviewer",
    "review_comment",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_results(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    return {
        (str(row["dataset_id"]), str(row["fault_class_id"])): row
        for row in receipt["results"]
    }


def _result_status(row: dict[str, Any]) -> str:
    return str(row.get("candidate_status", row.get("status", "")))


def _bundle_path(root: Path, dataset_id: str, fault_id: str) -> Path:
    direct = root / dataset_id / fault_id / "construction_bundle.json"
    stable = root / dataset_id / fault_id / "candidate" / "construction_bundle.json"
    return stable if stable.is_file() else direct


def _validate_stable_candidate_if_present(
    root: Path, dataset_id: str, fault_id: str, result: dict[str, Any]
) -> None:
    candidate_root = root / dataset_id / fault_id / "candidate"
    receipt_path = candidate_root / "receipt.json"
    if not receipt_path.is_file():
        return
    expected_receipt = result.get("candidate_receipt_file_sha256")
    if expected_receipt and _sha(receipt_path) != expected_receipt:
        raise ValueError("stable candidate summary and per-document receipt differ")
    receipt = DocumentCandidateBuildReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    for filename, expected in receipt.artifacts.items():
        path = candidate_root / filename
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"stable candidate artifact is missing or changed: {filename}")


def build_document_graph_review_handoff(
    *,
    primary_root: str | Path,
    supplemental_roots: list[str | Path] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a review CSV without populating any human decision field."""

    primary_root = Path(primary_root).resolve()
    supplementals = [Path(item).resolve() for item in (supplemental_roots or [])]
    output_dir = Path(output_dir).resolve()
    primary_receipt_path = primary_root / "receipt.json"
    primary_receipt = json.loads(primary_receipt_path.read_text(encoding="utf-8"))
    primary_results = _load_results(primary_root)
    supplemental_results = [_load_results(root) for root in supplementals]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("refusing to overwrite nonempty document graph review directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    unresolved_documents: list[str] = []
    source_bundle_paths: dict[str, str] = {}
    for key, primary_result in sorted(primary_results.items()):
        dataset_id, fault_id = key
        selected_root = primary_root
        selected_result = primary_result
        bundle_path = _bundle_path(primary_root, dataset_id, fault_id)
        _validate_stable_candidate_if_present(
            primary_root, dataset_id, fault_id, primary_result
        )
        if not bundle_path.is_file():
            for root, candidates in zip(supplementals, supplemental_results, strict=True):
                replacement = candidates.get(key)
                candidate_path = _bundle_path(root, dataset_id, fault_id)
                if replacement is not None and candidate_path.is_file():
                    _validate_stable_candidate_if_present(
                        root, dataset_id, fault_id, replacement
                    )
                    selected_root = root
                    selected_result = replacement
                    bundle_path = candidate_path
                    break
        if not bundle_path.is_file():
            unresolved_documents.append(f"{dataset_id}:{fault_id}")
            continue
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        proposal = bundle["proposal"]
        source_bundle_paths[f"{dataset_id}:{fault_id}"] = str(
            bundle_path.relative_to(selected_root)
        )
        blockers = ";".join(selected_result.get("blocker_codes", []))
        bundle_sha = str(bundle["bundle_sha256"])

        def common(candidate: dict[str, Any], kind: str) -> dict[str, str]:
            properties = candidate.get("properties", {})
            item_id = str(candidate["node_id"] if kind == "NODE" else candidate["edge_id"])
            review_id = "DOC-KG-REVIEW-" + _canonical_sha(
                {"dataset_id": dataset_id, "fault_id": fault_id, "kind": kind, "id": item_id}
            )[:20].upper()
            return {
                "review_item_id": review_id,
                "item_kind": kind,
                "dataset_id": dataset_id,
                "fault_class_id": fault_id,
                "source_document_id": str(properties.get("source_document_id", "")),
                "source_document_sha256": str(properties.get("source_document_sha256", "")),
                "candidate_id": item_id,
                "node_type": "",
                "canonical_name": "",
                "relation_type": "",
                "source_candidate_id": "",
                "target_candidate_id": "",
                "description_or_rationale": "",
                "source_fragment_ids": ";".join(candidate.get("source_fragment_ids", [])),
                "confidence": format(float(candidate.get("confidence", 0.0)), ".6g"),
                "ambiguous": str(bool(candidate.get("ambiguous", False))).lower(),
                "batch_status": _result_status(selected_result),
                "blocker_codes": blockers,
                "bundle_sha256": bundle_sha,
                "decision": "",
                "approved_canonical_id": "",
                "reviewer": "",
                "review_comment": "",
            }

        for node in proposal.get("nodes", []):
            row = common(node, "NODE")
            row.update(
                {
                    "node_type": str(node["node_type"]),
                    "canonical_name": str(node["canonical_name"]),
                    "description_or_rationale": str(node.get("description", "")),
                }
            )
            rows.append(row)
        for relation in proposal.get("relations", []):
            row = common(relation, "RELATION")
            row.update(
                {
                    "relation_type": str(relation["relation_type"]),
                    "source_candidate_id": str(relation["source_id"]),
                    "target_candidate_id": str(relation["target_id"]),
                    "description_or_rationale": str(relation.get("rationale", "")),
                }
            )
            rows.append(row)
    rows.sort(key=lambda row: (row["dataset_id"], row["fault_class_id"], row["item_kind"], row["candidate_id"]))
    if len({row["review_item_id"] for row in rows}) != len(rows):
        raise ValueError("document graph review item IDs are not unique")
    for row in rows:
        if any(row[field] for field in ("decision", "approved_canonical_id", "reviewer", "review_comment")):
            raise ValueError("machine-generated review sheet contains a human decision")

    csv_path = output_dir / "human_review_decisions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    receipt_body = {
        "schema_version": "astrograph-document-graph-review-handoff.v1",
        "primary_receipt_sha256": _sha(primary_receipt_path),
        "primary_batch_receipt_sha256": primary_receipt["receipt_sha256"],
        "supplemental_receipt_sha256": [
            _sha(root / "receipt.json") for root in supplementals
        ],
        "source_document_count": len(primary_results),
        "review_item_count": len(rows),
        "node_review_item_count": sum(row["item_kind"] == "NODE" for row in rows),
        "relation_review_item_count": sum(row["item_kind"] == "RELATION" for row in rows),
        "blocked_source_document_count": sum(
            _result_status(row) != "CANDIDATE_READY_FOR_HUMAN_REVIEW"
            for row in primary_results.values()
        ),
        "unresolved_source_documents": unresolved_documents,
        "source_bundle_paths": source_bundle_paths,
        "human_decision_count": 0,
        "review_sheet_sha256": _sha(csv_path),
        "publication_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "Blank human handoff for machine-extracted document graph candidates. "
            "Rows from BLOCKED bundles remain review evidence only. Empty reviewer fields "
            "must not be interpreted as approval."
        ),
    }
    receipt = {**receipt_body, "receipt_sha256": _canonical_sha(receipt_body)}
    (output_dir / "review_handoff_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def build_single_document_graph_review_handoff(
    *, candidate_package_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Create the same blank review contract for one stable candidate package."""

    root = Path(candidate_package_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite nonempty document graph review directory")
    receipt_path = root / "receipt.json"
    candidate_receipt = DocumentCandidateBuildReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    for filename, expected in candidate_receipt.artifacts.items():
        path = root / filename
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"document candidate artifact is missing or changed: {filename}")
    bundle_path = root / "construction_bundle.json"
    if not bundle_path.is_file():
        if candidate_receipt.status != "BLOCKED_BY_EXCEPTION":
            raise ValueError("document candidate package has no construction bundle")
        bundle = None
    else:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("bundle_sha256") != candidate_receipt.construction_bundle_sha256:
            raise ValueError("candidate receipt and construction bundle identities differ")
    dataset_id = "document-scope"
    fault_id = candidate_receipt.document_id
    blockers = ";".join(candidate_receipt.blocker_codes)
    rows: list[dict[str, str]] = []

    def common(candidate: dict[str, Any], kind: str) -> dict[str, str]:
        properties = candidate.get("properties", {})
        item_id = str(candidate["node_id"] if kind == "NODE" else candidate["edge_id"])
        review_id = "DOC-KG-REVIEW-" + _canonical_sha(
            {"dataset_id": dataset_id, "fault_id": fault_id, "kind": kind, "id": item_id}
        )[:20].upper()
        return {
            "review_item_id": review_id,
            "item_kind": kind,
            "dataset_id": dataset_id,
            "fault_class_id": fault_id,
            "source_document_id": str(properties.get("source_document_id", "")),
            "source_document_sha256": str(properties.get("source_document_sha256", "")),
            "candidate_id": item_id,
            "node_type": "",
            "canonical_name": "",
            "relation_type": "",
            "source_candidate_id": "",
            "target_candidate_id": "",
            "description_or_rationale": "",
            "source_fragment_ids": ";".join(candidate.get("source_fragment_ids", [])),
            "confidence": format(float(candidate.get("confidence", 0.0)), ".6g"),
            "ambiguous": str(bool(candidate.get("ambiguous", False))).lower(),
            "batch_status": candidate_receipt.status,
            "blocker_codes": blockers,
            "bundle_sha256": str(bundle.get("bundle_sha256", "")) if bundle else "",
            "decision": "",
            "approved_canonical_id": "",
            "reviewer": "",
            "review_comment": "",
        }

    proposal = (bundle or {}).get("proposal", {})
    for node in proposal.get("nodes", []):
        row = common(node, "NODE")
        row.update({
            "node_type": str(node["node_type"]),
            "canonical_name": str(node["canonical_name"]),
            "description_or_rationale": str(node.get("description", "")),
        })
        rows.append(row)
    for relation in proposal.get("relations", []):
        row = common(relation, "RELATION")
        row.update({
            "relation_type": str(relation["relation_type"]),
            "source_candidate_id": str(relation["source_id"]),
            "target_candidate_id": str(relation["target_id"]),
            "description_or_rationale": str(relation.get("rationale", "")),
        })
        rows.append(row)
    rows.sort(key=lambda row: (row["item_kind"], row["candidate_id"]))
    if len({row["review_item_id"] for row in rows}) != len(rows):
        raise ValueError("document graph review item IDs are not unique")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "human_review_decisions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    body = {
        "schema_version": "astrograph-single-document-graph-review-handoff.v1",
        "status": (
            "READY_FOR_HUMAN_REVIEW"
            if rows and candidate_receipt.status == "CANDIDATE_READY_FOR_HUMAN_REVIEW"
            else (
                "BLOCKED_CANDIDATES_FOR_INSPECTION"
                if rows
                else "NO_CANDIDATES_BLOCKED"
            )
        ),
        "candidate_receipt_sha256": _sha(receipt_path),
        "construction_bundle_sha256": (
            candidate_receipt.construction_bundle_sha256
        ),
        "document_id": candidate_receipt.document_id,
        "candidate_status": candidate_receipt.status,
        "review_item_count": len(rows),
        "node_review_item_count": sum(row["item_kind"] == "NODE" for row in rows),
        "relation_review_item_count": sum(
            row["item_kind"] == "RELATION" for row in rows
        ),
        "human_decision_count": 0,
        "review_sheet_sha256": _sha(csv_path),
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "Blank human handoff for one source-bound candidate package. Empty fields are not "
            "approval; a blocked or empty candidate remains closed."
        ),
    }
    receipt = {**body, "receipt_sha256": _canonical_sha(body)}
    (output / "review_handoff_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "REVIEW_FIELDS",
    "build_document_graph_review_handoff",
    "build_single_document_graph_review_handoff",
]
