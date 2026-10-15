"""Validate human document-graph gold and compute strict extraction metrics."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from .document_graph_gold import (
    CANDIDATE_FIELDS,
    FRAGMENT_FIELDS,
    HUMAN_CANDIDATE_FIELDS,
    HUMAN_FRAGMENT_FIELDS,
    build_document_graph_gold_handoff,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != fields:
            raise ValueError(f"gold sheet columns changed: {path.name}")
        return list(reader)


def _norm(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum()
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_document_graph_gold(
    *, pilot_root: str | Path, handoff_root: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Fail closed until both candidate and exhaustive fragment annotations are complete."""

    pilot_root = Path(pilot_root).resolve()
    handoff_root = Path(handoff_root).resolve()
    output_path = Path(output_path).resolve()
    actual_candidate_path = handoff_root / "candidate_adjudication.csv"
    actual_fragment_path = handoff_root / "fragment_gold_annotations.csv"
    actual_candidates = _read(actual_candidate_path, CANDIDATE_FIELDS)
    actual_fragments = _read(actual_fragment_path, FRAGMENT_FIELDS)
    with tempfile.TemporaryDirectory(prefix="astrograph-gold-baseline-") as temp:
        baseline_root = Path(temp)
        build_document_graph_gold_handoff(pilot_root=pilot_root, output_root=baseline_root)
        expected_candidates = _read(baseline_root / "candidate_adjudication.csv", CANDIDATE_FIELDS)
        expected_fragments = _read(baseline_root / "fragment_gold_annotations.csv", FRAGMENT_FIELDS)

    blockers: list[dict[str, str]] = []
    machine_candidate_fields = [field for field in CANDIDATE_FIELDS if field not in HUMAN_CANDIDATE_FIELDS]
    machine_fragment_fields = [field for field in FRAGMENT_FIELDS if field not in HUMAN_FRAGMENT_FIELDS]
    if len(actual_candidates) != len(expected_candidates):
        blockers.append({"code": "CANDIDATE_ROW_COUNT_CHANGED", "item_id": "*"})
    if len(actual_fragments) != len(expected_fragments):
        blockers.append({"code": "FRAGMENT_ROW_COUNT_CHANGED", "item_id": "*"})
    for index, (actual, expected) in enumerate(zip(actual_candidates, expected_candidates, strict=False)):
        changed = [field for field in machine_candidate_fields if actual[field] != expected[field]]
        if changed:
            blockers.append({"code": "CANDIDATE_MACHINE_FIELD_CHANGED", "item_id": str(index),
                             "detail": ";".join(changed)})
    for index, (actual, expected) in enumerate(zip(actual_fragments, expected_fragments, strict=False)):
        changed = [field for field in machine_fragment_fields if actual[field] != expected[field]]
        if changed:
            blockers.append({"code": "FRAGMENT_MACHINE_FIELD_CHANGED", "item_id": str(index),
                             "detail": ";".join(changed)})

    for row in actual_candidates:
        decision = row["decision"].strip().upper()
        item_id = row["candidate_id"]
        if decision not in {"CORRECT", "INCORRECT", "PARTIAL"}:
            blockers.append({"code": "CANDIDATE_DECISION_REQUIRED", "item_id": item_id})
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append({"code": "CANDIDATE_REVIEWER_COMMENT_REQUIRED", "item_id": item_id})
        corrected = row["corrected_candidate_json"].strip()
        if decision == "PARTIAL" and not corrected:
            blockers.append({"code": "PARTIAL_CORRECTION_REQUIRED", "item_id": item_id})
        if corrected:
            try:
                parsed = json.loads(corrected)
                if not isinstance(parsed, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                blockers.append({"code": "INVALID_CORRECTED_CANDIDATE_JSON", "item_id": item_id})

    gold_nodes: set[tuple[str, str, str]] = set()
    gold_relations: set[tuple[str, str, str, str]] = set()
    for row in actual_fragments:
        fragment_id = row["fragment_id"]
        if row["annotation_complete"].strip().lower() != "true":
            blockers.append({"code": "FRAGMENT_ANNOTATION_INCOMPLETE", "item_id": fragment_id})
        if not row["reviewer"].strip() or not row["review_comment"].strip():
            blockers.append({"code": "FRAGMENT_REVIEWER_COMMENT_REQUIRED", "item_id": fragment_id})
        parsed_values: dict[str, list[Any]] = {}
        for field in ("gold_nodes_json", "gold_relations_json"):
            try:
                value = json.loads(row[field])
                if not isinstance(value, list):
                    raise ValueError
                parsed_values[field] = value
            except (json.JSONDecodeError, ValueError):
                blockers.append({"code": "INVALID_GOLD_JSON", "item_id": fragment_id,
                                 "detail": field})
                parsed_values[field] = []
        for node in parsed_values["gold_nodes_json"]:
            if not isinstance(node, dict) or not all(node.get(key) for key in ("node_type", "canonical_name")):
                blockers.append({"code": "INVALID_GOLD_NODE", "item_id": fragment_id})
                continue
            gold_nodes.add((row["document_id"], str(node["node_type"]), _norm(str(node["canonical_name"]))))
        for relation in parsed_values["gold_relations_json"]:
            keys = ("relation_type", "source_id", "target_id")
            if not isinstance(relation, dict) or not all(relation.get(key) for key in keys):
                blockers.append({"code": "INVALID_GOLD_RELATION", "item_id": fragment_id})
                continue
            gold_relations.add((row["document_id"], str(relation["relation_type"]),
                                str(relation["source_id"]), str(relation["target_id"])))

    predicted_nodes: set[tuple[str, str, str]] = set()
    predicted_relations: set[tuple[str, str, str, str]] = set()
    decision_counts = {key: 0 for key in ("CORRECT", "INCORRECT", "PARTIAL")}
    for row in actual_candidates:
        decision = row["decision"].strip().upper()
        if decision in decision_counts:
            decision_counts[decision] += 1
        try:
            candidate = json.loads(row["candidate_json"])
        except json.JSONDecodeError:
            continue
        if row["item_kind"] == "NODE":
            predicted_nodes.add((row["document_id"], str(candidate["node_type"]),
                                 _norm(str(candidate["canonical_name"]))))
        else:
            predicted_relations.add((row["document_id"], str(candidate["relation_type"]),
                                     str(candidate["source_id"]), str(candidate["target_id"])))

    ready = not blockers
    node_matches = len(predicted_nodes & gold_nodes)
    relation_matches = len(predicted_relations & gold_relations)
    exact_matches = node_matches + relation_matches
    predicted_total = len(predicted_nodes) + len(predicted_relations)
    gold_total = len(gold_nodes) + len(gold_relations)
    exact_precision = _safe_ratio(exact_matches, predicted_total) if ready else None
    exact_recall = _safe_ratio(exact_matches, gold_total) if ready else None
    exact_f1 = (
        2 * exact_precision * exact_recall / (exact_precision + exact_recall)
        if ready and exact_precision is not None and exact_recall is not None
        and exact_precision + exact_recall > 0
        else None
    )
    body = {
        "schema_version": "astrograph-document-graph-gold-evaluation.v1",
        "status": "PASS" if ready else "BLOCKED",
        "pilot_receipt_sha256": _sha(pilot_root / "receipt.json"),
        "candidate_sheet_sha256": _sha(actual_candidate_path),
        "fragment_sheet_sha256": _sha(actual_fragment_path),
        "decision_counts": decision_counts,
        "annotation_counts": {
            "candidate_rows": len(actual_candidates), "fragment_rows": len(actual_fragments),
            "predicted_nodes": len(predicted_nodes), "predicted_relations": len(predicted_relations),
            "gold_nodes": len(gold_nodes), "gold_relations": len(gold_relations),
        },
        "strict_exact_metrics": {
            "node_matches": node_matches if ready else None,
            "relation_matches": relation_matches if ready else None,
            "precision": exact_precision, "recall": exact_recall, "f1": exact_f1,
        },
        "candidate_adjudication_precision": (
            _safe_ratio(decision_counts["CORRECT"], len(actual_candidates)) if ready else None
        ),
        "blockers": blockers,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "claim_boundary": (
            "Human-gold extraction metrics only. PASS does not establish diagnostic gain, "
            "causal truth, Active KG publication or Runtime voting authority."
        ),
    }
    result = {**body, "evaluation_sha256": _canonical_sha(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["evaluate_document_graph_gold"]
