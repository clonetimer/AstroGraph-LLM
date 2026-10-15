"""Create a truth-isolated human gold handoff for document graph extraction."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal

from .construction import DocumentKnowledgeStagingBundle


CANDIDATE_FIELDS = [
    "document_id", "lane", "dataset_id", "graph_scope", "source_sha256",
    "item_kind", "candidate_id", "candidate_type", "source_candidate_id",
    "target_candidate_id", "source_fragment_ids", "candidate_json",
    "source_fragments_json", "decision", "corrected_candidate_json", "reviewer", "review_comment",
]
FRAGMENT_FIELDS = [
    "document_id", "lane", "dataset_id", "graph_scope", "source_sha256",
    "fragment_id", "location", "text_sha256", "source_text",
    "gold_nodes_json", "gold_relations_json", "annotation_complete", "reviewer", "review_comment",
]
HUMAN_CANDIDATE_FIELDS = ("decision", "corrected_candidate_json", "reviewer", "review_comment")
HUMAN_FRAGMENT_FIELDS = (
    "gold_nodes_json", "gold_relations_json", "annotation_complete", "reviewer", "review_comment"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_document_graph_gold_handoff(
    *, pilot_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Write local detailed review sheets and a content-free machine receipt."""

    pilot_root = Path(pilot_root).resolve()
    output_root = Path(output_root).resolve()
    receipt_path = pilot_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("refusing to overwrite nonempty document graph gold handoff")
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, str]] = []
    fragment_rows: list[dict[str, str]] = []
    lane_counts: dict[str, dict[str, int]] = {}
    for row in receipt["results"]:
        directory = (
            pilot_root / row["lane"] / row["dataset_id"] / row["document_id"].replace(":", "_")
        )
        proposal = KnowledgeExtractionProposal.model_validate_json(
            (directory / "raw_extraction_proposal.json").read_text(encoding="utf-8")
        )
        staging = DocumentKnowledgeStagingBundle.model_validate_json(
            (directory / "knowledge_staging.json").read_text(encoding="utf-8")
        )
        profile = staging.source_profile
        fragments = {item.fragment_id: item for item in profile.fragments}
        common = {
            "document_id": row["document_id"], "lane": row["lane"],
            "dataset_id": row["dataset_id"], "graph_scope": row["graph_scope"],
            "source_sha256": row["source_sha256"],
        }
        for candidate in [*proposal.nodes, *proposal.relations]:
            is_node = hasattr(candidate, "node_id")
            cited = [fragments[fragment_id] for fragment_id in candidate.source_fragment_ids]
            candidate_rows.append({
                **common,
                "item_kind": "NODE" if is_node else "RELATION",
                "candidate_id": candidate.node_id if is_node else candidate.edge_id,
                "candidate_type": candidate.node_type.value if is_node else candidate.relation_type.value,
                "source_candidate_id": "" if is_node else candidate.source_id,
                "target_candidate_id": "" if is_node else candidate.target_id,
                "source_fragment_ids": ";".join(candidate.source_fragment_ids),
                "candidate_json": candidate.model_dump_json(),
                "source_fragments_json": json.dumps(
                    [{"fragment_id": item.fragment_id, "location": item.location, "text": item.text}
                     for item in cited], ensure_ascii=False, separators=(",", ":")
                ),
                **{field: "" for field in HUMAN_CANDIDATE_FIELDS},
            })
        for fragment in profile.fragments:
            fragment_rows.append({
                **common,
                "fragment_id": fragment.fragment_id,
                "location": fragment.location,
                "text_sha256": fragment.text_sha256,
                "source_text": fragment.text,
                **{field: "" for field in HUMAN_FRAGMENT_FIELDS},
            })
        counts = lane_counts.setdefault(row["lane"], {"documents": 0, "candidates": 0, "fragments": 0})
        counts["documents"] += 1
        counts["candidates"] += len(proposal.nodes) + len(proposal.relations)
        counts["fragments"] += len(profile.fragments)

    candidate_rows.sort(key=lambda item: (item["lane"], item["dataset_id"], item["document_id"], item["item_kind"], item["candidate_id"]))
    fragment_rows.sort(key=lambda item: (item["lane"], item["dataset_id"], item["document_id"], item["fragment_id"]))
    candidate_path = output_root / "candidate_adjudication.csv"
    fragment_path = output_root / "fragment_gold_annotations.csv"
    _write_csv(candidate_path, CANDIDATE_FIELDS, candidate_rows)
    _write_csv(fragment_path, FRAGMENT_FIELDS, fragment_rows)
    if any(row[field] for row in candidate_rows for field in HUMAN_CANDIDATE_FIELDS):
        raise ValueError("machine candidate handoff populated a human field")
    if any(row[field] for row in fragment_rows for field in HUMAN_FRAGMENT_FIELDS):
        raise ValueError("machine fragment handoff populated a human field")
    body = {
        "schema_version": "astrograph-document-graph-gold-handoff.v1",
        "status": "UNREVIEWED_HUMAN_GOLD_PENDING",
        "pilot_receipt_sha256": _sha(receipt_path),
        "document_count": len(receipt["results"]),
        "candidate_review_item_count": len(candidate_rows),
        "fragment_gold_item_count": len(fragment_rows),
        "lane_counts": lane_counts,
        "candidate_sheet_sha256": _sha(candidate_path),
        "fragment_sheet_sha256": _sha(fragment_path),
        "human_decision_count": 0,
        "local_detailed_artifacts_only": True,
        "quarantine_content_copied_to_governance_receipt": False,
        "precision_evaluation_ready": False,
        "recall_evaluation_ready": False,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "claim_boundary": (
            "Blank human annotation handoff. Candidate adjudication supports precision only; "
            "complete per-fragment node/relation gold is additionally required for recall. "
            "Detailed source text remains local and evaluator truth cannot enter Runtime."
        ),
    }
    result = {**body, "handoff_sha256": _canonical_sha(body)}
    (output_root / "handoff_receipt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["build_document_graph_gold_handoff"]
