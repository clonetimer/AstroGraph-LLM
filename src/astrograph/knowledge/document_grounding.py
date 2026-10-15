"""Field-level grounding checks for document-derived LLM graph candidates."""
from __future__ import annotations

import math
from typing import Any

from astrograph.agents.knowledge_curation import (
    KnowledgeExtractionProposal,
    KnowledgeSourceProfile,
)


def _table_rows(profile: KnowledgeSourceProfile) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for fragment in profile.fragments:
        for line in fragment.text.splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 7 and cells[0] not in {"Channel", "---"}:
                try:
                    rows[cells[0]] = {
                        "fragment_id": fragment.fragment_id,
                        "count": int(cells[1]),
                        "first": float(cells[2]),
                        "last": float(cells[3]),
                        "minimum": float(cells[4]),
                        "maximum": float(cells[5]),
                        "mean_or_distinct": float(cells[6]),
                    }
                except ValueError:
                    continue
    return rows


def _close(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=1e-8, abs_tol=1e-10)
    except (TypeError, ValueError):
        return False


def audit_document_proposal_grounding(
    *,
    source_profile: KnowledgeSourceProfile,
    proposal: KnowledgeExtractionProposal,
) -> dict[str, Any]:
    """Verify explicit table values without treating derived metadata as quoted text."""

    rows = _table_rows(source_profile)
    fragment_ids = {fragment.fragment_id for fragment in source_profile.fragments}
    findings: list[dict[str, str]] = []
    checked_fields = 0
    grounded_fields = 0
    cited_candidates = 0
    valid_cited_candidates = 0
    candidates = [*proposal.nodes, *proposal.relations]
    for candidate in candidates:
        candidate_id = getattr(candidate, "node_id", None) or getattr(
            candidate, "edge_id", "unknown"
        )
        cited_candidates += 1
        missing = sorted(set(candidate.source_fragment_ids) - fragment_ids)
        if missing:
            findings.append({
                "code": "SOURCE_FRAGMENT_NOT_FOUND",
                "candidate_id": candidate_id,
                "field": "source_fragment_ids",
                "detail": ",".join(missing),
            })
        else:
            valid_cited_candidates += 1
    for node in proposal.nodes:
        if node.node_type.value != "StateEstimate":
            continue
        channel = str(node.properties.get("channel") or node.canonical_name)
        row = rows.get(channel)
        if row is None:
            findings.append({
                "code": "STATE_ESTIMATE_ROW_NOT_FOUND",
                "candidate_id": node.node_id,
                "field": "channel",
                "detail": channel,
            })
            continue
        if row["fragment_id"] not in node.source_fragment_ids:
            findings.append({
                "code": "STATE_ESTIMATE_ROW_NOT_CITED",
                "candidate_id": node.node_id,
                "field": "source_fragment_ids",
                "detail": row["fragment_id"],
            })
        expected = {
            "count": row["count"],
            "sample_count": row["count"],
            "first": row["first"],
            "last": row["last"],
            "minimum": row["minimum"],
            "min_value": row["minimum"],
            "maximum": row["maximum"],
            "max_value": row["maximum"],
            "mean": row["mean_or_distinct"],
            "mean_value": row["mean_or_distinct"],
        }
        for field, expected_value in expected.items():
            if field not in node.properties:
                continue
            checked_fields += 1
            if _close(node.properties[field], float(expected_value)):
                grounded_fields += 1
            else:
                findings.append({
                    "code": "TABLE_VALUE_MISMATCH",
                    "candidate_id": node.node_id,
                    "field": field,
                    "detail": f"candidate={node.properties[field]!r}; source={expected_value!r}",
                })
        if "distinct" in node.properties:
            checked_fields += 1
            findings.append({
                "code": "UNSUPPORTED_DISTINCT_COUNT",
                "candidate_id": node.node_id,
                "field": "distinct",
                "detail": (
                    "The numeric Channel summaries row supplies one Mean / distinct value; "
                    "it does not separately supply a distinct count."
                ),
            })
    explicit_field_status = (
        "GROUNDED"
        if checked_fields > 0 and grounded_fields == checked_fields
        else "PARTIAL"
        if checked_fields > 0
        else "NO_EXPLICIT_FIELDS_TO_CHECK"
        if not proposal.nodes
        else "NO_SUPPORTED_FIELD_CHECK"
    )
    return {
        "schema_version": "astrograph-document-proposal-grounding-audit.v1",
        "status": "BLOCKED" if findings else "PASS",
        "proposal_node_count": len(proposal.nodes),
        "proposal_relation_count": len(proposal.relations),
        "cited_candidate_count": cited_candidates,
        "valid_cited_candidate_count": valid_cited_candidates,
        "source_trace_status": (
            "NOT_APPLICABLE"
            if not candidates
            else "COMPLETE"
            if valid_cited_candidates == cited_candidates
            else "INCOMPLETE"
        ),
        "checked_explicit_field_count": checked_fields,
        "grounded_explicit_field_count": grounded_fields,
        "explicit_field_grounding_status": explicit_field_status,
        "findings": findings,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "claim_boundary": (
            "Field-level source grounding only. PASS does not establish domain truth, "
            "calibration, human approval or Runtime evidence authority."
        ),
    }


def require_document_proposal_grounding(
    *,
    source_profile: KnowledgeSourceProfile,
    proposal: KnowledgeExtractionProposal,
) -> dict[str, Any]:
    audit = audit_document_proposal_grounding(
        source_profile=source_profile, proposal=proposal
    )
    if audit["status"] != "PASS":
        codes = sorted({item["code"] for item in audit["findings"]})
        raise ValueError(f"document proposal grounding failed: {codes}")
    return audit


__all__ = ["audit_document_proposal_grounding", "require_document_proposal_grounding"]
