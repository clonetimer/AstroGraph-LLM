from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from astrograph.evidence.models import EdgeReviewDecision, EvidenceLevel, EvidenceSourceStatus
from astrograph.evidence.registry import EvidenceRegistry


def validate_evidence_registry(registry: EvidenceRegistry) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, message: str, **details: Any) -> None:
        checks.append({
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "message": message,
            "details": details,
        })

    sources = registry.list_sources()
    reviews = registry.list_reviews()
    source_ids = [item.source_id for item in sources]
    review_ids = [item.review_id for item in reviews]
    add("source_ids_unique", len(source_ids) == len(set(source_ids)), "Evidence source IDs are unique.")
    add("review_ids_unique", len(review_ids) == len(set(review_ids)), "Edge review IDs are unique.")
    source_map = {item.source_id: item for item in sources}
    missing_sources = sorted({item.source_id for item in reviews if item.source_id not in source_map})
    add("review_sources_exist", not missing_sources, "All reviews reference registered sources.", missing=missing_sources)

    invalid_approvals: list[str] = []
    for review in reviews:
        if review.decision != EdgeReviewDecision.APPROVE:
            continue
        source = source_map.get(review.source_id)
        if source is None:
            continue
        if source.status != EvidenceSourceStatus.APPROVED or not source.eligible_for_edge_promotion:
            invalid_approvals.append(review.review_id)
    add(
        "approved_reviews_use_eligible_sources",
        not invalid_approvals,
        "Approved edge reviews use approved, promotion-eligible sources.",
        invalid=invalid_approvals,
    )
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {
        "schema_version": "astrograph-evidence-registry-validation.v1",
        "status": status,
        "source_count": len(sources),
        "review_count": len(reviews),
        "checks": checks,
    }


def validate_edges_against_evidence(
    edges_csv: str | Path,
    registry: EvidenceRegistry,
) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(edges_csv).open(encoding="utf-8-sig", newline="")))
    sources = {item.source_id: item for item in registry.list_sources()}
    reviews = registry.list_reviews()
    approved_keys = {
        (item.knowledge_version, item.fault_id, item.anomaly_id, item.source_id, item.proposed_evidence_level.value)
        for item in reviews
        if item.decision == EdgeReviewDecision.APPROVE
    }
    violations: list[dict[str, str]] = []
    for row in rows:
        level = row.get("evidence_level", "E0")
        if level == EvidenceLevel.E0.value:
            continue
        source_id = row.get("evidence_source", "")
        source = sources.get(source_id)
        key = (
            row.get("knowledge_version", ""),
            row.get("fault_id", ""),
            row.get("anomaly_id", ""),
            source_id,
            level,
        )
        if (
            source is None
            or source.status != EvidenceSourceStatus.APPROVED
            or not source.eligible_for_edge_promotion
            or key not in approved_keys
        ):
            violations.append({
                "fault_id": row.get("fault_id", ""),
                "anomaly_id": row.get("anomaly_id", ""),
                "evidence_level": level,
                "evidence_source": source_id,
            })
    return {
        "schema_version": "astrograph-edge-evidence-validation.v1",
        "status": "PASS" if not violations else "FAIL",
        "edge_count": len(rows),
        "violations": violations,
        "rule": "E1-E4 edges require an approved promotion-eligible source and an approved edge review.",
    }
