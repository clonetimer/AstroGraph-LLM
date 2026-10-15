"""Compile an explicit owner-approved TBox decision into scoped document claims.

This is not a second approval mechanism.  It reconstructs the 22 association
edges and their source fragments from the frozen graph/construction bundle,
then translates the existing owner decision into publishable
``ONTOLOGY_ASSOCIATION`` claims.  Episode evidence, causal truth, Active-KG
mutation and runtime voting remain prohibited.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from astrograph.retrieval.document_governance import (
    ClaimEpistemicStatus,
    ClaimTemporalScope,
    DocumentAuthority,
    DocumentEvidenceClaim,
    DocumentReviewState,
    DocumentVisibilityClass,
)

from .construction import KnowledgeConstructionBundle, canonical_sha256
from .schema_v3 import KnowledgeGraphV3, KnowledgeRelationTypeV3
from .tbox_approval import TBoxVocabularyApproval
from .source_paths import resolve_bound_source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_approved_tbox_claim_release(
    *,
    graph_path: str | Path,
    construction_bundle_path: str | Path,
    approval_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Reconstruct and atomically persist the approved TBox claim set."""

    graph_path = Path(graph_path).resolve()
    construction_path = Path(construction_bundle_path).resolve()
    approval_path = Path(approval_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite approved TBox claim release")
    graph = KnowledgeGraphV3.model_validate_json(graph_path.read_text(encoding="utf-8"))
    construction = KnowledgeConstructionBundle.model_validate_json(
        construction_path.read_text(encoding="utf-8")
    )
    approval = TBoxVocabularyApproval.model_validate_json(
        approval_path.read_text(encoding="utf-8")
    )
    graph_sha = canonical_sha256(graph.model_dump(mode="json"))
    if approval.graph_id != graph.graph_id or approval.graph_sha256 != graph_sha:
        raise ValueError("TBox approval does not bind the supplied graph")
    if construction.status != "CANDIDATE_READY_FOR_HUMAN_REVIEW":
        raise ValueError("TBox construction bundle is not a reviewable candidate")
    if construction.candidate_graph is None:
        raise ValueError("TBox construction bundle has no candidate graph")
    if canonical_sha256(construction.candidate_graph.model_dump(mode="json")) != graph_sha:
        raise ValueError("TBox construction bundle and supplied graph differ")
    if (
        approval.association_semantics != "TBOX_ASSOCIATION_ONLY"
        or approval.episode_evidence_approved
        or approval.actual_causal_propagation_approved
        or approval.runtime_diagnosis_or_voting_allowed
        or approval.active_graph_modified
    ):
        raise ValueError("TBox approval exceeds the permitted association-only scope")

    edges = {edge.edge_id: edge for edge in graph.edges}
    fragment_index: dict[str, tuple[Any, Any]] = {}
    for profile in construction.source_profiles:
        for fragment in profile.fragments:
            if fragment.fragment_id in fragment_index:
                raise ValueError(f"duplicate source fragment identity: {fragment.fragment_id}")
            fragment_index[fragment.fragment_id] = (profile, fragment)

    claims: list[DocumentEvidenceClaim] = []
    bindings: list[dict[str, str]] = []
    for edge_id in approval.approved_tbox_association_edge_ids:
        edge = edges.get(edge_id)
        if edge is None:
            raise ValueError(f"approved TBox edge is absent: {edge_id}")
        if edge.relation_type != KnowledgeRelationTypeV3.MANIFESTS_AS:
            raise ValueError(f"approved edge is not a MANIFESTS_AS association: {edge_id}")
        fragment_ids = list(edge.properties.get("source_fragment_ids", []))
        if len(fragment_ids) != 1 or fragment_ids[0] not in fragment_index:
            raise ValueError(f"approved TBox edge lacks one reconstructable source fragment: {edge_id}")
        profile, fragment = fragment_index[fragment_ids[0]]
        source_file = resolve_bound_source(profile.source_path, profile.source_sha256, anchor=construction_path)
        if profile.source_sha256 != _sha(source_file):
            raise ValueError(f"approved TBox source file changed: {profile.source_id}")
        if fragment.text_sha256 != hashlib.sha256(fragment.text.encode("utf-8")).hexdigest():
            raise ValueError(f"approved TBox source span hash changed: {fragment.fragment_id}")
        claim_id = "claim:tbox-association:" + _canonical_sha(
            {
                "approval_sha256": approval.approval_sha256,
                "edge_id": edge.edge_id,
                "fragment_id": fragment.fragment_id,
            }
        )[:24]
        claim = DocumentEvidenceClaim(
            claim_id=claim_id,
            subject=edge.source_id,
            predicate=edge.relation_type.value,
            object=edge.target_id,
            source_document_id=profile.source_id,
            source_document_sha256=profile.source_sha256,
            source_locator=fragment.location,
            source_fragment_id=fragment.fragment_id,
            source_visibility_class=DocumentVisibilityClass.DESIGN_PUBLIC,
            source_provenance_group_id=profile.independence_group,
            epistemic_status=ClaimEpistemicStatus.ONTOLOGY_ASSOCIATION,
            source_review_state=DocumentReviewState.HUMAN_REVIEWED,
            source_authority=DocumentAuthority.APPROVED_SOURCE,
            synthetic_from_simulation=False,
            temporal_scope=ClaimTemporalScope.TIME_INDEPENDENT,
            applicability=[
                "graph-scope:TBOX_ASSOCIATION_ONLY",
                f"approval:{approval.approval_sha256}",
            ],
            source_span_sha256=fragment.text_sha256,
            publication_eligible=True,
            source_runtime_vote_allowed=False,
            runtime_evidence_eligible=False,
        )
        claims.append(claim)
        bindings.append(
            {
                "claim_id": claim_id,
                "edge_id": edge.edge_id,
                "source_id": profile.source_id,
                "source_file_sha256": profile.source_sha256,
                "source_fragment_id": fragment.fragment_id,
                "source_locator": fragment.location,
                "source_span_sha256": fragment.text_sha256,
            }
        )
    claims.sort(key=lambda item: item.claim_id)
    bindings.sort(key=lambda item: item["claim_id"])
    if len(claims) != 22 or len({claim.claim_id for claim in claims}) != len(claims):
        raise ValueError("approved TBox claim release must contain exactly 22 unique claims")
    if {binding["claim_id"] for binding in bindings} != {claim.claim_id for claim in claims}:
        raise ValueError("approved TBox claim/source bindings are incomplete")

    body = {
        "schema_version": "astrograph-approved-tbox-document-claim-release.v1",
        "release_id": "document-claims:tbox-association:" + approval.approval_sha256[:20],
        "status": "APPROVED_TBOX_CLAIM_SET",
        "graph_id": graph.graph_id,
        "graph_sha256": graph_sha,
        "graph_file_sha256": _sha(graph_path),
        "construction_bundle_sha256": construction.bundle_sha256,
        "construction_bundle_file_sha256": _sha(construction_path),
        "approval_sha256": approval.approval_sha256,
        "approval_file_sha256": _sha(approval_path),
        "decision_actor": approval.decision_actor,
        "decision_date": approval.decision_date,
        "association_semantics": approval.association_semantics,
        "claim_count": len(claims),
        "source_binding_count": len(bindings),
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "source_bindings": bindings,
        "claim_set_publication_eligible": True,
        "active_graph_publication_allowed": False,
        "episode_evidence_approved": False,
        "actual_causal_propagation_approved": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "The 22 claims publish only the project-owner-approved MANIFESTS_AS TBox "
            "associations. They are not observations, actual causal propagation, diagnosis "
            "rules, calibration evidence, Active-KG publication or runtime votes."
        ),
    }
    result = {**body, "release_sha256": _canonical_sha(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


__all__ = ["build_approved_tbox_claim_release"]
