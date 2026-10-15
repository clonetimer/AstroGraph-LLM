"""Materialize and gate document-graph relations as formal claim candidates."""
from __future__ import annotations

import hashlib

from astrograph.knowledge.construction import GovernedDocumentProcessingBundle
from astrograph.knowledge.schema_v3 import KnowledgeGraphV3

from .document_governance import (
    ClaimEpistemicStatus,
    DocumentEvidenceClaim,
    DocumentAuthority,
    DocumentVisibilityClass,
    bind_document_evidence_claim,
)
from .models import (
    DocumentClaimAdmission,
    DocumentClaimAdmissionPolicy,
    DocumentGraphTraceLink,
)


def materialize_document_graph_claims(
    processed: GovernedDocumentProcessingBundle,
    *,
    graph: KnowledgeGraphV3,
    trace_links: list[DocumentGraphTraceLink],
) -> list[DocumentEvidenceClaim]:
    """Convert linked graph relations to sourced claims without granting authority."""

    links_by_item: dict[str, list[DocumentGraphTraceLink]] = {}
    for link in trace_links:
        if link.graph_id != graph.graph_id:
            continue
        for item_id in link.graph_item_ids:
            links_by_item.setdefault(item_id, []).append(link)
    sections = {
        item.fragment_id: item for item in processed.knowledge_staging.source_tree.sections
    }
    descriptor = processed.knowledge_staging.document_descriptor
    claims: list[DocumentEvidenceClaim] = []
    for edge in sorted(graph.edges, key=lambda item: item.edge_id):
        for link in sorted(
            links_by_item.get(edge.edge_id, []),
            key=lambda item: (item.knowledge_fragment_id, item.retrieval_chunk_id),
        ):
            section = sections.get(link.knowledge_fragment_id)
            if section is None:
                raise ValueError("linked graph relation references an unknown source fragment")
            raw_status = edge.properties.get(
                "epistemic_status",
                processed.knowledge_staging.default_epistemic_status.value,
            )
            try:
                epistemic = ClaimEpistemicStatus(str(raw_status))
            except ValueError as exc:
                raise ValueError("graph relation has an unsupported epistemic status") from exc
            identity = "|".join(
                [graph.graph_id, edge.edge_id, link.knowledge_fragment_id, section.text_sha256]
            )
            applicability = [f"graph-scope:{link.graph_scope}"]
            if descriptor.episode_id:
                applicability.append(f"episode:{descriptor.episode_id}")
            if descriptor.provenance_group_id:
                applicability.append(f"provenance-group:{descriptor.provenance_group_id}")
            claims.append(
                bind_document_evidence_claim(
                    claim_id="document-claim:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    subject=edge.source_id,
                    predicate=edge.relation_type.value,
                    object=edge.target_id,
                    source_span=section.text,
                    source_locator=link.source_locators[0],
                    source_chunk_id=link.retrieval_chunk_id,
                    source_fragment_id=link.knowledge_fragment_id,
                    descriptor=descriptor,
                    epistemic_status=epistemic,
                    applicability=applicability,
                    publication_eligible=False,
                    runtime_evidence_eligible=False,
                )
            )
    if len({item.claim_id for item in claims}) != len(claims):
        raise ValueError("materialized document claim IDs are not unique")
    return claims


def evaluate_document_claim_admissions(
    claims: list[DocumentEvidenceClaim],
    policy: DocumentClaimAdmissionPolicy,
) -> list[DocumentClaimAdmission]:
    """Fail closed on authority, calibration, independence and applicability."""

    calibrated = set(policy.calibrated_claim_ids)
    independent = set(policy.independent_provenance_group_ids)
    context = set(policy.applicability_context)
    admissions: list[DocumentClaimAdmission] = []
    for claim in claims:
        reasons: list[str] = []
        if not policy.formal_document_fusion_enabled:
            reasons.append("FORMAL_DOCUMENT_FUSION_DISABLED")
        if not claim.runtime_evidence_eligible:
            reasons.append("CLAIM_NOT_RUNTIME_EVIDENCE")
        if claim.source_authority != DocumentAuthority.APPROVED_SOURCE:
            reasons.append("SOURCE_AUTHORITY_NOT_APPROVED")
        if claim.source_visibility_class is None:
            reasons.append("SOURCE_VISIBILITY_NOT_BOUND")
        if claim.source_visibility_class in {
            DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
            DocumentVisibilityClass.TRUTH_QUARANTINED,
        }:
            reasons.append("SOURCE_TRUTH_OR_TRAIN_ONLY_NOT_RUNTIME_ELIGIBLE")
        if claim.claim_id not in calibrated:
            reasons.append("CLAIM_NOT_CALIBRATED")
        if (
            claim.source_provenance_group_id is None
            or claim.source_provenance_group_id not in independent
        ):
            reasons.append("SOURCE_INDEPENDENCE_NOT_ESTABLISHED")
        # Each applicability entry is a *constraint*, not an interchangeable
        # search keyword. An episode-A claim with matching graph scope must
        # never be accepted into episode B by an OR-over-tags check.
        if not claim.applicability or not set(claim.applicability).issubset(context):
            reasons.append("APPLICABILITY_NOT_ESTABLISHED")
        if (claim.source_visibility_class == DocumentVisibilityClass.OBSERVATION_CAUSAL
                and not any(item.startswith('episode:') for item in claim.applicability)):
            reasons.append("OBSERVATION_EPISODE_BINDING_MISSING")
        eligible = not reasons
        admissions.append(
            DocumentClaimAdmission(
                claim_id=claim.claim_id,
                formal_fusion_eligible=eligible,
                reason_codes=reasons or ["FORMAL_FUSION_ELIGIBLE"],
                policy_sha256=policy.policy_sha256,
                fusion_approval_receipt_sha256=policy.fusion_approval_receipt_sha256,
                calibration_receipt_sha256=policy.calibration_receipt_sha256,
                source_independence_receipt_sha256=(
                    policy.source_independence_receipt_sha256
                ),
            )
        )
    return admissions


__all__ = [
    "DocumentClaimAdmission",
    "DocumentClaimAdmissionPolicy",
    "evaluate_document_claim_admissions",
    "materialize_document_graph_claims",
]
