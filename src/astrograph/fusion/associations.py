from __future__ import annotations

from typing import Iterable

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceRole
from astrograph.knowledge.query import GraphEvidenceAssociation
from astrograph.knowledge.spatiotemporal_fault_graph import SpatiotemporalTemplateMatch

from .models import FaultRanking
from .ranking import build_backward_evidence_ranking, build_forward_graph_ranking

GRAPH_ASSOCIATION_RANKING_PRODUCER_VERSION = (
    "0.7.1.post24+b006.wip.kgcoverage.graph.v1"
)


def validate_and_filter_graph_associations(
    bundle: EpisodeEvidenceBundle,
    associations: Iterable[GraphEvidenceAssociation],
) -> EpisodeEvidenceBundle:
    association_list = [
        item if isinstance(item, GraphEvidenceAssociation) else GraphEvidenceAssociation.model_validate(item)
        for item in associations
    ]
    definitions = {item.evidence_definition_id: item for item in bundle.evidence_definitions}
    role_relation = {
        EvidenceRole.REQUIRED: "REQUIRES_EVIDENCE",
        EvidenceRole.SUPPORTING: "SUPPORTED_BY",
        EvidenceRole.COUNTER: "CONTRADICTED_BY",
        EvidenceRole.EXCLUSION: "EXCLUDES",
    }
    allowed_pairs: set[tuple[str, str]] = set()
    for association in association_list:
        definition = definitions.get(association.evidence_definition_id)
        if definition is None:
            raise ValueError(
                "graph association references evidence definition outside the episode bundle: "
                f"{association.evidence_definition_id}"
            )
        if definition.target_fault_id != association.fault_id:
            raise ValueError(
                f"graph association fault/evidence mismatch: {association.fault_id} -> "
                f"{association.evidence_definition_id} belongs to {definition.target_fault_id}"
            )
        expected_relation = role_relation[definition.role]
        if association.relation_type != expected_relation:
            raise ValueError(
                f"graph association relation mismatch for {association.evidence_definition_id}: "
                f"{association.relation_type} != {expected_relation}"
            )
        allowed_pairs.add((association.fault_id, association.evidence_definition_id))
    filtered_definitions = [
        item
        for item in bundle.evidence_definitions
        if (item.target_fault_id, item.evidence_definition_id) in allowed_pairs
    ]
    allowed_definition_ids = {item.evidence_definition_id for item in filtered_definitions}
    filtered_facts = [
        item for item in bundle.evidence_facts if item.evidence_definition_id in allowed_definition_ids
    ]
    return bundle.model_copy(
        update={
            "evidence_definitions": filtered_definitions,
            "evidence_facts": filtered_facts,
            "evaluation_metadata": {
                **bundle.evaluation_metadata,
                "graph_association_source": "PARAMETERIZED_READ_ONLY_QUERY",
                "graph_association_count": len(association_list),
            },
        }
    )


def build_forward_graph_ranking_from_associations(
    bundle: EpisodeEvidenceBundle,
    associations: Iterable[GraphEvidenceAssociation],
    *,
    spatiotemporal_matches: Iterable[SpatiotemporalTemplateMatch] | None = None,
    spatiotemporal_weight: float = 0.0,
) -> FaultRanking:
    filtered = validate_and_filter_graph_associations(bundle, associations)
    ranking = build_forward_graph_ranking(
        filtered,
        spatiotemporal_matches=spatiotemporal_matches,
        spatiotemporal_weight=spatiotemporal_weight,
    )
    return ranking.model_copy(
        update={
            "producer_id": "neo4j-read-only-forward-evidence-ranking",
            "producer_version": GRAPH_ASSOCIATION_RANKING_PRODUCER_VERSION,
            "metadata": {
                **ranking.metadata,
                "graph_query_backed": True,
                "association_count": len(filtered.evidence_definitions),
            },
        }
    )


def build_backward_evidence_ranking_from_associations(
    bundle: EpisodeEvidenceBundle,
    associations: Iterable[GraphEvidenceAssociation],
    *,
    candidate_fault_ids: Iterable[str] | None = None,
) -> FaultRanking:
    filtered = validate_and_filter_graph_associations(bundle, associations)
    ranking = build_backward_evidence_ranking(filtered, candidate_fault_ids=candidate_fault_ids)
    return ranking.model_copy(
        update={
            "producer_id": "neo4j-read-only-backward-evidence-ranking",
            "producer_version": GRAPH_ASSOCIATION_RANKING_PRODUCER_VERSION,
            "metadata": {
                **ranking.metadata,
                "graph_query_backed": True,
                "association_count": len(filtered.evidence_definitions),
            },
        }
    )
