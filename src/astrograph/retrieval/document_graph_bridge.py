"""Trace the two governed document branches without promoting candidate knowledge."""
from __future__ import annotations

from astrograph.knowledge.construction import (
    GovernedDocumentProcessingBundle,
    canonical_sha256,
)
from astrograph.knowledge.schema import ReviewStatus
from astrograph.knowledge.schema_v3 import KnowledgeGraphV3, LifecycleStatus

from .models import DocumentGraphTraceLink


def build_document_graph_trace_links(
    processed: GovernedDocumentProcessingBundle,
    *,
    candidate_graph: KnowledgeGraphV3 | None = None,
    reviewed_graph: KnowledgeGraphV3 | None = None,
) -> list[DocumentGraphTraceLink]:
    """Build chunk↔fragment links with explicit staging/candidate/reviewed state.

    Passing no graph yields staging trace links.  This function never publishes a
    graph, grants evidence status, or enables a runtime vote.
    """

    source = processed.retrieval_ingestion
    descriptor = processed.knowledge_staging.document_descriptor
    known_fragments = {item.knowledge_fragment_id for item in processed.crosswalk}

    def matching_fragments(properties: dict, item_id: str) -> set[str]:
        fragments = set(map(str, properties.get("source_fragment_ids", []))) & known_fragments
        if fragments and (
            properties.get("source_document_id") != descriptor.document_id
            or properties.get("source_document_sha256") != source.source_sha256
        ):
            raise ValueError(
                f"graph item {item_id} matches a fragment but not its governed document source"
            )
        return fragments

    candidate_items_by_fragment: dict[str, set[str]] = {}
    if candidate_graph is not None:
        for node in candidate_graph.nodes:
            if node.lifecycle_status not in {LifecycleStatus.DRAFT, LifecycleStatus.ACTIVE}:
                continue
            for fragment_id in matching_fragments(node.properties, node.node_id):
                candidate_items_by_fragment.setdefault(fragment_id, set()).add(node.node_id)
        for edge in candidate_graph.edges:
            if edge.lifecycle_status not in {LifecycleStatus.DRAFT, LifecycleStatus.ACTIVE}:
                continue
            for fragment_id in matching_fragments(edge.properties, edge.edge_id):
                candidate_items_by_fragment.setdefault(fragment_id, set()).add(edge.edge_id)

    reviewed_items_by_fragment: dict[str, set[str]] = {}
    if reviewed_graph is not None:
        for node in reviewed_graph.nodes:
            if node.lifecycle_status is not LifecycleStatus.ACTIVE:
                continue
            for fragment_id in matching_fragments(node.properties, node.node_id):
                reviewed_items_by_fragment.setdefault(fragment_id, set()).add(node.node_id)
        for edge in reviewed_graph.edges:
            if (
                edge.lifecycle_status is not LifecycleStatus.ACTIVE
                or edge.review_status is not ReviewStatus.HUMAN_REVIEWED
            ):
                continue
            for fragment_id in matching_fragments(edge.properties, edge.edge_id):
                reviewed_items_by_fragment.setdefault(fragment_id, set()).add(edge.edge_id)

    scope = processed.knowledge_staging.graph_scope.value
    links: list[DocumentGraphTraceLink] = []
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for crosswalk in sorted(
        processed.crosswalk,
        key=lambda item: (item.retrieval_chunk_id, item.knowledge_fragment_id),
    ):
        key = (
            crosswalk.retrieval_chunk_id,
            crosswalk.knowledge_section_id,
            crosswalk.knowledge_fragment_id,
        )
        grouped.setdefault(key, set()).add(crosswalk.source_locator)

    candidate_sha = (
        canonical_sha256(candidate_graph.model_dump(mode="json"))
        if candidate_graph is not None else None
    )
    reviewed_sha = (
        canonical_sha256(reviewed_graph.model_dump(mode="json"))
        if reviewed_graph is not None else None
    )
    for (chunk_id, section_id, fragment_id), locators in sorted(grouped.items()):
        reviewed_ids = sorted(reviewed_items_by_fragment.get(fragment_id, set()))
        candidate_ids = sorted(candidate_items_by_fragment.get(fragment_id, set()))
        if reviewed_ids:
            link_status = "REVIEWED_GRAPH_LINK"
            graph = reviewed_graph
            graph_sha = reviewed_sha
            graph_item_ids = reviewed_ids
        elif candidate_ids:
            link_status = "CANDIDATE_GRAPH_LINK"
            graph = candidate_graph
            graph_sha = candidate_sha
            graph_item_ids = candidate_ids
        else:
            link_status = "STAGING_TRACE_ONLY"
            graph = None
            graph_sha = None
            graph_item_ids = []
        links.append(
            DocumentGraphTraceLink(
                retrieval_chunk_id=chunk_id,
                source_document_id=source.source_id,
                source_document_sha256=source.source_sha256,
                knowledge_section_id=section_id,
                knowledge_fragment_id=fragment_id,
                source_locators=sorted(locators),
                graph_scope=scope,
                link_status=link_status,
                graph_id=graph.graph_id if graph is not None else None,
                graph_sha256=graph_sha,
                graph_item_ids=graph_item_ids,
            )
        )
    return links


__all__ = ["build_document_graph_trace_links"]
