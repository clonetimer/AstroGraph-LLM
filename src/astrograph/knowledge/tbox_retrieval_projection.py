"""Project owner-approved TBox vocabulary into a retrieval-only V3 graph."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import ReviewStatus
from .schema_v3 import (
    DEFAULT_GRAPH_SHAPE_VALIDATOR,
    GraphEdgeV3,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
)
from .tbox_approval import TBoxVocabularyApproval, canonical_sha256


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reviewed_tbox_retrieval_projection(
    *,
    source_graph_path: str | Path,
    approval_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create exactly the approved 12+10 vocabulary and association subgraph."""

    source_path = Path(source_graph_path).resolve()
    decision_path = Path(approval_path).resolve()
    output = Path(output_dir).resolve()
    source = KnowledgeGraphV3.model_validate_json(source_path.read_text(encoding="utf-8"))
    approval = TBoxVocabularyApproval.model_validate_json(
        decision_path.read_text(encoding="utf-8")
    )
    source_sha = canonical_sha256(source.model_dump(mode="json"))
    if approval.graph_id != source.graph_id or approval.graph_sha256 != source_sha:
        raise ValueError("TBox approval does not target the supplied source graph")
    if approval.runtime_diagnosis_or_voting_allowed:
        raise ValueError("TBox vocabulary approval unexpectedly permits runtime voting")

    approved_nodes = set(
        approval.approved_failure_mechanism_ids + approval.approved_effect_pattern_ids
    )
    node_by_id = {node.node_id: node for node in source.nodes}
    if approved_nodes - set(node_by_id):
        raise ValueError("TBox approval references missing nodes")
    nodes = [
        node_by_id[node_id].model_copy(
            update={
                "properties": {
                    **node_by_id[node_id].properties,
                    "approval_sha256": approval.approval_sha256,
                    "association_semantics": approval.association_semantics,
                    "retrieval_only": True,
                    "runtime_vote_allowed": False,
                }
            }
        )
        for node_id in sorted(approved_nodes)
    ]
    edge_by_id = {edge.edge_id: edge for edge in source.edges}
    approved_edges = set(approval.approved_tbox_association_edge_ids)
    if approved_edges - set(edge_by_id):
        raise ValueError("TBox approval references missing edges")
    edges: list[GraphEdgeV3] = []
    for edge_id in sorted(approved_edges):
        edge = edge_by_id[edge_id]
        if (
            edge.relation_type is not KnowledgeRelationTypeV3.MANIFESTS_AS
            or edge.source_id not in approved_nodes
            or edge.target_id not in approved_nodes
        ):
            raise ValueError("approved TBox association has an out-of-scope shape")
        edges.append(
            edge.model_copy(
                update={
                    "review_status": ReviewStatus.HUMAN_REVIEWED,
                    "properties": {
                        **edge.properties,
                        "approval_sha256": approval.approval_sha256,
                        "association_semantics": approval.association_semantics,
                        "retrieval_only": True,
                        "runtime_vote_allowed": False,
                    },
                }
            )
        )
    if len(nodes) != 22 or len(edges) != len(approval.approved_tbox_association_edge_ids):
        raise ValueError("reviewed TBox projection cardinality differs from approval")
    projected = KnowledgeGraphV3(
        graph_id="graph:astrograph.reviewed-tbox-retrieval:2026-09-02",
        nodes=nodes,
        edges=edges,
    )
    shape = DEFAULT_GRAPH_SHAPE_VALIDATOR.validate(projected)
    if shape.status != "PASS":
        raise ValueError("reviewed TBox projection failed V3 shape validation")

    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "reviewed_tbox_retrieval_graph.json"
    graph_path.write_text(projected.model_dump_json(indent=2) + "\n", encoding="utf-8")
    body = {
        "schema_version": "astrograph-reviewed-tbox-retrieval-projection.v1",
        "status": "RETRIEVAL_ONLY_READY",
        "builder_source_sha256": _file_sha(Path(__file__).resolve()),
        "source_graph_id": source.graph_id,
        "source_graph_sha256": source_sha,
        "source_graph_file_sha256": _file_sha(source_path),
        "approval_sha256": approval.approval_sha256,
        "approval_file_sha256": _file_sha(decision_path),
        "projected_graph_id": projected.graph_id,
        "projected_graph_sha256": canonical_sha256(projected.model_dump(mode="json")),
        "projected_graph_file_sha256": _file_sha(graph_path),
        "failure_mechanism_count": sum(
            node.node_type is KnowledgeNodeTypeV3.FAILURE_MECHANISM for node in nodes
        ),
        "effect_pattern_count": sum(
            node.node_type is KnowledgeNodeTypeV3.EFFECT_PATTERN for node in nodes
        ),
        "association_edge_count": len(edges),
        "all_edges_human_reviewed": all(
            edge.review_status is ReviewStatus.HUMAN_REVIEWED for edge in edges
        ),
        "association_semantics": approval.association_semantics,
        "retrieval_allowed": True,
        "diagnostic_evidence_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "Exactly the project-owner-approved 12 FailureMechanism terms, 10 EffectPattern "
            "terms and their TBox MANIFESTS_AS associations. The projection is available only "
            "for governed retrieval and explanation; it is not episode evidence, actual causal "
            "proof, a dataset-fault mapping, or runtime voting authority."
        ),
    }
    receipt = {**body, "projection_sha256": canonical_sha256(body)}
    (output / "build_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = ["build_reviewed_tbox_retrieval_projection"]
