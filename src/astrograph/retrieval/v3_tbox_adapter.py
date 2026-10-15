"""Adapt a governed retrieval-only V3 TBox projection to OAG assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.knowledge.schema import ReviewStatus
from astrograph.knowledge.schema_v3 import (
    DEFAULT_GRAPH_SHAPE_VALIDATOR,
    KnowledgeGraphV3,
    LifecycleStatus,
)
from astrograph.knowledge.tbox_approval import canonical_sha256

from .models import OntologyConcept, OntologyRelation


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reviewed_v3_tbox_retrieval_assets(
    *, graph_path: str | Path, projection_receipt_path: str | Path
) -> tuple[list[OntologyConcept], list[OntologyRelation], dict[str, object]]:
    graph_path = Path(graph_path).resolve()
    receipt_path = Path(projection_receipt_path).resolve()
    graph = KnowledgeGraphV3.model_validate_json(graph_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "astrograph-reviewed-tbox-retrieval-projection.v1":
        raise ValueError("unsupported reviewed TBox retrieval projection receipt")
    expected_receipt = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "projection_sha256"}
    )
    if receipt.get("projection_sha256") != expected_receipt:
        raise ValueError("reviewed TBox projection receipt hash mismatch")
    graph_sha = canonical_sha256(graph.model_dump(mode="json"))
    if (
        receipt.get("status") != "RETRIEVAL_ONLY_READY"
        or receipt.get("projected_graph_id") != graph.graph_id
        or receipt.get("projected_graph_sha256") != graph_sha
        or receipt.get("projected_graph_file_sha256") != _file_sha(graph_path)
        or receipt.get("retrieval_allowed") is not True
        or receipt.get("diagnostic_evidence_allowed") is not False
        or receipt.get("runtime_vote_allowed") is not False
    ):
        raise ValueError("reviewed TBox projection is not admissible for retrieval-only use")
    shape = DEFAULT_GRAPH_SHAPE_VALIDATOR.validate(graph)
    if shape.status != "PASS":
        raise ValueError("reviewed TBox retrieval graph failed shape validation")

    concepts: list[OntologyConcept] = []
    for node in graph.nodes:
        if node.lifecycle_status is not LifecycleStatus.ACTIVE:
            continue
        name = str(node.properties.get("canonical_name") or node.node_id)
        aliases = [str(value) for value in node.properties.get("aliases", []) if str(value)]
        labels = list(dict.fromkeys([node.node_id, name]))
        concepts.append(
            OntologyConcept(
                concept_id=node.node_id,
                labels=labels,
                aliases=list(dict.fromkeys(aliases)),
                source_id=node.source_id,
                source_sha256=node.source_sha256,
                knowledge_version=graph.graph_id,
                lifecycle_status="ACTIVE",
                review_status="HUMAN_REVIEWED",
            )
        )
    concept_ids = {item.concept_id for item in concepts}
    relations = [
        OntologyRelation(
            relation_id=edge.edge_id,
            source_concept_id=edge.source_id,
            relation_type=edge.relation_type.value,
            target_concept_id=edge.target_id,
            source_id=edge.provenance_id,
            source_sha256=edge.provenance_sha256,
            knowledge_version=graph.graph_id,
            lifecycle_status="ACTIVE",
            review_status="HUMAN_REVIEWED",
            source_fragment_ids=list(dict.fromkeys(map(
                str, edge.properties.get("source_fragment_ids", [])
            ))),
            epistemic_status=(
                str(edge.properties["epistemic_status"])
                if edge.properties.get("epistemic_status") is not None else None
            ),
            association_semantics=(
                str(edge.properties["association_semantics"])
                if edge.properties.get("association_semantics") is not None else None
            ),
            retrieval_only=bool(edge.properties.get("retrieval_only", False)),
            diagnostic_evidence_allowed=False,
            runtime_vote_allowed=bool(edge.properties.get("runtime_vote_allowed", False)),
        )
        for edge in graph.edges
        if edge.lifecycle_status is LifecycleStatus.ACTIVE
        and edge.review_status is ReviewStatus.HUMAN_REVIEWED
        and edge.source_id in concept_ids
        and edge.target_id in concept_ids
    ]
    audit = {
        "schema_version": "astrograph-v3-tbox-oag-adapter-audit.v1",
        "graph_id": graph.graph_id,
        "graph_sha256": graph_sha,
        "projection_sha256": receipt["projection_sha256"],
        "concept_count": len(concepts),
        "relation_count": len(relations),
        "retrieval_only": True,
        "diagnostic_evidence_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
    }
    return concepts, relations, audit


__all__ = ["load_reviewed_v3_tbox_retrieval_assets"]
