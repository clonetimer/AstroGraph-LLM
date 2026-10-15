"""Translate frozen document-graph review into a typed V3 patch proposal.

The legacy :mod:`astrograph.knowledge.patches` contract mutates only the
fault--anomaly CSV projection.  Document extraction produces schema-v3 nodes
and relations, so silently coercing it into that contract would discard type,
fragment and epistemic provenance.  This module deliberately stops at a
content-addressed, non-publishing V3 proposal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.agents.knowledge_curation import (
    KnowledgeNodeCandidate,
    KnowledgeRelationCandidate,
)

from .schema_v3 import (
    DEFAULT_GRAPH_SHAPE_VALIDATOR,
    GraphEdgeV3,
    GraphNodeV3,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
    RELATION_ALLOWLIST,
)
from .schema import AssertionStatus, ReviewStatus
from .causal_paths import CausalMetaPathQueryService


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_review_freeze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "astrograph-document-graph-review-freeze.v1":
        raise ValueError("unsupported document graph review freeze schema")
    freeze_sha = payload.get("freeze_sha256")
    expected = _canonical_sha({key: value for key, value in payload.items() if key != "freeze_sha256"})
    if freeze_sha != expected:
        raise ValueError("document graph review freeze_sha256 mismatch")
    return payload


def _load_patch_proposal(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "astrograph-v3-knowledge-patch-proposal.v1":
        raise ValueError("unsupported V3 knowledge patch proposal schema")
    actual = payload.get("patch_proposal_sha256")
    expected = _canonical_sha({
        key: value for key, value in payload.items() if key != "patch_proposal_sha256"
    })
    if actual != expected:
        raise ValueError("V3 knowledge patch proposal hash mismatch")
    return payload


def _node_type(
    endpoint_id: str,
    *,
    approved_nodes: dict[str, tuple[str, KnowledgeNodeTypeV3]],
    base_nodes: dict[str, KnowledgeNodeTypeV3],
) -> tuple[str | None, KnowledgeNodeTypeV3 | None]:
    if endpoint_id.startswith("candidate:"):
        approved = approved_nodes.get(endpoint_id)
        return approved if approved is not None else (None, None)
    return (
        (endpoint_id, base_nodes[endpoint_id])
        if endpoint_id in base_nodes
        else (None, None)
    )


def build_document_graph_patch_proposal(
    *,
    review_freeze_path: str | Path,
    base_graph_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a non-publishing V3 patch proposal or an explicit blocked receipt."""

    freeze_path = Path(review_freeze_path).resolve()
    graph_path = Path(base_graph_path).resolve()
    target = Path(output_path).resolve()
    freeze = _load_review_freeze(freeze_path)
    graph = KnowledgeGraphV3.model_validate_json(graph_path.read_text(encoding="utf-8"))
    shape = DEFAULT_GRAPH_SHAPE_VALIDATOR.validate(graph)
    if shape.status != "PASS":
        raise ValueError("base V3 graph failed shape validation")

    blockers: list[dict[str, str]] = []
    if freeze["status"] != "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL":
        blockers.append({
            "code": "DOCUMENT_REVIEW_FREEZE_NOT_READY",
            "detail": str(freeze["status"]),
        })
    if not freeze.get("knowledge_patch_proposal_allowed", False):
        blockers.append({
            "code": "DOCUMENT_REVIEW_PATCH_PERMISSION_MISSING",
            "detail": "knowledge_patch_proposal_allowed is not true",
        })

    approved_items = freeze.get("approved_items", []) if not blockers else []
    base_nodes = {node.node_id: node.node_type for node in graph.nodes}
    base_edges = {
        (edge.source_id, edge.relation_type, edge.target_id): edge.edge_id
        for edge in graph.edges
    }
    approved_nodes: dict[str, tuple[str, KnowledgeNodeTypeV3]] = {}
    proposed_canonical_nodes: dict[str, str] = {}
    operations: list[dict[str, Any]] = []
    reviewers: set[str] = set()
    source_bundle_sha256s: set[str] = set()
    review_item_ids: set[str] = set()

    for item in approved_items:
        review_id = str(item.get("review_item_id", ""))
        if not review_id or review_id in review_item_ids:
            blockers.append({"code": "INVALID_OR_DUPLICATE_REVIEW_ITEM", "detail": review_id})
            continue
        review_item_ids.add(review_id)
        reviewers.add(str(item.get("reviewer", "")))
        source_bundle_sha256s.add(str(item.get("bundle_sha256", "")))
        if item.get("item_kind") != "NODE":
            continue
        candidate = KnowledgeNodeCandidate.model_validate(item["candidate"])
        canonical_id = str(item.get("approved_canonical_id") or "")
        if not canonical_id:
            blockers.append({"code": "APPROVED_NODE_CANONICAL_ID_MISSING", "detail": review_id})
            continue
        existing = base_nodes.get(canonical_id)
        if existing is not None and existing != candidate.node_type:
            blockers.append({
                "code": "CANONICAL_NODE_TYPE_CONFLICT",
                "detail": f"{canonical_id}:{existing.value}!={candidate.node_type.value}",
            })
            continue
        if candidate.node_id in approved_nodes:
            blockers.append({"code": "DUPLICATE_APPROVED_CANDIDATE_NODE", "detail": candidate.node_id})
            continue
        if canonical_id in proposed_canonical_nodes:
            blockers.append({
                "code": "DUPLICATE_CANONICAL_NODE_IN_PATCH",
                "detail": (
                    f"{canonical_id}:{proposed_canonical_nodes[canonical_id]}:{candidate.node_id}"
                ),
            })
            continue
        proposed_canonical_nodes[canonical_id] = candidate.node_id
        approved_nodes[candidate.node_id] = (canonical_id, candidate.node_type)
        operation = {
            "operation_kind": "ALIGN_EXISTING_NODE" if existing is not None else "ADD_NODE",
            "review_item_id": review_id,
            "candidate_id": candidate.node_id,
            "canonical_id": canonical_id,
            "node_type": candidate.node_type.value,
            "canonical_name": candidate.canonical_name,
            "description": candidate.description,
            "aliases": candidate.aliases,
            "properties": candidate.properties,
            "source_fragment_ids": candidate.source_fragment_ids,
            "reviewer": item["reviewer"],
            "review_comment": item["review_comment"],
            "bundle_sha256": item["bundle_sha256"],
        }
        operations.append(operation)

    proposed_triples: dict[tuple[str, KnowledgeRelationTypeV3, str], str] = {}
    for item in approved_items:
        if item.get("item_kind") != "RELATION":
            continue
        review_id = str(item["review_item_id"])
        candidate = KnowledgeRelationCandidate.model_validate(item["candidate"])
        source_id, source_type = _node_type(
            candidate.source_id, approved_nodes=approved_nodes, base_nodes=base_nodes
        )
        target_id, target_type = _node_type(
            candidate.target_id, approved_nodes=approved_nodes, base_nodes=base_nodes
        )
        if source_id is None or target_id is None or source_type is None or target_type is None:
            blockers.append({
                "code": "APPROVED_RELATION_ENDPOINT_UNRESOLVED",
                "detail": review_id,
            })
            continue
        shape_key = (source_type, candidate.relation_type, target_type)
        if shape_key not in RELATION_ALLOWLIST:
            blockers.append({
                "code": "APPROVED_RELATION_SHAPE_INVALID",
                "detail": (
                    f"{source_type.value}->{candidate.relation_type.value}->{target_type.value}"
                ),
            })
            continue
        triple = (source_id, candidate.relation_type, target_id)
        if triple in proposed_triples:
            blockers.append({
                "code": "DUPLICATE_SEMANTIC_RELATION_IN_PATCH",
                "detail": f"{proposed_triples[triple]}:{review_id}",
            })
            continue
        proposed_triples[triple] = review_id
        if triple in base_edges:
            operation_kind = "ALIGN_EXISTING_EDGE"
            canonical_edge_id = base_edges[triple]
        else:
            operation_kind = "ADD_EDGE"
            canonical_edge_id = "edge:document-review:" + _canonical_sha({
                "source_id": source_id,
                "relation_type": candidate.relation_type.value,
                "target_id": target_id,
                "review_item_id": review_id,
            })[:20]
        operations.append({
            "operation_kind": operation_kind,
            "review_item_id": review_id,
            "candidate_id": candidate.edge_id,
            "canonical_id": canonical_edge_id,
            "source_id": source_id,
            "relation_type": candidate.relation_type.value,
            "target_id": target_id,
            "predicate_definition": candidate.predicate_definition,
            "rationale": candidate.rationale,
            "properties": candidate.properties,
            "source_fragment_ids": candidate.source_fragment_ids,
            "reviewer": item["reviewer"],
            "review_comment": item["review_comment"],
            "bundle_sha256": item["bundle_sha256"],
        })

    meaningful = [
        item for item in operations
        if item["operation_kind"] in {"ADD_NODE", "ADD_EDGE"}
    ]
    if not blockers and not operations:
        blockers.append({"code": "NO_APPROVED_DOCUMENT_GRAPH_ITEMS", "detail": "0"})

    status = "BLOCKED" if blockers else (
        "READY_FOR_IMPACT_REPLAY" if meaningful else "NO_GRAPH_DELTA"
    )
    body = {
        "schema_version": "astrograph-v3-knowledge-patch-proposal.v1",
        "status": status,
        "builder_source_sha256": _file_sha(Path(__file__).resolve()),
        "review_freeze_sha256": freeze["freeze_sha256"],
        "review_freeze_file_sha256": _file_sha(freeze_path),
        "base_graph_id": graph.graph_id,
        "base_graph_sha256": _canonical_sha(graph.model_dump(mode="json")),
        "base_graph_file_sha256": _file_sha(graph_path),
        "approved_item_count": len(approved_items),
        "operation_count": len(operations) if not blockers else 0,
        "operations": operations if not blockers else [],
        "reviewers": sorted(value for value in reviewers if value) if not blockers else [],
        "source_bundle_sha256s": (
            sorted(value for value in source_bundle_sha256s if value) if not blockers else []
        ),
        "blockers": blockers,
        "legacy_fault_anomaly_edge_patch_compatible": False,
        "impact_replay_required": status == "READY_FOR_IMPACT_REPLAY",
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "A type-preserving V3 graph delta proposal bound to a source-reconstructed human "
            "review freeze. It cannot publish, activate graph items, change runtime voting or "
            "bypass impact replay and the existing governance workflow."
        ),
    }
    receipt = {**body, "patch_proposal_sha256": _canonical_sha(body)}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def compile_document_graph_patch_candidate(
    *,
    patch_proposal_path: str | Path,
    base_graph_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile a READY V3 delta into an isolated graph for impact replay."""

    proposal_path = Path(patch_proposal_path).resolve()
    graph_path = Path(base_graph_path).resolve()
    output = Path(output_dir).resolve()
    proposal = _load_patch_proposal(proposal_path)
    if proposal["status"] != "READY_FOR_IMPACT_REPLAY":
        raise ValueError("V3 knowledge patch proposal is not ready for impact replay")
    base = KnowledgeGraphV3.model_validate_json(graph_path.read_text(encoding="utf-8"))
    base_sha = _canonical_sha(base.model_dump(mode="json"))
    if proposal["base_graph_id"] != base.graph_id or proposal["base_graph_sha256"] != base_sha:
        raise ValueError("V3 knowledge patch base graph identity is stale")

    nodes = {item.node_id: item for item in base.nodes}
    edges = {item.edge_id: item for item in base.edges}
    triples = {
        (item.source_id, item.relation_type.value, item.target_id): item.edge_id
        for item in base.edges
    }
    applied: list[str] = []
    for operation in proposal["operations"]:
        kind = operation["operation_kind"]
        if kind == "ALIGN_EXISTING_NODE":
            existing = nodes.get(operation["canonical_id"])
            if existing is None or existing.node_type.value != operation["node_type"]:
                raise ValueError("aligned V3 node drifted after proposal")
            continue
        if kind == "ALIGN_EXISTING_EDGE":
            triple = (
                operation["source_id"], operation["relation_type"], operation["target_id"]
            )
            if triples.get(triple) != operation["canonical_id"]:
                raise ValueError("aligned V3 edge drifted after proposal")
            continue
        properties = dict(operation.get("properties") or {})
        source_document_id = str(properties.get("source_document_id") or "")
        source_document_sha = str(properties.get("source_document_sha256") or "")
        if not source_document_id or ":" not in source_document_id:
            raise ValueError("V3 patch operation lacks a namespaced source document ID")
        if len(source_document_sha) != 64 or any(
            char not in "0123456789abcdef" for char in source_document_sha
        ):
            raise ValueError("V3 patch operation lacks a valid source document SHA256")
        audit_properties = {
            **properties,
            "source_fragment_ids": operation["source_fragment_ids"],
            "review_item_id": operation["review_item_id"],
            "reviewer": operation["reviewer"],
            "review_comment": operation["review_comment"],
            "source_bundle_sha256": operation["bundle_sha256"],
            "compiled_for_impact_replay": True,
        }
        if kind == "ADD_NODE":
            node_id = operation["canonical_id"]
            if node_id in nodes:
                raise ValueError("ADD_NODE collided with the base graph")
            nodes[node_id] = GraphNodeV3(
                node_id=node_id,
                node_type=operation["node_type"],
                source_id=source_document_id,
                source_sha256=source_document_sha,
                lifecycle_status=LifecycleStatus.ACTIVE,
                properties={
                    **audit_properties,
                    "canonical_name": operation["canonical_name"],
                    "description": operation["description"],
                    "aliases": operation["aliases"],
                },
            )
            applied.append(operation["review_item_id"])
        elif kind == "ADD_EDGE":
            edge_id = operation["canonical_id"]
            if edge_id in edges:
                raise ValueError("ADD_EDGE collided with the base graph")
            if operation["source_id"] not in nodes or operation["target_id"] not in nodes:
                raise ValueError("ADD_EDGE endpoint is absent after node operations")
            edge = GraphEdgeV3(
                edge_id=edge_id,
                source_id=operation["source_id"],
                relation_type=operation["relation_type"],
                target_id=operation["target_id"],
                provenance_id=source_document_id,
                provenance_sha256=source_document_sha,
                lifecycle_status=LifecycleStatus.ACTIVE,
                assertion_status=AssertionStatus.EXTRACTED,
                review_status=ReviewStatus.HUMAN_REVIEWED,
                properties={
                    **audit_properties,
                    "predicate_definition": operation["predicate_definition"],
                    "rationale": operation["rationale"],
                },
            )
            triple = (edge.source_id, edge.relation_type.value, edge.target_id)
            if triple in triples:
                raise ValueError("ADD_EDGE duplicates an existing semantic triple")
            edges[edge_id] = edge
            triples[triple] = edge_id
            applied.append(operation["review_item_id"])
        else:
            raise ValueError(f"unsupported V3 patch operation: {kind}")

    graph = KnowledgeGraphV3(
        graph_id="graph:document-patch-candidate:" + proposal["patch_proposal_sha256"][:20],
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        edges=sorted(edges.values(), key=lambda item: item.edge_id),
    )
    shape = DEFAULT_GRAPH_SHAPE_VALIDATOR.validate(graph)
    if shape.status != "PASS":
        raise ValueError(
            "compiled V3 document patch failed shape validation: "
            + ";".join(f"{item.code}:{item.subject_id}" for item in shape.issues)
        )
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_graph.json"
    candidate_path.write_text(graph.model_dump_json(indent=2) + "\n", encoding="utf-8")
    body = {
        "schema_version": "astrograph-v3-knowledge-patch-compile-receipt.v1",
        "status": "READY_FOR_IMPACT_REPLAY",
        "compiler_source_sha256": _file_sha(Path(__file__).resolve()),
        "patch_proposal_sha256": proposal["patch_proposal_sha256"],
        "base_graph_sha256": base_sha,
        "candidate_graph_id": graph.graph_id,
        "candidate_graph_sha256": _canonical_sha(graph.model_dump(mode="json")),
        "candidate_graph_file_sha256": _file_sha(candidate_path),
        "base_node_count": len(base.nodes),
        "base_edge_count": len(base.edges),
        "candidate_node_count": len(graph.nodes),
        "candidate_edge_count": len(graph.edges),
        "applied_review_item_ids": sorted(applied),
        "shape_status": shape.status,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "An isolated schema-v3 graph compiled solely for deterministic impact replay. "
            "ACTIVE lifecycle inside this candidate file makes new paths queryable in the sandbox; "
            "it does not publish or replace the deployed Active KG."
        ),
    }
    receipt = {**body, "compile_sha256": _canonical_sha(body)}
    (output / "compile_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def replay_document_graph_patch_topology(
    *,
    base_graph_path: str | Path,
    candidate_graph_path: str | Path,
    seed_node_ids: list[str],
    target_node_types: list[str],
    expected_added_path_signatures: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare governed causal-path topology before and after a V3 patch."""

    if not seed_node_ids or len(seed_node_ids) != len(set(seed_node_ids)):
        raise ValueError("impact replay seed IDs must be nonempty and unique")
    if len(expected_added_path_signatures) != len(set(expected_added_path_signatures)):
        raise ValueError("expected path signatures must be unique")
    base_path = Path(base_graph_path).resolve()
    candidate_path = Path(candidate_graph_path).resolve()
    base = KnowledgeGraphV3.model_validate_json(base_path.read_text(encoding="utf-8"))
    candidate = KnowledgeGraphV3.model_validate_json(candidate_path.read_text(encoding="utf-8"))
    service = CausalMetaPathQueryService()
    before = service.query(
        base, seed_node_ids=seed_node_ids, target_node_types=target_node_types
    )
    after = service.query(
        candidate, seed_node_ids=seed_node_ids, target_node_types=target_node_types
    )

    def signatures(result: Any) -> set[str]:
        output: set[str] = set()
        for path in result.paths:
            tokens = [path.node_ids[0]]
            for hop in path.hops:
                tokens.extend([hop.relation_type.value, hop.target_id])
            output.add("|".join(tokens))
        return output

    before_paths = signatures(before)
    after_paths = signatures(after)
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    expected = sorted(expected_added_path_signatures)
    unexpected = sorted(set(added) - set(expected))
    missing = sorted(set(expected) - set(added))
    body = {
        "schema_version": "astrograph-v3-knowledge-patch-topology-replay.v1",
        "status": "PASS" if not removed and not unexpected and not missing else "BLOCKED",
        "replay_source_sha256": _file_sha(Path(__file__).resolve()),
        "base_graph_sha256": _canonical_sha(base.model_dump(mode="json")),
        "candidate_graph_sha256": _canonical_sha(candidate.model_dump(mode="json")),
        "seed_node_ids": sorted(seed_node_ids),
        "target_node_types": sorted(target_node_types),
        "expected_added_path_signatures": expected,
        "actual_added_path_signatures": added,
        "removed_path_signatures": removed,
        "unexpected_added_path_signatures": unexpected,
        "missing_expected_path_signatures": missing,
        "base_query_sha256": before.query_sha256,
        "candidate_query_sha256": after.query_sha256,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "A deterministic topology-only sandbox replay. PASS proves only that declared V3 "
            "retrieval paths changed as expected; diagnostic metrics and runtime publication remain separate gates."
        ),
    }
    receipt = {**body, "replay_sha256": _canonical_sha(body)}
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "build_document_graph_patch_proposal",
    "compile_document_graph_patch_candidate",
    "replay_document_graph_patch_topology",
]
