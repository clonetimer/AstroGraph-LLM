from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import ReviewStatus
from .schema_v3 import (
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReviewSourceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    evidence_scope: str = Field(min_length=1)


class CausalEdgeReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_item_id: str = Field(pattern=r"^review-item:[^\s]+$")
    source_node_id: str = Field(min_length=1)
    relation_type: KnowledgeRelationTypeV3
    target_node_id: str = Field(min_length=1)
    current_review_status: ReviewStatus
    source_anchors: list[ReviewSourceAnchor] = Field(min_length=2)
    automated_recommendation: Literal["READY", "DEFER"]
    recommendation_reasons: list[str] = Field(min_length=1)
    human_decision: None = None

    @model_validator(mode="after")
    def distinct_sources(self) -> "CausalEdgeReviewItem":
        paths = [anchor.source_path for anchor in self.source_anchors]
        if len(paths) != len(set(paths)):
            raise ValueError("review anchors must use distinct source paths")
        return self


class CausalEdgeReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-causal-edge-review-packet.v1"] = (
        "astrograph-causal-edge-review-packet.v1"
    )
    packet_id: str = Field(min_length=1)
    ontology_path: str = Field(min_length=1)
    ontology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_items: list[CausalEdgeReviewItem] = Field(min_length=1)
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    claim_boundary: str = Field(min_length=1)


class ReviewPacketValidationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-review-packet-validation.v1"] = (
        "astrograph-review-packet-validation.v1"
    )
    packet_id: str
    status: Literal["READY_FOR_HUMAN_REVIEW", "BLOCKED"]
    checked_item_count: int = Field(ge=0)
    findings: list[str]
    ready_item_ids: list[str]
    deferred_item_ids: list[str]
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False


def validate_causal_edge_review_packet(
    packet_path: str | Path,
    *,
    project_root: str | Path,
    graph: KnowledgeGraphV3,
) -> ReviewPacketValidationReceipt:
    root = Path(project_root).resolve()
    packet = CausalEdgeReviewPacket.model_validate_json(
        Path(packet_path).read_text(encoding="utf-8")
    )
    findings: list[str] = []
    ontology_path = (root / packet.ontology_path).resolve()
    if not ontology_path.is_relative_to(root) or not ontology_path.is_file():
        findings.append("ONTOLOGY_PATH_INVALID")
    elif _sha256_file(ontology_path) != packet.ontology_sha256:
        findings.append("ONTOLOGY_HASH_MISMATCH")

    edges = {
        (edge.source_id, edge.relation_type, edge.target_id): edge
        for edge in graph.edges
    }
    item_ids: set[str] = set()
    for item in packet.review_items:
        if item.review_item_id in item_ids:
            findings.append(f"DUPLICATE_REVIEW_ITEM:{item.review_item_id}")
        item_ids.add(item.review_item_id)
        edge = edges.get((item.source_node_id, item.relation_type, item.target_node_id))
        if edge is None:
            findings.append(f"EDGE_NOT_IN_GRAPH:{item.review_item_id}")
        elif edge.review_status != item.current_review_status:
            findings.append(f"REVIEW_STATUS_DRIFT:{item.review_item_id}")
        for anchor in item.source_anchors:
            source_path = (root / anchor.source_path).resolve()
            if not source_path.is_relative_to(root) or not source_path.is_file():
                findings.append(f"SOURCE_PATH_INVALID:{item.review_item_id}:{anchor.source_path}")
            elif _sha256_file(source_path) != anchor.source_sha256:
                findings.append(f"SOURCE_HASH_MISMATCH:{item.review_item_id}:{anchor.source_path}")

    return ReviewPacketValidationReceipt(
        packet_id=packet.packet_id,
        status="BLOCKED" if findings else "READY_FOR_HUMAN_REVIEW",
        checked_item_count=len(packet.review_items),
        findings=findings,
        ready_item_ids=[
            item.review_item_id
            for item in packet.review_items
            if item.automated_recommendation == "READY"
        ],
        deferred_item_ids=[
            item.review_item_id
            for item in packet.review_items
            if item.automated_recommendation == "DEFER"
        ],
    )


class StructuredCandidateBenchmarkInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_type: KnowledgeNodeTypeV3
    canonical_name: str = Field(min_length=1)
    description: str = ""


class StructuredEntityLinkingBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^link-case:[^\s]+$")
    structured_candidate: StructuredCandidateBenchmarkInput
    expected_action: Literal["LINK", "ABSTAIN"]
    expected_node_id: str | None = None
    label_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def target_matches_action(self) -> "StructuredEntityLinkingBenchmarkCase":
        if (self.expected_action == "LINK") != (self.expected_node_id is not None):
            raise ValueError("LINK requires expected_node_id and ABSTAIN forbids it")
        return self


class StructuredEntityLinkingBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-structured-entity-linking-benchmark.v1"] = (
        "astrograph-structured-entity-linking-benchmark.v1"
    )
    benchmark_id: str = Field(min_length=1)
    ontology_id: str = Field(min_length=1)
    ontology_version: str = Field(min_length=1)
    ontology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_stage: Literal["POST_LLM_STRUCTURED_OUTPUT"] = "POST_LLM_STRUCTURED_OUTPUT"
    usage: Literal["DEVELOPMENT_GATE_ONLY"] = "DEVELOPMENT_GATE_ONLY"
    label_review_status: Literal["MACHINE_DRAFT"] = "MACHINE_DRAFT"
    human_label_review_required: Literal[True] = True
    frozen: Literal[True] = True
    cases: list[StructuredEntityLinkingBenchmarkCase] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


def load_frozen_structured_entity_linking_benchmark(
    benchmark_path: str | Path,
    checksum_path: str | Path,
    *,
    ontology_path: str | Path,
    graph: KnowledgeGraphV3,
) -> StructuredEntityLinkingBenchmark:
    benchmark_path = Path(benchmark_path)
    expected_sha = Path(checksum_path).read_text(encoding="utf-8").split()[0]
    if _sha256_file(benchmark_path) != expected_sha:
        raise ValueError("structured entity-linking benchmark checksum mismatch")
    benchmark = StructuredEntityLinkingBenchmark.model_validate_json(
        benchmark_path.read_text(encoding="utf-8")
    )
    ontology_path = Path(ontology_path)
    if _sha256_file(ontology_path) != benchmark.ontology_sha256:
        raise ValueError("structured entity-linking benchmark ontology hash mismatch")
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    if (benchmark.ontology_id, benchmark.ontology_version) != (
        ontology.get("ontology_id"),
        ontology.get("version"),
    ):
        raise ValueError("structured entity-linking benchmark ontology identity mismatch")
    nodes = {node.node_id: node for node in graph.nodes}
    case_ids: set[str] = set()
    for case in benchmark.cases:
        if case.case_id in case_ids:
            raise ValueError(f"duplicate benchmark case: {case.case_id}")
        case_ids.add(case.case_id)
        if case.expected_action == "LINK":
            target = nodes.get(str(case.expected_node_id))
            if target is None:
                raise ValueError(f"benchmark target not in graph: {case.expected_node_id}")
            if target.node_type != case.structured_candidate.node_type:
                raise ValueError(f"benchmark target type mismatch: {case.case_id}")
    return benchmark


__all__ = [
    "CausalEdgeReviewItem",
    "CausalEdgeReviewPacket",
    "ReviewPacketValidationReceipt",
    "ReviewSourceAnchor",
    "StructuredCandidateBenchmarkInput",
    "StructuredEntityLinkingBenchmark",
    "StructuredEntityLinkingBenchmarkCase",
    "load_frozen_structured_entity_linking_benchmark",
    "validate_causal_edge_review_packet",
]
