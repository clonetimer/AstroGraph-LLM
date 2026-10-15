from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.knowledge.schema import ReviewStatus
from astrograph.knowledge.schema_v3 import (
    DEFAULT_GRAPH_SHAPE_VALIDATOR,
    GraphEdgeV3,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
)


class CausalMetaPathTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str = Field(min_length=1)
    source_node_type: KnowledgeNodeTypeV3
    relation_types: list[KnowledgeRelationTypeV3] = Field(min_length=1, max_length=2)
    target_node_type: KnowledgeNodeTypeV3


def _default_templates() -> list[CausalMetaPathTemplate]:
    n = KnowledgeNodeTypeV3
    r = KnowledgeRelationTypeV3
    return [
        CausalMetaPathTemplate(
            template_id="component-to-mechanism",
            source_node_type=n.COMPONENT_ARCHETYPE,
            relation_types=[r.SUSCEPTIBLE_TO],
            target_node_type=n.FAILURE_MECHANISM,
        ),
        CausalMetaPathTemplate(
            template_id="component-to-manifested-effect",
            source_node_type=n.COMPONENT_ARCHETYPE,
            relation_types=[r.SUSCEPTIBLE_TO, r.MANIFESTS_AS],
            target_node_type=n.EFFECT_PATTERN,
        ),
        CausalMetaPathTemplate(
            template_id="component-to-propagated-effect",
            source_node_type=n.COMPONENT_ARCHETYPE,
            relation_types=[r.SUSCEPTIBLE_TO, r.PROPAGATES_TO],
            target_node_type=n.EFFECT_PATTERN,
        ),
        CausalMetaPathTemplate(
            template_id="mechanism-to-manifested-effect",
            source_node_type=n.FAILURE_MECHANISM,
            relation_types=[r.MANIFESTS_AS],
            target_node_type=n.EFFECT_PATTERN,
        ),
        CausalMetaPathTemplate(
            template_id="mechanism-to-propagated-effect",
            source_node_type=n.FAILURE_MECHANISM,
            relation_types=[r.PROPAGATES_TO],
            target_node_type=n.EFFECT_PATTERN,
        ),
        CausalMetaPathTemplate(
            template_id="effect-to-indicator",
            source_node_type=n.EFFECT_PATTERN,
            relation_types=[r.INDICATED_BY],
            target_node_type=n.QUANTITY_KIND,
        ),
        CausalMetaPathTemplate(
            template_id="mechanism-to-manifestation-indicator",
            source_node_type=n.FAILURE_MECHANISM,
            relation_types=[r.MANIFESTS_AS, r.INDICATED_BY],
            target_node_type=n.QUANTITY_KIND,
        ),
        CausalMetaPathTemplate(
            template_id="mechanism-to-propagation-indicator",
            source_node_type=n.FAILURE_MECHANISM,
            relation_types=[r.PROPAGATES_TO, r.INDICATED_BY],
            target_node_type=n.QUANTITY_KIND,
        ),
    ]


class CausalPathQueryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-causal-path-query-policy.v1"
    policy_id: str = "astrograph-v3-causal-metapaths-v1"
    templates: list[CausalMetaPathTemplate] = Field(default_factory=_default_templates)
    maximum_depth: int = Field(default=2, ge=1, le=2)
    maximum_paths: int = Field(default=100, ge=1, le=500)
    maximum_exclusions: int = Field(default=200, ge=1, le=1000)
    accepted_lifecycle_statuses: list[LifecycleStatus] = Field(
        default_factory=lambda: [LifecycleStatus.ACTIVE]
    )
    accepted_review_statuses: list[ReviewStatus] = Field(
        default_factory=lambda: [ReviewStatus.MACHINE_VALIDATED, ReviewStatus.HUMAN_REVIEWED]
    )
    human_review_required_relations: list[KnowledgeRelationTypeV3] = Field(
        default_factory=lambda: [
            KnowledgeRelationTypeV3.PROPAGATES_TO,
            KnowledgeRelationTypeV3.MANIFESTS_AS,
            KnowledgeRelationTypeV3.SUSCEPTIBLE_TO,
        ]
    )

    @field_validator(
        "accepted_lifecycle_statuses",
        "accepted_review_statuses",
        "human_review_required_relations",
    )
    @classmethod
    def unique_nonempty(cls, value: list) -> list:
        if not value or len(value) != len(set(value)):
            raise ValueError("causal path policy lists must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def validate_templates(self) -> "CausalPathQueryPolicy":
        if not self.templates or len({item.template_id for item in self.templates}) != len(self.templates):
            raise ValueError("causal meta-path templates must be nonempty and uniquely identified")
        if any(len(item.relation_types) > self.maximum_depth for item in self.templates):
            raise ValueError("causal meta-path template exceeds maximum depth")
        return self


class CausalPathHop(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str
    source_id: str
    source_node_type: KnowledgeNodeTypeV3
    relation_type: KnowledgeRelationTypeV3
    target_id: str
    target_node_type: KnowledgeNodeTypeV3
    provenance_id: str
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assertion_status: str
    review_status: ReviewStatus


class CausalEvidencePath(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str
    seed_node_id: str
    target_node_id: str
    node_ids: list[str] = Field(min_length=2, max_length=3)
    relation_types: list[KnowledgeRelationTypeV3] = Field(min_length=1, max_length=2)
    hops: list[CausalPathHop] = Field(min_length=1, max_length=2)
    complete: bool = True
    claim_boundary: str = (
        "A complete path matches a reviewed schema meta-path. It is retrievable support, not a calibrated "
        "probability and not proof that the represented mechanism caused a particular episode."
    )


class CausalPathExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_id: str
    code: str
    detail: str


class CausalPathQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-causal-path-query-result.v1"
    graph_id: str
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_id: str
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_node_ids: list[str]
    target_node_types: list[KnowledgeNodeTypeV3]
    paths: list[CausalEvidencePath]
    missing_seed_node_ids: list[str]
    unresolved_seed_node_ids: list[str]
    exclusions: list[CausalPathExclusion]
    truncated: bool = False
    exclusions_truncated: bool = False
    active_graph_modified: bool = False
    claim_boundary: str = (
        "Deterministic read-only retrieval over a validated V3 graph. Empty results remain unresolved and must "
        "not trigger unconstrained LLM fallback."
    )


def _graph_sha256(graph: KnowledgeGraphV3) -> str:
    payload = graph.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class CausalMetaPathQueryService:
    def __init__(self, policy: CausalPathQueryPolicy | None = None) -> None:
        self.policy = policy or CausalPathQueryPolicy()

    def query(
        self,
        graph: KnowledgeGraphV3,
        *,
        seed_node_ids: list[str],
        target_node_types: list[KnowledgeNodeTypeV3 | str],
    ) -> CausalPathQueryResult:
        if not seed_node_ids or len(seed_node_ids) != len(set(seed_node_ids)):
            raise ValueError("seed_node_ids must be nonempty and unique")
        targets = [KnowledgeNodeTypeV3(item) for item in target_node_types]
        if not targets or len(targets) != len(set(targets)):
            raise ValueError("target_node_types must be nonempty and unique")
        shape = DEFAULT_GRAPH_SHAPE_VALIDATOR.validate(graph)
        if shape.status != "PASS":
            summary = [f"{item.code}:{item.subject_id}" for item in shape.issues]
            raise ValueError(f"V3 graph shape validation failed: {summary}")

        node_by_id = {item.node_id: item for item in graph.nodes}
        missing = sorted(set(seed_node_ids) - set(node_by_id))
        active_statuses = set(self.policy.accepted_lifecycle_statuses)
        accepted_reviews = set(self.policy.accepted_review_statuses)
        human_required = set(self.policy.human_review_required_relations)
        exclusions: dict[tuple[str, str], CausalPathExclusion] = {}
        adjacency: dict[str, list[GraphEdgeV3]] = defaultdict(list)
        for edge in graph.edges:
            reason: tuple[str, str] | None = None
            if edge.lifecycle_status not in active_statuses:
                reason = ("EDGE_LIFECYCLE_NOT_ACCEPTED", edge.lifecycle_status.value)
            elif edge.review_status not in accepted_reviews:
                reason = ("EDGE_REVIEW_STATUS_NOT_ACCEPTED", edge.review_status.value)
            elif edge.relation_type in human_required and edge.review_status != ReviewStatus.HUMAN_REVIEWED:
                reason = ("CAUSAL_EDGE_NOT_HUMAN_REVIEWED", edge.review_status.value)
            target = node_by_id[edge.target_id]
            source = node_by_id[edge.source_id]
            if reason is None and (
                source.lifecycle_status not in active_statuses
                or target.lifecycle_status not in active_statuses
            ):
                reason = (
                    "ENDPOINT_LIFECYCLE_NOT_ACCEPTED",
                    f"{source.lifecycle_status.value}->{target.lifecycle_status.value}",
                )
            if reason is not None:
                exclusions[(edge.edge_id, reason[0])] = CausalPathExclusion(
                    subject_id=edge.edge_id,
                    code=reason[0],
                    detail=reason[1],
                )
                continue
            adjacency[edge.source_id].append(edge)
        for edges in adjacency.values():
            edges.sort(key=lambda item: (item.relation_type.value, item.target_id, item.edge_id))

        templates_by_source: dict[KnowledgeNodeTypeV3, list[CausalMetaPathTemplate]] = defaultdict(list)
        for template in self.policy.templates:
            if template.target_node_type in targets:
                templates_by_source[template.source_node_type].append(template)

        found: list[CausalEvidencePath] = []
        seeds_with_paths: set[str] = set()
        for seed_id in seed_node_ids:
            seed = node_by_id.get(seed_id)
            if seed is None or seed.lifecycle_status not in active_statuses:
                continue
            templates = templates_by_source.get(seed.node_type, [])
            for template in sorted(templates, key=lambda item: item.template_id):
                self._walk_template(
                    seed_id=seed_id,
                    current_id=seed_id,
                    template=template,
                    relation_index=0,
                    node_ids=[seed_id],
                    hops=[],
                    node_by_id=node_by_id,
                    adjacency=adjacency,
                    output=found,
                )
            if any(path.seed_node_id == seed_id for path in found):
                seeds_with_paths.add(seed_id)

        found.sort(
            key=lambda item: (
                item.seed_node_id,
                len(item.hops),
                item.template_id,
                item.target_node_id,
                tuple(hop.edge_id for hop in item.hops),
            )
        )
        truncated = len(found) > self.policy.maximum_paths
        paths = found[: self.policy.maximum_paths]
        graph_sha = _graph_sha256(graph)
        query_payload = {
            "graph_sha256": graph_sha,
            "policy": self.policy.model_dump(mode="json"),
            "seed_node_ids": seed_node_ids,
            "target_node_types": [item.value for item in targets],
        }
        query_sha = hashlib.sha256(
            json.dumps(query_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        unresolved = sorted(
            seed_id
            for seed_id in seed_node_ids
            if seed_id not in missing and seed_id not in seeds_with_paths
        )
        ordered_exclusions = sorted(exclusions.values(), key=lambda item: (item.subject_id, item.code))
        return CausalPathQueryResult(
            graph_id=graph.graph_id,
            graph_sha256=graph_sha,
            policy_id=self.policy.policy_id,
            query_sha256=query_sha,
            seed_node_ids=seed_node_ids,
            target_node_types=targets,
            paths=paths,
            missing_seed_node_ids=missing,
            unresolved_seed_node_ids=unresolved,
            exclusions=ordered_exclusions[: self.policy.maximum_exclusions],
            truncated=truncated,
            exclusions_truncated=len(ordered_exclusions) > self.policy.maximum_exclusions,
        )

    def _walk_template(
        self,
        *,
        seed_id: str,
        current_id: str,
        template: CausalMetaPathTemplate,
        relation_index: int,
        node_ids: list[str],
        hops: list[CausalPathHop],
        node_by_id: dict,
        adjacency: dict[str, list[GraphEdgeV3]],
        output: list[CausalEvidencePath],
    ) -> None:
        expected_relation = template.relation_types[relation_index]
        for edge in adjacency.get(current_id, []):
            if edge.relation_type != expected_relation or edge.target_id in node_ids:
                continue
            source = node_by_id[edge.source_id]
            target = node_by_id[edge.target_id]
            next_hops = [
                *hops,
                CausalPathHop(
                    edge_id=edge.edge_id,
                    source_id=edge.source_id,
                    source_node_type=source.node_type,
                    relation_type=edge.relation_type,
                    target_id=edge.target_id,
                    target_node_type=target.node_type,
                    provenance_id=edge.provenance_id,
                    provenance_sha256=edge.provenance_sha256,
                    assertion_status=edge.assertion_status.value,
                    review_status=edge.review_status,
                ),
            ]
            next_nodes = [*node_ids, edge.target_id]
            is_last = relation_index + 1 == len(template.relation_types)
            if is_last:
                if target.node_type == template.target_node_type:
                    output.append(
                        CausalEvidencePath(
                            template_id=template.template_id,
                            seed_node_id=seed_id,
                            target_node_id=edge.target_id,
                            node_ids=next_nodes,
                            relation_types=list(template.relation_types),
                            hops=next_hops,
                        )
                    )
                continue
            self._walk_template(
                seed_id=seed_id,
                current_id=edge.target_id,
                template=template,
                relation_index=relation_index + 1,
                node_ids=next_nodes,
                hops=next_hops,
                node_by_id=node_by_id,
                adjacency=adjacency,
                output=output,
            )


__all__ = [
    "CausalEvidencePath",
    "CausalMetaPathQueryService",
    "CausalMetaPathTemplate",
    "CausalPathExclusion",
    "CausalPathHop",
    "CausalPathQueryPolicy",
    "CausalPathQueryResult",
]
