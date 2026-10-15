from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.anomaly.episodes import AnomalyEpisode


class SpatiotemporalRelationType(str, Enum):
    PRECEDES = "PRECEDES"
    OVERLAPS = "OVERLAPS"
    STARTS_WITHIN = "STARTS_WITHIN"
    SAME_COMPONENT = "SAME_COMPONENT"
    PHYSICALLY_CONNECTED_TO = "PHYSICALLY_CONNECTED_TO"
    TRANSFERS_POWER_TO = "TRANSFERS_POWER_TO"
    TRANSFERS_DATA_TO = "TRANSFERS_DATA_TO"
    TRANSFERS_HEAT_TO = "TRANSFERS_HEAT_TO"
    TRANSFERS_CONTROL_TO = "TRANSFERS_CONTROL_TO"


class EpisodeComponentBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(min_length=1)
    component_instance_id: str = Field(min_length=1)
    component_archetype_id: str = Field(min_length=1)
    subsystem_id: str = Field(min_length=1)
    effect_pattern_ids: list[str] = Field(default_factory=list)

    @field_validator("effect_pattern_ids")
    @classmethod
    def unique_effect_patterns(cls, value: list[str]) -> list[str]:
        return sorted(set(value))


class ComponentTopologyEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str = Field(min_length=1)
    source_component_instance_id: str = Field(min_length=1)
    target_component_instance_id: str = Field(min_length=1)
    relation_type: SpatiotemporalRelationType
    bidirectional: bool = False
    review_status: str = "HUMAN_REVIEWED"
    evidence_level: str = "E0"

    @field_validator("relation_type")
    @classmethod
    def topology_relation_only(cls, value: SpatiotemporalRelationType) -> SpatiotemporalRelationType:
        if value in {
            SpatiotemporalRelationType.PRECEDES,
            SpatiotemporalRelationType.OVERLAPS,
            SpatiotemporalRelationType.STARTS_WITHIN,
            SpatiotemporalRelationType.SAME_COMPONENT,
        }:
            raise ValueError("topology edges require a physical relation type")
        return value


class TemplateNodeRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)
    anomaly_ids: list[str] = Field(default_factory=list)
    effect_pattern_ids: list[str] = Field(default_factory=list)
    component_archetype_ids: list[str] = Field(default_factory=list)
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    required: bool = True

    @model_validator(mode="after")
    def require_semantic_selector(self) -> "TemplateNodeRequirement":
        if not (self.anomaly_ids or self.effect_pattern_ids or self.component_archetype_ids):
            raise ValueError("template node requires at least one semantic selector")
        return self


class TemplateEdgeRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    target_role: str = Field(min_length=1)
    allowed_relations: list[SpatiotemporalRelationType] = Field(min_length=1)
    maximum_lag_s: float | None = Field(default=None, ge=0.0)
    required: bool = True


class SpatiotemporalFaultTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str = Field(min_length=1)
    fault_id: str = Field(min_length=1)
    node_requirements: list[TemplateNodeRequirement] = Field(min_length=1)
    edge_requirements: list[TemplateEdgeRequirement] = Field(default_factory=list)
    node_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    edge_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    minimum_score: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_level: str = "E0"
    review_status: str = "HUMAN_REVIEWED"
    rationale: str = ""

    @model_validator(mode="after")
    def validate_roles_and_weights(self) -> "SpatiotemporalFaultTemplate":
        roles = [item.role for item in self.node_requirements]
        if len(roles) != len(set(roles)):
            raise ValueError("template node roles must be unique")
        known = set(roles)
        for edge in self.edge_requirements:
            if edge.source_role not in known or edge.target_role not in known:
                raise ValueError(f"template edge {edge.edge_id} refers to an unknown node role")
        if self.node_weight + self.edge_weight <= 0.0:
            raise ValueError("template node and edge weights cannot both be zero")
        return self


class SpatiotemporalFaultTemplateCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-spatiotemporal-fault-template-catalog.v1"
    catalog_version: str = Field(min_length=1)
    bindings: list[EpisodeComponentBinding] = Field(default_factory=list)
    topology_edges: list[ComponentTopologyEdge] = Field(default_factory=list)
    templates: list[SpatiotemporalFaultTemplate] = Field(default_factory=list)
    maximum_temporal_lag_s: float = Field(default=30.0, ge=0.0)
    claim_boundary: str = (
        "Engineering spatiotemporal templates rank hypotheses; temporal association does not establish causality."
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "SpatiotemporalFaultTemplateCatalog":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def unique_catalog_ids(self) -> "SpatiotemporalFaultTemplateCatalog":
        for name, values in {
            "binding anomaly IDs": [item.anomaly_id for item in self.bindings],
            "topology edge IDs": [item.edge_id for item in self.topology_edges],
            "template IDs": [item.template_id for item in self.templates],
        }.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class ComponentEpisodeNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    episode_id: str
    run_id: str
    anomaly_id: str
    component_instance_id: str
    component_archetype_id: str
    subsystem_id: str
    effect_pattern_ids: list[str] = Field(default_factory=list)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    peak_confidence: float = Field(ge=0.0, le=1.0)


class ComponentEpisodeEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: SpatiotemporalRelationType
    lag_s: float = Field(default=0.0, ge=0.0)
    overlap_s: float = Field(default=0.0, ge=0.0)
    source: str
    causal_claim: bool = False
    review_status: str = "NOT_APPLICABLE"
    evidence_level: str = "E0"


class ComponentSpatiotemporalFaultGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-component-spatiotemporal-fault-graph.v1"
    run_id: str
    nodes: list[ComponentEpisodeNode] = Field(default_factory=list)
    edges: list[ComponentEpisodeEdge] = Field(default_factory=list)
    unmapped_episode_ids: list[str] = Field(default_factory=list)
    binding_catalog_version: str
    claim_boundary: str


class SpatiotemporalTemplateMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    fault_id: str
    score: float = Field(ge=0.0, le=1.0)
    eligible_for_rerank: bool
    node_score: float = Field(ge=0.0, le=1.0)
    edge_score: float = Field(ge=0.0, le=1.0)
    matched_roles: dict[str, str] = Field(default_factory=dict)
    missing_required_roles: list[str] = Field(default_factory=list)
    matched_edge_ids: list[str] = Field(default_factory=list)
    missing_required_edge_ids: list[str] = Field(default_factory=list)
    supporting_episode_ids: list[str] = Field(default_factory=list)
    evidence_level: str = "E0"
    review_status: str
    claim_boundary: str = (
        "Approximate template match for hypothesis ranking only; it is not a causal proof or calibrated probability."
    )


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _temporal_edge(left: ComponentEpisodeNode, right: ComponentEpisodeNode, max_lag_s: float) -> ComponentEpisodeEdge | None:
    overlap = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    if overlap > 0.0 or (left.start_s == right.start_s and left.end_s == right.end_s):
        relation = SpatiotemporalRelationType.OVERLAPS
        lag = 0.0
    elif left.end_s <= right.start_s and right.start_s - left.end_s <= max_lag_s:
        relation = SpatiotemporalRelationType.PRECEDES
        lag = right.start_s - left.end_s
    elif abs(right.start_s - left.start_s) <= max_lag_s:
        relation = SpatiotemporalRelationType.STARTS_WITHIN
        lag = abs(right.start_s - left.start_s)
    else:
        return None
    return ComponentEpisodeEdge(
        edge_id=_stable_id("st-edge", left.node_id, right.node_id, relation.value),
        source_node_id=left.node_id,
        target_node_id=right.node_id,
        relation_type=relation,
        lag_s=lag,
        overlap_s=overlap,
        source="episode_interval_relation",
    )


def build_component_spatiotemporal_graph(
    episodes: Iterable[AnomalyEpisode],
    catalog: SpatiotemporalFaultTemplateCatalog,
) -> ComponentSpatiotemporalFaultGraph:
    items = list(episodes)
    run_ids = {item.run_id for item in items}
    if len(run_ids) > 1:
        raise ValueError("spatiotemporal graph requires episodes from exactly one run")
    run_id = next(iter(run_ids), "EMPTY")
    bindings = {item.anomaly_id: item for item in catalog.bindings}
    nodes: list[ComponentEpisodeNode] = []
    unmapped: list[str] = []
    for episode in items:
        binding = bindings.get(episode.anomaly_id)
        if binding is None:
            unmapped.append(episode.episode_id)
            continue
        nodes.append(ComponentEpisodeNode(
            node_id=f"component-episode:{episode.episode_id}",
            episode_id=episode.episode_id,
            run_id=episode.run_id,
            anomaly_id=episode.anomaly_id,
            component_instance_id=binding.component_instance_id,
            component_archetype_id=binding.component_archetype_id,
            subsystem_id=binding.subsystem_id,
            effect_pattern_ids=binding.effect_pattern_ids,
            start_s=episode.start_s,
            end_s=episode.end_s,
            peak_confidence=episode.peak_confidence,
        ))
    nodes.sort(key=lambda item: (item.start_s, item.component_instance_id, item.anomaly_id, item.node_id))
    edges: list[ComponentEpisodeEdge] = []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            temporal = _temporal_edge(left, right, catalog.maximum_temporal_lag_s)
            if temporal is not None:
                edges.append(temporal)
                if temporal.relation_type in {
                    SpatiotemporalRelationType.OVERLAPS,
                    SpatiotemporalRelationType.STARTS_WITHIN,
                }:
                    edges.append(temporal.model_copy(update={
                        "edge_id": _stable_id(
                            "st-edge", right.node_id, left.node_id, temporal.relation_type.value
                        ),
                        "source_node_id": right.node_id,
                        "target_node_id": left.node_id,
                    }))
            if left.component_instance_id == right.component_instance_id:
                for source, target in ((left, right), (right, left)):
                    edges.append(ComponentEpisodeEdge(
                        edge_id=_stable_id("st-edge", source.node_id, target.node_id, "SAME_COMPONENT"),
                        source_node_id=source.node_id,
                        target_node_id=target.node_id,
                        relation_type=SpatiotemporalRelationType.SAME_COMPONENT,
                        source="component_binding",
                    ))
    by_component: dict[str, list[ComponentEpisodeNode]] = {}
    for node in nodes:
        by_component.setdefault(node.component_instance_id, []).append(node)
    for topology in catalog.topology_edges:
        pairs = [
            (source, target)
            for source in by_component.get(topology.source_component_instance_id, [])
            for target in by_component.get(topology.target_component_instance_id, [])
        ]
        if topology.bidirectional:
            pairs += [(target, source) for source, target in pairs]
        for source, target in pairs:
            edges.append(ComponentEpisodeEdge(
                edge_id=_stable_id("st-edge", topology.edge_id, source.node_id, target.node_id),
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                relation_type=topology.relation_type,
                source=f"topology:{topology.edge_id}",
                causal_claim=False,
                review_status=topology.review_status,
                evidence_level=topology.evidence_level,
            ))
    unique_edges = {item.edge_id: item for item in edges}
    return ComponentSpatiotemporalFaultGraph(
        run_id=run_id,
        nodes=nodes,
        edges=sorted(unique_edges.values(), key=lambda item: item.edge_id),
        unmapped_episode_ids=sorted(unmapped),
        binding_catalog_version=catalog.catalog_version,
        claim_boundary=catalog.claim_boundary,
    )


def _node_matches(node: ComponentEpisodeNode, requirement: TemplateNodeRequirement) -> bool:
    return (
        node.peak_confidence >= requirement.minimum_confidence
        and (not requirement.anomaly_ids or node.anomaly_id in requirement.anomaly_ids)
        and (not requirement.component_archetype_ids or node.component_archetype_id in requirement.component_archetype_ids)
        and (not requirement.effect_pattern_ids or bool(set(node.effect_pattern_ids) & set(requirement.effect_pattern_ids)))
    )


def match_spatiotemporal_templates(
    graph: ComponentSpatiotemporalFaultGraph,
    catalog: SpatiotemporalFaultTemplateCatalog,
) -> list[SpatiotemporalTemplateMatch]:
    graph_edges = list(graph.edges)
    results: list[SpatiotemporalTemplateMatch] = []
    for template in catalog.templates:
        role_candidates = {
            requirement.role: [node for node in graph.nodes if _node_matches(node, requirement)]
            for requirement in template.node_requirements
        }
        matched_roles = {
            role: sorted(candidates, key=lambda item: (-item.peak_confidence, item.start_s, item.node_id))[0].node_id
            for role, candidates in role_candidates.items()
            if candidates
        }
        required_nodes = [item for item in template.node_requirements if item.required]
        matched_required_nodes = sum(item.role in matched_roles for item in required_nodes)
        node_score = matched_required_nodes / len(required_nodes) if required_nodes else 1.0
        missing_roles = sorted(item.role for item in required_nodes if item.role not in matched_roles)
        matched_edge_ids: list[str] = []
        missing_edges: list[str] = []
        required_edges = [item for item in template.edge_requirements if item.required]
        for requirement in template.edge_requirements:
            source_candidates = role_candidates.get(requirement.source_role, [])
            target_candidates = role_candidates.get(requirement.target_role, [])
            matched = next((
                edge for edge in graph_edges
                if edge.relation_type in requirement.allowed_relations
                and (requirement.maximum_lag_s is None or edge.lag_s <= requirement.maximum_lag_s)
                and any(edge.source_node_id == source.node_id for source in source_candidates)
                and any(edge.target_node_id == target.node_id for target in target_candidates)
            ), None)
            if matched is not None:
                matched_edge_ids.append(requirement.edge_id)
                matched_roles[requirement.source_role] = matched.source_node_id
                matched_roles[requirement.target_role] = matched.target_node_id
            elif requirement.required:
                missing_edges.append(requirement.edge_id)
        edge_score = (
            (len(required_edges) - len(missing_edges)) / len(required_edges)
            if required_edges else 1.0
        )
        denominator = template.node_weight + template.edge_weight
        score = (template.node_weight * node_score + template.edge_weight * edge_score) / denominator
        eligible = score >= template.minimum_score and not missing_roles and not missing_edges
        supporting_nodes = {node_id for node_id in matched_roles.values()}
        supporting_episode_ids = sorted(
            node.episode_id for node in graph.nodes if node.node_id in supporting_nodes
        )
        results.append(SpatiotemporalTemplateMatch(
            template_id=template.template_id,
            fault_id=template.fault_id,
            score=score,
            eligible_for_rerank=eligible,
            node_score=node_score,
            edge_score=edge_score,
            matched_roles=matched_roles,
            missing_required_roles=missing_roles,
            matched_edge_ids=sorted(matched_edge_ids),
            missing_required_edge_ids=sorted(missing_edges),
            supporting_episode_ids=supporting_episode_ids,
            evidence_level=template.evidence_level,
            review_status=template.review_status,
        ))
    return sorted(results, key=lambda item: (-item.score, item.fault_id, item.template_id))


__all__ = [
    "ComponentEpisodeEdge",
    "ComponentEpisodeNode",
    "ComponentSpatiotemporalFaultGraph",
    "ComponentTopologyEdge",
    "EpisodeComponentBinding",
    "SpatiotemporalFaultTemplate",
    "SpatiotemporalFaultTemplateCatalog",
    "SpatiotemporalRelationType",
    "SpatiotemporalTemplateMatch",
    "TemplateEdgeRequirement",
    "TemplateNodeRequirement",
    "build_component_spatiotemporal_graph",
    "match_spatiotemporal_templates",
]
