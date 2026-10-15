from __future__ import annotations

from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.semantics.units import DEFAULT_UNIT_REGISTRY, DimensionVector

from .schema import AssertionStatus, ReviewStatus


class GraphLayer(str, Enum):
    TBOX = "TBox"
    ABOX = "ABox"
    MODEL = "Model"
    EVIDENCE = "Evidence"
    DECISION = "Decision"


class LifecycleStatus(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class KnowledgeNodeTypeV3(str, Enum):
    ASSET_ARCHETYPE = "AssetArchetype"
    COMPONENT_ARCHETYPE = "ComponentArchetype"
    FUNCTIONAL_ROLE = "FunctionalRole"
    PORT_ROLE = "PortRole"
    QUANTITY_KIND = "QuantityKind"
    FAILURE_MECHANISM = "FailureMechanism"
    EFFECT_PATTERN = "EffectPattern"
    ASSET_INSTANCE = "AssetInstance"
    COMPONENT_INSTANCE = "ComponentInstance"
    PORT_INSTANCE = "PortInstance"
    SENSOR_INSTANCE = "SensorInstance"
    CHANNEL_BINDING = "ChannelBinding"
    DYNAMICS_CONTRACT = "DynamicsContract"
    PARAMETER_SET = "ParameterSet"
    STATE_ESTIMATOR_ARTIFACT = "StateEstimatorArtifact"
    CALIBRATION = "Calibration"
    OBSERVATION_EPISODE = "ObservationEpisode"
    STATE_ESTIMATE = "StateEstimate"
    RESIDUAL = "Residual"
    CONSTRAINT_VIOLATION = "ConstraintViolation"
    EVIDENCE_FACT = "EvidenceFact"
    EVIDENCE_SOURCE = "EvidenceSource"
    DIAGNOSTIC_HYPOTHESIS = "DiagnosticHypothesis"
    DECISION_RECORD = "DecisionRecord"
    REVIEW_RECORD = "ReviewRecord"


NODE_LAYERS: dict[KnowledgeNodeTypeV3, GraphLayer] = {
    **{kind: GraphLayer.TBOX for kind in (
        KnowledgeNodeTypeV3.ASSET_ARCHETYPE, KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,
        KnowledgeNodeTypeV3.FUNCTIONAL_ROLE, KnowledgeNodeTypeV3.PORT_ROLE,
        KnowledgeNodeTypeV3.QUANTITY_KIND, KnowledgeNodeTypeV3.FAILURE_MECHANISM,
        KnowledgeNodeTypeV3.EFFECT_PATTERN,
    )},
    **{kind: GraphLayer.ABOX for kind in (
        KnowledgeNodeTypeV3.ASSET_INSTANCE, KnowledgeNodeTypeV3.COMPONENT_INSTANCE,
        KnowledgeNodeTypeV3.PORT_INSTANCE, KnowledgeNodeTypeV3.SENSOR_INSTANCE,
        KnowledgeNodeTypeV3.CHANNEL_BINDING,
    )},
    **{kind: GraphLayer.MODEL for kind in (
        KnowledgeNodeTypeV3.DYNAMICS_CONTRACT, KnowledgeNodeTypeV3.PARAMETER_SET,
        KnowledgeNodeTypeV3.STATE_ESTIMATOR_ARTIFACT, KnowledgeNodeTypeV3.CALIBRATION,
    )},
    **{kind: GraphLayer.EVIDENCE for kind in (
        KnowledgeNodeTypeV3.OBSERVATION_EPISODE, KnowledgeNodeTypeV3.STATE_ESTIMATE,
        KnowledgeNodeTypeV3.RESIDUAL, KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION,
        KnowledgeNodeTypeV3.EVIDENCE_FACT, KnowledgeNodeTypeV3.EVIDENCE_SOURCE,
    )},
    **{kind: GraphLayer.DECISION for kind in (
        KnowledgeNodeTypeV3.DIAGNOSTIC_HYPOTHESIS, KnowledgeNodeTypeV3.DECISION_RECORD,
        KnowledgeNodeTypeV3.REVIEW_RECORD,
    )},
}


class KnowledgeRelationTypeV3(str, Enum):
    INSTANCE_OF = "INSTANCE_OF"
    PART_OF = "PART_OF"
    HAS_PORT_ROLE = "HAS_PORT_ROLE"
    HAS_FUNCTIONAL_ROLE = "HAS_FUNCTIONAL_ROLE"
    BINDS_COMPONENT = "BINDS_COMPONENT"
    BINDS_PORT = "BINDS_PORT"
    BINDS_QUANTITY = "BINDS_QUANTITY"
    DESCRIBES_DYNAMICS = "DESCRIBES_DYNAMICS"
    IMPLEMENTS = "IMPLEMENTS"
    USES_PARAMETER_SET = "USES_PARAMETER_SET"
    CALIBRATED_BY = "CALIBRATED_BY"
    IN_EPISODE = "IN_EPISODE"
    DERIVED_FROM = "DERIVED_FROM"
    VIOLATES = "VIOLATES"
    HAS_HYPOTHESIS = "HAS_HYPOTHESIS"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    DECIDES_EPISODE = "DECIDES_EPISODE"
    ACCEPTS_HYPOTHESIS = "ACCEPTS_HYPOTHESIS"
    REVIEWS = "REVIEWS"
    PROPAGATES_TO = "PROPAGATES_TO"
    SUSCEPTIBLE_TO = "SUSCEPTIBLE_TO"
    MANIFESTS_AS = "MANIFESTS_AS"
    INDICATED_BY = "INDICATED_BY"


def _triples(
    relation: KnowledgeRelationTypeV3,
    sources: tuple[KnowledgeNodeTypeV3, ...],
    targets: tuple[KnowledgeNodeTypeV3, ...],
) -> set[tuple[KnowledgeNodeTypeV3, KnowledgeRelationTypeV3, KnowledgeNodeTypeV3]]:
    return {(source, relation, target) for source in sources for target in targets}


RELATION_ALLOWLIST = set().union(
    _triples(KnowledgeRelationTypeV3.INSTANCE_OF,
             (KnowledgeNodeTypeV3.ASSET_INSTANCE,), (KnowledgeNodeTypeV3.ASSET_ARCHETYPE,)),
    _triples(KnowledgeRelationTypeV3.INSTANCE_OF,
             (KnowledgeNodeTypeV3.COMPONENT_INSTANCE,), (KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,)),
    _triples(KnowledgeRelationTypeV3.INSTANCE_OF,
             (KnowledgeNodeTypeV3.PORT_INSTANCE,), (KnowledgeNodeTypeV3.PORT_ROLE,)),
    _triples(KnowledgeRelationTypeV3.PART_OF,
             (KnowledgeNodeTypeV3.COMPONENT_INSTANCE,),
             (KnowledgeNodeTypeV3.ASSET_INSTANCE, KnowledgeNodeTypeV3.COMPONENT_INSTANCE)),
    _triples(KnowledgeRelationTypeV3.PART_OF,
             (KnowledgeNodeTypeV3.PORT_INSTANCE, KnowledgeNodeTypeV3.SENSOR_INSTANCE),
             (KnowledgeNodeTypeV3.COMPONENT_INSTANCE,)),
    _triples(KnowledgeRelationTypeV3.HAS_PORT_ROLE,
             (KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,), (KnowledgeNodeTypeV3.PORT_ROLE,)),
    _triples(KnowledgeRelationTypeV3.HAS_FUNCTIONAL_ROLE,
             (KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,), (KnowledgeNodeTypeV3.FUNCTIONAL_ROLE,)),
    _triples(KnowledgeRelationTypeV3.BINDS_COMPONENT,
             (KnowledgeNodeTypeV3.CHANNEL_BINDING,), (KnowledgeNodeTypeV3.COMPONENT_INSTANCE,)),
    _triples(KnowledgeRelationTypeV3.BINDS_PORT,
             (KnowledgeNodeTypeV3.CHANNEL_BINDING,), (KnowledgeNodeTypeV3.PORT_INSTANCE,)),
    _triples(KnowledgeRelationTypeV3.BINDS_QUANTITY,
             (KnowledgeNodeTypeV3.CHANNEL_BINDING,), (KnowledgeNodeTypeV3.QUANTITY_KIND,)),
    _triples(KnowledgeRelationTypeV3.DESCRIBES_DYNAMICS,
             (KnowledgeNodeTypeV3.DYNAMICS_CONTRACT,), (KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,)),
    _triples(KnowledgeRelationTypeV3.IMPLEMENTS,
             (KnowledgeNodeTypeV3.STATE_ESTIMATOR_ARTIFACT,), (KnowledgeNodeTypeV3.DYNAMICS_CONTRACT,)),
    _triples(KnowledgeRelationTypeV3.USES_PARAMETER_SET,
             (KnowledgeNodeTypeV3.STATE_ESTIMATOR_ARTIFACT,), (KnowledgeNodeTypeV3.PARAMETER_SET,)),
    _triples(KnowledgeRelationTypeV3.CALIBRATED_BY,
             (KnowledgeNodeTypeV3.STATE_ESTIMATOR_ARTIFACT,), (KnowledgeNodeTypeV3.CALIBRATION,)),
    _triples(KnowledgeRelationTypeV3.IN_EPISODE,
             (KnowledgeNodeTypeV3.STATE_ESTIMATE, KnowledgeNodeTypeV3.RESIDUAL,
              KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION, KnowledgeNodeTypeV3.EVIDENCE_FACT),
             (KnowledgeNodeTypeV3.OBSERVATION_EPISODE,)),
    _triples(KnowledgeRelationTypeV3.DERIVED_FROM,
             (KnowledgeNodeTypeV3.STATE_ESTIMATE, KnowledgeNodeTypeV3.RESIDUAL,
              KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION, KnowledgeNodeTypeV3.EVIDENCE_FACT),
             (KnowledgeNodeTypeV3.EVIDENCE_SOURCE, KnowledgeNodeTypeV3.STATE_ESTIMATOR_ARTIFACT,
              KnowledgeNodeTypeV3.STATE_ESTIMATE, KnowledgeNodeTypeV3.RESIDUAL,
              KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION)),
    _triples(KnowledgeRelationTypeV3.VIOLATES,
             (KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION,), (KnowledgeNodeTypeV3.DYNAMICS_CONTRACT,)),
    _triples(KnowledgeRelationTypeV3.HAS_HYPOTHESIS,
             (KnowledgeNodeTypeV3.OBSERVATION_EPISODE,), (KnowledgeNodeTypeV3.DIAGNOSTIC_HYPOTHESIS,)),
    _triples(KnowledgeRelationTypeV3.SUPPORTED_BY,
             (KnowledgeNodeTypeV3.DIAGNOSTIC_HYPOTHESIS,), (KnowledgeNodeTypeV3.EVIDENCE_FACT,)),
    _triples(KnowledgeRelationTypeV3.CONTRADICTED_BY,
             (KnowledgeNodeTypeV3.DIAGNOSTIC_HYPOTHESIS,), (KnowledgeNodeTypeV3.EVIDENCE_FACT,)),
    _triples(KnowledgeRelationTypeV3.DECIDES_EPISODE,
             (KnowledgeNodeTypeV3.DECISION_RECORD,), (KnowledgeNodeTypeV3.OBSERVATION_EPISODE,)),
    _triples(KnowledgeRelationTypeV3.ACCEPTS_HYPOTHESIS,
             (KnowledgeNodeTypeV3.DECISION_RECORD,), (KnowledgeNodeTypeV3.DIAGNOSTIC_HYPOTHESIS,)),
    _triples(KnowledgeRelationTypeV3.REVIEWS,
             (KnowledgeNodeTypeV3.REVIEW_RECORD,),
             (KnowledgeNodeTypeV3.DECISION_RECORD, KnowledgeNodeTypeV3.DIAGNOSTIC_HYPOTHESIS)),
    _triples(KnowledgeRelationTypeV3.PROPAGATES_TO,
             (KnowledgeNodeTypeV3.FAILURE_MECHANISM,), (KnowledgeNodeTypeV3.EFFECT_PATTERN,)),
    _triples(KnowledgeRelationTypeV3.SUSCEPTIBLE_TO,
             (KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,), (KnowledgeNodeTypeV3.FAILURE_MECHANISM,)),
    _triples(KnowledgeRelationTypeV3.MANIFESTS_AS,
             (KnowledgeNodeTypeV3.FAILURE_MECHANISM,), (KnowledgeNodeTypeV3.EFFECT_PATTERN,)),
    _triples(KnowledgeRelationTypeV3.INDICATED_BY,
             (KnowledgeNodeTypeV3.EFFECT_PATTERN,), (KnowledgeNodeTypeV3.QUANTITY_KIND,)),
)

_RAW_ARRAY_KEYS = {"values", "samples", "timestamps", "raw_values", "telemetry_values"}
_EPISODE_CHILD_TYPES = {
    KnowledgeNodeTypeV3.STATE_ESTIMATE,
    KnowledgeNodeTypeV3.RESIDUAL,
    KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION,
    KnowledgeNodeTypeV3.EVIDENCE_FACT,
}


def _validate_namespaced_id(value: str) -> str:
    prefix, separator, local = value.partition(":")
    if not separator or not prefix or not local or any(char.isspace() for char in value):
        raise ValueError("graph identifiers must be stable namespaced IDs")
    return value


def _raw_array_paths(value: Any, prefix: str = "properties") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if key in _RAW_ARRAY_KEYS:
                paths.append(path)
            paths.extend(_raw_array_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_raw_array_paths(child, f"{prefix}[{index}]"))
    return paths


class GraphNodeV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: KnowledgeNodeTypeV3
    schema_version: Literal["astrograph-knowledge-schema.v3"] = "astrograph-knowledge-schema.v3"
    source_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycle_status: LifecycleStatus
    properties: dict[str, Any] = Field(default_factory=dict)

    _node_id = field_validator("node_id")(_validate_namespaced_id)
    _source_id = field_validator("source_id")(_validate_namespaced_id)

    @property
    def layer(self) -> GraphLayer:
        return NODE_LAYERS[self.node_type]


class GraphEdgeV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    source_id: str
    relation_type: KnowledgeRelationTypeV3
    target_id: str
    schema_version: Literal["astrograph-knowledge-schema.v3"] = "astrograph-knowledge-schema.v3"
    provenance_id: str
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycle_status: LifecycleStatus
    assertion_status: AssertionStatus = AssertionStatus.DERIVED
    review_status: ReviewStatus = ReviewStatus.MACHINE_VALIDATED
    properties: dict[str, Any] = Field(default_factory=dict)

    _edge_id = field_validator("edge_id")(_validate_namespaced_id)
    _source_id = field_validator("source_id")(_validate_namespaced_id)
    _target_id = field_validator("target_id")(_validate_namespaced_id)
    _provenance_id = field_validator("provenance_id")(_validate_namespaced_id)


class KnowledgeGraphV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-knowledge-schema.v3"] = "astrograph-knowledge-schema.v3"
    graph_id: str
    nodes: list[GraphNodeV3]
    edges: list[GraphEdgeV3]
    persistent_raw_telemetry: bool = False

    _graph_id = field_validator("graph_id")(_validate_namespaced_id)

    @model_validator(mode="after")
    def raw_telemetry_stays_external(self) -> "KnowledgeGraphV3":
        if self.persistent_raw_telemetry:
            raise ValueError("raw high-rate telemetry must remain outside the knowledge graph")
        return self


class GraphShapeIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    subject_id: str
    detail: str


class GraphShapeReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-graph-shape-report.v3"] = "astrograph-graph-shape-report.v3"
    status: Literal["PASS", "FAIL"]
    node_count: int
    edge_count: int
    issues: list[GraphShapeIssue]


class GraphShapeValidator:
    def validate(self, graph: KnowledgeGraphV3) -> GraphShapeReport:
        issues: list[GraphShapeIssue] = []
        node_counts = Counter(node.node_id for node in graph.nodes)
        edge_counts = Counter(edge.edge_id for edge in graph.edges)
        for node_id, count in node_counts.items():
            if count > 1:
                issues.append(GraphShapeIssue(code="DUPLICATE_NODE_ID", subject_id=node_id, detail=str(count)))
        for edge_id, count in edge_counts.items():
            if count > 1:
                issues.append(GraphShapeIssue(code="DUPLICATE_EDGE_ID", subject_id=edge_id, detail=str(count)))

        nodes = {node.node_id: node for node in graph.nodes}
        outgoing: dict[str, list[GraphEdgeV3]] = defaultdict(list)
        for node in graph.nodes:
            raw_paths = _raw_array_paths(node.properties)
            if raw_paths:
                issues.append(GraphShapeIssue(
                    code="RAW_TELEMETRY_IN_GRAPH", subject_id=node.node_id, detail=",".join(raw_paths),
                ))
            if node.node_type == KnowledgeNodeTypeV3.QUANTITY_KIND:
                self._validate_quantity_kind(node, issues)

        for edge in graph.edges:
            source = nodes.get(edge.source_id)
            target = nodes.get(edge.target_id)
            if source is None or target is None:
                missing = edge.source_id if source is None else edge.target_id
                issues.append(GraphShapeIssue(code="MISSING_EDGE_ENDPOINT", subject_id=edge.edge_id, detail=missing))
                continue
            outgoing[edge.source_id].append(edge)
            triple = (source.node_type, edge.relation_type, target.node_type)
            if triple not in RELATION_ALLOWLIST:
                issues.append(GraphShapeIssue(
                    code="RELATION_SHAPE_NOT_ALLOWED",
                    subject_id=edge.edge_id,
                    detail=f"{source.node_type.value}->{edge.relation_type.value}->{target.node_type.value}",
                ))
            if edge.relation_type == KnowledgeRelationTypeV3.PROPAGATES_TO and edge.review_status != ReviewStatus.HUMAN_REVIEWED:
                issues.append(GraphShapeIssue(
                    code="UNREVIEWED_CAUSAL_ASSERTION", subject_id=edge.edge_id, detail=edge.review_status.value,
                ))

        self._validate_bindings(graph.nodes, outgoing, issues)
        self._validate_hierarchy(graph.nodes, outgoing, issues)
        self._validate_episode_membership(graph.nodes, outgoing, issues)
        return GraphShapeReport(
            status="PASS" if not issues else "FAIL",
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            issues=issues,
        )

    @staticmethod
    def _validate_quantity_kind(node: GraphNodeV3, issues: list[GraphShapeIssue]) -> None:
        unit = node.properties.get("canonical_unit")
        dimension = node.properties.get("dimension")
        try:
            expected = DimensionVector.model_validate(dimension)
            DEFAULT_UNIT_REGISTRY.require_dimension(str(unit), expected)
        except (TypeError, ValueError) as exc:
            issues.append(GraphShapeIssue(code="INVALID_QUANTITY_UNIT", subject_id=node.node_id, detail=str(exc)))

    @staticmethod
    def _validate_bindings(
        nodes: list[GraphNodeV3], outgoing: dict[str, list[GraphEdgeV3]], issues: list[GraphShapeIssue],
    ) -> None:
        required = {
            KnowledgeRelationTypeV3.BINDS_COMPONENT,
            KnowledgeRelationTypeV3.BINDS_PORT,
            KnowledgeRelationTypeV3.BINDS_QUANTITY,
        }
        for node in nodes:
            if node.node_type != KnowledgeNodeTypeV3.CHANNEL_BINDING:
                continue
            counts = Counter(edge.relation_type for edge in outgoing[node.node_id])
            for relation in required:
                if counts[relation] != 1:
                    issues.append(GraphShapeIssue(
                        code="INVALID_CHANNEL_BINDING", subject_id=node.node_id,
                        detail=f"{relation.value} count={counts[relation]}",
                    ))

    @staticmethod
    def _validate_hierarchy(
        nodes: list[GraphNodeV3], outgoing: dict[str, list[GraphEdgeV3]], issues: list[GraphShapeIssue],
    ) -> None:
        parent: dict[str, str] = {}
        for node in nodes:
            if node.node_type not in {
                KnowledgeNodeTypeV3.COMPONENT_INSTANCE,
                KnowledgeNodeTypeV3.PORT_INSTANCE,
                KnowledgeNodeTypeV3.SENSOR_INSTANCE,
            }:
                continue
            parents = [edge.target_id for edge in outgoing[node.node_id] if edge.relation_type == KnowledgeRelationTypeV3.PART_OF]
            if len(parents) != 1:
                issues.append(GraphShapeIssue(
                    code="INVALID_INSTANCE_PARENT", subject_id=node.node_id, detail=f"count={len(parents)}",
                ))
            else:
                parent[node.node_id] = parents[0]
        for start in parent:
            seen: set[str] = set()
            current = start
            while current in parent:
                if current in seen:
                    issues.append(GraphShapeIssue(code="INSTANCE_HIERARCHY_CYCLE", subject_id=start, detail=current))
                    break
                seen.add(current)
                current = parent[current]

    @staticmethod
    def _validate_episode_membership(
        nodes: list[GraphNodeV3], outgoing: dict[str, list[GraphEdgeV3]], issues: list[GraphShapeIssue],
    ) -> None:
        for node in nodes:
            if node.node_type not in _EPISODE_CHILD_TYPES:
                continue
            count = sum(edge.relation_type == KnowledgeRelationTypeV3.IN_EPISODE for edge in outgoing[node.node_id])
            if count != 1:
                issues.append(GraphShapeIssue(
                    code="ORPHAN_OR_AMBIGUOUS_EVIDENCE", subject_id=node.node_id, detail=f"episode_count={count}",
                ))


DEFAULT_GRAPH_SHAPE_VALIDATOR = GraphShapeValidator()


__all__ = [
    "DEFAULT_GRAPH_SHAPE_VALIDATOR",
    "GraphEdgeV3",
    "GraphLayer",
    "GraphNodeV3",
    "GraphShapeIssue",
    "GraphShapeReport",
    "GraphShapeValidator",
    "KnowledgeGraphV3",
    "KnowledgeNodeTypeV3",
    "KnowledgeRelationTypeV3",
    "LifecycleStatus",
    "NODE_LAYERS",
    "RELATION_ALLOWLIST",
]
