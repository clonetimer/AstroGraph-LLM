from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from astrograph.knowledge import (
    GraphEdgeV3,
    GraphLayer,
    GraphNodeV3,
    GraphShapeValidator,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3 as NodeType,
    KnowledgeRelationTypeV3 as Relation,
    LifecycleStatus,
    NODE_LAYERS,
    ReviewStatus,
)
from astrograph.semantics import POWER

SHA = "a" * 64


def node(node_id: str, node_type: NodeType, **properties: object) -> GraphNodeV3:
    return GraphNodeV3(
        node_id=node_id,
        node_type=node_type,
        source_id="source:test",
        source_sha256=SHA,
        lifecycle_status=LifecycleStatus.ACTIVE,
        properties=properties,
    )


def edge(
    edge_id: str,
    source_id: str,
    relation: Relation,
    target_id: str,
    *,
    review_status: ReviewStatus = ReviewStatus.MACHINE_VALIDATED,
) -> GraphEdgeV3:
    return GraphEdgeV3(
        edge_id=edge_id,
        source_id=source_id,
        relation_type=relation,
        target_id=target_id,
        provenance_id="source:test",
        provenance_sha256=SHA,
        lifecycle_status=LifecycleStatus.ACTIVE,
        review_status=review_status,
    )


def valid_graph() -> KnowledgeGraphV3:
    nodes = [
        node("type:asset", NodeType.ASSET_ARCHETYPE),
        node("type:component", NodeType.COMPONENT_ARCHETYPE),
        node("type:port", NodeType.PORT_ROLE),
        node("quantity:power", NodeType.QUANTITY_KIND, canonical_unit="W", dimension=POWER.model_dump()),
        node("mechanism:loss", NodeType.FAILURE_MECHANISM),
        node("effect:power_drop", NodeType.EFFECT_PATTERN),
        node("asset:demo", NodeType.ASSET_INSTANCE),
        node("component:battery", NodeType.COMPONENT_INSTANCE),
        node("port:battery_output", NodeType.PORT_INSTANCE),
        node("binding:battery_power", NodeType.CHANNEL_BINDING, source_field="telemetry.power"),
        node("episode:1", NodeType.OBSERVATION_EPISODE, timeseries_ref="sha256:external"),
        node("evidence_source:1", NodeType.EVIDENCE_SOURCE),
        node("fact:1", NodeType.EVIDENCE_FACT, summary="active power residual is negative"),
        node("hypothesis:1", NodeType.DIAGNOSTIC_HYPOTHESIS),
        node("decision:1", NodeType.DECISION_RECORD),
    ]
    edges = [
        edge("edge:asset_type", "asset:demo", Relation.INSTANCE_OF, "type:asset"),
        edge("edge:component_type", "component:battery", Relation.INSTANCE_OF, "type:component"),
        edge("edge:component_parent", "component:battery", Relation.PART_OF, "asset:demo"),
        edge("edge:port_type", "port:battery_output", Relation.INSTANCE_OF, "type:port"),
        edge("edge:port_parent", "port:battery_output", Relation.PART_OF, "component:battery"),
        edge("edge:binding_component", "binding:battery_power", Relation.BINDS_COMPONENT, "component:battery"),
        edge("edge:binding_port", "binding:battery_power", Relation.BINDS_PORT, "port:battery_output"),
        edge("edge:binding_quantity", "binding:battery_power", Relation.BINDS_QUANTITY, "quantity:power"),
        edge("edge:fact_episode", "fact:1", Relation.IN_EPISODE, "episode:1"),
        edge("edge:fact_source", "fact:1", Relation.DERIVED_FROM, "evidence_source:1"),
        edge("edge:episode_hypothesis", "episode:1", Relation.HAS_HYPOTHESIS, "hypothesis:1"),
        edge("edge:hypothesis_fact", "hypothesis:1", Relation.SUPPORTED_BY, "fact:1"),
        edge("edge:decision_episode", "decision:1", Relation.DECIDES_EPISODE, "episode:1"),
        edge("edge:decision_hypothesis", "decision:1", Relation.ACCEPTS_HYPOTHESIS, "hypothesis:1"),
        edge(
            "edge:causal",
            "mechanism:loss",
            Relation.PROPAGATES_TO,
            "effect:power_drop",
            review_status=ReviewStatus.HUMAN_REVIEWED,
        ),
    ]
    return KnowledgeGraphV3(graph_id="graph:valid", nodes=nodes, edges=edges)


def issue_codes(graph: KnowledgeGraphV3) -> set[str]:
    return {issue.code for issue in GraphShapeValidator().validate(graph).issues}


def test_valid_hierarchical_graph_passes() -> None:
    report = GraphShapeValidator().validate(valid_graph())
    assert report.status == "PASS", report.model_dump(mode="json")
    assert set(NODE_LAYERS.values()) == set(GraphLayer)
    assert len(NODE_LAYERS) == len(NodeType)


def test_flat_telemetry_channel_is_not_a_v3_node_type() -> None:
    with pytest.raises(ValidationError):
        GraphNodeV3(
            node_id="channel:raw",
            node_type="TelemetryChannel",
            source_id="source:test",
            source_sha256=SHA,
            lifecycle_status="ACTIVE",
        )


def test_relation_endpoint_allowlist_fails_closed() -> None:
    graph = valid_graph()
    graph.edges.append(edge("edge:bad", "binding:battery_power", Relation.PART_OF, "quantity:power"))
    assert "RELATION_SHAPE_NOT_ALLOWED" in issue_codes(graph)


def test_channel_binding_requires_component_port_and_quantity_once() -> None:
    graph = valid_graph()
    graph.edges = [item for item in graph.edges if item.edge_id != "edge:binding_port"]
    assert "INVALID_CHANNEL_BINDING" in issue_codes(graph)


def test_quantity_unit_and_dimension_are_validated() -> None:
    graph = valid_graph()
    quantity = next(item for item in graph.nodes if item.node_id == "quantity:power")
    quantity.properties["canonical_unit"] = "K"
    assert "INVALID_QUANTITY_UNIT" in issue_codes(graph)


def test_evidence_requires_exactly_one_episode() -> None:
    graph = valid_graph()
    graph.edges = [item for item in graph.edges if item.edge_id != "edge:fact_episode"]
    assert "ORPHAN_OR_AMBIGUOUS_EVIDENCE" in issue_codes(graph)


def test_unreviewed_causal_assertion_is_rejected() -> None:
    graph = valid_graph()
    causal = next(item for item in graph.edges if item.edge_id == "edge:causal")
    causal.review_status = ReviewStatus.MACHINE_VALIDATED
    assert "UNREVIEWED_CAUSAL_ASSERTION" in issue_codes(graph)


def test_raw_high_rate_arrays_are_rejected() -> None:
    graph = valid_graph()
    episode = next(item for item in graph.nodes if item.node_id == "episode:1")
    episode.properties["payload"] = {"samples": [1.0, 2.0]}
    assert "RAW_TELEMETRY_IN_GRAPH" in issue_codes(graph)


def test_sensor_requires_component_parent() -> None:
    graph = valid_graph()
    graph.nodes.append(node("sensor:power", NodeType.SENSOR_INSTANCE))
    assert "INVALID_INSTANCE_PARENT" in issue_codes(graph)


def test_component_hierarchy_cycle_is_rejected() -> None:
    graph = valid_graph()
    graph.nodes.append(node("component:bus", NodeType.COMPONENT_INSTANCE))
    parent = next(item for item in graph.edges if item.edge_id == "edge:component_parent")
    parent.target_id = "component:bus"
    graph.edges.append(edge("edge:bus_parent", "component:bus", Relation.PART_OF, "component:battery"))
    assert "INSTANCE_HIERARCHY_CYCLE" in issue_codes(graph)


def test_missing_endpoint_and_duplicate_identity_are_reported() -> None:
    graph = valid_graph()
    graph.nodes.append(deepcopy(graph.nodes[0]))
    graph.edges.append(edge("edge:missing", "decision:1", Relation.ACCEPTS_HYPOTHESIS, "hypothesis:missing"))
    assert {"DUPLICATE_NODE_ID", "MISSING_EDGE_ENDPOINT"}.issubset(issue_codes(graph))


def test_identifiers_provenance_and_schema_version_fail_closed() -> None:
    with pytest.raises(ValidationError):
        node("not-namespaced", NodeType.ASSET_ARCHETYPE)
    payload = valid_graph().nodes[0].model_dump(mode="json")
    payload["source_sha256"] = "not-a-hash"
    with pytest.raises(ValidationError):
        GraphNodeV3.model_validate(payload)
    payload = valid_graph().nodes[0].model_dump(mode="json")
    payload["lifecycle_status"] = "UNCONTROLLED"
    with pytest.raises(ValidationError):
        GraphNodeV3.model_validate(payload)
    payload = valid_graph().model_dump(mode="json")
    payload["schema_version"] = "astrograph-knowledge-schema.v2"
    with pytest.raises(ValidationError):
        KnowledgeGraphV3.model_validate(payload)
