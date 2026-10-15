from __future__ import annotations

import pytest

from astrograph.knowledge.causal_paths import (
    CausalMetaPathQueryService,
    CausalPathQueryPolicy,
)
from astrograph.knowledge.schema import AssertionStatus, ReviewStatus
from astrograph.knowledge.schema_v3 import (
    GraphEdgeV3,
    GraphNodeV3,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
)


SHA = "a" * 64


def _node(node_id: str, node_type: KnowledgeNodeTypeV3) -> GraphNodeV3:
    return GraphNodeV3(
        node_id=node_id,
        node_type=node_type,
        source_id="source:test-causal-paths",
        source_sha256=SHA,
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def _edge(
    edge_id: str,
    source_id: str,
    relation_type: KnowledgeRelationTypeV3,
    target_id: str,
    *,
    review_status: ReviewStatus = ReviewStatus.HUMAN_REVIEWED,
) -> GraphEdgeV3:
    return GraphEdgeV3(
        edge_id=edge_id,
        source_id=source_id,
        relation_type=relation_type,
        target_id=target_id,
        provenance_id="source:test-causal-paths",
        provenance_sha256=SHA,
        lifecycle_status=LifecycleStatus.ACTIVE,
        assertion_status=AssertionStatus.EXTRACTED,
        review_status=review_status,
    )


def _graph(*, edge_review: ReviewStatus = ReviewStatus.HUMAN_REVIEWED) -> KnowledgeGraphV3:
    component = "component-archetype:spinning-rotor"
    mechanism = "mechanism:friction-increase"
    effect = "effect:torque-authority-loss"
    return KnowledgeGraphV3(
        graph_id="graph:causal-path-test",
        nodes=[
            _node(component, KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE),
            _node(mechanism, KnowledgeNodeTypeV3.FAILURE_MECHANISM),
            _node(effect, KnowledgeNodeTypeV3.EFFECT_PATTERN),
        ],
        edges=[
            _edge(
                "edge:component-susceptible-friction",
                component,
                KnowledgeRelationTypeV3.SUSCEPTIBLE_TO,
                mechanism,
            ),
            _edge(
                "edge:friction-manifests-torque-loss",
                mechanism,
                KnowledgeRelationTypeV3.MANIFESTS_AS,
                effect,
                review_status=edge_review,
            ),
        ],
    )


def test_query_returns_direct_and_two_hop_paths_with_provenance_without_mutation() -> None:
    graph = _graph()
    before = graph.model_dump(mode="json")
    result = CausalMetaPathQueryService().query(
        graph,
        seed_node_ids=["component-archetype:spinning-rotor", "mechanism:friction-increase"],
        target_node_types=[KnowledgeNodeTypeV3.EFFECT_PATTERN],
    )

    assert graph.model_dump(mode="json") == before
    assert result.active_graph_modified is False
    assert result.missing_seed_node_ids == []
    assert result.unresolved_seed_node_ids == []
    assert [path.template_id for path in result.paths] == [
        "component-to-manifested-effect",
        "mechanism-to-manifested-effect",
    ]
    assert [len(path.hops) for path in result.paths] == [2, 1]
    assert all(hop.provenance_sha256 == SHA for path in result.paths for hop in path.hops)
    assert all(path.complete for path in result.paths)


def test_missing_and_reverse_direction_seeds_remain_explicitly_unresolved() -> None:
    result = CausalMetaPathQueryService().query(
        _graph(),
        seed_node_ids=["effect:torque-authority-loss", "mechanism:missing"],
        target_node_types=[KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE],
    )
    assert result.paths == []
    assert result.missing_seed_node_ids == ["mechanism:missing"]
    assert result.unresolved_seed_node_ids == ["effect:torque-authority-loss"]


def test_machine_validated_mechanism_edge_is_excluded_not_treated_as_causal() -> None:
    result = CausalMetaPathQueryService().query(
        _graph(edge_review=ReviewStatus.MACHINE_VALIDATED),
        seed_node_ids=["mechanism:friction-increase"],
        target_node_types=[KnowledgeNodeTypeV3.EFFECT_PATTERN],
    )
    assert result.paths == []
    assert result.unresolved_seed_node_ids == ["mechanism:friction-increase"]
    assert [(item.subject_id, item.code) for item in result.exclusions] == [
        ("edge:friction-manifests-torque-loss", "CAUSAL_EDGE_NOT_HUMAN_REVIEWED")
    ]


def test_maximum_paths_is_deterministic_and_reports_truncation() -> None:
    graph = _graph()
    graph.nodes.append(_node("effect:wheel-heating", KnowledgeNodeTypeV3.EFFECT_PATTERN))
    graph.edges.append(
        _edge(
            "edge:friction-manifests-heating",
            "mechanism:friction-increase",
            KnowledgeRelationTypeV3.MANIFESTS_AS,
            "effect:wheel-heating",
        )
    )
    service = CausalMetaPathQueryService(CausalPathQueryPolicy(maximum_paths=1))
    first = service.query(
        graph,
        seed_node_ids=["mechanism:friction-increase"],
        target_node_types=[KnowledgeNodeTypeV3.EFFECT_PATTERN],
    )
    second = service.query(
        graph,
        seed_node_ids=["mechanism:friction-increase"],
        target_node_types=[KnowledgeNodeTypeV3.EFFECT_PATTERN],
    )
    assert first.truncated is True
    assert len(first.paths) == 1
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_invalid_v3_graph_shape_fails_closed_before_traversal() -> None:
    graph = _graph()
    graph.edges.append(
        _edge(
            "edge:invalid-shape",
            "effect:torque-authority-loss",
            KnowledgeRelationTypeV3.MANIFESTS_AS,
            "mechanism:friction-increase",
        )
    )
    with pytest.raises(ValueError, match="V3 graph shape validation failed"):
        CausalMetaPathQueryService().query(
            graph,
            seed_node_ids=["effect:torque-authority-loss"],
            target_node_types=[KnowledgeNodeTypeV3.FAILURE_MECHANISM],
        )


def test_query_inputs_and_policy_reject_ambiguous_or_excessive_scope() -> None:
    service = CausalMetaPathQueryService()
    with pytest.raises(ValueError, match="seed_node_ids"):
        service.query(
            _graph(),
            seed_node_ids=["mechanism:friction-increase", "mechanism:friction-increase"],
            target_node_types=[KnowledgeNodeTypeV3.EFFECT_PATTERN],
        )
    with pytest.raises(ValueError, match="target_node_types"):
        service.query(
            _graph(),
            seed_node_ids=["mechanism:friction-increase"],
            target_node_types=[],
        )
    with pytest.raises(ValueError):
        CausalPathQueryPolicy(maximum_depth=3)


def test_exclusion_audit_is_bounded() -> None:
    graph = _graph(edge_review=ReviewStatus.MACHINE_VALIDATED)
    graph.edges[0] = graph.edges[0].model_copy(
        update={"review_status": ReviewStatus.MACHINE_VALIDATED}
    )
    policy = CausalPathQueryPolicy(maximum_exclusions=1)
    result = CausalMetaPathQueryService(policy).query(
        graph,
        seed_node_ids=["component-archetype:spinning-rotor"],
        target_node_types=[KnowledgeNodeTypeV3.EFFECT_PATTERN],
    )
    assert len(result.exclusions) == 1
    assert result.exclusions_truncated is True
