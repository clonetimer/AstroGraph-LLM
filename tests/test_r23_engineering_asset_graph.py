from __future__ import annotations

from astrograph.knowledge.engineering_asset_graph import (
    EngineeringEdgeKind,
    build_engineering_asset_graph,
)


def test_engineering_registry_graph_links_only_exact_unique_entities():
    graph = build_engineering_asset_graph(
        source_artifact_sha256="a" * 64,
        component_registry={"components": [
            {"id": "adcs", "display_name": "姿态", "type": "subsystem"},
            {"id": "adcs.rw1", "display_name": "反作用轮1", "type": "actuator"},
        ]},
        fault_catalog=[
            {"domain": "反作用轮", "component": "反作用轮1", "fault": "卡滞"},
            {"domain": "传感器", "component": "未登记传感器", "fault": "偏差"},
        ],
        signal_registry={"signals": [
            {"id": "sat.adcs.x", "telemetry_group": "adcs", "unit": "1"},
            {"id": "sat.unknown.x", "telemetry_group": "unknown", "unit": "1"},
        ]},
    )
    assert "未登记传感器" in graph.unresolved_fault_component_labels
    assert "unknown" in graph.unresolved_signal_groups
    relations = {(edge.source_id, edge.relation, edge.target_id) for edge in graph.edges}
    assert any(rel == EngineeringEdgeKind.IDENTIFIER_HIERARCHY_PARENT for _, rel, _ in relations)
    assert any(rel == EngineeringEdgeKind.HAS_DECLARED_FAULT for _, rel, _ in relations)
    assert any(rel == EngineeringEdgeKind.HAS_TELEMETRY_SIGNAL for _, rel, _ in relations)
