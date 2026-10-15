"""Tests for the V3 five-layer graph projection of episodes and tokens."""

from __future__ import annotations

from pathlib import Path

import pytest

from astrograph.knowledge.projection import project_episode_to_graph
from astrograph.knowledge.schema_v3 import (
    GraphLayer,
    GraphShapeValidator,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
)
from astrograph.windowing.episode_builder import build_episodes_from_profile_csvs
from astrograph.windowing.episode_models import EpisodeManifest, EpisodeManifestCatalog

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CHANNEL_MAP_PATH = CONFIGS / "semantics" / "sat_agent_platform_channel_maps.s0.v1.json"
SCHEMA_PATH = CONFIGS / "semantics" / "core_semantic_schema.json"
PROFILE_PATH = CONFIGS / "data_sources" / "sat_agent_platform_schema_profiles.s0.v1.json"

# Public-telemetry snapshot (committed, no external Sat-Agent path required).
SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "data" / "sat_agent_snapshots"
BATTERY_CSV = SNAPSHOT_ROOT / "traces" / "sat-agent.component.battery.trace.v1.csv"


def _battery_episode() -> EpisodeManifest:
    cat = build_episodes_from_profile_csvs(
        profile_file_map={"sat-agent.component.battery.trace.v1": str(BATTERY_CSV)},
        profile_path=str(PROFILE_PATH),
        channel_map_catalog_path=str(CHANNEL_MAP_PATH),
        semantic_schema_path=str(SCHEMA_PATH),
        catalog_id="test-graph",
    )
    assert isinstance(cat, EpisodeManifestCatalog)
    assert len(cat.episodes) == 1
    return cat.episodes[0]


def _battery_quantity_kinds() -> dict[str, dict]:
    """Map semantic_channel_id → quantity-kind info from the core schema."""
    import json

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    kinds = {
        q["quantity_kind_id"]: {
            "quantity_kind_id": q["quantity_kind_id"],
            "canonical_unit": q["canonical_unit"],
            "dimension": q["dimension"],
        }
        for q in schema["quantity_kinds"]
    }
    # Battery channel → quantity kind mapping from the channel map.
    channel_map = json.loads(CHANNEL_MAP_PATH.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for m in channel_map["maps"]:
        if "battery" not in m["mapping_id"]:
            continue
        for e in m["entries"]:
            qk = kinds.get(e["quantity_kind_id"])
            if qk is not None:
                result[e["semantic_channel_id"]] = qk
    return result


class TestGraphProjection:
    def test_graph_shape_validates(self):
        episode = _battery_episode()
        qks = _battery_quantity_kinds()
        graph = project_episode_to_graph(
            episode, source_id="astrograph:test", quantity_kinds=qks
        )
        report = GraphShapeValidator().validate(graph)
        assert report.status == "PASS", report.model_dump(mode="json")

    def test_abox_nodes_present(self):
        episode = _battery_episode()
        graph = project_episode_to_graph(episode, quantity_kinds=_battery_quantity_kinds())
        types = {node.node_type for node in graph.nodes}
        assert KnowledgeNodeTypeV3.ASSET_INSTANCE in types
        assert KnowledgeNodeTypeV3.COMPONENT_INSTANCE in types
        assert KnowledgeNodeTypeV3.CHANNEL_BINDING in types
        assert KnowledgeNodeTypeV3.OBSERVATION_EPISODE in types
        assert KnowledgeNodeTypeV3.QUANTITY_KIND in types

    def test_every_node_in_declared_layer(self):
        episode = _battery_episode()
        graph = project_episode_to_graph(episode, quantity_kinds=_battery_quantity_kinds())
        for node in graph.nodes:
            assert node.layer in set(GraphLayer), f"node {node.node_id} has no layer"
        # ABox nodes stay in ABox, Evidence nodes in Evidence.
        for node in graph.nodes:
            if node.node_type in {
                KnowledgeNodeTypeV3.ASSET_INSTANCE,
                KnowledgeNodeTypeV3.COMPONENT_INSTANCE,
                KnowledgeNodeTypeV3.CHANNEL_BINDING,
                KnowledgeNodeTypeV3.PORT_INSTANCE,
            }:
                assert node.layer == GraphLayer.ABOX, node.node_id
            if node.node_type in {
                KnowledgeNodeTypeV3.OBSERVATION_EPISODE,
                KnowledgeNodeTypeV3.EVIDENCE_FACT,
                KnowledgeNodeTypeV3.STATE_ESTIMATE,
            }:
                assert node.layer == GraphLayer.EVIDENCE, node.node_id

    def test_no_raw_telemetry_in_properties(self):
        episode = _battery_episode()
        graph = project_episode_to_graph(episode, quantity_kinds=_battery_quantity_kinds())
        # GraphNodeV3 forbids persistent_raw_telemetry at graph level; also
        # ensure no node carries a values/timestamps array.
        for node in graph.nodes:
            for key in ("values", "samples", "timestamps", "raw_values", "telemetry_values"):
                assert key not in node.properties, f"{node.node_id} leaks {key}"

    def test_episode_id_opaque_and_stable(self):
        episode = _battery_episode()
        graph = project_episode_to_graph(episode, quantity_kinds=_battery_quantity_kinds())
        assert graph.graph_id.startswith("graph:")
        assert graph.graph_id == f"graph:{episode.episode_id}"

    def test_standalone_manifest_no_tokens(self):
        """An episode without tokens still validates (Evidence has just episode)."""
        episode = _battery_episode()
        graph = project_episode_to_graph(
            episode, source_id="astrograph:bare", quantity_kinds=_battery_quantity_kinds()
        )
        report = GraphShapeValidator().validate(graph)
        assert report.status == "PASS", report.model_dump(mode="json")
