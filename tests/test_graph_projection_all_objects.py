"""Tests for the loop-13 all-object V3 graph projection + Evidence layer.

Loop 13 work item ④.  Verifies, on committed fixtures and a small synthetic
battery frame only (no external Sat-Agent path):

1. ``project_episode_to_graph`` with a gray-box state estimate projects the
   Evidence layer: StateEstimate + Residual (summary statistics only) +
   ConstraintViolation (aggregated per code), and the graph still passes
   ``GraphShapeValidator``.
2. Raw residual ``values`` arrays never enter graph node properties (DEV_GOAL
   rule ③: abstract nodes only).
3. The all-object projection script output (``data/graph_projection/``) is
   self-consistent: one graph per object, all PASS shape validation, and the
   evidence-bearing objects (storage.energy_buffer / spinning_rotor /
   lumped_capacity) have non-empty StateEstimate / Residual records while
   non-contract archetypes are honestly recorded as NO_DYNAMICS_CONTRACT.

S0: zero-fit only (GLS linear identification), no calibration-dependent fitting.
"""

from __future__ import annotations

import json
from pathlib import Path

from astrograph.knowledge.projection import project_episode_to_graph
from astrograph.knowledge.schema_v3 import (
    GraphShapeValidator,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
)
from astrograph.physics import default_dynamics_contracts
from astrograph.state_estimation import GrayBoxStateEstimator
from astrograph.windowing.episode_builder import build_episodes_from_semantic_frames

ROOT = Path(__file__).resolve().parents[1]
PROJECTION_DIR = ROOT / "data" / "graph_projection"
SUMMARY_PATH = PROJECTION_DIR / "projection_summary.json"


def _battery_frame():
    import hashlib

    from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame

    n = 200
    timestamps = [float(i) for i in range(n)]
    # Charge/discharge alternating excitation so the storage v1.1 efficiency
    # branch (charge FIXED 0.98, discharge LEARNABLE) is identifiable; the soc
    # integrates P*eff matching the generator battery model.
    power = [35.0 if (i // 25) % 2 == 0 else -35.0 for i in range(n)]
    soc = [0.6]
    for i in range(1, n):
        p = (power[i - 1] + power[i]) * 0.5
        eff = 0.98 if p >= 0.0 else 1.0 / 0.97
        soc.append(max(0.0, min(1.0, soc[-1] + p * eff / 576000.0)))
    signals = [
        SemanticSignal(
            semantic_channel_id="asset.battery.soc",
            asset_instance_id="asset-under-test",
            component_archetype_id="storage.energy_buffer",
            port_role_id="state.energy_fraction",
            quantity_kind_id="energy.state_of_charge",
            canonical_unit="1",
            direction=1,
            values=soc,
            missing_mask=[False] * n,
        ),
        SemanticSignal(
            semantic_channel_id="asset.battery.power",
            asset_instance_id="asset-under-test",
            component_archetype_id="storage.energy_buffer",
            port_role_id="power.exchange",
            quantity_kind_id="electrical.active_power",
            canonical_unit="W",
            direction=1,
            values=power,
            missing_mask=[False] * n,
        ),
    ]
    payload = {
        "timestamps": timestamps,
        "signals": [item.model_dump(mode="json") for item in signals],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SemanticTelemetryFrame(
        frame_id="test-battery",
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=timestamps,
        signals=signals,
        sampling=SamplingProfile(row_count=n, nominal_period_s=1.0, irregular=False, missing_value_count=0),
        semantic_data_sha256=digest,
        labels_used=False,
        truth_used=False,
        claim_boundary="test only",
    )


def _battery_episode(frame):
    catalog = build_episodes_from_semantic_frames(
        frames={"battery": frame},
        source_file_map={"battery": "source:battery"},
        catalog_id="astrograph:test-evidence",
        source_id="SAT_AGENT_PLATFORM",
        channel_map_sha256=None,
        semantic_schema_id="astrograph.core.physical-quantities.v1",
    )
    return catalog.episodes[0]


class TestEvidenceLayerProjection:
    def test_state_estimate_projects_residual_and_violation(self):
        frame = _battery_frame()
        episode = _battery_episode(frame)
        contract = default_dynamics_contracts()["storage.energy_buffer"]
        result = GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract)
        graph = project_episode_to_graph(
            episode,
            source_id="astrograph:test",
            state_estimate=result,
            contract=contract,
            quantity_kinds={
                "asset.battery.soc": {
                    "quantity_kind_id": "energy.state_of_charge",
                    "canonical_unit": "1",
                    "dimension": {"mass": 0, "length": 0, "time": 0},
                },
                "asset.battery.power": {
                    "quantity_kind_id": "electrical.active_power",
                    "canonical_unit": "W",
                    "dimension": {"mass": 1, "length": 2, "time": -3},
                },
            },
        )
        report = GraphShapeValidator().validate(graph)
        assert report.status == "PASS", report.model_dump(mode="json")
        types = {node.node_type for node in graph.nodes}
        assert KnowledgeNodeTypeV3.STATE_ESTIMATE in types
        assert KnowledgeNodeTypeV3.RESIDUAL in types
        assert KnowledgeNodeTypeV3.DYNAMICS_CONTRACT in types
        assert KnowledgeNodeTypeV3.STATE_ESTIMATOR_ARTIFACT in types
        # residuals carry summary statistics only
        residual_nodes = [n for n in graph.nodes if n.node_type == KnowledgeNodeTypeV3.RESIDUAL]
        assert len(residual_nodes) == 3  # state_reconstruction / balance / innovation
        for node in residual_nodes:
            assert "root_mean_square" in node.properties
            assert "values" not in node.properties, node.node_id
        # each residual has one IN_EPISODE + one DERIVED_FROM edge
        for node in residual_nodes:
            edges = [e for e in graph.edges if e.source_id == node.node_id]
            rels = {e.relation_type for e in edges}
            assert rels == {KnowledgeRelationTypeV3.IN_EPISODE, KnowledgeRelationTypeV3.DERIVED_FROM}, node.node_id

    def test_constraint_violation_aggregates_by_code(self):
        frame = _battery_frame()
        episode = _battery_episode(frame)
        contract = default_dynamics_contracts()["storage.energy_buffer"]
        result = GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract)
        graph = project_episode_to_graph(
            episode,
            source_id="astrograph:test",
            state_estimate=result,
            contract=contract,
        )
        violations = [n for n in graph.nodes if n.node_type == KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION]
        # one aggregated node per code, carrying count + max_magnitude
        for node in violations:
            assert node.properties["count"] >= 1
            assert "max_magnitude" in node.properties
            assert "values" not in node.properties
            # each violation has IN_EPISODE + VIOLATES + DERIVED_FROM edges
            rels = {e.relation_type for e in graph.edges if e.source_id == node.node_id}
            assert rels == {
                KnowledgeRelationTypeV3.IN_EPISODE,
                KnowledgeRelationTypeV3.VIOLATES,
                KnowledgeRelationTypeV3.DERIVED_FROM,
            }, node.node_id

    def test_no_raw_arrays_anywhere(self):
        frame = _battery_frame()
        episode = _battery_episode(frame)
        contract = default_dynamics_contracts()["storage.energy_buffer"]
        result = GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract)
        graph = project_episode_to_graph(
            episode,
            source_id="astrograph:test",
            state_estimate=result,
            contract=contract,
        )
        for node in graph.nodes:
            for key in ("values", "samples", "timestamps", "raw_values", "telemetry_values"):
                assert key not in node.properties, f"{node.node_id} leaks {key}"


class TestAllObjectsProjectionOutput:
    def test_projection_summary_all_objects_pass_shape(self):
        if not SUMMARY_PATH.is_file():
            import pytest

            pytest.skip("loop-13 graph projection output not present in this checkout")
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        assert summary["object_count"] == 9
        assert summary["all_shape_pass"] is True
        by_object = {r["component_instance_id"]: r for r in summary["records"]}
        assert set(by_object) == {
            "battery", "reaction_wheel", "thermal", "imu", "mtb",
            "data_queue", "onboard_storage", "solar_panel", "eps",
        }
        # Evidence-bearing archetypes have non-empty estimates.
        for obj in ("battery", "reaction_wheel", "thermal", "eps"):
            assert by_object[obj]["estimate_status"] == "OK"
            assert by_object[obj]["parameter_count"] >= 1
            assert by_object[obj]["residual_count"] >= 1
        # Non-contract archetypes are recorded honestly (not fabricated).
        for obj in ("imu", "data_queue", "onboard_storage", "solar_panel"):
            assert by_object[obj]["estimate_status"] == "NO_DYNAMICS_CONTRACT"
        # MTB: contract exists but the dipole-command physics does not satisfy
        # the torque-command actuator contract — fail-closed, honestly recorded.
        assert by_object["mtb"]["estimate_status"] == "NO_ESTIMATION_CHANNELS"

    def test_each_object_has_a_graph_artifact(self):
        if not SUMMARY_PATH.is_file():
            import pytest

            pytest.skip("loop-13 graph projection output not present in this checkout")
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        for record in summary["records"]:
            artifact = PROJECTION_DIR / f"graph_{record['component_instance_id']}.json"
            assert artifact.is_file(), artifact
            graph = json.loads(artifact.read_text(encoding="utf-8"))
            assert graph["graph_id"] == record["graph_id"]
            # no raw telemetry arrays in any node property
            for node in graph["nodes"]:
                for key in ("values", "samples", "timestamps", "raw_values", "telemetry_values"):
                    assert key not in node["properties"], (record["component_instance_id"], node["node_id"], key)
