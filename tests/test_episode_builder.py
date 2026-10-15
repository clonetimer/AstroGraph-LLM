"""Tests for the generic episode builder — real batch-A data end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.semantics.contracts import SemanticTelemetryFrame
from astrograph.windowing.episode_builder import (
    _assess_time_axis,
    _opaque_episode_id,
    build_episodes_from_semantic_frames,
    build_episodes_from_profile_csvs,
)
from astrograph.windowing.episode_models import (
    EpisodeComponent,
    EpisodeManifest,
    EpisodeManifestCatalog,
)

# Paths
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CHANNEL_MAP_PATH = CONFIGS / "semantics" / "sat_agent_platform_channel_maps.s0.v1.json"
SCHEMA_PATH = CONFIGS / "semantics" / "core_semantic_schema.json"
PROFILE_PATH = CONFIGS / "data_sources" / "sat_agent_platform_schema_profiles.s0.v1.json"

# Real Sat-Agent files confirmed in Step 1 (batch A only), served from the
# committed public-telemetry snapshot (no external Sat-Agent path required).
SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "data" / "sat_agent_snapshots"

BATCH_A_FILES = {
    "sat-agent.component.battery.trace.v1": SNAPSHOT_ROOT
    / "traces/sat-agent.component.battery.trace.v1.csv",
    "sat-agent.subsystem.adcs-rw.astrograph-5ch.v2": SNAPSHOT_ROOT
    / "traces/sat-agent.subsystem.adcs-rw.astrograph-5ch.v2.csv",
    "sat-agent.subsystem.thermal.source-native.v1": SNAPSHOT_ROOT
    / "traces/sat-agent.subsystem.thermal.source-native.v1.csv",
    "sat-agent.component.imu.trace.v1": SNAPSHOT_ROOT
    / "traces/sat-agent.component.imu.trace.v1.csv",
    "sat-agent.component.pdu.trace.v1": SNAPSHOT_ROOT
    / "traces/sat-agent.component.pdu.trace.v1.csv",
    "sat-agent.component.data-queue.trace.v1": SNAPSHOT_ROOT
    / "traces/sat-agent.component.data-queue.trace.v1.csv",
}


def _frame(component_id: str = "storage_0") -> SemanticTelemetryFrame:
    """Minimal valid semantic frame for unit tests."""
    from astrograph.semantics.contracts import SamplingProfile, SemanticSignal

    return SemanticTelemetryFrame(
        frame_id="test-frame-1",
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=[0.0, 1.0, 2.0],
        signals=[
            SemanticSignal(
                semantic_channel_id=f"asset.eps.{component_id}.state_of_charge",
                asset_instance_id="asset-under-test",
                component_archetype_id="storage.energy_buffer",
                port_role_id="state.energy_fraction",
                quantity_kind_id="energy.state_of_charge",
                canonical_unit="1",
                direction=1,
                values=[0.8, 0.79, 0.78],
                missing_mask=[False, False, False],
            ),
        ],
        sampling=SamplingProfile(
            row_count=3,
            nominal_period_s=1.0,
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256="a" * 64,
        labels_used=False,
        truth_used=False,
        claim_boundary="test only",
    )


# --- Unit tests ---


class TestTimeAxisQuality:
    def test_uniform_frame_counter_detected(self):
        import numpy as np
        tao = _assess_time_axis(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert tao.is_uniform_frame_counter is True
        assert tao.is_strictly_monotonic is True
        assert tao.median_period_s == 1.0

    def test_non_uniform_monotonic(self):
        import numpy as np
        tao = _assess_time_axis(np.array([0.0, 1.5, 3.0, 4.5]))
        assert tao.is_monotonic is True
        assert tao.is_strictly_monotonic is True
        assert tao.is_uniform_frame_counter is False
        assert tao.median_period_s == 1.5

    def test_non_monotonic(self):
        import numpy as np
        tao = _assess_time_axis(np.array([0.0, 2.0, 1.0, 3.0]))
        assert tao.is_monotonic is False

    def test_duplicate_timestamps(self):
        import numpy as np
        tao = _assess_time_axis(np.array([0.0, 1.0, 1.0, 2.0]))
        assert tao.duplicate_count == 1

    def test_empty(self):
        import numpy as np
        tao = _assess_time_axis(np.array([]))
        assert tao.is_monotonic is False


class TestOpaqueEpisodeId:
    def test_deterministic(self):
        a = _opaque_episode_id(0.0, 60.0, ["comp_a", "comp_b"])
        b = _opaque_episode_id(0.0, 60.0, ["comp_b", "comp_a"])
        assert a == b  # order-independent

    def test_different_time_span(self):
        a = _opaque_episode_id(0.0, 60.0, ["comp_a"])
        b = _opaque_episode_id(1.0, 60.0, ["comp_a"])
        assert a != b


class TestBuildEpisodesFromFrames:
    def test_single_frame_produces_valid_episode(self):
        f = _frame()
        cat = build_episodes_from_semantic_frames(
            frames={"storage_0": f},
            source_file_map={"storage_0": "/tmp/test.csv"},
            catalog_id="test-catalog",
        )
        assert isinstance(cat, EpisodeManifestCatalog)
        assert len(cat.episodes) == 1
        ep = cat.episodes[0]
        assert ep.start_s == 0.0
        assert ep.end_s == 2.0
        assert len(ep.components) == 1
        comp = ep.components[0]
        assert comp.component_instance_id == "storage_0"
        assert comp.component_archetype_id == "storage.energy_buffer"
        assert comp.row_count == 3
        assert comp.data_quality is not None
        assert comp.data_quality.status == "VALID"

    def test_two_frames_same_window_grouped(self):
        f1 = _frame("battery")
        f2 = _frame("rw_0")  # same time span
        cat = build_episodes_from_semantic_frames(
            frames={"battery": f1, "rw_0": f2},
            source_file_map={"battery": "/a.csv", "rw_0": "/b.csv"},
        )
        assert len(cat.episodes) == 1
        assert len(cat.episodes[0].components) == 2

    def test_frames_must_not_encode_labels(self):
        f = _frame()
        cat = build_episodes_from_semantic_frames(
            frames={"storage_0": f},
            source_file_map={"storage_0": "/tmp/test.csv"},
        )
        for ep in cat.episodes:
            assert ep.labels_used is False
            assert ep.truth_used is False

    def test_empty_frames_raises(self):
        with pytest.raises(ValueError):
            build_episodes_from_semantic_frames(
                frames={},
                source_file_map={},
            )


# --- Integration tests (real Sat-Agent data) ---


@pytest.mark.parametrize(
    "profile_id, csv_path",
    [
        (pid, path)
        for pid, path in BATCH_A_FILES.items()
        if path.exists()
    ],
)
class TestRealBatchA:
    """End-to-end: real Sat-Agent CSV → SemanticTelemetryFrame → EpisodeManifest."""

    def test_builds_episode_from_real_file(self, profile_id, csv_path):
        """Each batch-A profile produces a valid episode manifest."""
        mapping = {profile_id: str(csv_path)}
        cat = build_episodes_from_profile_csvs(
            profile_file_map=mapping,
            profile_path=str(PROFILE_PATH),
            channel_map_catalog_path=str(CHANNEL_MAP_PATH),
            semantic_schema_path=str(SCHEMA_PATH),
            catalog_id=f"test-{profile_id}",
        )
        assert len(cat.episodes) == 1
        ep = cat.episodes[0]
        assert ep.episode_id != ""
        assert ep.source_id == "SAT_AGENT_PLATFORM"
        assert ep.start_s < ep.end_s
        assert len(ep.components) == 1

        comp = ep.components[0]
        assert comp.row_count > 0, f"{profile_id} has empty data"
        assert comp.data_quality is not None
        assert comp.data_quality.status in ("VALID", "DEGRADED"), (
            f"{profile_id} quality: {comp.data_quality.reason_codes}"
        )
        assert len(comp.semantic_channel_ids) > 0, f"{profile_id} has no semantic channels"

    def test_time_axis_assessed(self, profile_id, csv_path):
        """Time axis quality metadata is populated."""
        mapping = {profile_id: str(csv_path)}
        cat = build_episodes_from_profile_csvs(
            profile_file_map=mapping,
            profile_path=str(PROFILE_PATH),
            channel_map_catalog_path=str(CHANNEL_MAP_PATH),
            semantic_schema_path=str(SCHEMA_PATH),
            catalog_id=f"test-time-{profile_id}",
        )
        comp = cat.episodes[0].components[0]
        assert comp.data_quality is not None
        tao = comp.data_quality.time_quality
        assert tao is not None, f"time axis quality not set for {profile_id}"
        assert tao.is_monotonic is True, f"time is not monotonic for {profile_id}"

    def test_no_label_contamination(self, profile_id, csv_path):
        """Episode must not carry labels or truth."""
        mapping = {profile_id: str(csv_path)}
        cat = build_episodes_from_profile_csvs(
            profile_file_map=mapping,
            profile_path=str(PROFILE_PATH),
            channel_map_catalog_path=str(CHANNEL_MAP_PATH),
            semantic_schema_path=str(SCHEMA_PATH),
            catalog_id=f"test-guard-{profile_id}",
        )
        ep = cat.episodes[0]
        assert ep.labels_used is False
        assert ep.truth_used is False
