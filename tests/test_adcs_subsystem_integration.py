"""ADCS subsystem integration tests — multi-component episode from snapshots.

This is the "part → subsystem" step (project-owner guide #2). The seven already
bound ADCS components (RW subsystem / CMG / MTB actuators, IMU / star tracker /
sun sensor / magnetometer sensors) are aggregated into a single subsystem-level
episode and proven to share one replayable observation interval while each
keeps its native sampling rate and per-component missing mask (no global
resampling).

Coverage:

1. one episode groups multiple components; each retains its own
   ``native_sample_period_s`` and row count;
2. no label / truth leakage anywhere in the episode record;
3. cross-prototype interface universality: an actuator + a sensor + a third
   prototype (storage/battery) share the same episode frame, so the binding is
   not specialised to a single system (G3 requires >= 3 archetypes in one
   SemanticTelemetryFrame/EpisodeManifest);
4. every ADCS component passes its archetype feature operators and emits
   truth-blind EvidenceTokens;
5. time axes are classified correctly (uniform frame counter vs monotonic);
6. per-component missing masks are preserved (verified at the frame level).

Data source: the committed public-telemetry snapshot under
``data/sat_agent_snapshots/``, indexed by ``manifest.json``. No external
Sat-Agent path is required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from astrograph.semantics.channel_mapping import ChannelMapper
from astrograph.semantics.contracts import (
    ChannelMapCatalog,
    SemanticSchema,
    SemanticTelemetryFrame,
)
from astrograph.windowing.episode_builder import (
    build_episodes_from_profile_csvs,
    build_episodes_from_semantic_frames,
)
from astrograph.windowing.episode_models import EpisodeManifest, EpisodeManifestCatalog

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "data" / "sat_agent_snapshots"
MANIFEST_PATH = SNAPSHOT_ROOT / "manifest.json"
CONFIGS = ROOT / "configs"
CHANNEL_MAP_PATH = CONFIGS / "semantics" / "sat_agent_platform_channel_maps.s0.v1.json"
SCHEMA_PATH = CONFIGS / "semantics" / "core_semantic_schema.json"
PROFILE_PATH = CONFIGS / "data_sources" / "sat_agent_platform_schema_profiles.s0.v1.json"
FEATURE_CONTRACT_PATH = CONFIGS / "semantics" / "sat_agent_archetype_feature_contracts.s0.v1.json"

# Source distributions intentionally exclude the committed telemetry snapshot.
# These are data-backed integration tests, so absence of the independently
# checksummed snapshot is a declared SKIP rather than a source-code failure.
pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.is_file(),
    reason="Sat-Agent snapshot data artifact not present in source-only package",
)

# Snapshot profile ids for every bound ADCS object + the cross-prototype third
# prototype (battery = storage.energy_buffer).
ADCS_RW = "sat-agent.subsystem.adcs-rw.astrograph-5ch.v2"
ADCS_CMG = "sat-agent.component.cmg.trace.v1"
ADCS_MTB = "sat-agent.component.mtb.trace.v1"
ADCS_IMU = "sat-agent.component.imu.trace.v1"
ADCS_STAR = "sat-agent.component.star-tracker.trace.v1"
ADCS_SUN = "sat-agent.component.sun-sensor.trace.v1"
ADCS_MAG = "sat-agent.component.magnetometer.trace.v1"
BATTERY = "sat-agent.component.battery.trace.v1"

ADCS_PROFILES = [
    ADCS_RW,
    ADCS_CMG,
    ADCS_MTB,
    ADCS_IMU,
    ADCS_STAR,
    ADCS_SUN,
    ADCS_MAG,
]

ARCHETYPE = {
    ADCS_RW: "actuator.controlled_effort",
    ADCS_CMG: "actuator.controlled_effort",
    ADCS_MTB: "actuator.controlled_effort",
    ADCS_IMU: "sensor.measurement",
    ADCS_STAR: "sensor.measurement",
    ADCS_SUN: "sensor.measurement",
    ADCS_MAG: "sensor.measurement",
    BATTERY: "storage.energy_buffer",
}


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_entry(profile_id: str) -> dict:
    for entry in _manifest()["files"]:
        if entry["profile_id"] == profile_id:
            return entry
    raise KeyError(f"profile {profile_id} not in snapshot manifest")


def _snapshot_csv(profile_id: str) -> Path:
    return SNAPSHOT_ROOT / _manifest_entry(profile_id)["local_path"]


def _profile_file_map(profile_ids: list[str]) -> dict[str, str]:
    return {profile_id: str(_snapshot_csv(profile_id)) for profile_id in profile_ids}


def _build_catalog(
    profile_ids: list[str], catalog_id: str = "adcs-subsystem"
) -> EpisodeManifestCatalog:
    return build_episodes_from_profile_csvs(
        profile_file_map=_profile_file_map(profile_ids),
        profile_path=str(PROFILE_PATH),
        channel_map_catalog_path=str(CHANNEL_MAP_PATH),
        semantic_schema_path=str(SCHEMA_PATH),
        catalog_id=catalog_id,
    )


def _schema_and_maps() -> tuple[SemanticSchema, ChannelMapCatalog]:
    schema = SemanticSchema.model_validate_json(SCHEMA_PATH.read_text(encoding="utf-8"))
    maps = ChannelMapCatalog.model_validate_json(CHANNEL_MAP_PATH.read_text(encoding="utf-8"))
    return schema, maps


def _map_frame(profile_id: str, df: pd.DataFrame) -> SemanticTelemetryFrame:
    """Map an arbitrary source DataFrame through the profile's channel map."""
    entry = _manifest_entry(profile_id)
    schema, maps = _schema_and_maps()
    channel_map = maps.channel_map(entry["mapping_id"])
    mapped_fields = {e.source_field for e in channel_map.entries}
    selected = ["time_s"] + [c for c in sorted(mapped_fields) if c in df.columns]
    mapper = ChannelMapper(semantic_schema=schema, channel_map=channel_map)
    return mapper.map_frame(df[selected]).frame


def _frame_for(profile_id: str) -> SemanticTelemetryFrame:
    """Map the committed snapshot CSV for a profile into a semantic frame."""
    return _map_frame(profile_id, pd.read_csv(_snapshot_csv(profile_id)))


def _tokens_for(profile_id: str, frame: SemanticTelemetryFrame):
    from astrograph.features.generic.token_builder import TokenBuilder

    builder = TokenBuilder(archetype=ARCHETYPE[profile_id])
    builder.load_contract(FEATURE_CONTRACT_PATH)
    return builder.build(frame, component_archetype_id=ARCHETYPE[profile_id])


def _component_by_id(episode: EpisodeManifest, profile_id: str):
    for component in episode.components:
        if component.component_instance_id == profile_id:
            return component
    raise KeyError(profile_id)


def _assert_truth_blind_manifest(episode: EpisodeManifest) -> None:
    assert episode.labels_used is False
    assert episode.truth_used is False


# ---------------------------------------------------------------------------
# 1. Multi-component aggregation
# ---------------------------------------------------------------------------


def test_adcs_subsystem_episode_groups_all_components_in_one_window() -> None:
    profiles = [ADCS_RW, ADCS_IMU, ADCS_STAR, ADCS_CMG]
    catalog = _build_catalog(profiles, catalog_id="adcs-subsystem-4comp")
    assert isinstance(catalog, EpisodeManifestCatalog)
    assert len(catalog.episodes) == 1, "all ADCS components must share one episode"

    episode = catalog.episodes[0]
    assert isinstance(episode, EpisodeManifest)
    assert set(c.component_instance_id for c in episode.components) == set(profiles)

    # Union time span: RW ends at 60 s, the sensors at 12 s, CMG at 20 s.
    assert episode.start_s == pytest.approx(0.0, abs=1e-9)
    assert episode.end_s == pytest.approx(60.0, abs=1e-9)

    _assert_truth_blind_manifest(episode)


def test_each_component_keeps_native_sampling_rate_and_row_count() -> None:
    catalog = _build_catalog([ADCS_RW, ADCS_IMU, ADCS_STAR, ADCS_CMG])
    episode = catalog.episodes[0]

    expected_period = {
        ADCS_RW: 1.0,
        ADCS_IMU: 0.1,
        ADCS_STAR: 1.0,
        ADCS_CMG: 10.0,
    }
    for profile_id, period in expected_period.items():
        component = _component_by_id(episode, profile_id)
        assert component.native_sample_period_s == pytest.approx(period, abs=1e-6), profile_id
        assert component.row_count == _manifest_entry(profile_id)["row_count"], profile_id
        assert component.start_s < component.end_s, profile_id


def test_episode_has_no_global_resampling_and_keeps_channel_identity() -> None:
    profiles = [ADCS_RW, ADCS_IMU, ADCS_STAR, ADCS_CMG]
    catalog = _build_catalog(profiles)
    episode = catalog.episodes[0]

    for profile_id in profiles:
        component = _component_by_id(episode, profile_id)
        frame = _frame_for(profile_id)
        # The manifest keeps each component's own semantic channels and row
        # count; nothing is upsampled/downsampled to a common grid.
        assert len(component.semantic_channel_ids) == len(frame.signals)
        assert set(component.semantic_channel_ids) == {
            s.semantic_channel_id for s in frame.signals
        }


def test_source_references_point_only_into_the_snapshot() -> None:
    profiles = [ADCS_RW, ADCS_IMU, ADCS_STAR, ADCS_CMG]
    catalog = _build_catalog(profiles)
    episode = catalog.episodes[0]

    for profile_id in profiles:
        reference = episode.source_file_references[profile_id]
        assert "sat_agent_snapshots" in reference, reference
        assert "D:/Project/test_sat" not in reference and "/test_sat/" not in reference


# ---------------------------------------------------------------------------
# 2. No label leakage
# ---------------------------------------------------------------------------


def test_no_label_leakage_in_subsystem_episode() -> None:
    catalog = _build_catalog(ADCS_PROFILES, catalog_id="adcs-all-7")
    episode = catalog.episodes[0]
    _assert_truth_blind_manifest(episode)
    assert episode.overall_data_quality is not None

    for component in episode.components:
        assert component.data_quality is not None
        # Raw component ids are profile ids, never labels/truth/case ids.
        assert "label" not in component.component_instance_id.lower()
        assert "truth" not in component.component_instance_id.lower()


# ---------------------------------------------------------------------------
# 3. Cross-prototype interface universality
# ---------------------------------------------------------------------------


def test_cross_prototype_actuator_sensor_storage_share_one_episode() -> None:
    profiles = [ADCS_RW, ADCS_IMU, BATTERY]
    catalog = _build_catalog(profiles, catalog_id="adcs-cross-prototype")
    assert len(catalog.episodes) == 1
    episode = catalog.episodes[0]

    assert len(episode.components) == 3
    archetypes = {c.component_archetype_id for c in episode.components}
    assert archetypes == {
        "actuator.controlled_effort",
        "sensor.measurement",
        "storage.energy_buffer",
    }
    assert episode.start_s < episode.end_s
    _assert_truth_blind_manifest(episode)


def test_cross_prototype_mixed_actuators_and_sensors_adcs_only() -> None:
    profiles = [ADCS_RW, ADCS_CMG, ADCS_IMU, ADCS_STAR, ADCS_MAG]
    catalog = _build_catalog(profiles, catalog_id="adcs-mixed")
    episode = catalog.episodes[0]
    archetypes = {c.component_archetype_id for c in episode.components}
    assert archetypes == {"actuator.controlled_effort", "sensor.measurement"}


# ---------------------------------------------------------------------------
# 4. Evidence tokens per component
# ---------------------------------------------------------------------------


def test_each_adcs_component_produces_truth_blind_evidence_tokens() -> None:
    for profile_id in ADCS_PROFILES:
        frame = _frame_for(profile_id)
        result = _tokens_for(profile_id, frame)
        assert result.token_count > 0, profile_id
        assert result.labels_used is False and result.truth_used is False

        # Every token is truth-blind and its values are finite (zero-fit safe).
        for token in result.tokens:
            assert token.labels_used is False
            assert token.truth_used is False
            assert token.identity_fields_allowed is False
            assert token.component_archetype_id == ARCHETYPE[profile_id]
            assert math.isfinite(token.value), token.model_dump()

        # Every token is either a real operator output (availability > 0) or an
        # honest masked placeholder (availability == 0 with a quality flag) —
        # no silent fake confidence.
        for token in result.tokens:
            if token.availability > 0.0:
                continue
            assert any(
                flag in token.quality_flags
                for flag in (
                    "missing_required_port",
                    "operator_error",
                    "constant_input",
                    "insufficient_samples",
                )
            ), token.model_dump()

        # At least one real operator output is expected for every component that
        # maps the archetype's declared feature ports. MTB is the documented
        # exception: its magnetic-actuator ports (command.magnetic_dipole /
        # response.effort) are not declared in the actuator feature contract
        # (command.effort / state.rate), so all its tokens are honestly masked.
        if profile_id != ADCS_MTB:
            assert any(t.availability > 0.0 for t in result.tokens), profile_id


def test_cross_prototype_tokens_share_uniform_truth_blind_structure() -> None:
    """actuator + sensor + storage tokens share the same EvidenceToken contract."""
    token_sets = []
    for profile_id in [ADCS_RW, ADCS_IMU, BATTERY]:
        result = _tokens_for(profile_id, _frame_for(profile_id))
        assert result.token_count > 0
        token_sets.append(result.tokens)

    for token in [t for tokens in token_sets for t in tokens]:
        assert token.schema_version == "astrograph-evidence-token.v1"
        assert token.labels_used is False
        assert token.truth_used is False
        assert token.identity_fields_allowed is False


def test_adcs_subsystem_token_yield_is_recorded() -> None:
    """Freeze the observable token yield, separating total from valid tokens.

    Valid = availability > 0 (real operator output). Masked placeholder tokens
    (availability == 0, ``missing_required_port``) are counted separately so
    the demo numbers are never overstated. MTB and CMG keep only their masked
    placeholders for the actuator primitives whose declared ports they do not
    map; sensors keep one masked placeholder for the innovation primitive,
    which requires a public reference (``observation.input``) that no snapshot
    component declares.
    """
    observed: dict[str, tuple[int, int]] = {}
    for profile_id in ADCS_PROFILES + [BATTERY]:
        result = _tokens_for(profile_id, _frame_for(profile_id))
        valid = sum(1 for t in result.tokens if t.availability > 0.0)
        assert valid <= result.token_count
        observed[profile_id] = (result.token_count, valid)

    # (total, valid) per profile.
    assert observed[ADCS_RW] == (6, 6)
    assert observed[ADCS_CMG] == (5, 1)   # rate_dynamics only
    assert observed[ADCS_MTB] == (5, 0)   # all actuator ports unmapped → masked
    assert observed[ADCS_IMU] == (10, 9)  # innovation masked (no observation.input)
    assert observed[ADCS_STAR] == (10, 9)
    assert observed[ADCS_SUN] == (13, 12)
    assert observed[ADCS_MAG] == (13, 12)
    assert observed[BATTERY] == (5, 5)


def test_mtb_command_response_gain_is_masked_not_fake_negative() -> None:
    """Regression (fix round): MTB declares no ``command.effort`` port (its
    command port is ``command.magnetic_dipole``), so the actuator
    command-response-gain primitive must be masked rather than fed two
    response signals in place of the absent command signal. Feeding two
    response signals previously produced a fabricated -1.0 confidence via a
    NaN ``np.corrcoef`` on a constant input.
    """
    result = _tokens_for(ADCS_MTB, _frame_for(ADCS_MTB))
    gain = [
        t for t in result.tokens if t.operator_id == "actuator.command_response_gain"
    ]
    assert len(gain) == 1
    token = gain[0]
    assert token.availability == 0.0
    assert "missing_required_port" in token.quality_flags
    assert token.value == 0.0


# ---------------------------------------------------------------------------
# 5. Time-axis quality classification
# ---------------------------------------------------------------------------


def test_time_axis_quality_classified_per_component() -> None:
    catalog = _build_catalog([ADCS_RW, ADCS_IMU, ADCS_STAR, ADCS_CMG])
    episode = catalog.episodes[0]

    time_quality = {
        c.component_instance_id: c.data_quality.time_quality for c in episode.components
    }
    assert time_quality[ADCS_RW].is_uniform_frame_counter is True
    assert time_quality[ADCS_STAR].is_uniform_frame_counter is True
    # Decimal 0.1 s steps are monotonic but not an integer frame counter.
    assert time_quality[ADCS_IMU].is_uniform_frame_counter is False
    assert time_quality[ADCS_IMU].is_strictly_monotonic is True
    assert time_quality[ADCS_IMU].median_period_s == pytest.approx(0.1, abs=1e-6)
    # CMG 0/10/20 s is monotonic with a coarse period, not a frame counter.
    assert time_quality[ADCS_CMG].is_uniform_frame_counter is False
    assert time_quality[ADCS_CMG].is_strictly_monotonic is True
    assert time_quality[ADCS_CMG].median_period_s == pytest.approx(10.0, abs=1e-6)

    for component in episode.components:
        assert component.data_quality.time_quality.is_monotonic is True
        assert component.data_quality.status == "VALID"


def test_all_adcs_snapshot_time_axes_are_monotonic() -> None:
    for profile_id in ADCS_PROFILES + [BATTERY]:
        component = _component_by_id(_build_catalog([profile_id]).episodes[0], profile_id)
        assert component.data_quality.time_quality.is_monotonic is True, profile_id


# ---------------------------------------------------------------------------
# 6. Per-component missing masks preserved
# ---------------------------------------------------------------------------


def test_component_missing_mask_preserved_per_component() -> None:
    """Dropouts in one component never bleed into or resample another."""
    # IMU snapshot with one dropout injected at the source level.
    imu_df = pd.read_csv(_snapshot_csv(ADCS_IMU)).copy()
    imu_df.loc[10, "adcs.imu.gyro_x_rad_s"] = float("nan")
    imu_frame = _map_frame(ADCS_IMU, imu_df)

    # RW frame stays clean.
    rw_frame = _frame_for(ADCS_RW)

    catalog = build_episodes_from_semantic_frames(
        frames={"imu": imu_frame, "rw": rw_frame},
        source_file_map={
            "imu": str(_snapshot_csv(ADCS_IMU)),
            "rw": str(_snapshot_csv(ADCS_RW)),
        },
        catalog_id="mask-preservation",
    )
    episode = catalog.episodes[0]
    assert len(episode.components) == 2
    assert set(c.component_instance_id for c in episode.components) == {"imu", "rw"}

    # The imu frame keeps its own mask at the injected dropout position.
    imu_signal = next(
        s for s in imu_frame.signals if s.semantic_channel_id.endswith(".angular_rate_x")
    )
    assert imu_signal.missing_mask[10] is True
    assert len(imu_signal.missing_mask) == len(imu_frame.timestamps_s)
    assert sum(imu_signal.missing_mask) == 1

    # The rw frame is untouched: all signals carry a fully-present mask.
    for signal in rw_frame.signals:
        assert signal.missing_mask == [False] * len(rw_frame.timestamps_s)
