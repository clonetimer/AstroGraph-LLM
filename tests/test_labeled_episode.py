"""Tests for the independent label adapter and labeled-episode materialization.

G2 work item "物化带标签 episode".  These tests verify:

1. the label adapter extracts labels (health state / fault ids / degradation)
   from a Sat-Agent source trace and pairs them with an episode via
   ``episode_id``;
2. the KEY guarantee: labels NEVER appear inside an ``EpisodeManifest``
   (``labels_used`` / ``truth_used`` stay False and frozen);
3. grouping fields exist and are enumerable (instance / config / scenario
   family), ready for a future grouped holdout (S1-authorized);
4. materialization is deterministic / replayable.

Tests use committed fixtures only — no external Sat-Agent path is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.windowing.episode_builder import build_episodes_from_profile_csvs
from astrograph.windowing.label_adapter import (
    build_labeled_episode,
    extract_labels_from_trace,
    load_episode_manifest,
    load_labeled_episode_catalog,
    materialize_labeled_episodes,
)

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CHANNEL_MAP_PATH = CONFIGS / "semantics" / "sat_agent_platform_channel_maps.s0.v1.json"
SCHEMA_PATH = CONFIGS / "semantics" / "core_semantic_schema.json"
PROFILE_PATH = CONFIGS / "data_sources" / "sat_agent_platform_schema_profiles.s0.v1.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "labeled_traces"
DATASET = ROOT / "data" / "labeled_episodes"

IMU_PROFILE = "sat-agent.component.imu.trace.v1"

IMU_NOMINAL = FIXTURES / "component_imu__nominal__source.csv"
IMU_FAULT = FIXTURES / "component_imu__fault__source.csv"
IMU_DEGRAD = FIXTURES / "component_imu__degradation__source.csv"
MTB_FAULT = FIXTURES / "component_mtb__fault__source.csv"

# The committed public-telemetry snapshot has NO label columns — it is the
# honest "nominal-only" public view used for the scenario-declared fallback.
IMU_SNAPSHOT = ROOT / "data" / "sat_agent_snapshots" / "traces/sat-agent.component.imu.trace.v1.csv"

_SCENARIO_IDS = {
    "nominal": "component_imu_v1_12s_nominal",
    "fault": "component_imu_v1_bias_step_fault",
    "degradation": "component_imu_v1_bias_degradation",
}


def _build_manifest(csv_path: Path, profile_id: str = IMU_PROFILE):
    catalog = build_episodes_from_profile_csvs(
        profile_file_map={profile_id: str(csv_path)},
        profile_path=str(PROFILE_PATH),
        channel_map_catalog_path=str(CHANNEL_MAP_PATH),
        semantic_schema_path=str(SCHEMA_PATH),
        catalog_id="test-labeled-episode",
    )
    return catalog.episodes[0]


def _imu_sources() -> dict[str, dict[str, str]]:
    return {
        IMU_PROFILE: {
            "nominal": str(IMU_NOMINAL),
            "fault": str(IMU_FAULT),
            "degradation": str(IMU_DEGRAD),
        }
    }


# --- Label extraction ---


class TestLabelExtraction:
    def test_extracts_degradation_labels(self):
        trace = pd.read_csv(IMU_DEGRAD)
        labels = extract_labels_from_trace(
            trace,
            scenario_family="degradation",
            scenario_id=_SCENARIO_IDS["degradation"],
            label_source_ref=str(IMU_DEGRAD),
        )
        assert labels.label_source == "trace_columns"
        assert labels.scenario_family == "degradation"
        assert labels.health_state == "degraded"
        assert "degraded_unverified" in labels.health_states_seen
        assert labels.degradation_active is True
        assert labels.fault_active is False
        assert labels.fault_ids == ["gyro_bias_drift"]
        assert "label.health_state" in labels.label_columns_present

    def test_extracts_fault_labels(self):
        trace = pd.read_csv(IMU_FAULT)
        labels = extract_labels_from_trace(
            trace,
            scenario_family="fault",
            scenario_id=_SCENARIO_IDS["fault"],
            label_source_ref=str(IMU_FAULT),
        )
        assert labels.health_state == "fault"
        assert labels.fault_active is True
        assert labels.fault_ids == ["imu_gyro_bias_step"]

    def test_extracts_nominal_labels(self):
        trace = pd.read_csv(IMU_NOMINAL)
        labels = extract_labels_from_trace(
            trace,
            scenario_family="nominal",
            scenario_id=_SCENARIO_IDS["nominal"],
            label_source_ref=str(IMU_NOMINAL),
        )
        assert labels.health_state == "nominal"
        assert labels.fault_active is False
        assert labels.degradation_active is False
        assert labels.fault_ids == []

    def test_different_object_same_adapter(self):
        trace = pd.read_csv(MTB_FAULT)
        labels = extract_labels_from_trace(
            trace,
            scenario_family="fault",
            scenario_id="mtb_fault_simulation_12s",
            label_source_ref=str(MTB_FAULT),
        )
        assert labels.health_state == "fault"
        assert labels.fault_ids == ["mtb_coil_open"]

    def test_no_label_columns_is_scenario_declared_not_fabricated(self):
        """A public trace with no label.* columns yields the declared scenario
        label only — nothing is fabricated."""
        trace = pd.read_csv(IMU_SNAPSHOT)
        assert not any(c.startswith("label.") for c in trace.columns)
        labels = extract_labels_from_trace(
            trace,
            scenario_family="nominal",
            scenario_id="component_imu_v1_12s_nominal",
            label_source_ref=str(IMU_SNAPSHOT),
        )
        assert labels.label_source == "scenario_declared"
        assert labels.health_state == "nominal"
        assert labels.fault_active is False
        assert labels.degradation_active is False
        assert labels.fault_ids == []


# --- The KEY isolation guarantee ---


class TestLabelIsolation:
    def test_labels_never_enter_episode_manifest(self):
        manifest = _build_manifest(IMU_FAULT)
        assert manifest.labels_used is False
        assert manifest.truth_used is False

        dumped = manifest.model_dump()
        # No supervision-carrying key may exist on the manifest.
        for forbidden in (
            "labels",
            "health_state",
            "fault_ids",
            "fault_active",
            "degradation_active",
            "modifier_active",
            "scenario_id",
            "scenario_family",
            "label_columns_present",
        ):
            assert forbidden not in dumped, forbidden

        # Semantic channels are public quantities only.
        for comp in manifest.components:
            for channel in comp.semantic_channel_ids:
                lowered = channel.lower()
                assert "label" not in lowered
                assert "fault" not in lowered
                assert "health" not in lowered

    def test_manifest_json_has_no_label_column(self):
        manifest = _build_manifest(IMU_DEGRAD)
        raw = json.dumps(manifest.model_dump())
        assert raw.find("fault_active") == -1
        assert raw.find("health_state") == -1

    def test_labeled_episode_is_separate_record(self):
        """The labeled record is a distinct document that only *references* the
        manifest through episode_id."""
        trace = pd.read_csv(IMU_FAULT)
        manifest = _build_manifest(IMU_FAULT)
        labeled = build_labeled_episode(
            manifest=manifest,
            trace=trace,
            scenario_family="fault",
            scenario_id=_SCENARIO_IDS["fault"],
            label_source_ref=str(IMU_FAULT),
            config_id="component.imu.v1",
            seed=0,
        )
        assert labeled.episode_id == manifest.episode_id
        # Labels live only in the labeled record.
        assert labeled.labels.health_state == "fault"
        assert labeled.labels.fault_ids == ["imu_gyro_bias_step"]
        assert labeled.labels.fault_active is True

    def test_materialized_manifest_source_ref_is_neutral(self, tmp_path):
        """P0 regression (fix round): the public manifest must NOT carry a
        scenario-bearing source path — only a neutral source:<profile_id> ref."""
        catalog = materialize_labeled_episodes(
            profile_sources=_imu_sources(),
            profile_path=PROFILE_PATH,
            channel_map_catalog_path=CHANNEL_MAP_PATH,
            semantic_schema_path=SCHEMA_PATH,
            output_dir=tmp_path,
        )
        for ep in catalog.episodes:
            manifest = load_episode_manifest(tmp_path, ep.episode_id)
            comp_id = manifest.components[0].component_instance_id
            for ref in manifest.source_file_references.values():
                assert ref == f"source:{comp_id}", ref
                assert "reports/" not in ref
                assert "__" not in ref
                assert "trace.csv" not in ref

    def test_committed_manifests_never_leak_scenario_tokens(self):
        """P0 regression on the committed dataset: grep every public manifest and
        assert no scenario token / reports path / trace.csv appears."""
        if not (DATASET / "manifests").is_dir():
            pytest.skip("materialized dataset not present in this checkout")
        scenario_tokens = ("__fault__", "__degradation__", "__nominal__")
        files = sorted((DATASET / "manifests").glob("*.json"))
        assert files, "expected at least one committed manifest"
        for path in files:
            text = path.read_text(encoding="utf-8")
            for token in scenario_tokens:
                assert token not in text, f"{path.name} leaks scenario token {token!r}"
            assert "reports/" not in text, path.name
            assert "trace.csv" not in text, path.name
            manifest = load_episode_manifest(DATASET, path.stem)
            assert manifest.labels_used is False and manifest.truth_used is False
            for ref in manifest.source_file_references.values():
                assert ref.startswith("source:"), ref

    def test_scenario_tokens_live_only_in_label_records(self):
        """Positive contrast: scenario provenance is legitimate in the label
        record (NOT a model input) and must never appear in the public manifest."""
        if not (DATASET / "catalog.json").is_file():
            pytest.skip("materialized dataset not present in this checkout")
        catalog = load_labeled_episode_catalog(DATASET)
        assert catalog.episodes, "empty catalog"
        for ep in catalog.episodes:
            label_text = (DATASET / "labels" / f"{ep.episode_id}.json").read_text(encoding="utf-8")
            manifest_text = (DATASET / "manifests" / f"{ep.episode_id}.json").read_text(encoding="utf-8")
            # The label record is the one legitimate home for scenario provenance.
            assert any(
                t in label_text for t in ("__fault__", "__degradation__", "__nominal__")
            ), ep.episode_id
            for token in ("__fault__", "__degradation__", "__nominal__"):
                assert token not in manifest_text, f"{ep.episode_id} manifest leaks {token!r}"


# --- Pairing + grouping ---


class TestPairingAndGrouping:
    def test_grouping_fields_populated_from_manifest(self):
        trace = pd.read_csv(IMU_FAULT)
        manifest = _build_manifest(IMU_FAULT)
        labeled = build_labeled_episode(
            manifest=manifest,
            trace=trace,
            scenario_family="fault",
            scenario_id=_SCENARIO_IDS["fault"],
            label_source_ref=str(IMU_FAULT),
            config_id="component.imu.v1",
            seed=0,
        )
        grouping = labeled.grouping
        assert grouping.source_id == "SAT_AGENT_PLATFORM"
        assert grouping.asset_instance_id == "asset-under-test"
        assert grouping.component_instance_id == manifest.components[0].component_instance_id
        assert grouping.component_archetype_id == manifest.components[0].component_archetype_id
        assert grouping.config_id == "component.imu.v1"
        assert grouping.scenario_family == "fault"
        assert grouping.scenario_id == _SCENARIO_IDS["fault"]
        assert grouping.seed == 0
        # No dedicated operating-condition column exists in Sat-Agent traces;
        # it is deliberately left unset (not fabricated).
        assert grouping.operating_condition is None

    def test_public_summary_has_no_signal_values(self):
        trace = pd.read_csv(IMU_DEGRAD)
        manifest = _build_manifest(IMU_DEGRAD)
        labeled = build_labeled_episode(
            manifest=manifest,
            trace=trace,
            scenario_family="degradation",
            scenario_id=_SCENARIO_IDS["degradation"],
            label_source_ref=str(IMU_DEGRAD),
            config_id="component.imu.v1",
            seed=0,
        )
        # Summary is descriptive only — never carries telemetry samples.
        assert labeled.row_count > 0
        assert labeled.start_s < labeled.end_s
        assert labeled.public_signal_channel_count == 3
        assert labeled.native_sample_period_s is not None
        raw = labeled.model_dump_json()
        assert "gyro_x" not in raw  # no signal values leak into the label record


# --- Materialization + replay ---


class TestMaterialization:
    def test_materializes_pairs_and_files(self, tmp_path):
        catalog = materialize_labeled_episodes(
            profile_sources=_imu_sources(),
            profile_path=PROFILE_PATH,
            channel_map_catalog_path=CHANNEL_MAP_PATH,
            semantic_schema_path=SCHEMA_PATH,
            output_dir=tmp_path,
        )
        assert len(catalog.episodes) == 3
        assert (tmp_path / "catalog.json").is_file()
        for ep in catalog.episodes:
            assert (tmp_path / "manifests" / f"{ep.episode_id}.json").is_file()
            assert (tmp_path / "labels" / f"{ep.episode_id}.json").is_file()
            manifest = load_episode_manifest(tmp_path, ep.episode_id)
            assert manifest.episode_id == ep.episode_id
            assert manifest.labels_used is False
            assert manifest.truth_used is False

    def test_grouping_fields_exist_and_enumerable(self, tmp_path):
        catalog = materialize_labeled_episodes(
            profile_sources=_imu_sources(),
            profile_path=PROFILE_PATH,
            channel_map_catalog_path=CHANNEL_MAP_PATH,
            semantic_schema_path=SCHEMA_PATH,
            output_dir=tmp_path,
        )
        # All declared grouping dimensions exist on every episode.
        non_null = {
            "source_id",
            "asset_instance_id",
            "component_instance_id",
            "component_archetype_id",
            "config_id",
            "scenario_family",
            "scenario_id",
        }
        by_family: dict[str, list[str]] = {}
        for ep in catalog.episodes:
            grouping = ep.grouping.model_dump()
            assert set(catalog.grouping_dimensions) <= set(grouping)
            for dim in non_null:
                assert grouping[dim] not in (None, ""), dim
            by_family.setdefault(grouping["scenario_family"], []).append(ep.episode_id)
        # Enumerable by scenario family; ids distinct across scenarios.
        assert set(by_family) == {"nominal", "fault", "degradation"}
        assert len(by_family["nominal"]) == len(by_family["fault"]) == len(by_family["degradation"]) == 1
        assert not (set(by_family["fault"]) & set(by_family["degradation"]))

    def test_materialization_is_deterministic_and_replayable(self, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        materialize_labeled_episodes(
            profile_sources=_imu_sources(),
            profile_path=PROFILE_PATH,
            channel_map_catalog_path=CHANNEL_MAP_PATH,
            semantic_schema_path=SCHEMA_PATH,
            output_dir=out1,
        )
        materialize_labeled_episodes(
            profile_sources=_imu_sources(),
            profile_path=PROFILE_PATH,
            channel_map_catalog_path=CHANNEL_MAP_PATH,
            semantic_schema_path=SCHEMA_PATH,
            output_dir=out2,
        )
        files1 = sorted(p.relative_to(out1).as_posix() for p in out1.rglob("*.json"))
        files2 = sorted(p.relative_to(out2).as_posix() for p in out2.rglob("*.json"))
        assert files1 == files2
        for relative in files1:
            assert (out1 / relative).read_bytes() == (out2 / relative).read_bytes(), relative

    def test_committed_dataset_is_self_consistent(self):
        if not (DATASET / "catalog.json").is_file():
            pytest.skip("materialized dataset not present in this checkout")
        catalog = load_labeled_episode_catalog(DATASET)
        assert len(catalog.episodes) >= 3

        seen: set[str] = set()
        families: set[str] = set()
        archetypes: set[str] = set()
        for ep in catalog.episodes:
            assert ep.episode_id not in seen
            seen.add(ep.episode_id)
            assert (DATASET / "manifests" / f"{ep.episode_id}.json").is_file()
            assert (DATASET / "labels" / f"{ep.episode_id}.json").is_file()
            manifest = load_episode_manifest(DATASET, ep.episode_id)
            assert manifest.episode_id == ep.episode_id
            assert manifest.labels_used is False and manifest.truth_used is False
            families.add(ep.grouping.scenario_family)
            archetypes.add(ep.grouping.component_archetype_id)

        # The materialized dataset covers all three scenario families and
        # several archetypes (groupable for grouped holdout).
        assert {"nominal", "fault", "degradation"} <= families
        assert len(archetypes) >= 3
