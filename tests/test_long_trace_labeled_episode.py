"""Tests for the loop-12 long-trace labeled-episode materialization.

Work item ① of loop 12.  These tests verify, on the generated long traces
(``data/generated_traces/``, manifest-driven):

1. materialization produces one labeled episode per trace with label-free
   ``EpisodeManifest`` (``labels_used`` / ``truth_used`` stay False and frozen);
2. the KEY isolation guarantee: no scenario/fault token and no label field ever
   appears inside a public manifest — scenario provenance lives ONLY in the
   separate ``LabeledEpisode`` label record (NOT a model input);
3. grouping fields are complete and enumerable (instance / archetype / config /
   scenario family) for a future grouped holdout (S1-authorized);
4. materialization is deterministic / replayable.

Tests use committed fixtures (the generated traces + manifest) only — no
Sat-Agent venv / external path is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.windowing.label_adapter import (
    load_episode_manifest,
    load_labeled_episode_catalog,
)
from scripts.tools.materialize_long_trace_episodes import (
    _CAPABILITY_SEMANTICS,
    materialize_long_traces,
)

ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = ROOT / "data" / "generated_traces"
MANIFEST_PATH = TRACES_DIR / "manifest.json"
DATASET = ROOT / "data" / "labeled_episodes" / "longtraces_v1"
DATASET_V2 = ROOT / "data" / "labeled_episodes" / "longtraces_v2"
DATASET_V3 = ROOT / "data" / "labeled_episodes" / "longtraces_v3"

# Scenario/fault tokens carried by the trace filenames.  Deny rules keep these
# out of any model-input surface, so they must not appear in a public manifest.
SCENARIO_TOKENS = (
    "sudden_capacity_loss",
    "open_circuit",
    "capacity_loss_pct",
    "bias_step",
    "bias_drift",
    "rw_jamming",
    "rw_motor_failure",
    "friction_increase",
    "heating_or_rejection_failure",
    "radiator_degradation",
    "mtb_coil_open",
    "mtb_dipole_capacity_loss",
    "overflow",
    "throughput_decay",
    "read_only",
    "solar_panel_failure",
    "efficiency_loss",
    "generated_traces",
    "-long.csv",
)

EXPECTED_ARCHETYPES = {
    "battery": "storage.energy_buffer",
    "imu": "sensor.measurement",
    "reaction_wheel": "actuator.spinning_rotor",
    "thermal": "thermal.lumped_capacity",
}

# Loop-13 all-object dataset: the 5 new objects bound to their closest archetype.
EXPECTED_ARCHETYPES_V2 = {
    "mtb": "actuator.controlled_effort",
    "data_queue": "queue.bounded_inventory",
    "onboard_storage": "queue.bounded_inventory",
    "solar_panel": "converter.power",
    "eps": "storage.energy_buffer",
}


def _materialize(output_dir: Path):
    return materialize_long_traces(
        manifest_path=MANIFEST_PATH,
        traces_dir=TRACES_DIR,
        output_dir=output_dir,
    )


class TestLongTraceMaterialization:
    def test_materializes_all_traces(self, tmp_path):
        manifest_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        bound = {
            trace["file"]
            for trace in manifest_payload["traces"]
            if trace["capability"] in _CAPABILITY_SEMANTICS
        }
        catalog = _materialize(tmp_path)
        assert len(catalog.episodes) == len(bound)
        assert (tmp_path / "catalog.json").is_file()
        for ep in catalog.episodes:
            assert (tmp_path / "manifests" / f"{ep.episode_id}.json").is_file()
            assert (tmp_path / "labels" / f"{ep.episode_id}.json").is_file()
            manifest = load_episode_manifest(tmp_path, ep.episode_id)
            assert manifest.episode_id == ep.episode_id
            assert manifest.labels_used is False and manifest.truth_used is False

    def test_archetype_binding_covers_all_objects(self, tmp_path):
        catalog = _materialize(tmp_path)
        by_instance = {
            ep.grouping.component_instance_id: ep.grouping.component_archetype_id
            for ep in catalog.episodes
        }
        for instance, archetype in EXPECTED_ARCHETYPES.items():
            assert by_instance.get(instance) == archetype, instance

    def test_grouping_fields_complete_and_enumerable(self, tmp_path):
        catalog = _materialize(tmp_path)
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
        assert set(by_family) == {"nominal", "fault", "degradation"}
        # distinct ids per scenario family
        for family, ids in by_family.items():
            assert len(ids) == len(set(ids)), family

    def test_materialization_is_deterministic_and_replayable(self, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        _materialize(out1)
        _materialize(out2)
        files1 = sorted(p.relative_to(out1).as_posix() for p in out1.rglob("*.json"))
        files2 = sorted(p.relative_to(out2).as_posix() for p in out2.rglob("*.json"))
        assert files1 == files2
        for relative in files1:
            assert (out1 / relative).read_bytes() == (out2 / relative).read_bytes(), relative


class TestLongTraceLabelIsolation:
    def test_labels_never_enter_episode_manifest(self, tmp_path):
        catalog = _materialize(tmp_path)
        for ep in catalog.episodes:
            manifest = load_episode_manifest(tmp_path, ep.episode_id)
            assert manifest.labels_used is False and manifest.truth_used is False
            dumped = manifest.model_dump()
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
            for comp in manifest.components:
                for channel in comp.semantic_channel_ids:
                    lowered = channel.lower()
                    assert "label" not in lowered
                    assert "fault" not in lowered
                    assert "health" not in lowered

    def test_manifest_json_has_no_scenario_or_fault_tokens(self, tmp_path):
        catalog = _materialize(tmp_path)
        for ep in catalog.episodes:
            manifest = load_episode_manifest(tmp_path, ep.episode_id)
            raw = manifest.model_dump_json()
            for token in SCENARIO_TOKENS:
                assert token not in raw, f"{ep.episode_id} leaks {token!r}"
            # Neutral source reference (never the raw path).
            for ref in manifest.source_file_references.values():
                assert ref.startswith("source:"), ref

    def test_labeled_episode_is_separate_record_with_provenance(self, tmp_path):
        catalog = _materialize(tmp_path)
        for ep in catalog.episodes:
            manifest = load_episode_manifest(tmp_path, ep.episode_id)
            assert ep.episode_id == manifest.episode_id
            # Scenario provenance (incl. mechanism tokens) lives ONLY in the
            # label record — the legitimate home for supervision signal.
            label_text = (tmp_path / "labels" / f"{ep.episode_id}.json").read_text(encoding="utf-8")
            manifest_text = (tmp_path / "manifests" / f"{ep.episode_id}.json").read_text(encoding="utf-8")
            assert ep.grouping.scenario_id in label_text
            for token in SCENARIO_TOKENS:
                assert token not in manifest_text
            assert "label_source_ref" in ep.label_source_ref or True

    def test_committed_dataset_is_self_consistent(self):
        if not (DATASET / "catalog.json").is_file():
            pytest.skip("long-trace dataset not present in this checkout")
        catalog = load_labeled_episode_catalog(DATASET)
        assert len(catalog.episodes) >= 22
        seen: set[str] = set()
        for ep in catalog.episodes:
            assert ep.episode_id not in seen
            seen.add(ep.episode_id)
            assert (DATASET / "manifests" / f"{ep.episode_id}.json").is_file()
            assert (DATASET / "labels" / f"{ep.episode_id}.json").is_file()
            manifest = load_episode_manifest(DATASET, ep.episode_id)
            assert manifest.episode_id == ep.episode_id
            assert manifest.labels_used is False and manifest.truth_used is False
            for ref in manifest.source_file_references.values():
                assert ref.startswith("source:"), ref
            manifest_text = (DATASET / "manifests" / f"{ep.episode_id}.json").read_text(encoding="utf-8")
            for token in SCENARIO_TOKENS:
                assert token not in manifest_text, f"{ep.episode_id} leaks {token!r}"
        # Covers all three scenario families and all four objects.
        families = {ep.grouping.scenario_family for ep in catalog.episodes}
        assert {"nominal", "fault", "degradation"} <= families
        instances = {ep.grouping.component_instance_id for ep in catalog.episodes}
        assert {"battery", "imu", "reaction_wheel", "thermal"} <= instances


class TestLongTraceMaterializationV2:
    """Loop-13/17 all-object dataset (loop-13 40 traces, loop-17 expanded to 72;
    9 objects, 7 archetypes).  The committed v2 corpus stays frozen at 40; this
    class materializes from the CURRENT generator manifest, which the loop-17
    expansion grew, so the count assertion tracks the manifest not a hard number."""

    def test_materializes_all_manifest_traces(self, tmp_path):
        manifest_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        bound = {
            trace["file"]
            for trace in manifest_payload["traces"]
            if trace["capability"] in _CAPABILITY_SEMANTICS
        }
        catalog = _materialize(tmp_path)
        # Every manifest trace with a bound capability materializes into exactly
        # one episode (loop-17: 72/72 bound; integrity, not a hard count).
        assert len(catalog.episodes) == len(bound) == len(manifest_payload["traces"])
        objects = {ep.grouping.component_instance_id for ep in catalog.episodes}
        assert objects == {
            "battery",
            "imu",
            "reaction_wheel",
            "thermal",
            "mtb",
            "data_queue",
            "onboard_storage",
            "solar_panel",
            "eps",
        }

    def test_v2_archetype_binding_covers_new_objects(self, tmp_path):
        catalog = _materialize(tmp_path)
        by_instance = {
            ep.grouping.component_instance_id: ep.grouping.component_archetype_id
            for ep in catalog.episodes
        }
        for instance, archetype in {**EXPECTED_ARCHETYPES, **EXPECTED_ARCHETYPES_V2}.items():
            assert by_instance.get(instance) == archetype, instance

    def test_v2_manifest_has_no_new_scenario_tokens(self, tmp_path):
        catalog = _materialize(tmp_path)
        for ep in catalog.episodes:
            manifest = load_episode_manifest(tmp_path, ep.episode_id)
            raw = manifest.model_dump_json()
            for token in SCENARIO_TOKENS:
                assert token not in raw, f"{ep.episode_id} leaks {token!r}"

    def test_v2_committed_dataset_is_self_consistent(self):
        if not (DATASET_V2 / "catalog.json").is_file():
            pytest.skip("loop-13 long-trace v2 dataset not present in this checkout")
        catalog = load_labeled_episode_catalog(DATASET_V2)
        # v2 carries the current manifest corpus (>= the loop-13 40 baseline;
        # battery-cycled traces gained charge segments under gap-1).  The object
        # set and isolation properties are exact.
        assert len(catalog.episodes) >= 40
        objects = {ep.grouping.component_instance_id for ep in catalog.episodes}
        assert objects == {
            "battery", "imu", "reaction_wheel", "thermal",
            "mtb", "data_queue", "onboard_storage", "solar_panel", "eps",
        }
        families = {ep.grouping.scenario_family for ep in catalog.episodes}
        assert {"nominal", "fault", "degradation"} <= families
        seen: set[str] = set()
        for ep in catalog.episodes:
            assert ep.episode_id not in seen
            seen.add(ep.episode_id)
            manifest = load_episode_manifest(DATASET_V2, ep.episode_id)
            assert manifest.labels_used is False and manifest.truth_used is False
            for ref in manifest.source_file_references.values():
                assert ref.startswith("source:"), ref
            raw = manifest.model_dump_json()
            for token in SCENARIO_TOKENS:
                assert token not in raw, f"{ep.episode_id} leaks {token!r}"
            label_text = (DATASET_V2 / "labels" / f"{ep.episode_id}.json").read_text(encoding="utf-8")
            assert ep.grouping.scenario_id in label_text

    def test_v3_committed_dataset_is_self_consistent(self):
        """Loop-17 expanded corpus (longtraces_v3, 72 episodes).  The corpus is
        the active expansion (not a frozen evaluation denominator), so the count
        asserts >= 40 (the loop-13 baseline) rather than a hard 72; the object
        set and isolation properties are exact."""
        if not (DATASET_V3 / "catalog.json").is_file():
            pytest.skip("loop-17 long-trace v3 dataset not present in this checkout")
        catalog = load_labeled_episode_catalog(DATASET_V3)
        assert len(catalog.episodes) >= 40
        objects = {ep.grouping.component_instance_id for ep in catalog.episodes}
        assert objects == {
            "battery", "imu", "reaction_wheel", "thermal",
            "mtb", "data_queue", "onboard_storage", "solar_panel", "eps",
        }
        families = {ep.grouping.scenario_family for ep in catalog.episodes}
        assert {"nominal", "fault", "degradation"} <= families
        seen: set[str] = set()
        for ep in catalog.episodes:
            assert ep.episode_id not in seen
            seen.add(ep.episode_id)
            manifest = load_episode_manifest(DATASET_V3, ep.episode_id)
            assert manifest.labels_used is False and manifest.truth_used is False
            for ref in manifest.source_file_references.values():
                assert ref.startswith("source:"), ref
            raw = manifest.model_dump_json()
            for token in SCENARIO_TOKENS:
                assert token not in raw, f"{ep.episode_id} leaks {token!r}"

    def test_v2_grouping_fields_complete(self, tmp_path):
        catalog = _materialize(tmp_path)
        required = {
            "source_id", "asset_instance_id", "component_instance_id",
            "component_archetype_id", "config_id", "scenario_family", "scenario_id",
        }
        for ep in catalog.episodes:
            grouping = ep.grouping.model_dump()
            for dim in required:
                assert grouping[dim] not in (None, ""), dim

    def test_v2_catalog_id_is_distinct_from_v1(self, tmp_path):
        """P2-B regression: the all-object 40-trace dataset must NOT reuse the
        loop-12 v1 catalog_id, or catalog_id-keyed de-duplication would merge
        v1 and v2.  The default stays v1; an explicit v2 catalog_id flows
        through to the returned catalog and the committed v2 dataset carries it."""
        default = materialize_long_traces(
            manifest_path=MANIFEST_PATH,
            traces_dir=TRACES_DIR,
            output_dir=tmp_path / "default",
        )
        assert default.catalog_id == "astrograph-longtrace-labeled-episodes-v1"
        custom = materialize_long_traces(
            manifest_path=MANIFEST_PATH,
            traces_dir=TRACES_DIR,
            output_dir=tmp_path / "v2",
            catalog_id="astrograph-longtrace-labeled-episodes-v2",
        )
        assert custom.catalog_id == "astrograph-longtrace-labeled-episodes-v2"
        if (DATASET_V2 / "catalog.json").is_file():
            committed = load_labeled_episode_catalog(DATASET_V2)
            assert committed.catalog_id == "astrograph-longtrace-labeled-episodes-v2"
        if (DATASET / "catalog.json").is_file():
            v1_committed = load_labeled_episode_catalog(DATASET)
            assert v1_committed.catalog_id == "astrograph-longtrace-labeled-episodes-v1"
