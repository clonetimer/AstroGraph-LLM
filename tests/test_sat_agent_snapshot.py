"""Tests for the Sat-Agent public-telemetry snapshot infrastructure.

The snapshot copies only public telemetry columns (``time_s`` + channel-map
bound source fields) from real Sat-Agent trace CSVs into
``data/sat_agent_snapshots/`` with an idempotent SHA-256 manifest.  These tests
cover:

1. manifest + snapshot files are valid (exist, sha256 match, no forbidden
   columns, row_count > 0, quality facts preserved);
2. the snapshot is idempotent (a second run copies nothing);
3. snapshot files are consumable by the generic episode builder;
4. the on-disk committed deliverable manifest is intact.

Tests that regenerate from the source skip when the Sat-Agent tree is absent.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "data" / "sat_agent_snapshots"
MANIFEST_PATH = SNAPSHOT_ROOT / "manifest.json"

SCRIPT = ROOT / "scripts" / "tools" / "snapshot_sat_agent_data.py"
SPEC = importlib.util.spec_from_file_location("snapshot_sat_agent_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SNAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAP)

SCHEMA_VERSION = "astrograph-sat-agent-snapshot-manifest.v1"
# Loop 2.A closure (2026-08-04): eps-unified + whole-spacecraft composite were
# added to the snapshot list (real Sat-Agent data exists), 24 -> 26.
EXPECTED_PROFILE_COUNT = 26

# Independent deny list mirroring the schema-profile global deny rules, so the
# test does not trust the snapshot script's own filter.
_FORBIDDEN_PREFIXES = (
    "label.",
    "label_",
    "truth.",
    "ground_truth.",
    "effect.",
    "event.",
    "modifier.",
    "object.",
    "status.",
    "task_id",
    "case_id",
    "dataset_id",
    "scenario_id",
    "pair_id",
    "run_id",
    "seed",
    "split",
    "mode",
    "target_level",
    "target_name",
    "sample_index",
)
_FORBIDDEN_SUBSTRINGS = (
    "effective_",
    "effective.",
    ".effective_",
    ".bias",
    ".drift",
    ".degradation",
    ".fault",
    ".health",
    ".nominal",
    ".stuck",
    ".available",
    ".enabled",
    "fault_active",
    "degradation_active",
    "health_state",
)


def _forbidden(name: str) -> bool:
    lowered = str(name).strip().lower()
    if not lowered or lowered == "time_s":
        return False
    if lowered.startswith(_FORBIDDEN_PREFIXES):
        return True
    return any(token in lowered for token in _FORBIDDEN_SUBSTRINGS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [str(item).strip() for item in next(csv.reader(handle))]


def _sources_present() -> bool:
    sat_root = Path(
        os.environ.get("ASTROGRAPH_SAT_AGENT_ROOT", SNAP.DEFAULT_SAT_AGENT_ROOT)
    )
    return bool(sat_root.is_dir()) and all(
        (sat_root / rel).is_file() for rel in SNAP.DEFAULT_SOURCE_FILES.values()
    )


requires_source = pytest.mark.skipif(
    not _sources_present(), reason="Sat-Agent source tree not available"
)

# Snapshot files are regenerated into a fresh temp directory so tests never
# disturb the committed deliverable in data/sat_agent_snapshots/.


@requires_source
def test_snapshot_manifest_and_files_are_valid(tmp_path: Path) -> None:
    snap_root = tmp_path / "snap"
    report = SNAP.snapshot(snapshot_root=str(snap_root))
    assert report["counts"]["requested"] == EXPECTED_PROFILE_COUNT
    assert report["counts"]["missing_source"] == 0
    assert report["counts"]["errored"] == 0
    assert report["counts"]["copied"] == EXPECTED_PROFILE_COUNT

    manifest_path = snap_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["source_id"] == "SAT_AGENT_PLATFORM"
    assert manifest["source_count"] == 1
    assert manifest["snapshot_state"] == "complete"
    assert manifest["files_present"] == EXPECTED_PROFILE_COUNT
    assert manifest["preserved_quality_facts"]["empty_trajectories"] == 47
    assert manifest["preserved_quality_facts"]["duplicate_trajectories"] == 176
    assert manifest["preserved_quality_facts"]["cross_annotation_conflicts"] == 4

    assert len(manifest["files"]) == EXPECTED_PROFILE_COUNT
    for entry in manifest["files"]:
        assert entry["action"] in ("copied", "skipped")
        local = snap_root / entry["local_path"]
        assert local.is_file(), f"missing snapshot: {entry['local_path']}"
        assert _sha256(local) == entry["sha256"], f"sha mismatch: {entry['profile_id']}"
        headers = _headers(local)
        assert "time_s" in headers, f"no time axis: {entry['profile_id']}"
        assert not any(_forbidden(c) for c in headers), f"forbidden column: {headers}"
        assert entry["row_count"] > 0, f"empty trajectory: {entry['profile_id']}"
        assert len(headers) >= 2, headers  # time axis + at least one signal


@requires_source
def test_snapshot_is_idempotent(tmp_path: Path) -> None:
    snap_root = tmp_path / "snap"
    first = SNAP.snapshot(snapshot_root=str(snap_root))
    assert first["counts"]["copied"] == EXPECTED_PROFILE_COUNT

    second = SNAP.snapshot(snapshot_root=str(snap_root))
    assert second["counts"]["copied"] == 0, "second run must not re-copy"
    assert second["counts"]["skipped"] == EXPECTED_PROFILE_COUNT

    # Manifest bytes are unchanged between runs.
    before = (snap_root / "manifest.json").read_bytes()
    SNAP.snapshot(snapshot_root=str(snap_root))
    after = (snap_root / "manifest.json").read_bytes()
    assert before == after


@requires_source
def test_snapshot_files_consume_through_episode_builder(tmp_path: Path) -> None:
    from astrograph.windowing.episode_builder import build_episodes_from_profile_csvs

    snap_root = tmp_path / "snap"
    SNAP.snapshot(snapshot_root=str(snap_root))
    manifest = json.loads((snap_root / "manifest.json").read_text(encoding="utf-8"))

    picked = [
        "sat-agent.component.battery.trace.v1",
        "sat-agent.subsystem.adcs-rw.astrograph-5ch.v2",
    ]
    mapping = {}
    for entry in manifest["files"]:
        if entry["profile_id"] in picked:
            mapping[entry["profile_id"]] = str(snap_root / entry["local_path"])
    assert len(mapping) == 2

    catalog = build_episodes_from_profile_csvs(
        profile_file_map=mapping,
        profile_path=str(ROOT / "configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json"),
        channel_map_catalog_path=str(ROOT / "configs/semantics/sat_agent_platform_channel_maps.s0.v1.json"),
        semantic_schema_path=str(ROOT / "configs/semantics/core_semantic_schema.json"),
        catalog_id="test-snapshot-consumption",
    )
    assert len(catalog.episodes) >= 1
    components = [c for episode in catalog.episodes for c in episode.components]
    assert len(components) == 2
    for comp in components:
        assert comp.row_count > 0
        assert comp.data_quality is not None
    for episode in catalog.episodes:
        assert episode.labels_used is False
        assert episode.truth_used is False


def test_committed_snapshot_deliverable_is_intact() -> None:
    """Validate the on-disk committed manifest (data/sat_agent_snapshots)."""
    if not MANIFEST_PATH.is_file():
        pytest.skip("committed snapshot manifest not generated yet")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["source_id"] == "SAT_AGENT_PLATFORM"
    assert manifest["source_count"] == 1
    assert manifest["snapshot_state"] == "complete"
    assert manifest["files_present"] == EXPECTED_PROFILE_COUNT
    assert manifest["preserved_quality_facts"]["empty_trajectories"] == 47
    assert manifest["preserved_quality_facts"]["duplicate_trajectories"] == 176
    assert manifest["preserved_quality_facts"]["cross_annotation_conflicts"] == 4
    assert len(manifest["files"]) == EXPECTED_PROFILE_COUNT
    for entry in manifest["files"]:
        assert entry["action"] in ("copied", "skipped")
        local = SNAPSHOT_ROOT / entry["local_path"]
        assert local.is_file(), f"missing committed snapshot: {entry['local_path']}"
        assert _sha256(local) == entry["sha256"], f"committed sha mismatch: {entry['profile_id']}"
        assert not any(_forbidden(c) for c in _headers(local)), entry["profile_id"]
        assert entry["row_count"] > 0
