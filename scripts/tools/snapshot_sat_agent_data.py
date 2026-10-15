#!/usr/bin/env python3
"""Snapshot public Sat-Agent telemetry into the AstroGraph repo.

This is the project-owner-approved "data into this folder" work item.  It copies
only *public telemetry* columns (``time_s`` + the source fields already bound by
the channel-map catalog) from a curated set of real Sat-Agent trace CSVs into::

    data/sat_agent_snapshots/traces/<profile_id>.csv

and writes an idempotent ``data/sat_agent_snapshots/manifest.json`` recording the
profile, the source path, a SHA-256 per file, row count and time-axis
classification.

Hard guarantees (spec section 5.3):

* Label / truth / effect / event / object / identity / effective-* columns are
  NEVER copied.  Truth stays physically isolated.
* The source tree is opened read-only and never modified.
* No time axis is synthesised; the source ``time_s`` column is copied verbatim
  and only *classified* (uniform frame counter vs. real timestamps).
* Simulation scripts are not copied; they remain in the Sat-Agent environment.
* Source files already snapshotted with an unchanged SHA-256 are skipped on
  re-runs (byte-identical idempotency).
* The preserved quality facts (47 empty / 176 duplicate / 4 cross-annotation
  conflict) are recorded verbatim and never cleared or "fixed".

Run with the AstroGraph interpreter only (stdlib + pandas + numpy, no imports
of sat_agent_platform packages)::

    /d/Project/test_lunwen/.venv/Scripts/python.exe scripts/snapshot_sat_agent_data.py

The source root may be overridden with ``--sat-root`` or the
``ASTROGRAPH_SAT_AGENT_ROOT`` environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "data" / "sat_agent_snapshots"
TRACES_DIR = SNAPSHOT_DIR / "traces"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"

SCHEMA_VERSION = "astrograph-sat-agent-snapshot-manifest.v1"
SOURCE_ID = "SAT_AGENT_PLATFORM"
SOURCE_COUNT = 1

DEFAULT_SAT_AGENT_ROOT = r"D:/Project/test_sat/sat_agent_platform"

# profile_id -> channel-map mapping_id.  These bindings are the same objects the
# episode builder already consumes (configs/semantics/sat_agent_platform_channel_maps.s0.v1.json).
PROFILE_CHANNEL_MAPS: dict[str, str] = {
    "sat-agent.component.battery.trace.v1": "sat-agent-platform.component.battery.v1",
    "sat-agent.subsystem.adcs-rw.astrograph-5ch.v2": "sat-agent-platform.subsystem.adcs-rw.v2",
    "sat-agent.subsystem.thermal.source-native.v1": "sat-agent-platform.subsystem.thermal.v1",
    "sat-agent.component.imu.trace.v1": "sat-agent-platform.component.imu.v1",
    "sat-agent.component.pdu.trace.v1": "sat-agent-platform.component.pdu.v1",
    "sat-agent.component.data-queue.trace.v1": "sat-agent-platform.component.data-queue-link.v1",
    "sat-agent.component.cmg.trace.v1": "sat-agent-platform.component.cmg.v1",
    "sat-agent.component.mtb.trace.v1": "sat-agent-platform.component.mtb.v1",
    "sat-agent.component.star-tracker.trace.v1": "sat-agent-platform.component.star_tracker.v1",
    "sat-agent.component.sun-sensor.trace.v1": "sat-agent-platform.component.sun_sensor.v1",
    "sat-agent.component.magnetometer.trace.v1": "sat-agent-platform.component.magnetometer.v1",
    "sat-agent.component.heater.trace.v1": "sat-agent-platform.component.heater.v1",
    "sat-agent.component.radiator.trace.v1": "sat-agent-platform.component.radiator.v1",
    "sat-agent.component.transmitter.trace.v1": "sat-agent-platform.component.transmitter.v1",
    "sat-agent.component.link-budget.trace.v1": "sat-agent-platform.component.link_budget.v1",
    "sat-agent.component.onboard-storage.trace.v1": "sat-agent-platform.component.onboard-storage.v1",
    "sat-agent.component.solar-panel.trace.v1": "sat-agent-platform.component.solar_panel.v1",
    "sat-agent.component.power-sink.trace.v1": "sat-agent-platform.component.power_sink.v1",
    "sat-agent.component.antenna.trace.v1": "sat-agent-platform.component.antenna.v1",
    "sat-agent.component.ground-station.trace.v1": "sat-agent-platform.component.ground_station.v1",
    "sat-agent.component.fuel-tank.trace.v1": "sat-agent-platform.component.fuel_tank.v1",
    "sat-agent.component.payload.trace.v1": "sat-agent-platform.component.payload.v1",
    "sat-agent.component.payload-sensor.trace.v1": "sat-agent-platform.component.payload_sensor.v1",
    "sat-agent.component.thruster.trace.v1": "sat-agent-platform.component.thruster.v1",
    "sat-agent.subsystem.eps.unified-native.v1": "sat-agent-platform.subsystem.eps-unified.v1",
    "sat-agent.whole-spacecraft.composite.v1": "sat-agent-platform.whole-spacecraft.composite.v1",
}

# Confirmed real Sat-Agent files.  Batch-A paths come from
# ``tests/test_episode_builder.py`` (BATCH_A_FILES); the five remaining ADCS
# components come from the nominal traces resolved by
# ``tests/test_adcs_remaining_component_semantic_binding.py``
# (``**/component_<obj>__nominal__alias/runs/trace.csv``).  Paths are relative to
# the Sat-Agent platform root.
DEFAULT_SOURCE_FILES: dict[str, str] = {
    "sat-agent.component.battery.trace.v1": (
        "reports/capacity_benchmark/datasets/battery-fault-density-8/"
        "battery-fault-density-8-w01-s01/trace.csv"
    ),
    "sat-agent.subsystem.adcs-rw.astrograph-5ch.v2": (
        "reports/rw_native_validation/cases/"
        "rw_native_validation-validation-adcs_rw_torque_authority_loss-002-nominal/"
        "astrograph/telemetry.csv"
    ),
    "sat-agent.subsystem.thermal.source-native.v1": (
        "reports/capacity_benchmark/datasets/thermal-high-rate-600/"
        "thermal-high-rate-600-w01-s01/trace.csv"
    ),
    "sat-agent.component.imu.trace.v1": (
        "reports/vllm_nl_e2e_matrix_en_batch_039/evidence/cases/"
        "component_imu__nominal__en/runs/trace.csv"
    ),
    "sat-agent.component.pdu.trace.v1": (
        "reports/vllm_nl_e2e_matrix_en_batch_063/evidence/cases/"
        "component_pdu__nominal__en/runs/trace.csv"
    ),
    "sat-agent.component.data-queue.trace.v1": (
        "reports/vllm_nl_e2e_matrix_en_data_queue_recheck/evidence/cases/"
        "component_data_queue__degradation__en/runs/trace.csv"
    ),
    "sat-agent.component.cmg.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_027/evidence/cases/"
        "component_cmg__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.mtb.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_051/evidence/cases/"
        "component_mtb__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.star-tracker.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_075/evidence/cases/"
        "component_star_tracker__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.sun-sensor.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_075/evidence/cases/"
        "component_sun_sensor__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.magnetometer.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
        "component_magnetometer__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.heater.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
        "component_heater__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.radiator.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_063/evidence/cases/"
        "component_radiator__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.transmitter.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_087/evidence/cases/"
        "component_transmitter__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.link-budget.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
        "component_link_budget__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.onboard-storage.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_051/evidence/cases/"
        "component_onboard_storage__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.solar-panel.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_075/evidence/cases/"
        "component_solar_panel__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.power-sink.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_063/evidence/cases/"
        "component_power_sink__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.antenna.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_015/evidence/cases/"
        "component_antenna__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.ground-station.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_027/evidence/cases/"
        "component_ground_station__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.fuel-tank.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_027/evidence/cases/"
        "component_fuel_tank__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.payload.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_051/evidence/cases/"
        "component_payload__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.payload-sensor.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_051/evidence/cases/"
        "component_payload_sensor__nominal__alias/runs/trace.csv"
    ),
    "sat-agent.component.thruster.trace.v1": (
        "reports/vllm_nl_e2e_matrix_alias_batch_087/evidence/cases/"
        "component_thruster__nominal__alias/runs/trace.csv"
    ),
    # Loop 2.A closure (2026-08-04): eps-unified and whole-spacecraft composite
    # were "source-blocked" only because the snapshot list never included them;
    # the Sat-Agent source has exact real data under capacity_benchmark/datasets.
    # eps-basilisk-600 carries the eps.unified-native 3-field schema.
    "sat-agent.subsystem.eps.unified-native.v1": (
        "reports/capacity_benchmark/datasets/eps-basilisk-600/"
        "eps-basilisk-600-w01-s01/trace.csv"
    ),
    # composite-graph-120 carries the whole-spacecraft composite 7-field schema.
    "sat-agent.whole-spacecraft.composite.v1": (
        "reports/capacity_benchmark/datasets/composite-graph-120/"
        "composite-graph-120-w01-s01/trace.csv"
    ),
}

# Preserved platform-wide quality facts (DEV_GOAL hard constraint: "47 empty /
# 176 duplicate / 4 conflict kept as quality facts").  These describe the whole
# Sat-Agent report matrix, not just the snapshotted files.  They are kept as
# facts and are never cleared or "repaired" here.
PRESERVED_QUALITY_FACTS: dict[str, Any] = {
    "empty_trajectories": 47,
    "duplicate_trajectories": 176,
    "cross_annotation_conflicts": 4,
    "provenance": (
        "Platform-wide Sat-Agent report-matrix quality facts preserved verbatim "
        "(DEV_GOAL quality-fact constraint): 47 empty trajectories, 176 duplicate "
        "trajectories and 4 cross-annotation conflicts.  Data problems, not "
        "model-tuning problems; never cleared or repaired by the snapshot."
    ),
}

# Deny rules mirrored from configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json
# (global_model_input_deny_rules) and the conservative adapter guard.  These are a
# fail-closed safety net on top of the channel-map bound-field selection.
_FORBIDDEN_PREFIXES: tuple[str, ...] = (
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
_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "effective_",
    "effective.",
    ".effective_",
    ".bias",
    ".drift",
    ".dropout",
    ".degradation",
    ".fault",
    ".health",
    ".injected",
    ".nominal",
    ".stuck",
    ".available",
    ".enabled",
    "fault_active",
    "degradation_active",
    "health_state",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_forbidden_column(name: str) -> bool:
    """Return True when a *signal* column must never enter a snapshot.

    The ``time_s`` time axis is deliberately exempt: it is not a model input and
    is the column the snapshot is required to preserve.
    """
    lowered = str(name).strip().lower()
    if not lowered or lowered == "time_s":
        return False
    if lowered.startswith(_FORBIDDEN_PREFIXES):
        return True
    return any(token in lowered for token in _FORBIDDEN_SUBSTRINGS)


def _is_uniform_frame_counter(series: pd.Series) -> bool:
    """Classify the time axis exactly like episode_builder._assess_time_axis.

    A uniform frame counter is a strictly-increasing integer sequence with unit
    step (0, 1, 2, ...).  This is metadata only; the raw values are never
    altered.
    """
    arr = np.asarray(series.to_numpy(dtype=float), dtype=float)
    if len(arr) < 2:
        return False
    diffs = np.diff(arr)
    if not bool(np.all(diffs > 0)):
        return False
    int_ts = arr.astype(np.int64)
    if not bool(np.allclose(arr, int_ts.astype(float))):
        return False
    expected = np.arange(int_ts[0], int_ts[0] + len(int_ts))
    return bool(np.array_equal(int_ts, expected))


def _load_channel_map_source_fields(channel_map_catalog_path: Path) -> dict[str, set[str]]:
    """Return mapping_id -> set(source_field) from the channel-map catalog."""
    payload = json.loads(channel_map_catalog_path.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for cm in payload.get("maps", []):
        mapping_id = cm.get("mapping_id")
        if not mapping_id:
            continue
        result[mapping_id] = {entry["source_field"] for entry in cm.get("entries", [])}
    return result


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def snapshot(
    *,
    sat_root: str | Path | None = None,
    source_files: dict[str, str] | None = None,
    channel_map_catalog_path: str | Path | None = None,
    snapshot_root: str | Path | None = None,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    """Copy public telemetry columns for every configured profile into the repo.

    Returns a report dict with per-file actions (``copied`` / ``skipped`` /
    ``missing_source`` / ``error``) and whether the manifest was rewritten.
    """
    root_path = Path(sat_root) if sat_root is not None else Path(
        os.environ.get("ASTROGRAPH_SAT_AGENT_ROOT", DEFAULT_SAT_AGENT_ROOT)
    ).expanduser().resolve()
    map_path = Path(channel_map_catalog_path) if channel_map_catalog_path is not None else (
        ROOT / "configs" / "semantics" / "sat_agent_platform_channel_maps.s0.v1.json"
    )
    snapshot_path = Path(snapshot_root) if snapshot_root is not None else SNAPSHOT_DIR
    traces_path = snapshot_path / "traces"
    manifest_path = snapshot_path / "manifest.json"
    stamp = snapshot_date or date.today().isoformat()

    file_map = dict(DEFAULT_SOURCE_FILES if source_files is None else source_files)
    channel_fields = _load_channel_map_source_fields(map_path)
    existing = _load_existing_manifest(manifest_path)
    existing_files = {f["profile_id"]: f for f in existing.get("files", [])}

    traces_path.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    for profile_id in PROFILE_CHANNEL_MAPS:
        rel = file_map.get(profile_id)
        if rel is None:
            files.append({"profile_id": profile_id, "action": "missing_source", "reason": "no source path configured"})
            continue

        mapping_id = PROFILE_CHANNEL_MAPS[profile_id]
        source_path = root_path / rel
        local_path = traces_path / f"{profile_id}.csv"
        local_rel = f"traces/{profile_id}.csv"

        entry: dict[str, Any] = {
            "profile_id": profile_id,
            "source_path": rel.replace("\\", "/"),
            "local_path": local_rel,
            "mapping_id": mapping_id,
        }
        if not source_path.is_file():
            entry["action"] = "missing_source"
            entry["reason"] = f"source not a regular file: {source_path}"
            files.append(entry)
            continue

        mapped_fields = channel_fields.get(mapping_id, set())
        if not mapped_fields:
            entry["action"] = "error"
            entry["reason"] = f"no channel-map entries for {mapping_id}"
            files.append(entry)
            continue

        try:
            frame = pd.read_csv(source_path, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            entry["action"] = "error"
            entry["reason"] = f"read failed: {type(exc).__name__}: {exc}"
            files.append(entry)
            continue

        if "time_s" not in frame.columns:
            entry["action"] = "error"
            entry["reason"] = "source has no time_s column"
            files.append(entry)
            continue

        select = ["time_s", *sorted(f for f in mapped_fields if f in frame.columns)]
        forbidden_hit = [c for c in select if _is_forbidden_column(c)]
        if forbidden_hit:
            entry["action"] = "error"
            entry["reason"] = f"bound field would leak forbidden column: {forbidden_hit}"
            files.append(entry)
            continue

        selected = frame[select]
        row_count = int(len(selected))
        time_uniform = _is_uniform_frame_counter(selected["time_s"])

        # Deterministic serialization so idempotency is byte-identical.
        csv_bytes = selected.to_csv(
            index=False, encoding="utf-8", lineterminator="\n"
        ).encode("utf-8")
        local_sha = _sha256_bytes(csv_bytes)
        source_sha = _sha256_file(source_path)

        entry["sha256"] = local_sha
        entry["source_sha256"] = source_sha
        entry["row_count"] = row_count
        entry["time_axis_uniform_frame_counter"] = time_uniform

        prior = existing_files.get(profile_id)
        unchanged = (
            prior is not None
            and prior.get("source_sha256") == source_sha
            and prior.get("sha256") == local_sha
            and local_path.is_file()
            and _sha256_file(local_path) == local_sha
        )
        if unchanged:
            entry["action"] = "skipped"
        else:
            local_path.write_bytes(csv_bytes)
            entry["action"] = "copied"
        files.append(entry)

    counts = {
        "requested": len(PROFILE_CHANNEL_MAPS),
        "copied": sum(1 for f in files if f["action"] == "copied"),
        "skipped": sum(1 for f in files if f["action"] == "skipped"),
        "missing_source": sum(1 for f in files if f["action"] == "missing_source"),
        "errored": sum(1 for f in files if f["action"] == "error"),
    }

    files_present = sum(
        1 for f in files if f["action"] in ("copied", "skipped") and (snapshot_path / f.get("local_path", "")).is_file()
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "source_count": SOURCE_COUNT,
        "snapshot_date": stamp,
        "snapshot_state": "complete",
        "files_present": files_present,
        "files": files,
        "preserved_quality_facts": PRESERVED_QUALITY_FACTS,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": stamp,
        "manifest_path": str(manifest_path),
        "traces_dir": str(traces_path),
        "counts": counts,
        "files": files,
        "manifest_written": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sat-root",
        default=None,
        help="Sat-Agent platform root (read-only). Defaults to $ASTROGRAPH_SAT_AGENT_ROOT or the canonical dev path.",
    )
    parser.add_argument(
        "--snapshot-root",
        default=None,
        help="Output snapshot directory. Defaults to data/sat_agent_snapshots.",
    )
    parser.add_argument(
        "--channel-maps",
        default=None,
        help="Channel-map catalog JSON. Defaults to configs/semantics/sat_agent_platform_channel_maps.s0.v1.json.",
    )
    parser.add_argument(
        "--file-map",
        default=None,
        help="Optional JSON file mapping profile_id -> relative source path (overrides defaults).",
    )
    args = parser.parse_args(argv)

    source_files: dict[str, str] | None = None
    if args.file_map:
        source_files = json.loads(Path(args.file_map).read_text(encoding="utf-8"))

    report = snapshot(
        sat_root=args.sat_root,
        source_files=source_files,
        channel_map_catalog_path=args.channel_maps,
        snapshot_root=args.snapshot_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["counts"]["errored"] or report["counts"]["missing_source"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
