#!/usr/bin/env python3
"""Materialize the generated long traces into a labeled, groupable, replayable
episode dataset (loop-12 work item ①).

The generator produces up to 25 long traces in ``data/generated_traces/``
(13 original + 9 excitation-injected cycled + 3 additional cycled fault
variants added in loop 12 for experiments B/C), each carrying ``label.*``
columns.  This script materialises them into (``EpisodeManifest``,
``LabeledEpisode``) pairs with the same label-isolation discipline as the
loop-9 ``materialize_labeled_episodes.py``:

* The public ``EpisodeManifest`` is built from the *same* source trace through
  the semantic-frame path, selecting ONLY ``time_s`` + physical signal columns
  (never ``label.*`` / provenance columns).  ``labels_used`` / ``truth_used``
  stay ``False`` and frozen.  ``source_file_references`` are neutralised to
  ``source:<component_instance_id>`` so no scenario/fault token leaks.
* Labels (health state, fault/degradation flags, scenario family + mechanism)
  are extracted through the independent label adapter into a separate
  ``LabeledEpisode`` record linked by ``episode_id`` — NOT a model input.

S0 discipline: no fitting, no calibration, no split/seed.  Materialisation only.

Dataset-layer semantic binding (does NOT touch the frozen channel-map configs):

* battery  -> ``storage.energy_buffer``  (soc / storage_wh / net_power_w)
* IMU      -> ``sensor.measurement``     (gyro x/y/z)
* RW       -> ``actuator.spinning_rotor`` (speed_rad_s_0 / command / applied)
             -- the ``spinning_rotor`` archetype is registered in the core
             semantic catalog (loop-14 P2-A fix).  The production P018 snapshot
             channel map stays ``subsystem.adcs-rw.v2`` ->
             ``actuator.controlled_effort``: a second RW-spin map on the same RW
             profile fields would create a single-profile-per-mapping ambiguity
             (the EPS-basic lesson), so the decision is recorded, not forced.
             This dataset layer is the estimator-facing spinning_rotor binding.
* thermal  -> ``thermal.lumped_capacity`` (bus/battery temp + internal/solar/
             heater power).  Only physical signals are materialised; the
             single-node-vs-dual-node model-alignment gap is NOT judged here.

Output: ``data/labeled_episodes/longtraces_v1/`` (v2 dataset is written to
    ``data/labeled_episodes/longtraces_v2/`` and carries its OWN catalog id
    ``astrograph-longtrace-labeled-episodes-v2`` so tools that de-duplicate by
    catalog_id never merge it with the loop-12 v1 dataset — loop-14 P2-B fix).
    v3 dataset (``--dataset v3``) is the loop-17 expanded corpus (same 9-object
    grouping, more severity x onset scenarios) and carries catalog id
    ``astrograph-longtrace-labeled-episodes-v3``.
    catalog.json             # LabeledEpisodeCatalog (dataset index)
    manifests/<id>.json      # EpisodeManifest (label-free)
    labels/<id>.json         # LabeledEpisode (labels + grouping)

Run with the AstroGraph interpreter only::

    /d/Project/test_lunwen/.venv/Scripts/python.exe scripts/materialize_long_trace_episodes.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.windowing.episode_builder import build_episodes_from_semantic_frames
from astrograph.windowing.label_adapter import (
    LabeledEpisodeCatalog,
    build_labeled_episode,
)

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "generated_traces"
MANIFEST_PATH = TRACES_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "labeled_episodes" / "longtraces_v1"
# Loop 13: all-object materialization (40 traces, 9 capabilities) is written to
# longtraces_v2 so the loop-12 v1 dataset stays byte-identical.
DEFAULT_OUTPUT_DIR_V2 = ROOT / "data" / "labeled_episodes" / "longtraces_v2"
# Loop 17: expanded corpus (72 traces, same 9-object grouping, severity x onset
# variants) is written to longtraces_v3 with its own catalog id so catalog_id
# de-duplication never merges it with v1/v2.
DEFAULT_OUTPUT_DIR_V3 = ROOT / "data" / "labeled_episodes" / "longtraces_v3"

# channel spec: (source_column, port_role, quantity_kind, canonical_unit, convert, direction)
ChannelSpec = tuple[str, str, str, str, Callable[[str], float], int]


def _ch(source: str, port: str, kind: str, unit: str, direction: int = 1) -> tuple[str, str, str, str, Callable[[str], float], int]:
    """Build a channel spec with the default identity converter."""
    return (source, port, kind, unit, lambda v: float(v), direction)


def _ch_k(source: str, port: str, kind: str, unit: str, offset: float, direction: int = 1) -> tuple[str, str, str, str, Callable[[str], float], int]:
    """Build a channel spec with a unit offset converter (e.g. C -> K)."""
    return (source, port, kind, unit, lambda v: float(v) + offset, direction)


# Per-capability dataset-layer semantic binding.
_CAPABILITY_SEMANTICS: dict[str, dict[str, Any]] = {
    "component.battery.v1": {
        "archetype": "storage.energy_buffer",
        "component_instance_id": "battery",
        "channels": [
            _ch("eps.battery.soc", "state.energy_fraction", "energy.state_of_charge", "1"),
            _ch("eps.battery.storage_wh", "state.energy_reserve", "energy.stored_energy", "Wh"),
            _ch("eps.battery.net_power_w", "power.exchange", "electrical.active_power", "W"),
        ],
    },
    "component.imu.v1": {
        "archetype": "sensor.measurement",
        "component_instance_id": "imu",
        "channels": [
            _ch("adcs.imu.gyro_x_rad_s", "measurement.output", "mechanical.angular_rate", "rad/s"),
            _ch("adcs.imu.gyro_y_rad_s", "measurement.output", "mechanical.angular_rate", "rad/s"),
            _ch("adcs.imu.gyro_z_rad_s", "measurement.output", "mechanical.angular_rate", "rad/s"),
        ],
    },
    "component.reaction_wheel.v1": {
        "archetype": "actuator.spinning_rotor",
        "component_instance_id": "reaction_wheel",
        "channels": [
            _ch("adcs.reaction_wheel.speed_rad_s_0", "state.rate", "mechanical.angular_rate", "rad/s"),
            _ch("adcs.reaction_wheel.command_torque_nm_0", "command.effort", "mechanical.torque", "N*m"),
            _ch("adcs.reaction_wheel.applied_torque_nm_0", "response.effort", "mechanical.torque", "N*m"),
        ],
    },
    "subsystem.thermal.basic_lumped.v1": {
        "archetype": "thermal.lumped_capacity",
        "component_instance_id": "thermal",
        "channels": [
            _ch_k("thermal.node.bus_temp_c", "state.temperature", "thermal.temperature", "K", 273.15),
            _ch_k("thermal.node.battery_temp_c", "state.temperature", "thermal.temperature", "K", 273.15),
            _ch("thermal.heat.internal_power_w", "heat.exchange", "electrical.active_power", "W"),
            _ch("thermal.heat.solar_input_w", "heat.exchange", "electrical.active_power", "W"),
            _ch("thermal.heater.power_w", "heat.exchange", "electrical.active_power", "W"),
        ],
    },
    # ---- Loop 13 objects ----
    "component.mtb.v1": {
        "archetype": "actuator.controlled_effort",
        "component_instance_id": "mtb",
        "channels": [
            _ch("adcs.mtb.command_dipole_am2_0", "command.magnetic_dipole", "magnetic.dipole_moment", "A*m^2"),
            _ch("adcs.mtb.actual_dipole_am2_0", "response.magnetic_dipole", "magnetic.dipole_moment", "A*m^2"),
            _ch("adcs.mtb.torque_norm_nm", "response.effort", "mechanical.torque", "N*m"),
            _ch("adcs.mtb.magnetic_field_t_0", "context.magnetic_field", "magnetic.field", "T"),
        ],
    },
    "component.data_queue.v1": {
        "archetype": "queue.bounded_inventory",
        "component_instance_id": "data_queue",
        "channels": [
            _ch("comm.data_queue.queue_bits", "state.occupied_amount", "information.data_amount", "bit"),
            _ch("comm.data_queue.fill_ratio", "state.fill_fraction", "inventory.fill_fraction", "1"),
            _ch("comm.data_queue.generated_bps", "flow.exchange", "information.data_rate", "bit/s", direction=1),
            _ch("comm.data_queue.downlink_bps", "flow.exchange", "information.data_rate", "bit/s", direction=-1),
        ],
    },
    "component.onboard_storage.v1": {
        "archetype": "queue.bounded_inventory",
        "component_instance_id": "onboard_storage",
        "channels": [
            _ch("comm.storage.stored_bits", "state.occupied_amount", "information.data_amount", "bit"),
            _ch("comm.storage.fill_ratio", "state.fill_fraction", "inventory.fill_fraction", "1"),
            _ch("comm.storage.generated_bps", "flow.exchange", "information.data_rate", "bit/s", direction=1),
            _ch("comm.storage.downlink_bps", "flow.exchange", "information.data_rate", "bit/s", direction=-1),
        ],
    },
    "component.solar_panel.v1": {
        "archetype": "converter.power",
        "component_instance_id": "solar_panel",
        "channels": [
            _ch("eps.solar.array_power_w", "power.output", "electrical.active_power", "W"),
            _ch("eps.solar.efficiency", "performance.efficiency", "energy.conversion_efficiency", "1"),
        ],
    },
    "subsystem.eps.basic.v1": {
        "archetype": "storage.energy_buffer",
        "component_instance_id": "eps",
        "channels": [
            _ch("eps.battery.soc", "state.energy_fraction", "energy.state_of_charge", "1"),
            _ch("eps.battery.storage_wh", "state.energy_reserve", "energy.stored_energy", "Wh"),
            _ch("eps.battery.net_power_w", "power.exchange", "electrical.active_power", "W"),
            _ch("eps.battery.applied_power_w", "power.exchange", "electrical.active_power", "W"),
        ],
    },
}

def build_frame(
    *,
    component_instance_id: str,
    archetype: str,
    rows: list[dict[str, str]],
    channels: list[ChannelSpec],
) -> SemanticTelemetryFrame:
    """Build a label-free SemanticTelemetryFrame from trace rows."""
    timestamps = [float(row["time_s"]) for row in rows]
    signals: list[SemanticSignal] = []
    for index, (column, port_role, quantity_kind, unit, convert, direction) in enumerate(channels):
        values = [convert(row[column]) for row in rows]
        signals.append(
            SemanticSignal(
                semantic_channel_id=f"asset.{component_instance_id}.{column}",
                asset_instance_id="asset-under-test",
                component_archetype_id=archetype,
                port_role_id=port_role,
                quantity_kind_id=quantity_kind,
                canonical_unit=unit,
                direction=direction,
                values=values,
                missing_mask=[False] * len(values),
            )
        )
    payload = {
        "timestamps": timestamps,
        "signals": [item.model_dump(mode="json") for item in signals],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    nominal_period = float(sum(diffs) / len(diffs)) if diffs else 0.0
    return SemanticTelemetryFrame(
        frame_id=f"longtrace-{component_instance_id}",
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=timestamps,
        signals=signals,
        sampling=SamplingProfile(
            row_count=len(timestamps),
            nominal_period_s=nominal_period,
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256=digest,
        labels_used=False,
        truth_used=False,
        claim_boundary="long-trace labeled-episode materialization",
    )


def materialize_long_traces(
    *,
    manifest_path: str | Path,
    traces_dir: str | Path,
    output_dir: str | Path,
    catalog_id: str = "astrograph-longtrace-labeled-episodes-v1",
    source_id: str = "SAT_AGENT_PLATFORM",
) -> LabeledEpisodeCatalog:
    """Materialize every trace in the generator manifest into (manifest, label)
    pairs under ``output_dir``.  Deterministic: sorted iteration, opaque hashed
    episode ids, stable JSON field order, no timestamps."""
    manifest_path = Path(manifest_path)
    traces_dir = Path(traces_dir)
    out = Path(output_dir)
    manifests_dir = out / "manifests"
    labels_dir = out / "labels"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = []

    for trace in sorted(payload["traces"], key=lambda item: item["file"]):
        capability = trace["capability"]
        spec = _CAPABILITY_SEMANTICS.get(capability)
        if spec is None:
            continue  # a capability with no dataset-layer semantic binding
        component_id = spec["component_instance_id"]
        source_path = Path(traces_dir) / trace["file"]
        if not source_path.is_file():
            raise FileNotFoundError(f"long-trace source missing: {source_path}")

        with source_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        frame = build_frame(
            component_instance_id=component_id,
            archetype=spec["archetype"],
            rows=rows,
            channels=spec["channels"],
        )
        catalog = build_episodes_from_semantic_frames(
            frames={component_id: frame},
            source_file_map={component_id: str(source_path)},
            catalog_id=f"{catalog_id}:{capability}",
            source_id=source_id,
            channel_map_sha256=None,
            semantic_schema_id="astrograph.core.physical-quantities.v1",
        )
        manifest = catalog.episodes[0]

        scenario_family = trace["mode"]
        scenario_id = trace["scenario"]
        # case_id is unique per trace (e.g. BATT-CYC-DEG-CL30 vs BATT-DEG-CL30),
        # which prevents id collisions between cycled and non-cycled variants of
        # the same object+scenario+time-span.  The emitted id is an opaque hash
        # — no label/case token is embedded.
        episode_id = hashlib.sha256(
            f"{component_id}|{trace['case_id']}".encode("utf-8")
        ).hexdigest()[:32]
        # Neutralise the manifest source reference: the trace filename carries
        # scenario/fault tokens that deny rules keep out of any model-input
        # surface.  The label record keeps the honest path instead.
        manifest = manifest.model_copy(
            update={
                "episode_id": episode_id,
                "source_file_references": {component_id: f"source:{component_id}"},
            }
        )

        labeled = build_labeled_episode(
            manifest=manifest,
            trace=pd.DataFrame(rows),
            scenario_family=scenario_family,
            scenario_id=scenario_id,
            label_source_ref=f"data/generated_traces/{trace['file']}",
            config_id=capability,
            seed=None,  # deterministic adapters; no simulation seed exists
        )
        episodes.append(labeled)

        (manifests_dir / f"{episode_id}.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (labels_dir / f"{episode_id}.json").write_text(
            labeled.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    catalog = LabeledEpisodeCatalog(
        catalog_id=catalog_id,
        source_id=source_id,
        episodes=episodes,
    )
    (out / "catalog.json").write_text(
        catalog.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return catalog


def _stats(catalog: LabeledEpisodeCatalog) -> dict[str, Any]:
    by_scenario: dict[str, int] = {}
    by_archetype: dict[str, list[str]] = {}
    for ep in catalog.episodes:
        fam = ep.grouping.scenario_family
        by_scenario[fam] = by_scenario.get(fam, 0) + 1
        by_archetype.setdefault(ep.grouping.component_archetype_id, []).append(ep.episode_id)
    return {
        "episode_count": len(catalog.episodes),
        "object_count": len({ep.grouping.component_instance_id for ep in catalog.episodes}),
        "by_scenario_family": by_scenario,
        "by_archetype": {k: len(v) for k, v in by_archetype.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    parser.add_argument("--output-dir", default=None, help="explicit output dir (overrides --dataset)")
    parser.add_argument(
        "--dataset",
        choices=("v1", "v2", "v3"),
        default="v1",
        help="v1 = loop-12 25-trace dataset; v2 = loop-13 all-object 40-trace dataset; v3 = loop-17 expanded 72-trace dataset",
    )
    args = parser.parse_args(argv)

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        catalog_id = "astrograph-longtrace-labeled-episodes-v1"
    elif args.dataset == "v3":
        out_dir = DEFAULT_OUTPUT_DIR_V3
        # Loop-17 expanded corpus: its own catalog so catalog_id de-dup never
        # merges it with v1/v2; same 9-object grouping as v2.
        catalog_id = "astrograph-longtrace-labeled-episodes-v3"
    elif args.dataset == "v2":
        out_dir = DEFAULT_OUTPUT_DIR_V2
        # P2-B: the all-object 40-trace dataset is its OWN catalog, not a re-run
        # of the loop-12 25-trace v1 dataset.  A distinct catalog_id prevents
        # catalog_id-keyed de-duplication from merging v1 and v2.
        catalog_id = "astrograph-longtrace-labeled-episodes-v2"
    else:
        out_dir = DEFAULT_OUTPUT_DIR
        catalog_id = "astrograph-longtrace-labeled-episodes-v1"

    catalog = materialize_long_traces(
        manifest_path=args.manifest,
        traces_dir=args.traces_dir,
        output_dir=out_dir,
        catalog_id=catalog_id,
    )
    out = Path(out_dir)
    print(json.dumps({"catalog_path": str(out / "catalog.json"), "stats": _stats(catalog)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
