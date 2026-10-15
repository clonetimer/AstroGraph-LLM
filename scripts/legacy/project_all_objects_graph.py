#!/usr/bin/env python3
"""Project all loop-13 long-trace objects into the V3 five-layer knowledge graph.

Loop 13 work item ④.  For every object in ``data/generated_traces/`` that has a
dataset-layer semantic binding (9 objects / 7 archetypes), this script:

1. builds a label-free ``SemanticTelemetryFrame`` from the source CSV;
2. builds the public ``EpisodeManifest`` through ``episode_builder``;
3. runs the zero-fit feature operator family (``TokenBuilder``) into Evidence
   tokens;
4. runs ``GrayBoxStateEstimator`` when the archetype has a dynamics contract
   (battery / reaction_wheel / thermal / eps), projecting
   StateEstimate + Residual + ConstraintViolation evidence; archetypes without a
   contract (queue / converter / sensor) and estimation failures are recorded
   honestly as fail-closed (never fabricated);
5. projects the episode + evidence into a ``KnowledgeGraphV3`` with ONLY
   abstract nodes (AssetArchetype / QuantityKind / PortRole etc.) — raw
   telemetry arrays never enter the graph (DEV_GOAL rule ③);
6. validates each graph with ``GraphShapeValidator``;
7. writes JSON graphs + a CSV/JSON summary to ``data/graph_projection/`` and,
   when explicitly configured through ``ASTROGRAPH_NEO4J_*`` and reachable,
   writes the graphs there, falling back to CSV/JSON otherwise.

S0: zero-fit — no training, no calibration-dependent fitting in the evidence
path (the estimator runs uncalibrated; only GLS linear identification happens).

Run with the AstroGraph interpreter only::

    /d/Project/test_lunwen/.venv/Scripts/python.exe scripts/project_all_objects_graph.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.knowledge.projection import project_episode_to_graph
from astrograph.knowledge.schema_v3 import GraphShapeValidator, KnowledgeGraphV3
from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.state_estimation import GrayBoxStateEstimator
from astrograph.windowing.episode_builder import build_episodes_from_semantic_frames
from astrograph.features.generic.token_builder import TokenBuilder

from materialize_long_trace_episodes import _CAPABILITY_SEMANTICS

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "generated_traces"
MANIFEST_PATH = TRACES_DIR / "manifest.json"
OUT_DIR = ROOT / "data" / "graph_projection"
FEATURE_CONTRACT_PATH = ROOT / "configs" / "semantics" / "sat_agent_archetype_feature_contracts.s0.v1.json"
SCHEMA_PATH = ROOT / "configs" / "semantics" / "core_semantic_schema.json"
SOURCE_ID = "astrograph:loop13-all-object-projection"

# One nominal trace per object (deterministic pick; the nominal is the first
# non-fault/degradation trace for the capability in the manifest).
NOMINAL_PICKS = {
    "battery": "battery-cycled-nominal-long.csv",
    "reaction_wheel": "reaction_wheel-cycled-nominal-long.csv",
    "thermal": "thermal-cycled-nominal-long.csv",
    "imu": "imu-nominal-long.csv",
    "mtb": "mtb-nominal-long.csv",
    "data_queue": "data_queue-nominal-long.csv",
    "onboard_storage": "onboard_storage-nominal-long.csv",
    "solar_panel": "solar_panel-nominal-long.csv",
    "eps": "eps-nominal-long.csv",
}

# Estimation frame channels per archetype.  The gray-box contracts expect
# EXACTLY ONE signal per port role, while the materialised dataset-layer frames
# legitimately carry several (e.g. bus + battery temperature).  A reduced
# single-signal-per-role frame is built for estimation only; the ABox projection
# still uses the full materialised frame.
_ESTIMATION_CHANNELS: dict[str, list[tuple[str, str, str, str, Any, int]]] = {
    "storage.energy_buffer": [
        ("soc", "state.energy_fraction", "energy.state_of_charge", "1", "eps.battery.soc", 1),
        ("power", "power.exchange", "electrical.active_power", "W", "eps.battery.net_power_w", 1),
    ],
    "actuator.spinning_rotor": [
        ("rate", "state.rate", "mechanical.angular_rate", "rad/s", "adcs.reaction_wheel.speed_rad_s_0", 1),
        ("command", "command.effort", "mechanical.torque", "N*m", "adcs.reaction_wheel.command_torque_nm_0", 1),
    ],
    "thermal.lumped_capacity": [
        ("temp", "state.temperature", "thermal.temperature", "K", "thermal.node.bus_temp_c", 1),
        ("heat", "heat.exchange", "electrical.active_power", "W", "__combined_heat__", 1),
        ("ambient", "context.ambient_temperature", "thermal.temperature", "K", "__sink__", 1),
    ],
}


def _estimation_frame(archetype: str, rows: list[dict[str, str]]) -> SemanticTelemetryFrame:
    """Build a single-signal-per-role frame for gray-box estimation."""
    spec = _ESTIMATION_CHANNELS[archetype]
    timestamps = [float(row["time_s"]) for row in rows]
    signals: list[SemanticSignal] = []
    for index, (name, port_role, quantity_kind, unit, source, direction) in enumerate(spec):
        if source == "__combined_heat__":
            values = [
                float(row["thermal.heat.internal_power_w"])
                + float(row["thermal.heat.solar_input_w"])
                + float(row["thermal.heater.power_w"])
                for row in rows
            ]
        elif source == "__sink__":
            values = [-35.0 + 273.15] * len(rows)
        else:
            values = [float(row[source]) for row in rows]
            if archetype == "thermal.lumped_capacity" and source == "thermal.node.bus_temp_c":
                values = [value + 273.15 for value in values]
        signals.append(
            SemanticSignal(
                semantic_channel_id=f"est.{archetype}.{name}",
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
        frame_id=f"loop13-est-{archetype}",
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
        claim_boundary="loop-13 all-object graph projection (estimation frame)",
    )


def _frame(component_id: str, capability: str, archetype: str, rows: list[dict[str, str]]) -> SemanticTelemetryFrame:
    timestamps = [float(row["time_s"]) for row in rows]
    signals: list[SemanticSignal] = []
    spec = _CAPABILITY_SEMANTICS[capability]
    for column, port_role, quantity_kind, unit, convert, direction in spec["channels"]:
        values = [convert(row[column]) for row in rows]
        signals.append(
            SemanticSignal(
                semantic_channel_id=f"asset.{component_id}.{column}",
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
        frame_id=f"loop13-{component_id}",
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
        claim_boundary="loop-13 all-object graph projection (evidence)",
    )


def _quantity_kinds(frame: SemanticTelemetryFrame) -> dict[str, dict]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    kinds = {
        q["quantity_kind_id"]: {
            "quantity_kind_id": q["quantity_kind_id"],
            "canonical_unit": q["canonical_unit"],
            "dimension": q["dimension"],
        }
        for q in schema["quantity_kinds"]
    }
    out: dict[str, dict] = {}
    for signal in frame.signals:
        qk = kinds.get(signal.quantity_kind_id)
        if qk is not None:
            out[signal.semantic_channel_id] = qk
    return out


def _build_episode(component_id: str, frame: SemanticTelemetryFrame, source_path: str):
    catalog = build_episodes_from_semantic_frames(
        frames={component_id: frame},
        source_file_map={component_id: source_path},
        catalog_id=f"astrograph:loop13-projection:{component_id}",
        source_id="SAT_AGENT_PLATFORM",
        channel_map_sha256=None,
        semantic_schema_id="astrograph.core.physical-quantities.v1",
    )
    return catalog.episodes[0]


def _estimate(archetype: str, rows: list[dict[str, str]]):
    contracts = default_dynamics_contracts()
    contract = contracts.get(archetype)
    if contract is None:
        return None, contract, "NO_DYNAMICS_CONTRACT"
    if archetype not in _ESTIMATION_CHANNELS:
        return None, contract, "NO_ESTIMATION_CHANNELS"
    estimator = GrayBoxStateEstimator(use_frozen_calibration=False)
    try:
        est_frame = _estimation_frame(archetype, rows)
        result = estimator.estimate(est_frame, contract)
        return result, contract, None
    except Exception as exc:  # noqa: BLE001 - honest fail-closed
        return None, contract, str(exc)


def _tokens(archetype: str, frame: SemanticTelemetryFrame):
    builder = TokenBuilder(archetype=archetype, contract_path=str(FEATURE_CONTRACT_PATH))
    return builder.build(frame, component_archetype_id=archetype).tokens


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    global capability_by_instance
    capability_by_instance = {
        spec["component_instance_id"]: capability
        for capability, spec in _CAPABILITY_SEMANTICS.items()
    }

    records: list[dict[str, Any]] = []
    graphs: list[KnowledgeGraphV3] = []
    neo4j_ok = False
    writer = None
    try:
        from astrograph.knowledge.neo4j_projection import Neo4jGraphWriter
        writer = Neo4jGraphWriter()
        writer.verify_connectivity()
        neo4j_ok = True
    except Exception as exc:  # noqa: BLE001 - fall back to CSV/JSON
        print(f"Neo4j unavailable ({type(exc).__name__}): {str(exc)[:100]} — using CSV/JSON only")

    for component_id, filename in NOMINAL_PICKS.items():
        capability = capability_by_instance[component_id]
        spec = _CAPABILITY_SEMANTICS[capability]
        archetype = spec["archetype"]
        source_path = TRACES_DIR / filename
        with source_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        frame = _frame(component_id, capability, archetype, rows)
        episode = _build_episode(component_id, frame, str(source_path))
        qks = _quantity_kinds(frame)

        tokens: list = []
        try:
            tokens = _tokens(archetype, frame)
        except Exception as exc:  # noqa: BLE001
            tokens = []

        state_estimate, contract, estimate_error = _estimate(archetype, rows)
        estimate_status = (
            "OK"
            if state_estimate is not None
            else (estimate_error if estimate_error else "NO_ESTIMATE")
        )
        graph = project_episode_to_graph(
            episode,
            source_id=SOURCE_ID,
            tokens=tokens,
            state_estimate=state_estimate,
            quantity_kinds=qks,
            contract=contract,
        )
        report = GraphShapeValidator().validate(graph)
        graphs.append(graph)

        record = {
            "component_instance_id": component_id,
            "capability": capability,
            "archetype": archetype,
            "filename": filename,
            "rows": len(rows),
            "token_count": len(tokens),
            "estimate_status": estimate_status,
            "estimate_error": estimate_error,
            "parameter_count": len(getattr(state_estimate, "parameters", [])),
            "residual_count": len(getattr(state_estimate, "residuals", [])),
            "constraint_violation_count": (
                len(getattr(state_estimate, "constraint_violations", []))
                if state_estimate is not None else 0
            ),
            "ood_score": getattr(state_estimate, "ood_score", None),
            "shape_status": report.status,
            "shape_issues": [i.model_dump() for i in report.issues],
            "node_count": report.node_count,
            "edge_count": report.edge_count,
            "graph_id": graph.graph_id,
        }
        records.append(record)

        # JSON graph artifact (abstract nodes only).
        (OUT_DIR / f"graph_{component_id}.json").write_text(
            graph.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if writer is not None:
            writer.write_graph(graph)

        print(
            f"{record['shape_status']:4s} {component_id:15s} archetype={archetype:26s} "
            f"tokens={record['token_count']:3d} estimate={record['estimate_status']:20s} "
            f"params={record['parameter_count']} resid={record['residual_count']} "
            f"viol={record['constraint_violation_count']} nodes={report.node_count} edges={report.edge_count}"
        )

    if writer is not None:
        writer.close()

    summary = {
        "schema_version": "astrograph-loop13-graph-projection.v1",
        "source_id": SOURCE_ID,
        "projection_dir": str(OUT_DIR),
        "neo4j_written": neo4j_ok,
        "object_count": len(records),
        "all_shape_pass": all(r["shape_status"] == "PASS" for r in records),
        "records": records,
    }
    (OUT_DIR / "projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # CSV summary
    import csv as _csv

    csv_path = OUT_DIR / "projection_summary.csv"
    columns = [
        "component_instance_id", "capability", "archetype", "filename", "rows",
        "token_count", "estimate_status", "estimate_error", "parameter_count",
        "residual_count", "constraint_violation_count", "ood_score",
        "shape_status", "node_count", "edge_count", "graph_id",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = _csv.DictWriter(handle, fieldnames=columns)
        writer_csv.writeheader()
        for record in records:
            writer_csv.writerow({key: record.get(key) for key in columns})

    passed = sum(1 for r in records if r["shape_status"] == "PASS")
    print(f"\nprojection summary: {passed}/{len(records)} graphs PASS shape validation")
    print(f"neo4j: {'written' if neo4j_ok else 'fallback CSV/JSON'}")
    print(f"artifacts: {OUT_DIR}")
    return 0 if passed == len(records) else 2


capability_by_instance: dict[str, str] = {}


if __name__ == "__main__":
    raise SystemExit(main())
