"""Semantic binding tests for the remaining five ADCS component profiles.

Covers CMG + MTB (``actuator.controlled_effort``) and star tracker + sun
sensor + magnetometer (``sensor.measurement``).  For every object the five
mapping classes from ``test_battery_semantic_binding.py`` are reproduced:

1. minimal positive frame mapping;
2. field rename must fail closed (required) / be ignored cleanly (optional);
3. unit conversion preserves canonical semantics;
4. missing required field / time axis must fail closed;
5. prohibited label/truth/event/object fields must never reach a frame.

Real Sat-Agent nominal traces are verified end-to-end when available on disk.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from astrograph.semantics import (
    ChannelMapCatalog,
    ChannelMapEntry,
    ChannelMapper,
    ComponentArchetypeCatalog,
    DEFAULT_UNIT_REGISTRY,
    SemanticMappingError,
    SemanticSchema,
)

ROOT = Path(__file__).parents[1]
SEMANTICS = ROOT / "configs" / "semantics"

# Public-telemetry snapshot (committed, no external Sat-Agent path required).
SNAPSHOT_ROOT = ROOT / "data" / "sat_agent_snapshots" / "traces"
_OBJ_TO_SNAPSHOT = {
    "cmg": "sat-agent.component.cmg.trace.v1.csv",
    "mtb": "sat-agent.component.mtb.trace.v1.csv",
    "star_tracker": "sat-agent.component.star-tracker.trace.v1.csv",
    "sun_sensor": "sat-agent.component.sun-sensor.trace.v1.csv",
    "magnetometer": "sat-agent.component.magnetometer.trace.v1.csv",
}

# source_field -> values for a two-row minimal positive frame
CASES = {
    "cmg": {
        "mapping_id": "sat-agent-platform.component.cmg.v1",
        "fields": {
            "adcs.cmg.wheel_speed_rad_s": [100.0, 100.5],
            "adcs.cmg.gimbal_angle_rad": [0.0, 0.1],
            "adcs.cmg.wheel_momentum_nms": [10.0, 10.05],
        },
    },
    "mtb": {
        "mapping_id": "sat-agent-platform.component.mtb.v1",
        "fields": {
            "adcs.mtb.command_dipole_am2_0": [0.1, 0.1],
            "adcs.mtb.command_dipole_am2_1": [0.0, 0.0],
            "adcs.mtb.command_dipole_am2_2": [0.0, 0.0],
            "adcs.mtb.actual_dipole_am2_0": [0.05, 0.06],
            "adcs.mtb.actual_dipole_am2_1": [0.0, 0.0],
            "adcs.mtb.actual_dipole_am2_2": [0.0, 0.0],
            "adcs.mtb.torque_nm_0": [0.0, 1e-6],
            "adcs.mtb.torque_nm_1": [0.0, 0.0],
            "adcs.mtb.torque_nm_2": [0.0, 0.0],
            "adcs.mtb.torque_norm_nm": [1.05e-6, 1.06e-6],
            "adcs.mtb.magnetic_field_t_0": [2e-5, 2e-5],
            "adcs.mtb.magnetic_field_t_1": [-1e-5, -1e-5],
            "adcs.mtb.magnetic_field_t_2": [3e-5, 3e-5],
        },
    },
    "star_tracker": {
        "mapping_id": "sat-agent-platform.component.star_tracker.v1",
        "fields": {
            "adcs.star_tracker.sigma_x": [0.1, 0.2],
            "adcs.star_tracker.sigma_y": [0.0, 0.0],
            "adcs.star_tracker.sigma_z": [0.0, 0.0],
        },
    },
    "sun_sensor": {
        "mapping_id": "sat-agent-platform.component.sun_sensor.v1",
        "fields": {
            "adcs.sun_sensor.intensity": [1.0, 1.0],
            "adcs.sun_sensor.direction_x": [1.0, 0.9],
            "adcs.sun_sensor.direction_y": [0.0, 0.0],
            "adcs.sun_sensor.direction_z": [0.0, 0.0],
        },
    },
    "magnetometer": {
        "mapping_id": "sat-agent-platform.component.magnetometer.v1",
        "fields": {
            "adcs.magnetometer.field_x_t": [2e-5, 2.1e-5],
            "adcs.magnetometer.field_y_t": [0.0, 0.0],
            "adcs.magnetometer.field_z_t": [4e-5, 4e-5],
            "adcs.magnetometer.field_norm_t": [4.47213595e-5, 4.52213595e-5],
        },
    },
}

# semantic_channel_id -> (quantity_kind_id, canonical_unit) for each object
EXPECTED = {
    "cmg": {
        "asset.adcs.cmg_0.wheel_angular_rate": ("mechanical.angular_rate", "rad/s"),
        "asset.adcs.cmg_0.gimbal_angle": ("mechanical.angular_position", "rad"),
        "asset.adcs.cmg_0.wheel_momentum": ("mechanical.angular_momentum", "N*m*s"),
    },
    "mtb": {
        "asset.adcs.mtb_0.command_dipole_x": ("magnetic.dipole_moment", "A*m^2"),
        "asset.adcs.mtb_0.command_dipole_y": ("magnetic.dipole_moment", "A*m^2"),
        "asset.adcs.mtb_0.command_dipole_z": ("magnetic.dipole_moment", "A*m^2"),
        "asset.adcs.mtb_0.actual_dipole_x": ("magnetic.dipole_moment", "A*m^2"),
        "asset.adcs.mtb_0.actual_dipole_y": ("magnetic.dipole_moment", "A*m^2"),
        "asset.adcs.mtb_0.actual_dipole_z": ("magnetic.dipole_moment", "A*m^2"),
        "asset.adcs.mtb_0.torque_x": ("mechanical.torque", "N*m"),
        "asset.adcs.mtb_0.torque_y": ("mechanical.torque", "N*m"),
        "asset.adcs.mtb_0.torque_z": ("mechanical.torque", "N*m"),
        "asset.adcs.mtb_0.torque_norm": ("mechanical.torque", "N*m"),
        "asset.adcs.mtb_0.field_x": ("magnetic.field", "T"),
        "asset.adcs.mtb_0.field_y": ("magnetic.field", "T"),
        "asset.adcs.mtb_0.field_z": ("magnetic.field", "T"),
    },
    "star_tracker": {
        "asset.adcs.star_tracker_0.attitude_mrp_x": ("attitude.mrp", "MRP"),
        "asset.adcs.star_tracker_0.attitude_mrp_y": ("attitude.mrp", "MRP"),
        "asset.adcs.star_tracker_0.attitude_mrp_z": ("attitude.mrp", "MRP"),
    },
    "sun_sensor": {
        "asset.adcs.sun_sensor_0.intensity": ("optical.sun_intensity", "1"),
        "asset.adcs.sun_sensor_0.direction_x": ("direction.unit_vector_component", "1"),
        "asset.adcs.sun_sensor_0.direction_y": ("direction.unit_vector_component", "1"),
        "asset.adcs.sun_sensor_0.direction_z": ("direction.unit_vector_component", "1"),
    },
    "magnetometer": {
        "asset.adcs.magnetometer_0.field_x": ("magnetic.field", "T"),
        "asset.adcs.magnetometer_0.field_y": ("magnetic.field", "T"),
        "asset.adcs.magnetometer_0.field_z": ("magnetic.field", "T"),
        "asset.adcs.magnetometer_0.field_norm": ("magnetic.field", "T"),
    },
}

# For each object: swap one source field to an equivalent unit and assert the
# canonical semantic values are preserved.
UNIT_CONVERSIONS = {
    "cmg": {
        "target_field": "adcs.cmg.gimbal_angle_rad",
        "new_unit": "deg",
        "values": [180.0, 90.0],
        "channel": "asset.adcs.cmg_0.gimbal_angle",
        "expected": [math.pi, math.pi / 2.0],
    },
    "mtb": {
        "target_field": "adcs.mtb.magnetic_field_t_0",
        "new_unit": "nT",
        "values": [20000.0, 5000.0],
        "channel": "asset.adcs.mtb_0.field_x",
        "expected": [2e-5, 5e-6],
    },
    "star_tracker": {
        "target_field": "adcs.star_tracker.sigma_x",
        "new_unit": "MRP",
        "values": [0.1, 0.2],
        "channel": "asset.adcs.star_tracker_0.attitude_mrp_x",
        "expected": [0.1, 0.2],
    },
    "sun_sensor": {
        "target_field": "adcs.sun_sensor.intensity",
        "new_unit": "1",
        "values": [1.0, 0.9],
        "channel": "asset.adcs.sun_sensor_0.intensity",
        "expected": [1.0, 0.9],
    },
    "magnetometer": {
        "target_field": "adcs.magnetometer.field_x_t",
        "new_unit": "nT",
        "values": [20000.0, 5000.0],
        "channel": "asset.adcs.magnetometer_0.field_x",
        "expected": [2e-5, 5e-6],
    },
}

OBJECT_PARAMS = [pytest.param(name, id=name) for name in CASES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contracts() -> tuple[SemanticSchema, ComponentArchetypeCatalog, ChannelMapCatalog]:
    schema = SemanticSchema.model_validate_json(
        (SEMANTICS / "core_semantic_schema.json").read_text(encoding="utf-8")
    )
    archetypes = ComponentArchetypeCatalog.model_validate_json(
        (SEMANTICS / "core_component_archetypes.json").read_text(encoding="utf-8")
    )
    maps = ChannelMapCatalog.model_validate_json(
        (SEMANTICS / "sat_agent_platform_channel_maps.s0.v1.json").read_text(
            encoding="utf-8"
        )
    )
    return schema, archetypes, maps


def _mapper(mapping_id: str) -> ChannelMapper:
    schema, _, maps = _contracts()
    return ChannelMapper(semantic_schema=schema, channel_map=maps.channel_map(mapping_id))


def _entries(mapping_id: str) -> list:
    _, _, maps = _contracts()
    return list(maps.channel_map(mapping_id).entries)


def _by_channel(result: object) -> dict[str, object]:
    return {signal.semantic_channel_id: signal for signal in result.frame.signals}


def _full_frame(name: str) -> pd.DataFrame:
    fields = CASES[name]["fields"]
    return pd.DataFrame({"time_s": [0.0, 1.0], **{k: list(v) for k, v in fields.items()}})


def _mapper_with_source_unit(mapping_id: str, target_field: str, new_unit: str) -> ChannelMapper:
    schema, _, maps = _contracts()
    channel_map = maps.channel_map(mapping_id)
    payload = [entry.model_dump(mode="json") for entry in channel_map.entries]
    for entry in payload:
        if entry["source_field"] == target_field:
            entry["source_unit"] = new_unit
    modified = channel_map.model_copy(
        update={"entries": [ChannelMapEntry.model_validate(entry) for entry in payload]}
    )
    return ChannelMapper(semantic_schema=schema, channel_map=modified)


# ---------------------------------------------------------------------------
# 1. Minimal positive frames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_minimal_positive_frame_maps_all_declared_channels(name: str) -> None:
    mapping_id = CASES[name]["mapping_id"]
    result = _mapper(mapping_id).map_frame(_full_frame(name))
    signals = _by_channel(result)
    assert set(signals) == set(EXPECTED[name])

    for channel_id, (quantity_kind, canonical_unit) in EXPECTED[name].items():
        signal = signals[channel_id]
        assert signal.quantity_kind_id == quantity_kind
        assert signal.canonical_unit == canonical_unit
        assert signal.values == CASES[name]["fields"][
            next(e.source_field for e in _entries(mapping_id) if e.semantic_channel_id == channel_id)
        ]

    assert result.frame.labels_used is False
    assert result.frame.truth_used is False
    assert result.frame.semantic_data_sha256


@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_required_fields_only_map_to_required_channels(name: str) -> None:
    mapping_id = CASES[name]["mapping_id"]
    required = [e for e in _entries(mapping_id) if e.required]
    frame = pd.DataFrame({"time_s": [0.0, 1.0], **{e.source_field: [1.0, 2.0] for e in required}})
    result = _mapper(mapping_id).map_frame(frame)
    assert {signal.semantic_channel_id for signal in result.frame.signals} == {
        e.semantic_channel_id for e in required
    }


# ---------------------------------------------------------------------------
# 2. Field rename must never silently change meaning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_renamed_required_field_fails_closed(name: str) -> None:
    mapping_id = CASES[name]["mapping_id"]
    required_field = next(e.source_field for e in _entries(mapping_id) if e.required)
    frame = _full_frame(name)
    frame = frame.rename(columns={required_field: f"{required_field}_renamed"})
    with pytest.raises(SemanticMappingError) as exc:
        _mapper(mapping_id).map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert required_field in exc.value.fields


@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_renamed_optional_field_is_ignored_cleanly(name: str) -> None:
    mapping_id = CASES[name]["mapping_id"]
    optional = [e for e in _entries(mapping_id) if not e.required]
    if not optional:
        pytest.skip(f"{name} has no optional source fields")
    entry = optional[0]
    frame = _full_frame(name).rename(columns={entry.source_field: f"{entry.source_field}_renamed"})
    result = _mapper(mapping_id).map_frame(frame)
    assert entry.semantic_channel_id not in {s.semantic_channel_id for s in result.frame.signals}
    assert f"{entry.source_field}_renamed" in result.receipt.ignored_clean_fields


# ---------------------------------------------------------------------------
# 3. Unit conversion preserves canonical semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_unit_conversion_preserves_canonical_semantics(name: str) -> None:
    mapping_id = CASES[name]["mapping_id"]
    spec = UNIT_CONVERSIONS[name]
    fields = dict(CASES[name]["fields"])
    fields[spec["target_field"]] = spec["values"]
    frame = pd.DataFrame({"time_s": [0.0, 1.0], **fields})
    mapper = _mapper_with_source_unit(mapping_id, spec["target_field"], spec["new_unit"])
    result = mapper.map_frame(frame)
    signal = _by_channel(result)[spec["channel"]]
    assert signal.canonical_unit == EXPECTED[name][spec["channel"]][1]
    assert signal.values == pytest.approx(spec["expected"])


def test_new_unit_registry_conversions_are_registered() -> None:
    registry = DEFAULT_UNIT_REGISTRY
    assert registry.convert(5000.0, source="nT", target="T") == pytest.approx(5e-6)
    assert registry.convert(1.0, source="T", target="nT") == pytest.approx(1e9)
    assert registry.convert(2.0, source="N*m*s", target="N*m*s") == 2.0
    assert registry.convert(0.1, source="MRP", target="MRP") == 0.1
    assert registry.convert(0.5, source="A*m^2", target="A*m^2") == 0.5
    assert registry.convert(180.0, source="deg", target="rad") == pytest.approx(math.pi)


def test_new_quantity_kinds_are_declared_with_canonical_units() -> None:
    schema, _, _ = _contracts()
    kinds = {item.quantity_kind_id: item for item in schema.quantity_kinds}
    assert kinds["mechanical.angular_momentum"].canonical_unit == "N*m*s"
    assert kinds["magnetic.field"].canonical_unit == "T"
    assert kinds["magnetic.dipole_moment"].canonical_unit == "A*m^2"
    assert kinds["attitude.mrp"].canonical_unit == "MRP"
    assert kinds["optical.sun_intensity"].canonical_unit == "1"
    assert kinds["direction.unit_vector_component"].canonical_unit == "1"


# ---------------------------------------------------------------------------
# 4. Missing fields must fail closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_missing_required_signal_field_fails_closed(name: str) -> None:
    mapping_id = CASES[name]["mapping_id"]
    required_field = next(e.source_field for e in _entries(mapping_id) if e.required)
    frame = _full_frame(name).drop(columns=[required_field])
    with pytest.raises(SemanticMappingError) as exc:
        _mapper(mapping_id).map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert required_field in exc.value.fields


@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_missing_time_axis_fails_closed(name: str) -> None:
    mapping_id = CASES[name]["mapping_id"]
    frame = _full_frame(name).drop(columns=["time_s"])
    with pytest.raises(SemanticMappingError) as exc:
        _mapper(mapping_id).map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert "time_s" in exc.value.fields


# ---------------------------------------------------------------------------
# 5. Prohibited fields must never reach a semantic frame
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", OBJECT_PARAMS)
@pytest.mark.parametrize(
    "bad_field",
    ["label.health_state", "truth.fault_active", "effect.active", "event.onset", "object.case"],
)
def test_prohibited_field_is_rejected(name: str, bad_field: str) -> None:
    frame = _full_frame(name)
    frame[bad_field] = ["x", "y"]
    with pytest.raises(SemanticMappingError) as exc:
        _mapper(CASES[name]["mapping_id"]).map_frame(frame)
    assert exc.value.code == "PROHIBITED_SOURCE_FIELD"
    assert bad_field in exc.value.fields


# ---------------------------------------------------------------------------
# Real-file verification (skipped when the source trace is unavailable)
# ---------------------------------------------------------------------------

def _find_real_nominal_trace(name: str) -> Path | None:
    obj = {
        "cmg": "cmg",
        "mtb": "mtb",
        "star_tracker": "star_tracker",
        "sun_sensor": "sun_sensor",
        "magnetometer": "magnetometer",
    }[name]
    # Snapshot-backed public trace; no external Sat-Agent tree is required.
    path = SNAPSHOT_ROOT / _OBJ_TO_SNAPSHOT[obj]
    return path if path.is_file() else None


@pytest.mark.parametrize("name", OBJECT_PARAMS)
def test_real_sat_agent_nominal_trace_maps_through_declared_binding(name: str) -> None:
    trace_path = _find_real_nominal_trace(name)
    if trace_path is None:
        pytest.skip(f"real Sat-Agent {name} nominal trace unavailable")
    mapping_id = CASES[name]["mapping_id"]
    mapped_fields = {e.source_field for e in _entries(mapping_id)}
    raw = pd.read_csv(trace_path)
    selected = ["time_s", *[c for c in sorted(mapped_fields) if c in raw.columns]]
    frame = raw[selected].copy()
    result = _mapper(mapping_id).map_frame(frame)

    present = set(mapped_fields) & set(raw.columns)
    assert {s.semantic_channel_id for s in result.frame.signals} == {
        e.semantic_channel_id for e in _entries(mapping_id) if e.source_field in present
    }
    timestamps = result.frame.timestamps_s
    assert timestamps
    assert all(right > left for left, right in zip(timestamps, timestamps[1:]))
    assert result.frame.sampling.row_count == len(raw)
    assert result.frame.labels_used is False
    assert result.frame.truth_used is False
