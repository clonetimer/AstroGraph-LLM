"""Semantic binding tests for the batch-C component profiles.

Batch C binds four more real Sat-Agent component profiles onto two prototype
families that already carry bindings, exercising *same-archetype reuse across
objects*:

* solar panel + power sink -> ``converter.power`` (PDU already bound in Phase I)
* antenna + ground station -> ``link.transport`` (transmitter / link budget
  already bound in batch B)

For every object the five mapping classes from
``test_battery_semantic_binding.py`` are reproduced:

1. minimal positive frame mapping;
2. field rename must fail closed (required) / be ignored cleanly (optional);
3. unit conversion preserves canonical semantics;
4. missing required field / time axis must fail closed;
5. prohibited label/truth/event/object/effective-* fields must never reach a
   frame.

Real Sat-Agent nominal traces are verified end-to-end via the committed public
telemetry snapshots only (no external Sat-Agent path is required).
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
    "solar_panel": "sat-agent.component.solar-panel.trace.v1.csv",
    "power_sink": "sat-agent.component.power-sink.trace.v1.csv",
    "antenna": "sat-agent.component.antenna.trace.v1.csv",
    "ground_station": "sat-agent.component.ground-station.trace.v1.csv",
}

# source_field -> values (in source units) for a two-row minimal positive frame.
CASES = {
    "solar_panel": {
        "mapping_id": "sat-agent-platform.component.solar_panel.v1",
        "fields": {
            "eps.solar.array_power_w": [120.0, 120.0],
            "eps.solar.panel_power_w": [120.0, 120.0],
            "eps.solar.efficiency": [1.0, 0.95],
        },
    },
    "power_sink": {
        "mapping_id": "sat-agent-platform.component.power_sink.v1",
        "fields": {
            "eps.power_sink.demand_w": [10.0, 12.0],
        },
    },
    "antenna": {
        "mapping_id": "sat-agent-platform.component.antenna.v1",
        "fields": {
            "comm.antenna.gain_dbi": [8.0, 8.0],
            "comm.antenna.off_boresight_deg": [0.0, 5.0],
        },
    },
    "ground_station": {
        "mapping_id": "sat-agent-platform.component.ground_station.v1",
        "fields": {
            "ground_station.slant_range_m": [500000.0, 500053.787589],
            "ground_station.elevation_deg": [90.0, 89.1272667347],
        },
    },
}

# semantic_channel_id -> (quantity_kind_id, canonical_unit) for each object.
EXPECTED = {
    "solar_panel": {
        "asset.eps.solar_0.array_power": ("electrical.active_power", "W"),
        "asset.eps.solar_0.panel_power": ("electrical.active_power", "W"),
        "asset.eps.solar_0.conversion_efficiency": ("energy.conversion_efficiency", "1"),
    },
    "power_sink": {
        "asset.eps.power_sink_0.demand_power": ("electrical.active_power", "W"),
    },
    "antenna": {
        "asset.comm.antenna_0.gain": ("communication.antenna_gain", "dBi"),
        "asset.comm.antenna_0.off_boresight_angle": ("angle.angular_deviation", "rad"),
    },
    "ground_station": {
        "asset.comm.ground_station_0.slant_range": ("length.distance", "m"),
        "asset.comm.ground_station_0.elevation_angle": ("angle.angular_deviation", "rad"),
    },
}

# canonical-unit values produced from CASES by the declared bindings.  Angles
# are source degrees converted to canonical radians, so they are spelled out
# explicitly here (independent of the unit registry).
EXPECTED_VALUES = {
    "solar_panel": {
        "asset.eps.solar_0.array_power": [120.0, 120.0],
        "asset.eps.solar_0.panel_power": [120.0, 120.0],
        "asset.eps.solar_0.conversion_efficiency": [1.0, 0.95],
    },
    "power_sink": {
        "asset.eps.power_sink_0.demand_power": [10.0, 12.0],
    },
    "antenna": {
        "asset.comm.antenna_0.gain": [8.0, 8.0],
        "asset.comm.antenna_0.off_boresight_angle": [0.0, 5.0 * math.pi / 180.0],
    },
    "ground_station": {
        "asset.comm.ground_station_0.slant_range": [500000.0, 500053.787589],
        "asset.comm.ground_station_0.elevation_angle": [
            90.0 * math.pi / 180.0,
            89.1272667347 * math.pi / 180.0,
        ],
    },
}

# For each object: swap one source field to an equivalent unit and assert the
# canonical semantic values are preserved.
UNIT_CONVERSIONS = {
    "solar_panel": {
        "target_field": "eps.solar.efficiency",
        "new_unit": "%",
        "values": [100.0, 95.0],
        "channel": "asset.eps.solar_0.conversion_efficiency",
        "expected": [1.0, 0.95],
    },
    "power_sink": {
        "target_field": "eps.power_sink.demand_w",
        "new_unit": "kW",
        "values": [0.01, 0.012],
        "channel": "asset.eps.power_sink_0.demand_power",
        "expected": [10.0, 12.0],
    },
    "antenna": {
        "target_field": "comm.antenna.gain_dbi",
        "new_unit": "dB",
        "values": [8.0, 8.0],
        "channel": "asset.comm.antenna_0.gain",
        "expected": [8.0, 8.0],
    },
    "ground_station": {
        "target_field": "ground_station.slant_range_m",
        "new_unit": "km",
        "values": [500.0, 500.053787589],
        "channel": "asset.comm.ground_station_0.slant_range",
        "expected": [500000.0, 500053.787589],
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
# Batch theme: same archetype reused across objects
# ---------------------------------------------------------------------------

def test_batch_c_reuses_existing_archetypes_across_objects() -> None:
    _, archetypes, maps = _contracts()
    converter = archetypes.archetype("converter.power")
    assert {"power.input", "power.output", "performance.efficiency"} <= {
        role.port_role_id for role in converter.port_roles
    }
    link = archetypes.archetype("link.transport")
    assert {"context.antenna_gain", "context.pointing_angle", "context.elevation_angle", "context.slant_range"} <= {
        role.port_role_id for role in link.port_roles
    }

    by_mapping = {}
    for name in CASES:
        mapping_id = CASES[name]["mapping_id"]
        by_mapping[mapping_id] = {
            e.component_archetype_id for e in _entries(mapping_id)
        }
    assert by_mapping["sat-agent-platform.component.solar_panel.v1"] == {"converter.power"}
    assert by_mapping["sat-agent-platform.component.power_sink.v1"] == {"converter.power"}
    assert by_mapping["sat-agent-platform.component.antenna.v1"] == {"link.transport"}
    assert by_mapping["sat-agent-platform.component.ground_station.v1"] == {"link.transport"}


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
        assert signal.values == pytest.approx(EXPECTED_VALUES[name][channel_id])

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


def test_batch_c_unit_registry_conversions_are_registered() -> None:
    registry = DEFAULT_UNIT_REGISTRY
    assert registry.convert(100.0, source="%", target="1") == pytest.approx(1.0)
    assert registry.convert(0.01, source="kW", target="W") == pytest.approx(10.0)
    assert registry.convert(8.0, source="dBi", target="dB") == pytest.approx(8.0)
    assert registry.convert(500.0, source="km", target="m") == pytest.approx(500000.0)
    assert registry.convert(math.pi / 2.0, source="rad", target="deg") == pytest.approx(90.0)


def test_batch_c_new_quantity_kinds_are_declared_with_canonical_units() -> None:
    schema, _, _ = _contracts()
    kinds = {item.quantity_kind_id: item for item in schema.quantity_kinds}
    assert kinds["communication.antenna_gain"].canonical_unit == "dBi"
    assert kinds["energy.conversion_efficiency"].canonical_unit == "1"


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


def test_no_batch_c_bound_field_leaks_deny_table_terms() -> None:
    """Bound source fields must never carry deny-table tokens."""
    denied = (
        "label", "truth", "effect", "event", "object", "status",
        "task_id", "case_id", "mode", "target_level", "target_name",
        "fault", "bias", "degradation", "drift", "dropout", "injected",
        "effective_", "effective.", "stuck", "available", "enabled",
        "seed", "split", "nominal",
    )
    for name in CASES:
        mapping_id = CASES[name]["mapping_id"]
        for entry in _entries(mapping_id):
            lowered = entry.source_field.lower()
            assert not any(token in lowered for token in denied), (
                f"{name} bound field {entry.source_field} leaks deny-table token"
            )


def test_batch_c_fault_effective_nominal_fields_never_enter_snapshots() -> None:
    """Deny-listed real fields stay physically isolated from the snapshots.

    The real solar-panel trace exports ``eps.solar.fault_multiplier`` and
    ``eps.solar.nominal_*`` columns; none of them may appear in the committed
    public snapshots.
    """
    snapshot_text = "\n".join(
        (SNAPSHOT_ROOT / _OBJ_TO_SNAPSHOT[name]).read_text(encoding="utf-8")
        for name in CASES
    )
    for token in ("fault", "nominal", "effective_", "label.", "enabled", "degradation"):
        assert token not in snapshot_text, f"deny-token leaked into snapshot: {token}"


# ---------------------------------------------------------------------------
# Real-file verification (snapshot-backed only)
# ---------------------------------------------------------------------------

def _find_real_nominal_trace(name: str) -> Path | None:
    path = SNAPSHOT_ROOT / _OBJ_TO_SNAPSHOT[name]
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
