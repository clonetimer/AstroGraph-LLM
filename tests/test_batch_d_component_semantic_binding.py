"""Semantic binding tests for the batch-D component profiles.

Batch D binds three more real Sat-Agent component profiles onto three prototype
families that already carry bindings, exercising *same-archetype reuse across
objects*:

* fuel tank -> ``queue.bounded_inventory`` (onboard storage / data queue
  already bound; extends the queue with mass-kind inventory and a pressure
  state port)
* payload instrument -> ``link.transport`` (transmitter / link budget /
  antenna / ground station already bound; treats the payload as the emitting
  endpoint that turns power into a data flow)
* payload sensor -> ``sensor.measurement`` (IMU / star tracker / sun sensor /
  magnetometer already bound; adds angle, data-amount, cloud-fraction and
  quality-score measurement outputs)

The thruster (propulsion.thruster) is deliberately NOT bound here: its effort
is linear force/impulse and its state is propellant mass, none of which is
expressible by the seven existing archetypes.  Adding a propulsion archetype is
a governance change-request decision for agent 1# (``DEV_GOAL.md``), so batch D
reports the investigation only and binds the three objects that fit existing
archetypes.

For every bound object the five mapping classes from
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
    "fuel_tank": "sat-agent.component.fuel-tank.trace.v1.csv",
    "payload": "sat-agent.component.payload.trace.v1.csv",
    "payload_sensor": "sat-agent.component.payload-sensor.trace.v1.csv",
}

# source_field -> values (in source units) for a two-row minimal positive frame.
# Direction is handled by the entry direction (fuel-tank mass flow is outflow).
CASES = {
    "fuel_tank": {
        "mapping_id": "sat-agent-platform.component.fuel_tank.v1",
        "fields": {
            "propulsion.fuel_tank.propellant_mass_kg": [2.0, 1.999],
            "propulsion.fuel_tank.commanded_mass_flow_kg_s": [0.001, 0.001],
            "propulsion.fuel_tank.pressure_pa": [1716666.6666666665, 1715883.3333333333],
        },
    },
    "payload": {
        "mapping_id": "sat-agent-platform.component.payload.v1",
        "fields": {
            "payload.instrument.power_w": [3.0, 18.0],
            "payload.instrument.heat_w": [2.55, 15.3],
            "payload.instrument.generated_bps": [0.0, 250000.0],
            "payload.instrument.cumulative_data_bits": [0.0, 250000.0],
        },
    },
    "payload_sensor": {
        "mapping_id": "sat-agent-platform.component.payload_sensor.v1",
        "fields": {
            "payload.sensor.pointing_error_deg": [0.05, 0.05],
            "payload.sensor.cloud_fraction": [0.0, 0.0],
            "payload.sensor.quality_score": [0.8, 0.8],
            "payload.sensor.generated_bits": [2000000.0, 2000000.0],
            "payload.sensor.cumulative_generated_bits": [0.0, 2000000.0],
        },
    },
}

# semantic_channel_id -> (quantity_kind_id, canonical_unit) for each object.
EXPECTED = {
    "fuel_tank": {
        "asset.propulsion.fuel_tank_0.propellant_mass": ("mass.amount", "kg"),
        "asset.propulsion.fuel_tank_0.mass_flow_rate": ("mass.mass_flow_rate", "kg/s"),
        "asset.propulsion.fuel_tank_0.pressure": ("pressure.pressure", "Pa"),
    },
    "payload": {
        "asset.payload.instrument_0.power": ("electrical.active_power", "W"),
        "asset.payload.instrument_0.heat_power": ("electrical.active_power", "W"),
        "asset.payload.instrument_0.generated_rate": ("information.data_rate", "bit/s"),
        "asset.payload.instrument_0.cumulative_data_amount": ("information.data_amount", "bit"),
    },
    "payload_sensor": {
        "asset.payload.sensor_0.pointing_error": ("angle.angular_deviation", "rad"),
        "asset.payload.sensor_0.cloud_fraction": ("atmosphere.cloud_fraction", "1"),
        "asset.payload.sensor_0.quality_score": ("quality.measurement_score", "1"),
        "asset.payload.sensor_0.generated_bits": ("information.data_amount", "bit"),
        "asset.payload.sensor_0.cumulative_generated_bits": ("information.data_amount", "bit"),
    },
}

# canonical-unit values produced from CASES by the declared bindings.  Angles
# are source degrees converted to canonical radians and the fuel-tank mass flow
# is a signed outflow, so both are spelled out explicitly here (independent of
# the unit registry).
EXPECTED_VALUES = {
    "fuel_tank": {
        "asset.propulsion.fuel_tank_0.propellant_mass": [2.0, 1.999],
        "asset.propulsion.fuel_tank_0.mass_flow_rate": [-0.001, -0.001],
        "asset.propulsion.fuel_tank_0.pressure": [1716666.6666666665, 1715883.3333333333],
    },
    "payload": {
        "asset.payload.instrument_0.power": [3.0, 18.0],
        "asset.payload.instrument_0.heat_power": [2.55, 15.3],
        "asset.payload.instrument_0.generated_rate": [0.0, 250000.0],
        "asset.payload.instrument_0.cumulative_data_amount": [0.0, 250000.0],
    },
    "payload_sensor": {
        "asset.payload.sensor_0.pointing_error": [
            0.05 * math.pi / 180.0,
            0.05 * math.pi / 180.0,
        ],
        "asset.payload.sensor_0.cloud_fraction": [0.0, 0.0],
        "asset.payload.sensor_0.quality_score": [0.8, 0.8],
        "asset.payload.sensor_0.generated_bits": [2000000.0, 2000000.0],
        "asset.payload.sensor_0.cumulative_generated_bits": [0.0, 2000000.0],
    },
}

# For each object: swap one source field to an equivalent unit and assert the
# canonical semantic values are preserved.
UNIT_CONVERSIONS = {
    "fuel_tank": {
        "target_field": "propulsion.fuel_tank.pressure_pa",
        "new_unit": "kPa",
        "values": [1716.6666666666665, 1715.8833333333333],
        "channel": "asset.propulsion.fuel_tank_0.pressure",
        "expected": [1716666.6666666665, 1715883.3333333333],
    },
    "payload": {
        "target_field": "payload.instrument.power_w",
        "new_unit": "kW",
        "values": [0.003, 0.018],
        "channel": "asset.payload.instrument_0.power",
        "expected": [3.0, 18.0],
    },
    "payload_sensor": {
        "target_field": "payload.sensor.pointing_error_deg",
        "new_unit": "rad",
        "values": [0.05 * math.pi / 180.0, 0.05 * math.pi / 180.0],
        "channel": "asset.payload.sensor_0.pointing_error",
        "expected": [0.05 * math.pi / 180.0, 0.05 * math.pi / 180.0],
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
# Batch theme: existing archetypes reused across objects (no new archetype)
# ---------------------------------------------------------------------------

def test_batch_d_reuses_existing_archetypes_across_objects() -> None:
    schema, archetypes, maps = _contracts()
    # Batch D reused the seven existing archetypes across the three new objects;
    # the thrust_impulse prototype was added separately in batch E (it is a
    # distinct source-independent archetype and changes no existing prototype).
    assert {item.archetype_id for item in archetypes.archetypes} == {
        "storage.energy_buffer",
        "actuator.controlled_effort",
        "actuator.spinning_rotor",
        "thermal.lumped_capacity",
        "sensor.measurement",
        "converter.power",
        "link.transport",
        "queue.bounded_inventory",
        "actuator.thrust_impulse",
    }
    archetypes.validate_quantity_kinds({item.quantity_kind_id for item in schema.quantity_kinds})
    archetypes.validate_channel_maps(maps)

    by_mapping = {}
    for name in CASES:
        mapping_id = CASES[name]["mapping_id"]
        by_mapping[mapping_id] = {
            e.component_archetype_id for e in _entries(mapping_id)
        }
    assert by_mapping["sat-agent-platform.component.fuel_tank.v1"] == {"queue.bounded_inventory"}
    assert by_mapping["sat-agent-platform.component.payload.v1"] == {"link.transport"}
    assert by_mapping["sat-agent-platform.component.payload_sensor.v1"] == {"sensor.measurement"}


def test_batch_d_queue_gains_mass_and_pressure_ports() -> None:
    _, archetypes, _ = _contracts()
    queue = archetypes.archetype("queue.bounded_inventory")
    assert "mass.amount" in queue.port_role("state.occupied_amount").quantity_kind_ids
    assert "mass.mass_flow_rate" in queue.port_role("flow.exchange").quantity_kind_ids
    assert "pressure.pressure" in queue.port_role("state.pressure").quantity_kind_ids


def test_batch_d_link_output_gains_data_amount() -> None:
    _, archetypes, _ = _contracts()
    link = archetypes.archetype("link.transport")
    assert "information.data_amount" in link.port_role("flow.output").quantity_kind_ids


def test_batch_d_payload_heat_is_an_output_not_an_input() -> None:
    """Waste heat is a payload by-product, so it exits the transport boundary.

    3# review caught heat_w bound to ``flow.input`` ("flow entering the
    boundary"); waste heat is emitted by the instrument, so it must be the
    ``flow.output`` port (which already allows electrical.active_power).
    """
    mapping_id = CASES["payload"]["mapping_id"]
    entry = next(e for e in _entries(mapping_id) if e.source_field == "payload.instrument.heat_w")
    assert entry.port_role_id == "flow.output"
    assert entry.direction == 1

    result = _mapper(mapping_id).map_frame(_full_frame("payload"))
    heat = next(
        s for s in result.frame.signals
        if s.semantic_channel_id == "asset.payload.instrument_0.heat_power"
    )
    assert heat.port_role_id == "flow.output"
    assert heat.quantity_kind_id == "electrical.active_power"
    assert heat.values == pytest.approx([2.55, 15.3])


def test_batch_d_sensor_measurement_gains_payload_quality_kinds() -> None:
    _, archetypes, _ = _contracts()
    sensor = archetypes.archetype("sensor.measurement")
    for role_id in ("observation.input", "measurement.output"):
        allowed = sensor.port_role(role_id).quantity_kind_ids
        assert {"angle.angular_deviation", "information.data_amount",
                "atmosphere.cloud_fraction", "quality.measurement_score"} <= set(allowed)


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


def test_batch_d_unit_registry_conversions_are_registered() -> None:
    registry = DEFAULT_UNIT_REGISTRY
    assert registry.convert(1716.6666666666665, source="kPa", target="Pa") == pytest.approx(1716666.6666666665)
    assert registry.convert(0.003, source="kW", target="W") == pytest.approx(3.0)
    assert registry.convert(0.05 * math.pi / 180.0, source="rad", target="deg") == pytest.approx(0.05)
    assert registry.convert(0.001, source="kg/s", target="kg/s") == pytest.approx(0.001)
    assert registry.convert(2.0, source="kg", target="kg") == pytest.approx(2.0)


def test_batch_d_new_quantity_kinds_are_declared_with_canonical_units() -> None:
    schema, _, _ = _contracts()
    kinds = {item.quantity_kind_id: item for item in schema.quantity_kinds}
    assert kinds["mass.amount"].canonical_unit == "kg"
    assert kinds["mass.mass_flow_rate"].canonical_unit == "kg/s"
    assert kinds["pressure.pressure"].canonical_unit == "Pa"
    assert kinds["atmosphere.cloud_fraction"].canonical_unit == "1"
    assert kinds["quality.measurement_score"].canonical_unit == "1"


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


def test_no_batch_d_bound_field_leaks_deny_table_terms() -> None:
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


def test_batch_d_fault_effective_nominal_fields_never_enter_snapshots() -> None:
    """Unbound / deny-filtered real fields stay physically isolated.

    The real payload trace exports ``payload.instrument.enabled`` and
    ``payload.instrument.effective_rate_factor``, and the payload-sensor trace
    exports ``payload.sensor.valid_observation`` (a boolean gating field not in
    the declared mapping) and ``payload.sensor.effective_data_rate_bps``; none
    of them may appear in the committed public snapshots.
    """
    snapshot_text = "\n".join(
        (SNAPSHOT_ROOT / _OBJ_TO_SNAPSHOT[name]).read_text(encoding="utf-8")
        for name in CASES
    )
    for token in ("fault", "nominal", "effective_", "label.", "enabled", "degradation",
                  "valid_observation"):
        assert token not in snapshot_text, f"unbound/deny-filtered field leaked into snapshot: {token}"


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
