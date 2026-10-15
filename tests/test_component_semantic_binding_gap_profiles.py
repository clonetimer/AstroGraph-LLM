"""Semantic binding five-class tests for the remaining snapshot-backed profiles.

Closes the 2.A coverage gap (2026-08-04): five profiles had a real committed
public snapshot and a declared channel map, but no standalone five-class
binding test file.  They are:

* IMU    -> ``sensor.measurement``        (gyro x/y/z, all required)
* PDU    -> ``converter.power``           (requested/delivered power + shed count)
* data queue -> ``queue.bounded_inventory`` (generated/downlink rate, downlink is
             a negative-direction flow)
* thermal -> ``thermal.lumped_capacity``  (battery/electronics/payload temps + heater)
* ADCS-RW -> ``actuator.controlled_effort`` (torque/speed/current/pointing-error)

For every profile the five mapping classes from ``test_battery_semantic_binding.py``
are reproduced:

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
    "imu": "sat-agent.component.imu.trace.v1.csv",
    "pdu": "sat-agent.component.pdu.trace.v1.csv",
    "data_queue": "sat-agent.component.data-queue.trace.v1.csv",
    "thermal": "sat-agent.subsystem.thermal.source-native.v1.csv",
    "adcs_rw": "sat-agent.subsystem.adcs-rw.astrograph-5ch.v2.csv",
    # 2.A closure (2026-08-04): eps-unified and whole-spacecraft composite were
    # "source-blocked" only because the snapshot list never included them; real
    # Sat-Agent data now snapshots these exact schemas.
    "eps_unified": "sat-agent.subsystem.eps.unified-native.v1.csv",
    "whole_sc": "sat-agent.whole-spacecraft.composite.v1.csv",
}

# mapping_id -> two-row minimal positive frame (source units).
CASES = {
    "imu": {
        "mapping_id": "sat-agent-platform.component.imu.v1",
        "fields": {
            "adcs.imu.gyro_x_rad_s": [0.1, 0.2],
            "adcs.imu.gyro_y_rad_s": [-0.1, 0.0],
            "adcs.imu.gyro_z_rad_s": [0.0, 0.05],
        },
    },
    "pdu": {
        "mapping_id": "sat-agent-platform.component.pdu.v1",
        "fields": {
            "eps.pdu.requested_power_w": [10.0, 20.0],
            "eps.pdu.delivered_power_w": [10.0, 19.5],
            "eps.pdu.shed_count": [0, 1],
        },
    },
    "data_queue": {
        "mapping_id": "sat-agent-platform.component.data-queue-link.v1",
        "fields": {
            "comm.data_queue.generated_bps": [10000.0, 20000.0],
            "comm.data_queue.downlink_bps": [5000.0, 8000.0],
        },
    },
    "thermal": {
        "mapping_id": "sat-agent-platform.subsystem.thermal.v1",
        "fields": {
            "thermal.source_native.battery_temp_k": [300.0, 301.0],
            "thermal.source_native.electronics_temp_k": [295.0, 295.5],
            "thermal.source_native.payload_temp_k": [290.0, 290.2],
            "thermal.source_native.heater_power_w": [5.0, 0.0],
        },
    },
    "adcs_rw": {
        "mapping_id": "sat-agent-platform.subsystem.adcs-rw.v2",
        "fields": {
            "command.adcs.rw.motor_torque_nm_0": [0.01, -0.01],
            "estimate.adcs.rw.actual_torque_nm_0": [0.01, -0.009],
            "sensor.adcs.rw.speed_rad_s_0": [100.0, 100.5],
            "sensor.adcs.rw.motor_current_a_0": [1.0, 0.95],
            "estimate.adcs.pointing_error_deg": [0.5, 0.0],
        },
    },
    "eps_unified": {
        "mapping_id": "sat-agent-platform.subsystem.eps-unified.v1",
        "fields": {
            "eps.battery_soc": [0.62, 0.60],
            "eps.net_power_w": [-60.0, -55.0],
            "eps.solar_power_w": [25.0, 23.0],
        },
    },
    "whole_sc": {
        "mapping_id": "sat-agent-platform.whole-spacecraft.composite.v1",
        "fields": {
            "battery_soc": [0.62, 0.61],
            "thermal_temp_c": [21.0, 21.5],
            "attitude_error_deg": [0.5, 0.4],
            "payload_generated_bps": [2500000.0, 2500000.0],
            "downlink_delivered_bps": [0.0, 1000000.0],
            "data_storage_bits": [0.0, 1500000000.0],
            "propellant_remaining_kg": [1.0, 0.99],
        },
    },
}

# semantic_channel_id -> (quantity_kind_id, canonical_unit).
EXPECTED = {
    "imu": {
        "asset.adcs.imu_0.angular_rate_x": ("mechanical.angular_rate", "rad/s"),
        "asset.adcs.imu_0.angular_rate_y": ("mechanical.angular_rate", "rad/s"),
        "asset.adcs.imu_0.angular_rate_z": ("mechanical.angular_rate", "rad/s"),
    },
    "pdu": {
        "asset.eps.pdu_0.input_power": ("electrical.active_power", "W"),
        "asset.eps.pdu_0.output_power": ("electrical.active_power", "W"),
        "asset.eps.converter_0.shed_events": ("inventory.event_count", "1"),
    },
    "data_queue": {
        "asset.comm.queue_0.ingress_rate": ("information.data_rate", "bit/s"),
        "asset.comm.queue_0.egress_rate": ("information.data_rate", "bit/s"),
    },
    "thermal": {
        "asset.thermal.battery_0.temperature": ("thermal.temperature", "K"),
        "asset.thermal.electronics_0.temperature": ("thermal.temperature", "K"),
        "asset.thermal.payload_0.temperature": ("thermal.temperature", "K"),
        "asset.thermal.heater_0.exchange_power": ("electrical.active_power", "W"),
    },
    "adcs_rw": {
        "asset.adcs.actuator_0.command_torque": ("mechanical.torque", "N*m"),
        "asset.adcs.actuator_0.response_torque": ("mechanical.torque", "N*m"),
        "asset.adcs.actuator_0.angular_rate": ("mechanical.angular_rate", "rad/s"),
        "asset.adcs.actuator_0.motor_current": ("electrical.current", "A"),
        "asset.adcs.actuator_0.pointing_error": ("angle.angular_deviation", "rad"),
    },
    "eps_unified": {
        "asset.eps.storage_0.state_of_charge": ("energy.state_of_charge", "1"),
        "asset.eps.storage_0.exchange_power": ("electrical.active_power", "W"),
        "asset.eps.storage_0.solar_power": ("electrical.active_power", "W"),
    },
    "whole_sc": {
        "asset.eps.storage_0.state_of_charge": ("energy.state_of_charge", "1"),
        "asset.thermal.payload_0.temperature": ("thermal.temperature", "K"),
        "asset.adcs.actuator_0.pointing_error": ("angle.angular_deviation", "rad"),
        "asset.comm.queue_0.ingress_rate": ("information.data_rate", "bit/s"),
        "asset.comm.queue_0.egress_rate": ("information.data_rate", "bit/s"),
        "asset.comm.queue_0.occupied_amount": ("information.data_amount", "bit"),
        "asset.propulsion.tank_0.remaining_mass": ("mass.amount", "kg"),
    },
}

# Canonical values produced from CASES by the declared bindings.
EXPECTED_VALUES = {
    "imu": {
        "asset.adcs.imu_0.angular_rate_x": [0.1, 0.2],
        "asset.adcs.imu_0.angular_rate_y": [-0.1, 0.0],
        "asset.adcs.imu_0.angular_rate_z": [0.0, 0.05],
    },
    "pdu": {
        "asset.eps.pdu_0.input_power": [10.0, 20.0],
        "asset.eps.pdu_0.output_power": [10.0, 19.5],
        "asset.eps.converter_0.shed_events": [0.0, 1.0],
    },
    "data_queue": {
        "asset.comm.queue_0.ingress_rate": [10000.0, 20000.0],
        # downlink is a negative-direction (egress) flow, so the mapped value
        # carries the direction sign.
        "asset.comm.queue_0.egress_rate": [-5000.0, -8000.0],
    },
    "thermal": {
        "asset.thermal.battery_0.temperature": [300.0, 301.0],
        "asset.thermal.electronics_0.temperature": [295.0, 295.5],
        "asset.thermal.payload_0.temperature": [290.0, 290.2],
        "asset.thermal.heater_0.exchange_power": [5.0, 0.0],
    },
    "adcs_rw": {
        "asset.adcs.actuator_0.command_torque": [0.01, -0.01],
        "asset.adcs.actuator_0.response_torque": [0.01, -0.009],
        "asset.adcs.actuator_0.angular_rate": [100.0, 100.5],
        "asset.adcs.actuator_0.motor_current": [1.0, 0.95],
        # pointing error source deg converted to canonical rad.
        "asset.adcs.actuator_0.pointing_error": [
            math.radians(0.5),
            math.radians(0.0),
        ],
    },
    "eps_unified": {
        "asset.eps.storage_0.state_of_charge": [0.62, 0.60],
        "asset.eps.storage_0.exchange_power": [-60.0, -55.0],
        "asset.eps.storage_0.solar_power": [25.0, 23.0],
    },
    "whole_sc": {
        "asset.eps.storage_0.state_of_charge": [0.62, 0.61],
        # thermal_temp_c source degC converted to canonical K.
        "asset.thermal.payload_0.temperature": [21.0 + 273.15, 21.5 + 273.15],
        # attitude_error_deg source deg converted to canonical rad.
        "asset.adcs.actuator_0.pointing_error": [
            math.radians(0.5),
            math.radians(0.4),
        ],
        "asset.comm.queue_0.ingress_rate": [2500000.0, 2500000.0],
        # downlink_delivered is a negative-direction (egress) flow.
        "asset.comm.queue_0.egress_rate": [-0.0, -1000000.0],
        "asset.comm.queue_0.occupied_amount": [0.0, 1500000000.0],
        "asset.propulsion.tank_0.remaining_mass": [1.0, 0.99],
    },
}

# Unit conversions the registry supports for each profile (source-unit swap
# on a bound field must preserve the canonical value).
UNIT_CONVERSIONS = {
    "imu": {
        "target_field": "adcs.imu.gyro_x_rad_s",
        "new_unit": "deg/s",
        "values": [math.degrees(0.1), math.degrees(0.2)],
        "channel": "asset.adcs.imu_0.angular_rate_x",
        "expected": [0.1, 0.2],
    },
    "pdu": {
        "target_field": "eps.pdu.requested_power_w",
        "new_unit": "kW",
        "values": [0.01, 0.02],
        "channel": "asset.eps.pdu_0.input_power",
        "expected": [10.0, 20.0],
    },
    # data_queue has no registry-supported alternate rate unit (both bound
    # fields are canonical bit/s); its meaningful semantic property is the
    # flow-direction sign, covered by test_data_queue_downlink_is_a_negative_direction_flow.
    "thermal": {
        "target_field": "thermal.source_native.battery_temp_k",
        "new_unit": "degC",
        "values": [300.0 - 273.15, 301.0 - 273.15],
        "channel": "asset.thermal.battery_0.temperature",
        "expected": [300.0, 301.0],
    },
    "adcs_rw": {
        "target_field": "sensor.adcs.rw.speed_rad_s_0",
        "new_unit": "deg/s",
        "values": [math.degrees(100.0), math.degrees(100.5)],
        "channel": "asset.adcs.actuator_0.angular_rate",
        "expected": [100.0, 100.5],
    },
    "eps_unified": {
        "target_field": "eps.net_power_w",
        "new_unit": "kW",
        "values": [-0.060, -0.055],
        "channel": "asset.eps.storage_0.exchange_power",
        "expected": [-60.0, -55.0],
    },
    "whole_sc": {
        "target_field": "thermal_temp_c",
        "new_unit": "degC",
        "values": [21.0, 21.5],
        "channel": "asset.thermal.payload_0.temperature",
        "expected": [21.0 + 273.15, 21.5 + 273.15],
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
# Archetype theme: each gap profile binds to the archetype it declares
# ---------------------------------------------------------------------------

def test_gap_profiles_bind_to_declared_archetypes() -> None:
    expect_archetype = {
        "sat-agent-platform.component.imu.v1": "sensor.measurement",
        "sat-agent-platform.component.pdu.v1": "converter.power",
        "sat-agent-platform.component.data-queue-link.v1": "queue.bounded_inventory",
        "sat-agent-platform.subsystem.thermal.v1": "thermal.lumped_capacity",
        "sat-agent-platform.subsystem.adcs-rw.v2": "actuator.controlled_effort",
        # Multi-archetype maps: eps-unified is all storage; whole-spacecraft
        # composite spans storage / thermal / actuator / queue.
        "sat-agent-platform.subsystem.eps-unified.v1": "storage.energy_buffer",
        "sat-agent-platform.whole-spacecraft.composite.v1": {
            "storage.energy_buffer",
            "thermal.lumped_capacity",
            "actuator.controlled_effort",
            "queue.bounded_inventory",
        },
    }
    for name in CASES:
        mapping_id = CASES[name]["mapping_id"]
        expected = expect_archetype[mapping_id]
        actual = {e.component_archetype_id for e in _entries(mapping_id)}
        if isinstance(expected, set):
            assert actual == expected
        else:
            assert actual == {expected}


def test_whole_spacecraft_downlink_is_a_negative_direction_flow() -> None:
    """whole-spacecraft downlink_delivered is an egress flow (direction=-1)."""
    mapping_id = CASES["whole_sc"]["mapping_id"]
    directions = {e.source_field: e.direction for e in _entries(mapping_id)}
    assert directions["payload_generated_bps"] == 1
    assert directions["downlink_delivered_bps"] == -1


def test_data_queue_downlink_is_a_negative_direction_flow() -> None:
    """downlink is an egress flow (direction=-1); ingress is direction=+1."""
    mapping_id = CASES["data_queue"]["mapping_id"]
    directions = {e.source_field: e.direction for e in _entries(mapping_id)}
    assert directions["comm.data_queue.generated_bps"] == 1
    assert directions["comm.data_queue.downlink_bps"] == -1


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
    if name not in UNIT_CONVERSIONS:
        pytest.skip(f"{name} has no registry-supported alternate unit")
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


def test_gap_profiles_unit_registry_conversions_are_registered() -> None:
    registry = DEFAULT_UNIT_REGISTRY
    assert registry.convert(math.pi / 180.0, source="rad", target="deg") == pytest.approx(1.0)
    assert registry.convert(1.0, source="deg/s", target="rad/s") == pytest.approx(math.pi / 180.0)
    assert registry.convert(1.0, source="kW", target="W") == pytest.approx(1000.0)
    assert registry.convert(1.0, source="degC", target="K") == pytest.approx(274.15)


def test_gap_profiles_new_quantity_kinds_are_declared_with_canonical_units() -> None:
    schema, _, _ = _contracts()
    kinds = {item.quantity_kind_id: item for item in schema.quantity_kinds}
    assert kinds["mechanical.angular_rate"].canonical_unit == "rad/s"
    assert kinds["information.data_rate"].canonical_unit == "bit/s"
    assert kinds["electrical.current"].canonical_unit == "A"
    assert kinds["inventory.event_count"].canonical_unit == "1"


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


def test_no_gap_profile_bound_field_leaks_deny_table_terms() -> None:
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


def test_gap_profile_deny_fields_never_enter_snapshots() -> None:
    """Deny-listed real fields stay physically isolated from the snapshots."""
    missing = [name for name in CASES if not (SNAPSHOT_ROOT / _OBJ_TO_SNAPSHOT[name]).is_file()]
    if missing:
        pytest.skip("Sat-Agent snapshot data artifact not present in source-only package")
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
