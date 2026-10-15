"""Semantic binding tests for the batch-B component profiles.

Batch B binds two prototype families that were previously only partially
covered: thermal (``thermal.lumped_capacity``) heater + radiator, and the first
real link.transport (``link.transport``) bindings for transmitter + link budget,
plus the first real queue binding (``queue.bounded_inventory``) for onboard
storage.

For every object the five mapping classes from
``test_battery_semantic_binding.py`` are reproduced:

1. minimal positive frame mapping;
2. field rename must fail closed (required) / be ignored cleanly (optional);
3. unit conversion preserves canonical semantics;
4. missing required field / time axis must fail closed;
5. prohibited label/truth/event/object/effective-* fields must never reach a frame.

Real Sat-Agent nominal traces are verified end-to-end via the committed public
telemetry snapshots only (no external Sat-Agent path is required).
"""

from __future__ import annotations

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
    "heater": "sat-agent.component.heater.trace.v1.csv",
    "radiator": "sat-agent.component.radiator.trace.v1.csv",
    "transmitter": "sat-agent.component.transmitter.trace.v1.csv",
    "link_budget": "sat-agent.component.link-budget.trace.v1.csv",
    "onboard_storage": "sat-agent.component.onboard-storage.trace.v1.csv",
}

# source_field -> values for a two-row minimal positive frame.
# Direction is handled by multiplying by the entry direction.
CASES = {
    "heater": {
        "mapping_id": "sat-agent-platform.component.heater.v1",
        "fields": {
            "thermal.heater.node_temp_k": [280.0, 280.0],
            "thermal.heater.power_w": [25.0, 25.0],
        },
    },
    "radiator": {
        "mapping_id": "sat-agent-platform.component.radiator.v1",
        "fields": {
            "thermal.radiator.node_temp_k": [300.0, 300.0],
            "thermal.radiator.heat_load_w": [25.0, 25.0],
            "thermal.radiator.rejected_heat_w": [68.248981, 68.248981],
            "thermal.radiator.margin_w": [43.248981, 43.248981],
        },
    },
    "transmitter": {
        "mapping_id": "sat-agent-platform.component.transmitter.v1",
        "fields": {
            "comm.transmitter.requested_rate_bps": [500000.0, 500000.0],
            "comm.transmitter.tx_power_w": [0.96306, 0.96306],
            "comm.transmitter.power_draw_w": [2.390347, 2.390347],
        },
    },
    "link_budget": {
        "mapping_id": "sat-agent-platform.component.link_budget.v1",
        "fields": {
            "comm.link_budget.ebn0_db": [12.314063, 12.314063],
            "comm.link_budget.ber": [2.651363e-09, 2.651363e-09],
            "comm.link_budget.slant_range_m": [1000000.0, 1000000.0],
        },
    },
    "onboard_storage": {
        "mapping_id": "sat-agent-platform.component.onboard-storage.v1",
        "fields": {
            "comm.storage.fill_ratio": [0.0, 0.00005],
            "comm.storage.generated_bps": [1000.0, 1000.0],
            "comm.storage.downlink_bps": [0.0, 0.0],
            "comm.storage.stored_bits": [0.0, 1000.0],
        },
    },
}

# semantic_channel_id -> (quantity_kind_id, canonical_unit) for each object
EXPECTED = {
    "heater": {
        "asset.thermal.heater_0.temperature": ("thermal.temperature", "K"),
        "asset.thermal.heater_0.heat_power": ("electrical.active_power", "W"),
    },
    "radiator": {
        "asset.thermal.radiator_0.temperature": ("thermal.temperature", "K"),
        "asset.thermal.radiator_0.heat_load": ("electrical.active_power", "W"),
        "asset.thermal.radiator_0.rejected_heat": ("electrical.active_power", "W"),
        "asset.thermal.radiator_0.rejection_margin": ("electrical.active_power", "W"),
    },
    "transmitter": {
        "asset.comm.transmitter_0.requested_rate": ("information.data_rate", "bit/s"),
        "asset.comm.transmitter_0.tx_power": ("electrical.active_power", "W"),
        "asset.comm.transmitter_0.power_draw": ("electrical.active_power", "W"),
    },
    "link_budget": {
        "asset.comm.link_0.signal_to_noise_ratio": ("signal.signal_to_noise_ratio", "dB"),
        "asset.comm.link_0.bit_error_rate": ("signal.bit_error_rate", "1"),
        "asset.comm.link_0.slant_range": ("length.distance", "m"),
    },
    "onboard_storage": {
        "asset.comm.queue_0.fill_fraction": ("inventory.fill_fraction", "1"),
        "asset.comm.queue_0.arrival_rate": ("information.data_rate", "bit/s"),
        "asset.comm.queue_0.service_rate": ("information.data_rate", "bit/s"),
        "asset.comm.queue_0.stored_amount": ("information.data_amount", "bit"),
    },
}

# For each object: swap one source field to an equivalent unit and assert the
# canonical semantic values are preserved.
UNIT_CONVERSIONS = {
    "heater": {
        "target_field": "thermal.heater.node_temp_k",
        "new_unit": "degC",
        "values": [6.85, 6.85],
        "channel": "asset.thermal.heater_0.temperature",
        "expected": [280.0, 280.0],
    },
    "radiator": {
        "target_field": "thermal.radiator.rejected_heat_w",
        "new_unit": "kW",
        "values": [0.068248981, 0.068248981],
        "channel": "asset.thermal.radiator_0.rejected_heat",
        "expected": [68.248981, 68.248981],
    },
    "transmitter": {
        "target_field": "comm.transmitter.tx_power_w",
        "new_unit": "kW",
        "values": [0.00096306, 0.00096306],
        "channel": "asset.comm.transmitter_0.tx_power",
        "expected": [0.96306, 0.96306],
    },
    "link_budget": {
        "target_field": "comm.link_budget.ber",
        "new_unit": "%",
        "values": [2.651363e-07, 2.651363e-07],
        "channel": "asset.comm.link_0.bit_error_rate",
        "expected": [2.651363e-09, 2.651363e-09],
    },
    "onboard_storage": {
        "target_field": "comm.storage.fill_ratio",
        "new_unit": "%",
        "values": [0.0, 0.005],
        "channel": "asset.comm.queue_0.fill_fraction",
        "expected": [0.0, 0.00005],
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


def _directed(field: str, values: list[float]) -> list[float]:
    entry = _entry_by_source(field)
    return [entry.direction * value for value in values]


def _entry_by_source(source_field: str):
    for name in CASES:
        for entry in _entries(CASES[name]["mapping_id"]):
            if entry.source_field == source_field:
                return entry
    raise AssertionError(f"no entry for source field {source_field}")


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
        entry = next(
            e for e in _entries(mapping_id) if e.semantic_channel_id == channel_id
        )
        assert signal.values == pytest.approx(_directed(entry.source_field, CASES[name]["fields"][entry.source_field]))

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


def test_batch_b_unit_registry_conversions_are_registered() -> None:
    registry = DEFAULT_UNIT_REGISTRY
    assert registry.convert(280.0, source="K", target="degC") == pytest.approx(6.85)
    assert registry.convert(0.068248981, source="kW", target="W") == pytest.approx(68.248981)
    assert registry.convert(2.651363e-07, source="%", target="1") == pytest.approx(2.651363e-09)
    assert registry.convert(1000000.0, source="m", target="m") == 1000000.0
    assert registry.convert(12.314063, source="dB", target="dB") == pytest.approx(12.314063)


def test_batch_b_new_quantity_kinds_are_declared_with_canonical_units() -> None:
    schema, _, _ = _contracts()
    kinds = {item.quantity_kind_id: item for item in schema.quantity_kinds}
    assert kinds["signal.signal_to_noise_ratio"].canonical_unit == "dB"
    assert kinds["signal.bit_error_rate"].canonical_unit == "1"
    assert kinds["information.data_amount"].canonical_unit == "bit"
    assert kinds["length.distance"].canonical_unit == "m"


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


def test_no_batch_b_bound_field_leaks_deny_table_terms() -> None:
    """Bound source fields must never carry deny-table tokens."""
    denied = (
        "label", "truth", "effect", "event", "object", "status",
        "task_id", "case_id", "mode", "target_level", "target_name",
        "fault", "bias", "degradation", "drift", "dropout", "injected",
        "effective_", "effective.", "stuck", "available", "enabled", "seed", "split",
    )
    for name in CASES:
        mapping_id = CASES[name]["mapping_id"]
        for entry in _entries(mapping_id):
            lowered = entry.source_field.lower()
            assert not any(token in lowered for token in denied), (
                f"{name} bound field {entry.source_field} leaks deny-table token"
            )


def test_batch_b_fault_effective_fields_never_enter_snapshots() -> None:
    """Deny-listed real fields stay physically isolated from the snapshots."""
    assert "effective_" not in "\n".join(
        (SNAPSHOT_ROOT / _OBJ_TO_SNAPSHOT[name]).read_text(encoding="utf-8")
        for name in CASES
    )
    assert "label." not in "\n".join(
        (SNAPSHOT_ROOT / _OBJ_TO_SNAPSHOT[name]).read_text(encoding="utf-8")
        for name in CASES
    )


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
