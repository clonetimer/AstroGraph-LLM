"""Semantic binding tests for the new ``actuator.thrust_impulse`` prototype.

Batch E binds the real Sat-Agent thruster component onto a brand-new propulsion
archetype.  The thruster's effort is linear force/impulse and its state is
propellant mass, neither of which the seven existing archetypes can express, so
this work item (user-approved) introduces an eighth source-independent
archetype without altering any existing prototype's behavior.

Schema additions:
* dimensions ``FORCE`` (M, L, T^-2) and ``LINEAR_MOMENTUM`` (M, L, T^-1);
* units ``N`` (force) and ``N*s`` (linear momentum);
* quantity kinds ``mechanical.thrust``, ``mechanical.linear_impulse`` and
  ``time.duration`` (the last reuses the existing TIME dimension and ``s``
  unit that were already in the registry).

Channel binding (deny-safe, only clean trace fields):
* ``propulsion.thruster.cumulative_impulse_ns``    -> ``state.linear_impulse``
* ``propulsion.thruster.cumulative_propellant_kg`` -> ``state.propellant_consumed``
* ``propulsion.thruster.on_time_s``                -> ``state.on_time``

``propulsion.thruster.effective_thrust_n`` and
``propulsion.thruster.effective_isp_s`` carry the deny-table ``effective_``
token and are deliberately NOT bound.

The five mapping classes from the earlier batch tests are reproduced:
1. minimal positive frame mapping;
2. field rename must fail closed (required) / be ignored cleanly (optional);
3. unit conversion preserves canonical semantics;
4. missing required field / time axis must fail closed;
5. prohibited label/truth/event/object/effective-* fields must never reach a
   frame.

Real Sat-Agent nominal traces are verified end-to-end via the committed public
telemetry snapshot only (no external Sat-Agent path is required).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from astrograph.semantics import (
    ArchetypeFeatureCatalog,
    ChannelMapCatalog,
    ChannelMapEntry,
    ChannelMapper,
    ComponentArchetypeCatalog,
    DEFAULT_UNIT_REGISTRY,
    FORCE,
    LINEAR_MOMENTUM,
    SemanticMappingError,
    SemanticSchema,
)
from astrograph.features.generic.registry import default_registry

ROOT = Path(__file__).parents[1]
SEMANTICS = ROOT / "configs" / "semantics"

SNAPSHOT_ROOT = ROOT / "data" / "sat_agent_snapshots" / "traces"
THRUSTER_SNAPSHOT = SNAPSHOT_ROOT / "sat-agent.component.thruster.trace.v1.csv"

MAPPING_ID = "sat-agent-platform.component.thruster.v1"
ARCHETYPE_ID = "actuator.thrust_impulse"

# source_field -> two-row values (source units) for a minimal positive frame.
FIELDS = {
    "propulsion.thruster.cumulative_impulse_ns": [0.0, 0.1],
    "propulsion.thruster.cumulative_propellant_kg": [0.0, 3.641843617778316e-05],
    "propulsion.thruster.on_time_s": [0.0, 0.1],
}

# semantic_channel_id -> (quantity_kind_id, canonical_unit).
EXPECTED = {
    "asset.propulsion.thruster_0.cumulative_impulse": ("mechanical.linear_impulse", "N*s"),
    "asset.propulsion.thruster_0.propellant_consumed": ("mass.amount", "kg"),
    "asset.propulsion.thruster_0.on_time": ("time.duration", "s"),
}

EXPECTED_VALUES = {
    "asset.propulsion.thruster_0.cumulative_impulse": [0.0, 0.1],
    "asset.propulsion.thruster_0.propellant_consumed": [0.0, 3.641843617778316e-05],
    "asset.propulsion.thruster_0.on_time": [0.0, 0.1],
}

# on_time_s declared in milliseconds must map back to canonical seconds.
UNIT_CONVERSION = {
    "target_field": "propulsion.thruster.on_time_s",
    "new_unit": "ms",
    "values": [100.0, 100.0],
    "channel": "asset.propulsion.thruster_0.on_time",
    "expected": [0.1, 0.1],
}


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


def _mapper(mapping_id: str = MAPPING_ID) -> ChannelMapper:
    schema, _, maps = _contracts()
    return ChannelMapper(semantic_schema=schema, channel_map=maps.channel_map(mapping_id))


def _entries(mapping_id: str = MAPPING_ID) -> list:
    _, _, maps = _contracts()
    return list(maps.channel_map(mapping_id).entries)


def _by_channel(result: object) -> dict[str, object]:
    return {signal.semantic_channel_id: signal for signal in result.frame.signals}


def _full_frame() -> pd.DataFrame:
    return pd.DataFrame({"time_s": [0.0, 1.0], **{k: list(v) for k, v in FIELDS.items()}})


def _mapper_with_source_unit(target_field: str, new_unit: str) -> ChannelMapper:
    schema, _, maps = _contracts()
    channel_map = maps.channel_map(MAPPING_ID)
    payload = [entry.model_dump(mode="json") for entry in channel_map.entries]
    for entry in payload:
        if entry["source_field"] == target_field:
            entry["source_unit"] = new_unit
    modified = channel_map.model_copy(
        update={"entries": [ChannelMapEntry.model_validate(entry) for entry in payload]}
    )
    return ChannelMapper(semantic_schema=schema, channel_map=modified)


# ---------------------------------------------------------------------------
# Prototype and schema additions
# ---------------------------------------------------------------------------

def test_thrust_impulse_archetype_is_declared_and_validates_with_all_maps() -> None:
    schema, archetypes, maps = _contracts()
    assert {item.archetype_id for item in archetypes.archetypes} >= {
        "storage.energy_buffer",
        "actuator.controlled_effort",
        "thermal.lumped_capacity",
        "sensor.measurement",
        "converter.power",
        "link.transport",
        "queue.bounded_inventory",
        "actuator.thrust_impulse",
    }
    archetypes.validate_quantity_kinds({item.quantity_kind_id for item in schema.quantity_kinds})
    archetypes.validate_channel_maps(maps)

    proto = archetypes.archetype(ARCHETYPE_ID)
    assert proto.family == "ACTUATOR"
    assert proto.port_role("state.linear_impulse").quantity_kind_ids == ["mechanical.linear_impulse"]
    assert proto.port_role("state.propellant_consumed").quantity_kind_ids == ["mass.amount"]
    assert proto.port_role("state.on_time").quantity_kind_ids == ["time.duration"]
    assert proto.port_role("state.linear_impulse").required is True
    assert proto.port_role("state.propellant_consumed").required is True
    assert proto.port_role("state.on_time").required is False


def test_new_quantity_kinds_and_dimension_units_are_registered() -> None:
    schema, _, _ = _contracts()
    kinds = {item.quantity_kind_id: item for item in schema.quantity_kinds}
    assert kinds["mechanical.thrust"].canonical_unit == "N"
    assert kinds["mechanical.linear_impulse"].canonical_unit == "N*s"
    assert kinds["time.duration"].canonical_unit == "s"

    registry = DEFAULT_UNIT_REGISTRY
    assert registry.get("N").dimension == FORCE
    assert registry.get("N*s").dimension == LINEAR_MOMENTUM
    # FORCE = M L T^-2 ; LINEAR_MOMENTUM = M L T^-1 are distinct dimensions.
    assert FORCE != LINEAR_MOMENTUM
    # Existing angular momentum is M L^2 T^-1, distinct from linear momentum.
    assert registry.get("N*m*s").dimension != LINEAR_MOMENTUM
    assert registry.convert(1.0, source="N", target="N") == pytest.approx(1.0)
    assert registry.convert(1.0, source="N*s", target="N*s") == pytest.approx(1.0)


def test_thrust_feature_contract_is_truth_blind_and_operator_resolved() -> None:
    catalog = ArchetypeFeatureCatalog.model_validate_json(
        (SEMANTICS / "sat_agent_archetype_feature_contracts.s0.v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract = next(
        item for item in catalog.archetype_contracts
        if item.component_archetype_id == ARCHETYPE_ID
    )
    assert contract.required_port_role_ids == ["state.linear_impulse", "state.propellant_consumed"]
    assert contract.optional_port_role_ids == ["state.on_time"]
    assert {p.operator_family for p in contract.primitives} >= {
        "ROBUST_LEVEL_DISTRIBUTION",
        "SIGNED_SLOPE_DISTRIBUTION",
        "INTEGRAL_BALANCE_RESIDUAL",
    }
    # S1 fit authorized 2026-08-04; catalog records the materialized fit state.
    assert catalog.fitted_model_count == 1
    assert catalog.s1_fit_authorized is True

    registry = default_registry()
    resolved = registry.load_feature_contract_catalog(
        SEMANTICS / "sat_agent_archetype_feature_contracts.s0.v1.json"
    )
    assert ARCHETYPE_ID in resolved
    assert len(resolved[ARCHETYPE_ID]) >= 3


# ---------------------------------------------------------------------------
# 1. Minimal positive frames
# ---------------------------------------------------------------------------

def test_minimal_positive_frame_maps_all_declared_channels() -> None:
    result = _mapper().map_frame(_full_frame())
    signals = _by_channel(result)
    assert set(signals) == set(EXPECTED)

    for channel_id, (quantity_kind, canonical_unit) in EXPECTED.items():
        signal = signals[channel_id]
        assert signal.quantity_kind_id == quantity_kind
        assert signal.canonical_unit == canonical_unit
        assert signal.values == pytest.approx(EXPECTED_VALUES[channel_id])

    assert result.frame.labels_used is False
    assert result.frame.truth_used is False
    assert result.frame.semantic_data_sha256


def test_required_fields_only_map_to_required_channels() -> None:
    required = [e for e in _entries() if e.required]
    frame = pd.DataFrame({"time_s": [0.0, 1.0], **{e.source_field: [1.0, 2.0] for e in required}})
    result = _mapper().map_frame(frame)
    assert {signal.semantic_channel_id for signal in result.frame.signals} == {
        e.semantic_channel_id for e in required
    }


# ---------------------------------------------------------------------------
# 2. Field rename must never silently change meaning
# ---------------------------------------------------------------------------

def test_renamed_required_field_fails_closed() -> None:
    required_field = next(e.source_field for e in _entries() if e.required)
    frame = _full_frame().rename(columns={required_field: f"{required_field}_renamed"})
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert required_field in exc.value.fields


def test_renamed_optional_field_is_ignored_cleanly() -> None:
    optional = [e for e in _entries() if not e.required]
    assert optional, "thruster map should declare at least one optional port"
    entry = optional[0]
    frame = _full_frame().rename(columns={entry.source_field: f"{entry.source_field}_renamed"})
    result = _mapper().map_frame(frame)
    assert entry.semantic_channel_id not in {s.semantic_channel_id for s in result.frame.signals}
    assert f"{entry.source_field}_renamed" in result.receipt.ignored_clean_fields


# ---------------------------------------------------------------------------
# 3. Unit conversion preserves canonical semantics
# ---------------------------------------------------------------------------

def test_unit_conversion_preserves_canonical_semantics() -> None:
    spec = UNIT_CONVERSION
    fields = dict(FIELDS)
    fields[spec["target_field"]] = spec["values"]
    frame = pd.DataFrame({"time_s": [0.0, 1.0], **fields})
    mapper = _mapper_with_source_unit(spec["target_field"], spec["new_unit"])
    result = mapper.map_frame(frame)
    signal = _by_channel(result)[spec["channel"]]
    assert signal.canonical_unit == EXPECTED[spec["channel"]][1]
    assert signal.values == pytest.approx(spec["expected"])


def test_thruster_bound_fields_have_dimension_consistent_source_units() -> None:
    # Every bound source unit must match the declared quantity-kind dimension.
    schema, _, _ = _contracts()
    kinds = {item.quantity_kind_id: item for item in schema.quantity_kinds}
    registry = DEFAULT_UNIT_REGISTRY
    for entry in _entries():
        kind = kinds[entry.quantity_kind_id]
        registry.require_dimension(entry.source_unit, kind.dimension)
        registry.require_dimension(kind.canonical_unit, kind.dimension)


# ---------------------------------------------------------------------------
# 4. Missing fields must fail closed
# ---------------------------------------------------------------------------

def test_missing_required_signal_field_fails_closed() -> None:
    required_field = next(e.source_field for e in _entries() if e.required)
    frame = _full_frame().drop(columns=[required_field])
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert required_field in exc.value.fields


def test_missing_time_axis_fails_closed() -> None:
    frame = _full_frame().drop(columns=["time_s"])
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert "time_s" in exc.value.fields


# ---------------------------------------------------------------------------
# 5. Prohibited fields must never reach a semantic frame
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_field",
    ["label.health_state", "truth.fault_active", "effect.active", "event.onset", "object.case"],
)
def test_prohibited_field_is_rejected(bad_field: str) -> None:
    frame = _full_frame()
    frame[bad_field] = ["x", "y"]
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "PROHIBITED_SOURCE_FIELD"
    assert bad_field in exc.value.fields


def test_effective_thrust_and_isp_fields_are_not_bound() -> None:
    """Deny-table ``effective_*`` fields must never be declared as bound."""
    bound = {e.source_field for e in _entries()}
    assert "propulsion.thruster.effective_thrust_n" not in bound
    assert "propulsion.thruster.effective_isp_s" not in bound
    denied = ("label", "truth", "effect", "event", "object", "status",
              "task_id", "case_id", "mode", "target_level", "target_name",
              "fault", "bias", "degradation", "drift", "dropout", "injected",
              "effective_", "effective.", "stuck", "available", "enabled",
              "seed", "split", "nominal")
    for entry in _entries():
        lowered = entry.source_field.lower()
        assert not any(token in lowered for token in denied), (
            f"thruster bound field {entry.source_field} leaks deny-table token"
        )


# ---------------------------------------------------------------------------
# Real-file verification (snapshot-backed only)
# ---------------------------------------------------------------------------

def test_real_sat_agent_nominal_trace_maps_through_declared_binding() -> None:
    if not THRUSTER_SNAPSHOT.is_file():
        pytest.skip("thruster nominal snapshot unavailable")
    mapped_fields = {e.source_field for e in _entries()}
    raw = pd.read_csv(THRUSTER_SNAPSHOT)
    selected = ["time_s", *[c for c in sorted(mapped_fields) if c in raw.columns]]
    frame = raw[selected].copy()
    result = _mapper().map_frame(frame)

    present = set(mapped_fields) & set(raw.columns)
    assert {s.semantic_channel_id for s in result.frame.signals} == {
        e.semantic_channel_id for e in _entries() if e.source_field in present
    }
    timestamps = result.frame.timestamps_s
    assert timestamps
    assert all(right > left for left, right in zip(timestamps, timestamps[1:]))
    assert result.frame.sampling.row_count == len(raw)
    assert result.frame.labels_used is False
    assert result.frame.truth_used is False


def test_thruster_snapshot_never_contains_deny_filtered_columns() -> None:
    if not THRUSTER_SNAPSHOT.is_file():
        pytest.skip("thruster nominal snapshot unavailable")
    text = THRUSTER_SNAPSHOT.read_text(encoding="utf-8")
    for token in ("fault", "nominal", "effective_", "label.", "mode", "target_level",
                  "target_name", "task_id", "case_id"):
        assert token not in text, f"deny-filtered token leaked into thruster snapshot: {token}"
