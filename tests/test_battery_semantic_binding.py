from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from astrograph.semantics import (
    ChannelMapCatalog,
    ChannelMapper,
    ComponentArchetypeCatalog,
    SemanticMappingError,
    SemanticSchema,
)


ROOT = Path(__file__).parents[1]
SEMANTICS = ROOT / "configs" / "semantics"

MAPPING_ID = "sat-agent-platform.component.battery.v1"

EXPECTED_CHANNELS = {
    "asset.eps.storage_0.state_of_charge",
    "asset.eps.storage_0.exchange_power",
    "asset.eps.storage_0.applied_power",
    "asset.eps.storage_0.stored_energy",
}


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


def _mapper() -> ChannelMapper:
    schema, _, maps = _contracts()
    return ChannelMapper(
        semantic_schema=schema, channel_map=maps.channel_map(MAPPING_ID)
    )


def _by_channel(
    result: object,
) -> dict[str, object]:
    frame = result.frame  # type: ignore[attr-defined]
    return {signal.semantic_channel_id: signal for signal in frame.signals}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1. Minimal positive examples
# ---------------------------------------------------------------------------

def test_minimal_positive_frame_maps_all_four_profile_fields() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "eps.battery.soc": [0.8, 0.79],
            "eps.battery.net_power_w": [-10.0, -10.0],
            "eps.battery.applied_power_w": [-10.0, -10.0],
            "eps.battery.storage_wh": [128.0, 126.4],
        }
    )
    result = _mapper().map_frame(frame)
    signals = _by_channel(result)
    assert set(signals) == EXPECTED_CHANNELS

    soc = signals["asset.eps.storage_0.state_of_charge"]
    assert soc.quantity_kind_id == "energy.state_of_charge"
    assert soc.canonical_unit == "1"
    assert soc.values == [0.8, 0.79]

    exchange = signals["asset.eps.storage_0.exchange_power"]
    assert exchange.quantity_kind_id == "electrical.active_power"
    assert exchange.canonical_unit == "W"
    assert exchange.values == [-10.0, -10.0]

    stored = signals["asset.eps.storage_0.stored_energy"]
    assert stored.quantity_kind_id == "energy.stored_energy"
    assert stored.canonical_unit == "J"
    assert stored.values == [128.0 * 3600.0, 126.4 * 3600.0]

    assert result.frame.labels_used is False
    assert result.frame.truth_used is False
    assert result.frame.semantic_data_sha256


def test_minimal_positive_frame_with_required_fields_only() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "eps.battery.soc": [0.8, 0.79],
            "eps.battery.net_power_w": [-10.0, -10.0],
        }
    )
    result = _mapper().map_frame(frame)
    assert {signal.semantic_channel_id for signal in result.frame.signals} == {
        "asset.eps.storage_0.state_of_charge",
        "asset.eps.storage_0.exchange_power",
    }


# ---------------------------------------------------------------------------
# 2. Field rename must never silently change meaning
# ---------------------------------------------------------------------------

def test_renamed_required_field_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "eps.battery.soc_renamed": [0.8, 0.79],
            "eps.battery.net_power_w": [-10.0, -10.0],
        }
    )
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert "eps.battery.soc" in exc.value.fields


def test_renamed_non_required_field_is_ignored_cleanly() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "eps.battery.soc": [0.8, 0.79],
            "eps.battery.net_power_w_renamed": [-10.0, -10.0],
        }
    )
    result = _mapper().map_frame(frame)
    channels = {signal.semantic_channel_id for signal in result.frame.signals}
    assert "asset.eps.storage_0.state_of_charge" in channels
    assert "asset.eps.storage_0.exchange_power" not in channels
    assert "eps.battery.net_power_w_renamed" in result.receipt.ignored_clean_fields


# ---------------------------------------------------------------------------
# 3. Unit conversion: watt-hours to joules
# ---------------------------------------------------------------------------

def test_storage_energy_watthour_to_joule_conversion() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "eps.battery.soc": [0.8, 0.79],
            "eps.battery.storage_wh": [1.0, 2.0],
        }
    )
    result = _mapper().map_frame(frame)
    stored = next(
        signal
        for signal in result.frame.signals
        if signal.semantic_channel_id == "asset.eps.storage_0.stored_energy"
    )
    assert stored.canonical_unit == "J"
    assert stored.values == [3600.0, 7200.0]


# ---------------------------------------------------------------------------
# 4. Missing fields must fail closed
# ---------------------------------------------------------------------------

def test_missing_required_signal_field_rejected() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "eps.battery.net_power_w": [-10.0, -10.0],
        }
    )
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert "eps.battery.soc" in exc.value.fields


def test_missing_time_axis_rejected() -> None:
    frame = pd.DataFrame(
        {
            "eps.battery.soc": [0.8, 0.79],
            "eps.battery.net_power_w": [-10.0, -10.0],
        }
    )
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "MISSING_REQUIRED_SOURCE_FIELD"
    assert "time_s" in exc.value.fields


# ---------------------------------------------------------------------------
# 5. Illegal / prohibited fields must never reach a semantic frame
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_field",
    [
        "label.health_state",
        "truth.fault_active",
        "task_id",
        "object.thermal.safe_request",
    ],
)
def test_prohibited_field_is_rejected(bad_field: str) -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "eps.battery.soc": [0.8, 0.79],
            bad_field: ["x", "y"],
        }
    )
    with pytest.raises(SemanticMappingError) as exc:
        _mapper().map_frame(frame)
    assert exc.value.code == "PROHIBITED_SOURCE_FIELD"
    assert bad_field in exc.value.fields


# ---------------------------------------------------------------------------
# Real-file verification (skipped when the source trace is unavailable)
# ---------------------------------------------------------------------------

# Public-telemetry snapshot (committed, no external Sat-Agent path required).
BATTERY_TRACE = (
    ROOT / "data" / "sat_agent_snapshots" / "traces" / "sat-agent.component.battery.trace.v1.csv"
)
_MAPPED_SOURCE_FIELDS = [
    "time_s",
    "eps.battery.soc",
    "eps.battery.net_power_w",
    "eps.battery.applied_power_w",
    "eps.battery.storage_wh",
]


@pytest.mark.skipif(not BATTERY_TRACE.is_file(), reason="real Sat-Agent battery trace unavailable")
def test_real_sat_agent_battery_trace_maps_through_declared_binding() -> None:
    raw = pd.read_csv(BATTERY_TRACE)
    frame = raw[_MAPPED_SOURCE_FIELDS].copy()
    result = _mapper().map_frame(frame)
    assert {signal.semantic_channel_id for signal in result.frame.signals} == EXPECTED_CHANNELS
    assert result.frame.sampling.row_count == len(raw)
    timestamps = result.frame.timestamps_s
    assert timestamps
    assert all(right > left for left, right in zip(timestamps, timestamps[1:]))
    # the declared time axis is the only time used; no row-index surrogate is exposed
    assert result.frame.labels_used is False
    assert result.frame.truth_used is False
