from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.semantics import (
    ChannelMapCatalog,
    ChannelMapper,
    ComponentArchetypeCatalog,
    SemanticSchema,
)


ROOT = Path(__file__).parents[1]
SEMANTICS = ROOT / "configs" / "semantics"
PROFILE = ROOT / "configs" / "data_sources" / "sat_agent_platform_schema_profiles.s0.v1.json"


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


def test_sat_agent_platform_maps_bind_the_versioned_source_profile() -> None:
    schema, archetypes, maps = _contracts()
    expected = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    assert {item.source_schema_sha256 for item in maps.maps} == {expected}
    archetypes.validate_quantity_kinds({item.quantity_kind_id for item in schema.quantity_kinds})
    archetypes.validate_channel_maps(maps)
    assert len(maps.maps) == 26
    assert len({tuple(item.source_field for item in mapping.entries) for mapping in maps.maps}) == 26
    assert {
        entry.component_archetype_id
        for mapping in maps.maps
        for entry in mapping.entries
    } == {
        "storage.energy_buffer",
        "actuator.controlled_effort",
        "thermal.lumped_capacity",
        "sensor.measurement",
        "converter.power",
        "link.transport",
        "queue.bounded_inventory",
        "actuator.thrust_impulse",
    }

    profiles = json.loads(PROFILE.read_text(encoding="utf-8"))["profiles"]
    declared_field_sets = [
        {item["time_field"]["name"]}
        | {field["name"] for field in item["public_observation_fields"]}
        for item in profiles
    ]
    for mapping in maps.maps:
        mapped = {mapping.time_axis.source_field} | {
            entry.source_field for entry in mapping.entries
        }
        assert any(mapped <= declared for declared in declared_field_sets)


@pytest.mark.parametrize(
    ("mapping_id", "frame", "expected_archetypes", "expected_signal_count"),
    [
        (
            "sat-agent-platform.component.battery.v1",
            pd.DataFrame(
                {"time_s": [0.0, 1.0], "eps.battery.soc": [0.8, 0.79], "eps.battery.net_power_w": [-10.0, -10.0]}
            ),
            {"storage.energy_buffer"},
            2,
        ),
        (
            "sat-agent-platform.subsystem.adcs-rw.v2",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "sensor.adcs.rw.speed_rad_s_0": [20.0, 20.2],
                    "command.adcs.rw.motor_torque_nm_0": [0.1, 0.2],
                    "estimate.adcs.rw.actual_torque_nm_0": [0.09, 0.18],
                    "sensor.adcs.rw.motor_current_a_0": [4.5, 9.0],
                    "estimate.adcs.pointing_error_deg": [2.0, 1.8],
                }
            ),
            {"actuator.controlled_effort"},
            5,
        ),
        (
            "sat-agent-platform.subsystem.thermal.v1",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "thermal.source_native.battery_temp_k": [290.0, 290.1],
                    "thermal.source_native.electronics_temp_k": [300.0, 300.2],
                    "thermal.source_native.payload_temp_k": [295.0, 295.1],
                    "thermal.source_native.heater_power_w": [0.0, 5.0],
                }
            ),
            {"thermal.lumped_capacity"},
            4,
        ),
        (
            "sat-agent-platform.subsystem.eps-unified.v1",
            pd.DataFrame(
                {"time_s": [0.0, 1.0], "eps.battery_soc": [0.6, 0.59], "eps.net_power_w": [-5.0, -5.0], "eps.solar_power_w": [2.0, 2.1]}
            ),
            {"storage.energy_buffer"},
            3,
        ),
        (
            "sat-agent-platform.whole-spacecraft.composite.v1",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "battery_soc": [0.7, 0.69],
                    "thermal_temp_c": [20.0, 20.2],
                    "payload_generated_bps": [10.0, 10.0],
                }
            ),
            {"storage.energy_buffer", "thermal.lumped_capacity", "queue.bounded_inventory"},
            3,
        ),
        (
            "sat-agent-platform.component.imu.v1",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "adcs.imu.gyro_x_rad_s": [0.01, 0.02],
                    "adcs.imu.gyro_y_rad_s": [0.02, 0.03],
                    "adcs.imu.gyro_z_rad_s": [0.03, 0.04],
                }
            ),
            {"sensor.measurement"},
            3,
        ),
        (
            "sat-agent-platform.component.pdu.v1",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "eps.pdu.requested_power_w": [100.0, 120.0],
                    "eps.pdu.delivered_power_w": [95.0, 110.0],
                    "eps.pdu.shed_count": [0.0, 0.0],
                }
            ),
            {"converter.power"},
            3,
        ),
        (
            "sat-agent-platform.component.data-queue-link.v1",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "comm.data_queue.generated_bps": [2000.0, 2200.0],
                    "comm.data_queue.downlink_bps": [1800.0, 1900.0],
                }
            ),
            {"queue.bounded_inventory"},
            2,
        ),
        (
            "sat-agent-platform.component.onboard-storage.v1",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "comm.storage.fill_ratio": [0.3, 0.31],
                    "comm.storage.generated_bps": [2000.0, 2200.0],
                    "comm.storage.downlink_bps": [1800.0, 1900.0],
                }
            ),
            {"queue.bounded_inventory"},
            3,
        ),
    ],
)
def test_actual_sat_agent_fields_map_without_source_specific_python_branches(
    mapping_id: str,
    frame: pd.DataFrame,
    expected_archetypes: set[str],
    expected_signal_count: int,
) -> None:
    schema, _, maps = _contracts()
    result = ChannelMapper(
        semantic_schema=schema, channel_map=maps.channel_map(mapping_id)
    ).map_frame(frame)
    assert {item.component_archetype_id for item in result.frame.signals} == expected_archetypes
    assert len(result.frame.signals) == expected_signal_count
    assert result.frame.labels_used is False
    assert result.frame.truth_used is False
