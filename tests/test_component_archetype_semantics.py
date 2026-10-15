from __future__ import annotations

from pathlib import Path

import hashlib

import pandas as pd
import pytest

from astrograph.semantics import (
    AssetTopology,
    ChannelMapCatalog,
    ChannelMapper,
    ComponentArchetypeCatalog,
    ComponentInstance,
    PortBinding,
    SemanticSchema,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "semantics"


def _load_contracts() -> tuple[SemanticSchema, ComponentArchetypeCatalog, ChannelMapCatalog]:
    schema = SemanticSchema.model_validate_json(
        (CONFIG_ROOT / "core_semantic_schema.json").read_text(encoding="utf-8")
    )
    archetypes = ComponentArchetypeCatalog.model_validate_json(
        (CONFIG_ROOT / "core_component_archetypes.json").read_text(encoding="utf-8")
    )
    maps = ChannelMapCatalog.model_validate_json(
        (CONFIG_ROOT / "sat_agent_channel_maps.json").read_text(encoding="utf-8")
    )
    return schema, archetypes, maps


def test_source_independent_component_archetypes_are_valid() -> None:
    schema, archetypes, maps = _load_contracts()
    archetypes.validate_quantity_kinds({item.quantity_kind_id for item in schema.quantity_kinds})
    archetypes.validate_channel_maps(maps)

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
    archetype_text = (CONFIG_ROOT / "core_component_archetypes.json").read_text(encoding="utf-8")
    schema_text = (CONFIG_ROOT / "core_semantic_schema.json").read_text(encoding="utf-8")
    for source_marker in (
        "sensor.thermal.payload_temp",
        "estimate.eps.battery_soc",
        "adcs.rw",
        "sat-agent",
    ):
        assert source_marker not in archetype_text.lower()
        assert source_marker not in schema_text.lower()
    assert len(maps.maps) == 3


def test_all_required_archetype_families_are_present_once() -> None:
    _, archetypes, _ = _load_contracts()
    assert [item.family for item in archetypes.archetypes].count("STORAGE") == 1
    assert {item.family for item in archetypes.archetypes} == {
        "STORAGE",
        "ACTUATOR",
        "THERMAL_CAPACITY",
        "SENSOR",
        "CONVERTER",
        "TRANSPORT_LINK",
        "QUEUE_STORAGE",
    }


def test_channel_map_source_schema_hash_matches_declared_protocol() -> None:
    _, _, maps = _load_contracts()
    protocol = ROOT / "protocols" / "phm_project_a" / "telemetry_physical_semantics_v1.csv"
    expected = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert {item.source_schema_sha256 for item in maps.maps} == {expected}


def test_generic_mapper_source_has_no_sat_agent_or_concrete_channel_branches() -> None:
    source = (ROOT / "src" / "astrograph" / "semantics" / "channel_mapping.py").read_text(
        encoding="utf-8"
    ).lower()
    for marker in ("sat-agent", "battery_soc", "rw.speed", "payload_temp"):
        assert marker not in source


@pytest.mark.parametrize(
    ("mapping_id", "source", "expected_archetype", "expected_channels"),
    [
        (
            "sat-agent.storage.energy-buffer",
            pd.DataFrame({"time_s": [0.0, 5.0], "estimate.eps.battery_soc": [0.8, 0.79]}),
            "storage.energy_buffer",
            {"asset.eps.storage_0.state_of_charge"},
        ),
        (
            "sat-agent.actuator.controlled-effort",
            pd.DataFrame(
                {
                    "time_s": [0.0, 1.0],
                    "adcs.rw.command_torque_nm_0": [0.01, 0.02],
                    "estimate.adcs.rw.actual_torque_nm_0": [0.009, 0.019],
                    "sensor.adcs.rw.speed_rad_s_0": [100.0, 100.5],
                }
            ),
            "actuator.controlled_effort",
            {
                "asset.adcs.actuator_0.command_torque",
                "asset.adcs.actuator_0.response_torque",
                "asset.adcs.actuator_0.angular_rate",
            },
        ),
        (
            "sat-agent.thermal.lumped-capacity",
            pd.DataFrame({"time_s": [0.0, 30.0], "sensor.thermal.payload_temp_c": [20.0, 21.0]}),
            "thermal.lumped_capacity",
            {"asset.thermal.capacity_0.temperature"},
        ),
    ],
)
def test_sat_agent_maps_use_one_generic_mapper_without_source_branches(
    mapping_id: str,
    source: pd.DataFrame,
    expected_archetype: str,
    expected_channels: set[str],
) -> None:
    schema, _, maps = _load_contracts()
    result = ChannelMapper(
        semantic_schema=schema,
        channel_map=maps.channel_map(mapping_id),
    ).map_frame(source)

    assert {signal.component_archetype_id for signal in result.frame.signals} == {expected_archetype}
    assert {signal.semantic_channel_id for signal in result.frame.signals} == expected_channels
    assert result.frame.labels_used is False and result.frame.truth_used is False


def test_asset_topology_validates_archetype_ports_and_semantic_channels() -> None:
    _, archetypes, _ = _load_contracts()
    topology = AssetTopology(
        asset_instance_id="asset-under-test",
        asset_archetype_id="spacecraft.generic",
        components=[
            ComponentInstance(
                component_instance_id="storage-0",
                component_archetype_id="storage.energy_buffer",
                port_bindings=[
                    PortBinding(
                        port_role_id="state.energy_fraction",
                        semantic_channel_ids=["asset.eps.storage_0.state_of_charge"],
                    )
                ],
            ),
            ComponentInstance(
                component_instance_id="actuator-0",
                component_archetype_id="actuator.controlled_effort",
                port_bindings=[
                    PortBinding(
                        port_role_id="state.rate",
                        semantic_channel_ids=["asset.adcs.actuator_0.angular_rate"],
                    )
                ],
            ),
            ComponentInstance(
                component_instance_id="thermal-0",
                component_archetype_id="thermal.lumped_capacity",
                port_bindings=[
                    PortBinding(
                        port_role_id="state.temperature",
                        semantic_channel_ids=["asset.thermal.capacity_0.temperature"],
                    )
                ],
            ),
        ],
    )
    channels = {
        "asset.eps.storage_0.state_of_charge",
        "asset.adcs.actuator_0.angular_rate",
        "asset.thermal.capacity_0.temperature",
    }
    topology.validate_against(archetypes, known_semantic_channel_ids=channels)


def test_asset_topology_rejects_unknown_port_or_channel() -> None:
    _, archetypes, _ = _load_contracts()
    unknown_port = AssetTopology(
        asset_instance_id="asset-under-test",
        asset_archetype_id="spacecraft.generic",
        components=[
            ComponentInstance(
                component_instance_id="storage-0",
                component_archetype_id="storage.energy_buffer",
                port_bindings=[PortBinding(port_role_id="dataset.column", semantic_channel_ids=["x"])],
            )
        ],
    )
    with pytest.raises(KeyError, match="unknown port role"):
        unknown_port.validate_against(archetypes, known_semantic_channel_ids={"x"})

    unknown_channel = AssetTopology(
        asset_instance_id="asset-under-test",
        asset_archetype_id="spacecraft.generic",
        components=[
            ComponentInstance(
                component_instance_id="storage-0",
                component_archetype_id="storage.energy_buffer",
                port_bindings=[
                    PortBinding(port_role_id="state.energy_fraction", semantic_channel_ids=["missing"])
                ],
            )
        ],
    )
    with pytest.raises(ValueError, match="unknown semantic channels"):
        unknown_channel.validate_against(archetypes, known_semantic_channel_ids=set())
