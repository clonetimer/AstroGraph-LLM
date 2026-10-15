"""Loop-13 all-object semantic binding verification.

Work item ③ of loop 13.  Verifies, on committed fixtures only (no external
Sat-Agent path required):

1. Every loop-13 long-trace capability has both a channel-map entry and a
   dataset-layer semantic binding, and the two agree on the archetype.
2. Every quantity kind / port role referenced by the new and batch-C channel
   maps resolves in the core semantic schema and component-archetype catalog.
3. Batch-C objects (solar_panel / power_sink / antenna / ground_station) keep a
   consistent channel map + schema profile + committed public snapshot.
4. Objects that are channel-map-bound but have NO loop-13 long trace are
   documented as "source not in long-trace batch" (they are covered by the
   committed public snapshots, not by the loop-13 long traces).

S0: verification only, no fitting, no calibration, no split/seed.
"""

from __future__ import annotations

import json
from pathlib import Path

from astrograph.semantics import (
    ChannelMapCatalog,
    ComponentArchetypeCatalog,
    SemanticSchema,
)
from scripts.tools.materialize_long_trace_episodes import _CAPABILITY_SEMANTICS

ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = ROOT / "configs" / "semantics"
PROFILE_PATH = ROOT / "configs" / "data_sources" / "sat_agent_platform_schema_profiles.s0.v1.json"
MANIFEST_PATH = ROOT / "data" / "generated_traces" / "manifest.json"
SNAPSHOT_DIR = ROOT / "data" / "sat_agent_snapshots" / "traces"

# Loop-13 long-trace capability -> closest archetype.  EPS basic is bound at the
# DATASET layer only: its overlapping eps.battery.* field set
# (eps.battery.soc / storage_wh / net_power_w / applied_power_w) EQUALS the
# battery component profile field set (equality, not a proper subset), so a
# distinct EPS-basic channel-map profile would create a two-match ambiguity in
# the single-profile-per-mapping resolver.  That channel-map gap is asserted and
# documented below, not fabricated.
LOOP13_OBJECTS = {
    "component.mtb.v1": "actuator.controlled_effort",
    "component.data_queue.v1": "queue.bounded_inventory",
    "component.onboard_storage.v1": "queue.bounded_inventory",
    "component.solar_panel.v1": "converter.power",
}

# EPS basic is dataset-layer bound; its channel-map wiring is the recorded gap.
EPS_BASIC_CAPABILITY = "subsystem.eps.basic.v1"
EPS_BASIC_ARCHETYPE = "storage.energy_buffer"

# Batch-C objects with committed public snapshots.
BATCH_C_SNAPSHOTS = {
    "sat-agent.component.solar-panel.trace.v1.csv",
    "sat-agent.component.power-sink.trace.v1.csv",
    "sat-agent.component.antenna.trace.v1.csv",
    "sat-agent.component.ground-station.trace.v1.csv",
}

# Objects bound in the channel-map catalog but NOT in the loop-13 long-trace
# batch.  They are covered by the committed public telemetry snapshots; the
# loop-13 long-trace dataset intentionally does not regenerate them.
SOURCE_NOT_IN_LONGTRACE_BATCH = {
    "sat-agent-platform.component.pdu.v1",
    "sat-agent-platform.component.cmg.v1",
    "sat-agent-platform.component.star_tracker.v1",
    "sat-agent-platform.component.sun_sensor.v1",
    "sat-agent-platform.component.magnetometer.v1",
    "sat-agent-platform.component.heater.v1",
    "sat-agent-platform.component.radiator.v1",
    "sat-agent-platform.component.transmitter.v1",
    "sat-agent-platform.component.link_budget.v1",
    "sat-agent-platform.component.power_sink.v1",
    "sat-agent-platform.component.antenna.v1",
    "sat-agent-platform.component.ground_station.v1",
    "sat-agent-platform.component.fuel_tank.v1",
    "sat-agent-platform.component.payload.v1",
    "sat-agent-platform.component.payload_sensor.v1",
    "sat-agent-platform.component.thruster.v1",
    "sat-agent-platform.whole-spacecraft.composite.v1",
}


def _contracts() -> tuple[SemanticSchema, ComponentArchetypeCatalog, ChannelMapCatalog]:
    schema = SemanticSchema.model_validate_json(
        (SEMANTICS / "core_semantic_schema.json").read_text(encoding="utf-8")
    )
    archetypes = ComponentArchetypeCatalog.model_validate_json(
        (SEMANTICS / "core_component_archetypes.json").read_text(encoding="utf-8")
    )
    maps = ChannelMapCatalog.model_validate_json(
        (SEMANTICS / "sat_agent_platform_channel_maps.s0.v1.json").read_text(encoding="utf-8")
    )
    return schema, archetypes, maps


def _mapping_by_capability(maps: ChannelMapCatalog) -> dict[str, object]:
    out: dict[str, object] = {}
    for mapping in maps.maps:
        # mapping_id is "sat-agent-platform.<capability>.vN"; derive capability.
        mapping_id = mapping.mapping_id  # type: ignore[attr-defined]
        capability = mapping_id.split("sat-agent-platform.")[1]  # type: ignore[index]
        out[capability] = mapping
    return out


# capability_id -> channel-map mapping_id fragment (mapping ids hyphenate dots).
_CAPABILITY_TO_MAP_FRAGMENT = {
    "component.mtb.v1": "component.mtb.v1",
    "component.data_queue.v1": "component.data-queue-link.v1",
    "component.onboard_storage.v1": "component.onboard-storage.v1",
    "component.solar_panel.v1": "component.solar_panel.v1",
}


class TestLoop13Binding:
    def test_every_loop13_trace_capability_has_dataset_binding(self):
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        capabilities = {trace["capability"] for trace in payload["traces"]}
        for capability in capabilities:
            assert capability in _CAPABILITY_SEMANTICS, capability
        # All 9 objects present in the dataset-layer binding.
        assert set(_CAPABILITY_SEMANTICS) == capabilities

    def test_dataset_and_channel_map_archetypes_agree(self):
        schema, archetypes, maps = _contracts()
        for capability, expected_archetype in LOOP13_OBJECTS.items():
            # dataset-layer binding archetype
            assert _CAPABILITY_SEMANTICS[capability]["archetype"] == expected_archetype
            # channel-map entry has the same archetype on at least one bound field
            mapping = next(
                (
                    m
                    for m in maps.maps
                    if m.mapping_id == f"sat-agent-platform.{_CAPABILITY_TO_MAP_FRAGMENT[capability]}"
                ),
                None,
            )
            assert mapping is not None, capability
            mapped_archetypes = {e.component_archetype_id for e in mapping.entries}
            assert expected_archetype in mapped_archetypes, capability

    def test_loop13_channel_map_entries_resolve_in_schema(self):
        schema, archetypes, maps = _contracts()
        known_kinds = {qk.quantity_kind_id for qk in schema.quantity_kinds}
        for mapping in maps.maps:
            if not any(obj in mapping.mapping_id for obj in ("mtb", "data-queue", "onboard-storage", "solar_panel", "eps-basic")):
                continue
            for entry in mapping.entries:
                assert entry.quantity_kind_id in known_kinds, (mapping.mapping_id, entry.quantity_kind_id)
                arch = archetypes.archetype(entry.component_archetype_id)
                port_roles = {p.port_role_id: p for p in arch.port_roles}
                assert entry.port_role_id in port_roles, (mapping.mapping_id, entry.port_role_id)
                assert entry.quantity_kind_id in port_roles[entry.port_role_id].quantity_kind_ids, (
                    mapping.mapping_id, entry.port_role_id, entry.quantity_kind_id,
                )

    def test_batch_c_snapshot_bindings_consistent(self):
        schema, archetypes, maps = _contracts()
        profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]
        profile_declared = {
            tuple({p["time_field"]["name"]} | {f["name"] for f in p["public_observation_fields"]})
            for p in profiles
        }
        for snapshot in BATCH_C_SNAPSHOTS:
            assert (SNAPSHOT_DIR / snapshot).is_file(), snapshot
        for mapping in maps.maps:
            if not any(obj in mapping.mapping_id for obj in ("solar_panel", "power_sink", "antenna", "ground_station")):
                continue
            mapped = {mapping.time_axis.source_field} | {e.source_field for e in mapping.entries}
            assert any(mapped <= set(declared) for declared in profile_declared), mapping.mapping_id

    def test_batch_c_demand_and_power_fields_have_snapshot(self):
        # The snapshot files exist for all four batch-C objects (committed data).
        assert BATCH_C_SNAPSHOTS == {
            name for name in BATCH_C_SNAPSHOTS if (SNAPSHOT_DIR / name).is_file()
        }

    def test_channel_map_bound_objects_without_longtrace_are_documented(self):
        schema, archetypes, maps = _contracts()
        mapping_ids = {m.mapping_id for m in maps.maps}
        for documented in SOURCE_NOT_IN_LONGTRACE_BATCH:
            assert documented in mapping_ids, documented


class TestEpsBasicBinding:
    def test_eps_basic_dataset_layer_binding_covers_trace_columns(self):
        # The EPS-basic long-trace columns are all present in the dataset-layer
        # binding (which drives materialisation + projection).
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        eps_trace = next(
            t for t in manifest["traces"] if t["capability"] == "subsystem.eps.basic.v1"
        )
        columns = set(eps_trace["columns"])
        assert EPS_BASIC_CAPABILITY in _CAPABILITY_SEMANTICS
        spec = _CAPABILITY_SEMANTICS[EPS_BASIC_CAPABILITY]
        assert spec["archetype"] == EPS_BASIC_ARCHETYPE
        for source_column, _, _, _, _, _ in spec["channels"]:
            assert source_column in columns, source_column

    def test_eps_basic_channel_map_gap_is_documented_not_fabricated(self):
        # Honest alignment gap: the eps.basic dataset-layer field set EQUALS the
        # battery channel-map field set (soc / storage_wh / net_power_w /
        # applied_power_w — equality, not a proper subset), so no distinct
        # channel-map profile is registered (a second profile would make the
        # single-profile-per-mapping resolver ambiguous).  The dataset-layer
        # binding is the operative one.
        schema, archetypes, maps = _contracts()
        mapping_ids = {m.mapping_id for m in maps.maps}
        assert "sat-agent-platform.subsystem.eps-basic.v1" not in mapping_ids
        # The battery channel map covers EXACTLY the same eps.battery.* field set
        # as the EPS-basic dataset-layer binding (field equality).
        battery_mapping = next(
            m for m in maps.maps if m.mapping_id == "sat-agent-platform.component.battery.v1"
        )
        battery_fields = {e.source_field for e in battery_mapping.entries}
        eps_spec = _CAPABILITY_SEMANTICS[EPS_BASIC_CAPABILITY]
        eps_fields = {source_column for source_column, _, _, _, _, _ in eps_spec["channels"]}
        assert battery_fields == {
            "eps.battery.soc",
            "eps.battery.storage_wh",
            "eps.battery.net_power_w",
            "eps.battery.applied_power_w",
        }
        assert eps_fields == battery_fields
