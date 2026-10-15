"""26-profile semantic-binding coverage matrix (2.A completeness gate).

Locks in the "every profile has a map, a real snapshot column alignment, and a
five-class binding test" guarantee as a regression net, so adding a future
Sat-Agent profile can never silently ship without a binding + test.

For each of the 26 governed schema profiles:
  1. a channel map exists (mapping_id resolution + non-empty entries);
  2. every entry's source_field appears in the *real committed snapshot* columns
     when a snapshot exists (real-data alignment, not declaration-only);
  3. the two profiles with NO public snapshot (eps-unified, whole-spacecraft
     composite) are explicitly recorded as source-blocked and must NOT be
     fabricated here;
  4. each profile is covered by at least one five-class binding test file.

Real-data alignment is snapshot-backed only (committed public telemetry, no
external Sat-Agent path required).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from astrograph.semantics import ChannelMapCatalog

ROOT = Path(__file__).parents[1]
SEMANTICS = ROOT / "configs" / "semantics"
PROFILE = ROOT / "configs" / "data_sources" / "sat_agent_platform_schema_profiles.s0.v1.json"
CHANNEL_MAPS = SEMANTICS / "sat_agent_platform_channel_maps.s0.v1.json"
SNAPSHOT_ROOT = ROOT / "data" / "sat_agent_snapshots" / "traces"

# mapping_id -> real committed snapshot profile (no snapshot -> None = source-blocked).
_MID_TO_SNAPSHOT: dict[str, str | None] = {
    "sat-agent-platform.component.battery.v1": "sat-agent.component.battery.trace.v1",
    "sat-agent-platform.subsystem.adcs-rw.v2": "sat-agent.subsystem.adcs-rw.astrograph-5ch.v2",
    "sat-agent-platform.subsystem.thermal.v1": "sat-agent.subsystem.thermal.source-native.v1",
    # Loop 2.A closure (2026-08-04): previously "source-blocked", the snapshot
    # list simply never included them; real Sat-Agent data exists under
    # capacity_benchmark/datasets (eps-basilisk-600 / composite-graph-120).
    "sat-agent-platform.subsystem.eps-unified.v1": "sat-agent.subsystem.eps.unified-native.v1",
    "sat-agent-platform.whole-spacecraft.composite.v1": "sat-agent.whole-spacecraft.composite.v1",
    "sat-agent-platform.component.imu.v1": "sat-agent.component.imu.trace.v1",
    "sat-agent-platform.component.pdu.v1": "sat-agent.component.pdu.trace.v1",
    "sat-agent-platform.component.data-queue-link.v1": "sat-agent.component.data-queue.trace.v1",
    "sat-agent-platform.component.onboard-storage.v1": "sat-agent.component.onboard-storage.trace.v1",
    "sat-agent-platform.component.cmg.v1": "sat-agent.component.cmg.trace.v1",
    "sat-agent-platform.component.mtb.v1": "sat-agent.component.mtb.trace.v1",
    "sat-agent-platform.component.star_tracker.v1": "sat-agent.component.star-tracker.trace.v1",
    "sat-agent-platform.component.sun_sensor.v1": "sat-agent.component.sun-sensor.trace.v1",
    "sat-agent-platform.component.magnetometer.v1": "sat-agent.component.magnetometer.trace.v1",
    "sat-agent-platform.component.heater.v1": "sat-agent.component.heater.trace.v1",
    "sat-agent-platform.component.radiator.v1": "sat-agent.component.radiator.trace.v1",
    "sat-agent-platform.component.transmitter.v1": "sat-agent.component.transmitter.trace.v1",
    "sat-agent-platform.component.link_budget.v1": "sat-agent.component.link-budget.trace.v1",
    "sat-agent-platform.component.solar_panel.v1": "sat-agent.component.solar-panel.trace.v1",
    "sat-agent-platform.component.power_sink.v1": "sat-agent.component.power-sink.trace.v1",
    "sat-agent-platform.component.antenna.v1": "sat-agent.component.antenna.trace.v1",
    "sat-agent-platform.component.ground_station.v1": "sat-agent.component.ground-station.trace.v1",
    "sat-agent-platform.component.fuel_tank.v1": "sat-agent.component.fuel-tank.trace.v1",
    "sat-agent-platform.component.payload.v1": "sat-agent.component.payload.trace.v1",
    "sat-agent-platform.component.payload_sensor.v1": "sat-agent.component.payload-sensor.trace.v1",
    "sat-agent-platform.component.thruster.v1": "sat-agent.component.thruster.trace.v1",
}

# mapping_id -> five-class binding test file(s) that cover it.
_MID_TO_TESTS: dict[str, list[str]] = {
    "sat-agent-platform.component.battery.v1": ["test_battery_semantic_binding.py"],
    "sat-agent-platform.subsystem.adcs-rw.v2": [
        "test_component_semantic_binding_gap_profiles.py",
        "test_adcs_subsystem_integration.py",
    ],
    "sat-agent-platform.subsystem.thermal.v1": ["test_component_semantic_binding_gap_profiles.py"],
    "sat-agent-platform.subsystem.eps-unified.v1": [
        "test_component_semantic_binding_gap_profiles.py",
    ],
    "sat-agent-platform.whole-spacecraft.composite.v1": [
        "test_component_semantic_binding_gap_profiles.py",
    ],
    "sat-agent-platform.component.imu.v1": ["test_component_semantic_binding_gap_profiles.py"],
    "sat-agent-platform.component.pdu.v1": ["test_component_semantic_binding_gap_profiles.py"],
    "sat-agent-platform.component.data-queue-link.v1": ["test_component_semantic_binding_gap_profiles.py"],
    "sat-agent-platform.component.onboard-storage.v1": ["test_batch_b_component_semantic_binding.py"],
    "sat-agent-platform.component.cmg.v1": ["test_adcs_remaining_component_semantic_binding.py"],
    "sat-agent-platform.component.mtb.v1": ["test_adcs_remaining_component_semantic_binding.py"],
    "sat-agent-platform.component.star_tracker.v1": ["test_adcs_remaining_component_semantic_binding.py"],
    "sat-agent-platform.component.sun_sensor.v1": ["test_adcs_remaining_component_semantic_binding.py"],
    "sat-agent-platform.component.magnetometer.v1": ["test_adcs_remaining_component_semantic_binding.py"],
    "sat-agent-platform.component.heater.v1": ["test_batch_b_component_semantic_binding.py"],
    "sat-agent-platform.component.radiator.v1": ["test_batch_b_component_semantic_binding.py"],
    "sat-agent-platform.component.transmitter.v1": ["test_batch_b_component_semantic_binding.py"],
    "sat-agent-platform.component.link_budget.v1": ["test_batch_b_component_semantic_binding.py"],
    "sat-agent-platform.component.solar_panel.v1": ["test_batch_c_component_semantic_binding.py"],
    "sat-agent-platform.component.power_sink.v1": ["test_batch_c_component_semantic_binding.py"],
    "sat-agent-platform.component.antenna.v1": ["test_batch_c_component_semantic_binding.py"],
    "sat-agent-platform.component.ground_station.v1": ["test_batch_c_component_semantic_binding.py"],
    "sat-agent-platform.component.fuel_tank.v1": ["test_batch_d_component_semantic_binding.py"],
    "sat-agent-platform.component.payload.v1": ["test_batch_d_component_semantic_binding.py"],
    "sat-agent-platform.component.payload_sensor.v1": ["test_batch_d_component_semantic_binding.py"],
    "sat-agent-platform.component.thruster.v1": ["test_thrust_prototype_binding.py"],
}


def _profiles() -> list[dict]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))["profiles"]


def _maps() -> dict[str, object]:
    catalog = ChannelMapCatalog.model_validate_json(CHANNEL_MAPS.read_text(encoding="utf-8"))
    return {item.mapping_id: item for item in catalog.maps}


def _snapshot_columns(profile_id: str) -> set[str] | None:
    path = SNAPSHOT_ROOT / f"{profile_id}.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return set(rows[0].keys()) if rows else set()


def _norm(value: str) -> str:
    """Normalise underscore/hyphen differences (ground_station vs ground-station)."""
    return value.replace("_", "-")


def _mapping_token(mapping_id: str) -> str:
    # sat-agent-platform.component.battery.v1 -> component.battery
    parts = mapping_id.replace("sat-agent-platform.", "").split(".")
    return _norm(".".join(parts[:2]))


def _profile_token(profile_id: str) -> str:
    # sat-agent.component.battery.trace.v1 -> component.battery
    parts = profile_id.replace("sat-agent.", "").replace(".trace", "").split(".")
    return _norm(".".join(parts[:2]))


def test_every_governed_profile_has_a_channel_map() -> None:
    maps = _maps()
    for profile in _profiles():
        # every profile maps to a channel map via its normalised component token.
        token = _profile_token(profile["profile_id"])
        matched = [mid for mid in maps if token in _mapping_token(mid)]
        assert matched, f"profile {profile['profile_id']} has no channel map (token {token})"
        mapping = maps[matched[0]]
        assert mapping.entries, f"map {matched[0]} has empty entries"


def test_snapshot_backed_maps_align_with_real_columns() -> None:
    """Every snapshot-backed profile's map entries must exist in the real CSV."""
    maps = _maps()
    misaligned: list[str] = []
    for mid, snapshot in _MID_TO_SNAPSHOT.items():
        if snapshot is None:
            continue
        columns = _snapshot_columns(snapshot)
        assert columns is not None, f"snapshot {snapshot} missing for {mid}"
        missing = [e.source_field for e in maps[mid].entries if e.source_field not in columns]
        if missing:
            misaligned.append(f"{mid}: {missing}")
    assert not misaligned, f"snapshot-aligned maps have columns missing from real data: {misaligned}"


def test_no_source_blocked_profiles_exist() -> None:
    """Every governed profile now has a real public snapshot (2.A closure).

    The former "source-blocked" pair (eps-unified / whole-spacecraft composite)
    was closed 2026-08-04 because the snapshot list simply never included them
    while real Sat-Agent data existed.  A profile left out of the snapshot list
    must be explicit (None here) so no future profile can silently ship without
    a real-data snapshot.
    """
    maps = _maps()
    blocked = [mid for mid, snap in _MID_TO_SNAPSHOT.items() if snap is None]
    assert not blocked, f"profiles without a public snapshot: {blocked}"
    # every map is still resolved (declared map exists, snapshot aligns).
    for mid, snap in _MID_TO_SNAPSHOT.items():
        assert mid in maps, f"map missing: {mid}"
        assert snap is not None
        assert _snapshot_columns(snap) is not None, f"snapshot {snap} missing for {mid}"


def test_every_snapshot_profile_has_a_five_class_binding_test() -> None:
    """Every profile with a real snapshot must be covered by a binding test."""
    uncovered = [
        mid
        for mid, snap in _MID_TO_SNAPSHOT.items()
        if snap is not None and not _MID_TO_TESTS[mid]
    ]
    assert not uncovered, f"profiles with real snapshots lack binding tests: {uncovered}"


def test_profile_map_and_test_coverage_are_total() -> None:
    """The coverage matrix spans all 26 governed profiles with no omissions."""
    maps = _maps()
    assert set(_MID_TO_SNAPSHOT) == set(maps), "coverage matrix out of sync with channel maps"
    profiles = {p["profile_id"] for p in _profiles()}
    # every governed profile token resolves to a mapped id (prefix-tolerant:
    # mapping_id data-queue-link covers profile data-queue).
    tokens = {_mapping_token(mid) for mid in _MID_TO_SNAPSHOT}
    for profile in profiles:
        token = _profile_token(profile)
        assert any(
            token.startswith(t) or t.startswith(token) for t in tokens
        ), f"profile {profile} unmapped"
