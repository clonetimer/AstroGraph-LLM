"""B5: scenario-aware dual-path routing tests."""
from __future__ import annotations

from astrograph.orchestration.scenario_routing import RoutePolicy, ScenarioClass, ScenarioRouter


def test_critical_attitude_routes_fast_forced_human():
    router = ScenarioRouter(critical_keywords=('attitude', 'rw_jam'))
    route = router.route('ADCS_RW_JAM', subsystem='ADCS_ATTITUDE')
    assert route['scenario_class'] == ScenarioClass.CRITICAL.value
    assert route['route_policy'] == RoutePolicy.FAST_DETERMINISTIC_FORCE_HUMAN.value
    assert route['requires_human_gate'] is True
    assert route['depth'] == 'fast'


def test_standard_routes_deep_kg_llm():
    router = ScenarioRouter()
    route = router.route('EPS_BATTERY_CAPACITY_LOSS', subsystem='EPS')
    assert route['scenario_class'] == ScenarioClass.STANDARD.value
    assert route['route_policy'] == RoutePolicy.DEEP_KG_LLM.value
    assert route['requires_human_gate'] is False


def test_benign_routes_fast_rule_only():
    router = ScenarioRouter()
    route = router.route('NO_FAULT_NOMINAL', subsystem='ADCS')
    assert route['scenario_class'] == ScenarioClass.BENIGN.value
    assert route['route_policy'] == RoutePolicy.FAST_RULE_ONLY.value


def test_critical_thruster_recognized():
    router = ScenarioRouter(critical_keywords=('thruster',))
    assert router.classify('PROPULSION_THRUSTER_IGNITION_FAILURE', subsystem='PROPULSION') == ScenarioClass.CRITICAL


def test_custom_keywords():
    router = ScenarioRouter(critical_keywords=('battery',))
    assert router.classify('EPS_BATTERY_OPEN_CIRCUIT', subsystem='EPS') == ScenarioClass.CRITICAL
    assert router.classify('ADCS_RW_JAM', subsystem='ADCS') == ScenarioClass.STANDARD


def test_route_metadata_complete():
    router = ScenarioRouter()
    route = router.route('COMM_DATA_DOWNLINK_LINK_LOSS', subsystem='COMM')
    assert set(route) >= {'scenario_class', 'route_policy', 'requires_human_gate', 'depth', 'routed_at'}
