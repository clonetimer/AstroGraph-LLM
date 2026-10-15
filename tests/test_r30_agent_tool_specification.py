from astrograph.governance.agent_tool_specification import (
    AgentToolSurface, DiagnosisAuthority, EvidenceAuthority, TruthAccessPolicy,
    build_agent_tool_governance_catalog,
)


def test_r30_catalog_covers_all_current_governed_surfaces():
    c=build_agent_tool_governance_catalog()
    assert len(c.tools) >= 46
    surfaces={x.surface for x in c.tools}
    assert AgentToolSurface.MISSION_PLATFORM in surfaces
    assert AgentToolSurface.CONTROLLED_DIAGNOSTIC_CAPABILITY in surfaces
    assert AgentToolSurface.DIAGNOSTIC_PROBE in surfaces
    assert AgentToolSurface.LOW_LEVEL_AGENTIC_REGISTRY in surfaces
    assert c.global_invariants['agent_release_authority'] is False
    assert c.global_invariants['arbitrary_python'] is False
    assert c.global_invariants['core_detection_must_not_depend_on_optional_agent_probe'] is True


def test_all_probe_kinds_are_shadow_host_bound_and_non_authoritative():
    c=build_agent_tool_governance_catalog()
    probes=[x for x in c.tools if x.surface==AgentToolSurface.DIAGNOSTIC_PROBE]
    assert len(probes) == 12
    for p in probes:
        assert p.agent_selectable
        assert p.evidence_authority==EvidenceAuthority.SHADOW_DERIVED_OBSERVATION_ONLY
        assert p.diagnosis_authority==DiagnosisAuthority.SHADOW_REVIEW_ONLY
        assert not p.core_prediction_depends_on_agent_call
        assert not p.formal_release_authority
        assert not p.arbitrary_channel_allowed
        assert not p.arbitrary_window_allowed
        assert not p.arbitrary_threshold_allowed


def test_truth_aware_replay_is_offline_only_and_diagnosis_core_stays_deterministic():
    c=build_agent_tool_governance_catalog()
    by={x.tool_key:x for x in c.tools}
    replay=by['mission:dataset.replay']
    assert replay.truth_access_policy==TruthAccessPolicy.OFFLINE_EVALUATION_ONLY
    fuse=by['mission:diagnosis.fuse']
    assert fuse.diagnosis_authority==DiagnosisAuthority.DETERMINISTIC_CORE_EXECUTION
    assert fuse.core_prediction_depends_on_agent_call is False


def test_r29_review_is_catalogued_without_promoting_formal_evidence():
    c=build_agent_tool_governance_catalog()
    row=next(x for x in c.tools if x.tool_key=='probe:DXC_REDUCED_ORDER_SIMULATION_REVIEW')
    assert row.evidence_authority==EvidenceAuthority.SHADOW_DERIVED_OBSERVATION_ONLY
    assert row.diagnosis_authority==DiagnosisAuthority.SHADOW_REVIEW_ONLY


def test_r30_read_only_platform_endpoint_exposes_governance_catalog():
    from fastapi.testclient import TestClient
    from astrograph.api import create_app
    data=TestClient(create_app()).get('/v1/platform/agent-tool-specification').json()
    assert data['schema_version']=='astrograph-agent-tool-governance-catalog.v1'
    assert data['global_invariants']['agent_release_authority'] is False
    assert any(x['runtime_tool_id']=='DXC_REDUCED_ORDER_SIMULATION_REVIEW' for x in data['tools'])


def test_r31_pump_review_is_catalogued_without_promoting_formal_evidence():
    c=build_agent_tool_governance_catalog()
    row=next(x for x in c.tools if x.tool_key=='probe:DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW')
    assert row.evidence_authority==EvidenceAuthority.SHADOW_DERIVED_OBSERVATION_ONLY
    assert row.diagnosis_authority==DiagnosisAuthority.SHADOW_REVIEW_ONLY
    assert row.core_prediction_depends_on_agent_call is False
