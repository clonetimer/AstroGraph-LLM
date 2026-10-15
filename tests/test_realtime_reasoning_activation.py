from astrograph.evidence.executable import (
    ComparisonOperator, EpisodeEvidenceBundle, EpisodeWindow, EvidenceFact,
    EvidenceFactStatus, EvidenceRole,
)
from astrograph.orchestration.realtime_reasoning import (
    RealtimeReasoningActivationPolicy,
    decide_reasoning_activation,
)


def _bundle():
    episode = EpisodeWindow(
        episode_id="EP-1", run_id="run", paired_nominal_run_id="nominal",
        start_s=0, end_s=1, source="test",
    )
    fact = EvidenceFact(
        evidence_fact_id="EF-1", evidence_definition_id="ED-1", target_fault_id="FAULT-1",
        feature_result_id="FR-1", episode_id="EP-1", role=EvidenceRole.REQUIRED,
        status=EvidenceFactStatus.SATISFIED, observed_value=2, threshold_value=1,
        unit="1", comparison=ComparisonOperator.GREATER_THAN,
    )
    return EpisodeEvidenceBundle(
        episode=episode, analysis_component_instance="component:test",
        feature_definitions=[], feature_results=[], evidence_definitions=[], evidence_facts=[fact],
    )


def test_algorithm_event_activates_llm_advisor_without_control_authority():
    decision = decide_reasoning_activation(
        request_mode="hybrid", policy=RealtimeReasoningActivationPolicy(), bundle=_bundle(),
        deterministic_core_result={"diagnosis_status": "DIAGNOSED"},
        candidate_fusion_available=False, operator_requested=False,
    )
    assert decision.llm_invoked is True
    assert decision.deterministic_path_completed_first is True
    assert decision.hard_realtime_path_blocked_by_llm is False
    assert decision.llm_has_direct_algorithm_control is False


def test_rule_and_never_modes_do_not_invoke_llm():
    bundle = _bundle()
    rule = decide_reasoning_activation(
        request_mode="rule", policy=RealtimeReasoningActivationPolicy(mode="ALWAYS"), bundle=bundle,
        deterministic_core_result={"diagnosis_status": "DIAGNOSED"},
        candidate_fusion_available=True, operator_requested=True,
    )
    disabled = decide_reasoning_activation(
        request_mode="llm", policy=RealtimeReasoningActivationPolicy(mode="NEVER"), bundle=bundle,
        deterministic_core_result={"diagnosis_status": "DIAGNOSED"},
        candidate_fusion_available=True, operator_requested=True,
    )
    assert rule.llm_invoked is disabled.llm_invoked is False
