"""R11 LLM/tool boundary tests. Synthetic non-spaceflight knowledge only."""
from __future__ import annotations

import pytest

from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService,
    PlatformKnowledgeExplanation,
)
from astrograph.orchestration.models import PermissionScope, WorkflowContext
from astrograph.tools.material_review import (
    TOOL_IDS, MaterialToolPlan, build_material_review_tool_registry,
    execute_material_tool_plan, summarize_material_causal_review,
)


def sample_context():
    return {
        "competing_hypothesis_review": {"status": "COMPARE_ONLY", "shared_symptoms": ["temperature rise"]},
        "candidates": [
            {"fault_id": "PUMP_FAULT_A", "reference_retrieval": {"status": "SUCCEEDED", "hits": [
                {"knowledge_id": "K1", "quote": "Lab-only trace.", "source_sha256": "f" * 64,
                 "knowledge_validation_status": "EXPERT_APPROVED_HISTORICAL_REFERENCE"}]},
             "multi_hop": {"causal_chain_review": {
                 "chains": [{"path_id": "R7_PATH_1", "hop_count": 2, "terminal_symptom_id": "S1",
                             "joint_applicability_status": "REVIEWED_SEQUENTIAL_CONSTRAINTS_COMPATIBLE_FLIGHT_UNVERIFIED",
                             "reviewed_context_consistency": {"status": "CONFLICT", "conflict_reasons": ["OPERATING_MODES_DISJOINT"]},
                             "reviewed_temporal_consistency": {
                                 "status": "REVIEWED_SEQUENCE_COMPATIBLE_NOT_FLIGHT_VERIFIED",
                                 "mode": "REVIEWED_STAGES", "cumulative_lag_seconds": {"min": 8, "max": 16},
                                 "conflict_reasons": []},
                             "hops": [{"edge_id": "E1", "source_name": "wear", "target_name": "heat",
                                       "evidence_quote": "Wear can cause heat.", "direction": "FORWARD",
                                       "source_sha256": "a"*64, "source_locators": ["line:3"]}] }],
                 "blocked": [{"edge_id": "E2", "reasons": ["TEMPORAL_STATE_TRANSITION_INCOMPATIBLE"],
                              "path_edge_ids": ["E1", "E2"]}],
             }}},
            {"fault_id": "PUMP_FAULT_B", "multi_hop": {"causal_chain_review": {
                "chains": [], "blocked": []}}},
        ],
    }


def workflow_context(*permissions):
    return WorkflowContext(workflow_id="unit", run_id="r11-test", permissions=list(permissions))


def test_material_tools_are_sealed_task_bound_read_only_and_permission_checked():
    registry = build_material_review_tool_registry(sample_context())
    assert {x.tool_id for x in registry.list_specs()} == set(TOOL_IDS)
    assert all(x.deterministic and x.side_effect_level.value == "READ_ONLY" for x in registry.list_specs())
    with pytest.raises(PermissionError):
        registry.invoke(TOOL_IDS[0], {}, workflow_context())
    with pytest.raises(RuntimeError):
        registry.register(registry.get(TOOL_IDS[0]))
    authorized = workflow_context(PermissionScope.READ_KNOWLEDGE)
    with pytest.raises(ValueError, match="FAULT_ID_NOT_IN_FROZEN_CANDIDATES"):
        registry.invoke(TOOL_IDS[1], {"fault_id": "UNKNOWN_FAULT"}, authorized)
    with pytest.raises(ValueError, match="PATH_ID_NOT_IN_FROZEN_REVIEW"):
        registry.invoke(TOOL_IDS[2], {"fault_id": "PUMP_FAULT_B", "path_id": "R7_PATH_1"}, authorized)
    with pytest.raises(Exception):
        registry.invoke(TOOL_IDS[2], {"fault_id": "PUMP_FAULT_A", "path_id": "R7_PATH_1",
                                      "graph_snapshot": "forged"}, authorized)
    assert registry.invoke(TOOL_IDS[3], {"fault_id": "PUMP_FAULT_A", "path_id": "R7_PATH_1"},
                           authorized)["actual_telemetry_event_alignment_verified"] is False
    assert registry.invoke(TOOL_IDS[2], {"fault_id": "PUMP_FAULT_A", "path_id": "R7_PATH_1"},
                           authorized)["observed_flight_conditions_verified"] is False


def test_material_review_immutable_snapshot_and_context_block_reasons():
    material = sample_context()
    registry = build_material_review_tool_registry(material)
    material["candidates"][0]["multi_hop"]["causal_chain_review"]["chains"].clear()
    authorized = workflow_context(PermissionScope.READ_KNOWLEDGE)
    rows = registry.invoke(TOOL_IDS[1], {"fault_id": "PUMP_FAULT_A"}, authorized)
    assert rows["chains"][0]["path_id"] == "R7_PATH_1"
    assert rows["blocked"][0]["reasons"] == ["TEMPORAL_STATE_TRANSITION_INCOMPATIBLE"]
    assert rows["diagnostic_vote_allowed"] is False
    summary = summarize_material_causal_review(sample_context())
    assert summary["candidates"][0]["chains"][0]["reviewed_context_consistency"]["conflict_reasons"]
    assert summary["candidates"][0]["blocked"][0]["edge_id"] == "E2"
    assert summary["current_root_cause_proven"] is False


class TwoPhaseSpy:
    def __init__(self, actions):
        self.actions = actions
        self.payloads = []
        self.last_call_metadata = {}

    def generate_structured(self, *, agent_id, output_model, payload):
        self.payloads.append((agent_id, payload))
        if agent_id == "platform-knowledge-material-tool-selector":
            return MaterialToolPlan.model_validate({"actions": self.actions})
        return PlatformKnowledgeExplanation(candidate_fault_id=None, abstained=True,
                                            summary="Independent confirmation is unavailable.", claims=[])


def test_model_chooses_whitelisted_tools_with_denied_foreign_candidate():
    spy = TwoPhaseSpy([
        {"tool_id": TOOL_IDS[1], "fault_id": "PUMP_FAULT_A"},
        {"tool_id": TOOL_IDS[2], "fault_id": "PUMP_FAULT_B", "path_id": "R7_PATH_1"},
        {"tool_id": TOOL_IDS[3], "fault_id": "PUMP_FAULT_A", "path_id": "R7_PATH_1"},
    ])
    result = execute_material_tool_plan(backend=spy, material_review=sample_context())
    assert [r["status"] for r in result["calls"]] == ["SUCCEEDED", "DENIED", "SUCCEEDED"]
    assert result["calls"][-1]["output"]["reviewed_temporal_consistency"]["cumulative_lag_seconds"]["min"] == 8
    assert result["current_root_cause_proven"] is False
    assert result["receipts"][0]["status"] == "SUCCEEDED"
    selection_payload = spy.payloads[0][1]
    assert not any("source_path" in str(k) for k in selection_payload)


def test_material_llm_enrichment_passes_verdicts_and_tool_results(tmp_path):
    spy = TwoPhaseSpy([{"tool_id": TOOL_IDS[2], "fault_id": "PUMP_FAULT_A", "path_id": "R7_PATH_1"}])
    service = PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(a2o_enabled=False, material_tool_calls_enabled=True), llm_backend=spy)
    output = service.enrich(task="Compare frozen pump candidates.",
                            diagnosis_snapshot={"candidate_fault_id": None, "evidence": []},
                            document_paths=[], output_root=tmp_path / "knowledge",
                            material_review_context=sample_context())
    assert [agent for agent, _ in spy.payloads] == [
        "platform-knowledge-material-tool-selector", "platform-knowledge-explanation"]
    payload = spy.payloads[-1][1]["material_reasoning"]
    assert payload["reviewed_causal_chains"]["candidates"][0]["blocked"][0]["reasons"]
    assert payload["tool_investigation"]["calls"][0]["status"] == "SUCCEEDED"
    assert payload["tool_investigation"]["calls"][0]["output"]["observed_flight_conditions_verified"] is False
    assert output["generation"]["status"] == "SUCCEEDED"


def test_material_llm_legacy_single_call_when_opt_out(tmp_path):
    spy = TwoPhaseSpy([])
    service = PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(a2o_enabled=False, material_tool_calls_enabled=False), llm_backend=spy)
    service.enrich(task="Compare frozen pump candidates.",
                   diagnosis_snapshot={"candidate_fault_id": None, "evidence": []},
                   document_paths=[], output_root=tmp_path / "knowledge",
                   material_review_context=sample_context())
    assert [agent for agent, _ in spy.payloads] == ["platform-knowledge-explanation"]
    assert spy.payloads[-1][1]["material_reasoning"]["tool_investigation"]["status"] == "DISABLED"


def test_public_guard_only_releases_deterministic_verdicts():
    from astrograph.integrations.material_multihop import apply_material_answer_guard
    reasoning = sample_context()
    reasoning["answer_policy"] = {"all_candidates_abstained": True}
    knowledge = {"generation": {"status": "SUCCEEDED", "citation_reference_validation": "PASS",
                                "explanation": {"summary": "Model invents causal diagnosis",
                                                "candidate_fault_id": "PUMP_FAULT_A",
                                                "abstained": False, "claims": []}}}
    guarded = apply_material_answer_guard(knowledge, reasoning)
    rows = guarded["reviewed_causal_explanations"]["candidates"]
    assert rows[0]["blocked_reference_paths"][0]["reason_codes"] == ["TEMPORAL_STATE_TRANSITION_INCOMPATIBLE"]
    assert rows[0]["approved_reference_paths"][0]["temporal_status"] == "REVIEWED_SEQUENCE_COMPATIBLE_NOT_FLIGHT_VERIFIED"
    assert guarded["reviewed_causal_explanations"]["current_root_cause_proven"] is False
    assert "Model invents causal diagnosis" not in str(guarded)
    assert guarded["generation"]["explanation"] is None
