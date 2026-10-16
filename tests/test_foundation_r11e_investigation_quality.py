"""Synthetic R11-E tool-investigation quality tests; not flight/LLM reasoning proof."""
from __future__ import annotations

from copy import deepcopy

from astrograph.tools.material_review import (
    TOOL_IDS, MaterialToolPlan, material_investigation_opportunities,
    audit_material_investigation_plan, execute_material_tool_plan,
)


def reviewed_material():
    return {"competing_hypothesis_review": {"status": "COMPARE_ONLY"},
            "candidates": [
                {"fault_id": "BENCH_A", "multi_hop": {"causal_chain_review": {
                    "chains": [{"path_id": "A_STAGE_1",
                                "reviewed_context_consistency": {
                                    "status": "CONFLICT", "conflict_reasons": ["OPERATING_MODE_DISJOINT"]},
                                "reviewed_temporal_consistency": {
                                    "status": "REVIEWED_SEQUENCE_COMPATIBLE_NOT_FLIGHT_VERIFIED",
                                    "cumulative_lag_seconds": {"min": 8, "max": 16}},
                                "hops": []}],
                    "blocked": [{"edge_id": "A_BLOCKED", "reasons": ["NO_REVIEWED_CONTEXT"]}],
                }}},
                {"fault_id": "BENCH_B", "multi_hop": {"causal_chain_review": {
                    "chains": [], "blocked": []}}},
            ]}


class SyntheticSelector:
    def __init__(self, actions):
        self.actions = actions
        self.payload = None

    def generate_structured(self, *, agent_id, output_model, payload):
        assert agent_id == "platform-knowledge-material-tool-selector"
        self.payload = payload
        return MaterialToolPlan.model_validate({"actions": self.actions})


def test_review_opportunities_use_only_host_ids_not_current_root_cause():
    source = reviewed_material()
    before = deepcopy(source)
    checklist = material_investigation_opportunities(source)
    assert checklist["total_opportunities"] == 4
    assert [x["tool_id"] for x in checklist["opportunities"]] == list(TOOL_IDS)
    assert checklist["opportunities"][2]["path_id"] == "A_STAGE_1"
    assert checklist["actual_flight_evidence_verified"] is False
    assert checklist["diagnostic_vote_allowed"] is False
    assert source == before


def test_model_skips_available_tools_and_gap_is_explicit():
    selector = SyntheticSelector([])
    result = execute_material_tool_plan(backend=selector, material_review=reviewed_material())
    audit = result["review_coverage_audit"]
    assert result["calls"] == []
    assert audit["status"] == "MODEL_DECLINED_AVAILABLE_REVIEW"
    assert audit["checks_missing_in_visible_list"] == 4
    assert selector.payload["investigation_opportunities"]["total_opportunities"] == 4
    assert audit["root_cause_proven"] is False


def test_budget_limited_plan_cannot_claim_complete_inspection():
    result = execute_material_tool_plan(
        backend=SyntheticSelector([
            {"tool_id": TOOL_IDS[0]},
            {"tool_id": TOOL_IDS[1], "fault_id": "BENCH_A"},
            {"tool_id": TOOL_IDS[3], "fault_id": "BENCH_A", "path_id": "A_STAGE_1"},
        ]), material_review=reviewed_material())
    audit = result["review_coverage_audit"]
    assert audit["status"] == "REVIEW_COVERAGE_INCOMPLETE"
    assert audit["checks_matching_opportunities"] == 3
    assert audit["review_gaps"][0]["tool_id"] == TOOL_IDS[2]
    assert audit["checks_denied"] == 0
    assert result["current_root_cause_proven"] is False


def test_denials_and_repetitions_do_not_count_as_completeness():
    result = execute_material_tool_plan(
        backend=SyntheticSelector([
            {"tool_id": TOOL_IDS[1], "fault_id": "FOREIGN_FAULT"},
            {"tool_id": TOOL_IDS[0]}, {"tool_id": TOOL_IDS[0]},
        ]), material_review=reviewed_material())
    audit = result["review_coverage_audit"]
    assert [x["status"] for x in result["calls"]] == ["DENIED", "SUCCEEDED", "SUCCEEDED"]
    assert audit["checks_denied"] == 1
    assert audit["checks_repeated"] == 1
    assert audit["checks_matching_opportunities"] == 1
    assert "FOREIGN_FAULT" not in str(audit)


def test_no_reviewed_path_means_no_review_opportunity_not_negative_observation():
    source = {"candidates": [{"fault_id": "BENCH_A"}], "competing_hypothesis_review": {}}
    selector = SyntheticSelector([])
    result = execute_material_tool_plan(backend=selector, material_review=source)
    assert result["review_coverage_audit"]["status"] == "NO_REVIEWED_TOOL_OPPORTUNITIES"
    assert selector.payload["investigation_opportunities"]["total_opportunities"] == 0


def test_unnecessary_tool_call_is_not_counted_as_a_useful_check():
    source = {"candidates": [{"fault_id": "BENCH_A"}], "competing_hypothesis_review": {}}
    result = execute_material_tool_plan(backend=SyntheticSelector([
        {"tool_id": TOOL_IDS[0]},
    ]), material_review=source)
    audit = result["review_coverage_audit"]
    assert audit["status"] == "TOOL_CALLED_WITHOUT_REVIEW_OPPORTUNITY"
    assert audit["checks_without_host_opportunity"] == 1
    assert audit["root_cause_proven"] is False


def test_truncated_checklist_never_asserts_completeness():
    source = reviewed_material()
    for index in range(35):
        source["candidates"][0]["multi_hop"]["causal_chain_review"]["chains"].append({
            "path_id": f"MORE_{index}", "reviewed_temporal_consistency": {"status": "UNKNOWN"},
        })
    checklist = material_investigation_opportunities(source, max_opportunities=5)
    assert checklist["truncated"] is True
    audit = audit_material_investigation_plan(MaterialToolPlan(actions=[]), checklist, [])
    assert audit["status"] == "REVIEW_OPPORTUNITY_LIST_TRUNCATED"
    assert audit["opportunities_total"] > 5
    assert not audit["root_cause_proven"]
