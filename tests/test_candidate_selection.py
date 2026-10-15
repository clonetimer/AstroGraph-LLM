from __future__ import annotations

import pytest
from pydantic import ValidationError

from astrograph.llm import (
    CandidateSelectionInput,
    CandidateSignal,
    OpenAICompatibleConfig,
    OpenAICompatibleStructuredBackend,
    select_candidates,
)


def _signal(fault_id: str, status: str, *, support=0, contradict=0, missing=0, source="REFERENCE_HYPOTHESIS_SET"):
    return CandidateSignal(
        fault_id=fault_id,
        status=status,
        supporting_evidence_fact_ids=[f"EF-S-{fault_id}-{i}" for i in range(support)],
        contradicting_evidence_fact_ids=[f"EF-C-{fault_id}-{i}" for i in range(contradict)],
        missing_evidence_definition_ids=[f"ED-M-{fault_id}-{i}" for i in range(missing)],
        source=source,
    )


def test_candidate_interface_accepts_future_evidence_navigator_results():
    request = CandidateSelectionInput(
        registered_fault_ids=["F-A", "F-B", "F-C"],
        top_k=2,
        candidates=[
            _signal("F-A", "UNDER_VERIFICATION", support=1, source="EVIDENCE_NAVIGATOR"),
            _signal("F-B", "SUPPORTED", support=2, source="EVIDENCE_NAVIGATOR"),
            _signal("F-C", "CONTRADICTED", contradict=1),
        ],
    )
    result = select_candidates(request)
    assert result.selected_fault_ids == ["F-A", "F-B"]
    assert result.forced_candidate_count == 2
    assert result.ranked_candidates[0].fault_id == "F-B"
    assert result.nominal_short_circuit is False


def test_candidate_interface_rejects_unregistered_fault():
    with pytest.raises(ValidationError):
        CandidateSelectionInput(
            registered_fault_ids=["F-A"],
            candidates=[_signal("UNKNOWN", "SUPPORTED")],
        )


def test_nominal_all_contradicted_returns_no_llm_candidates():
    request = CandidateSelectionInput(
        registered_fault_ids=["F-A", "F-B"],
        top_k=2,
        candidates=[
            _signal("F-A", "CONTRADICTED", contradict=1),
            _signal("F-B", "CONTRADICTED", contradict=1),
        ],
    )
    result = select_candidates(request)
    assert result.selected_fault_ids == []
    assert result.nominal_short_circuit is True


def test_top_k_prefers_supported_then_insufficient_then_under_verification():
    request = CandidateSelectionInput(
        registered_fault_ids=["F-A", "F-B", "F-C", "F-D"],
        top_k=2,
        candidates=[
            _signal("F-A", "UNDER_VERIFICATION", support=3),
            _signal("F-B", "INSUFFICIENT_EVIDENCE", missing=1),
            _signal("F-C", "SUPPORTED", support=1),
            _signal("F-D", "PROPOSED", support=5),
        ],
    )
    result = select_candidates(request)
    assert result.selected_fault_ids == ["F-B", "F-C"]


def test_missing_evidence_never_improves_plausibility_within_same_status():
    request = CandidateSelectionInput(
        registered_fault_ids=["F-A", "F-B"],
        top_k=2,
        candidates=[
            _signal("F-A", "INSUFFICIENT_EVIDENCE", support=2, missing=1),
            _signal("F-B", "INSUFFICIENT_EVIDENCE", support=2, missing=5),
        ],
    )
    result = select_candidates(request)
    assert [item.fault_id for item in result.ranked_candidates] == ["F-A", "F-B"]
    assert result.ranked_candidates[0].plausibility_rank == 1
    assert result.ranked_candidates[1].investigation_priority > result.ranked_candidates[0].investigation_priority


def test_contradiction_precedes_support_count_within_same_status():
    request = CandidateSelectionInput(
        registered_fault_ids=["F-A", "F-B"],
        top_k=2,
        candidates=[
            _signal("F-A", "INSUFFICIENT_EVIDENCE", support=8, contradict=2, missing=1),
            _signal("F-B", "INSUFFICIENT_EVIDENCE", support=1, contradict=0, missing=1),
        ],
    )
    result = select_candidates(request)
    assert result.ranked_candidates[0].fault_id == "F-B"
    assert "HAS_CONTRADICTION" in result.ranked_candidates[1].ranking_reason_codes

def test_active_candidate_overflow_disables_compact_reasoning_without_dropping_audit_rank():
    request = CandidateSelectionInput(
        registered_fault_ids=["F-A", "F-B", "F-C", "F-D"],
        top_k=2,
        candidates=[
            _signal("F-A", "SUPPORTED", support=2),
            _signal("F-B", "INSUFFICIENT_EVIDENCE", support=1, missing=1),
            _signal("F-C", "UNDER_VERIFICATION", support=1),
            _signal("F-D", "CONTRADICTED", contradict=1),
        ],
    )
    result = select_candidates(request)
    assert len(result.selected_fault_ids) == 2
    assert result.forced_candidate_count == 3
    assert result.compact_reasoning_allowed is False
    assert result.candidate_set_status == "ACTIVE_CANDIDATE_OVERFLOW"
    assert result.escape_action == "DISABLE_COMPACT_REASONING"
    assert len(result.unselected_mandatory_fault_ids) == 1
    assert result.escape_reason_codes == ["ACTIVE_CANDIDATES_EXCEED_TOP_K"]
    assert len(result.ranked_candidates) == 4


def test_backend_candidate_escape_keeps_full_registered_context_instead_of_hard_topk_filter():
    backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        model="local",
        prompt_profile="evidence_protocol_v059_compact",
        candidate_top_k=2,
        candidate_top_k_mode="fixed",
        include_reference_hypothesis_set=True,
    ))
    hypotheses = [
        {
            "hypothesis_type": "REGISTERED_FAULT",
            "target_fault_id": fault_id,
            "status": status,
            "supporting_evidence_fact_ids": [f"EF-{fault_id}"] if status != "CONTRADICTED" else [],
            "contradicting_evidence_fact_ids": [f"EF-C-{fault_id}"] if status == "CONTRADICTED" else [],
            "missing_evidence_definition_ids": [],
        }
        for fault_id, status in [
            ("F-A", "SUPPORTED"),
            ("F-B", "INSUFFICIENT_EVIDENCE"),
            ("F-C", "UNDER_VERIFICATION"),
            ("F-D", "CONTRADICTED"),
        ]
    ]
    payload = {
        "registered_fault_ids": ["F-A", "F-B", "F-C", "F-D"],
        "reference_hypothesis_set": {"hypotheses": hypotheses},
        "evidence_definitions": [],
        "evidence_facts": [],
        "diagnostic_protocol": {"faults": {}},
    }
    effective, preferred, compact_applied = backend._prepare_effective_payload(payload)
    assert compact_applied is False
    assert len(preferred) == 2
    assert effective["registered_fault_ids"] == payload["registered_fault_ids"]
    assert effective["reference_hypothesis_set"] == payload["reference_hypothesis_set"]
    assert "compact_protocol" not in effective
    assert effective["candidate_selection_escape"]["candidate_set_status"] == "ACTIVE_CANDIDATE_OVERFLOW"
    assert effective["candidate_selection_escape"]["unselected_mandatory_fault_ids"]

