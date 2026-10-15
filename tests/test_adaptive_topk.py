from __future__ import annotations

import pytest
from pydantic import ValidationError

from astrograph.evaluation.topk_robustness import run_adaptive_topk_robustness
from astrograph.llm import (
    CandidateSelectionInput,
    CandidateSignal,
    OpenAICompatibleConfig,
    OpenAICompatibleStructuredBackend,
    resolve_candidate_top_k,
    select_candidates,
)


def _signal(fault_id: str, status: str, *, support: int = 0) -> CandidateSignal:
    return CandidateSignal(
        fault_id=fault_id,
        status=status,
        supporting_evidence_fact_ids=[f"EF-{fault_id}-{index}" for index in range(support)],
        contradicting_evidence_fact_ids=[],
        missing_evidence_definition_ids=[],
    )


def _hypothesis(fault_id: str, status: str, support: int = 0) -> dict:
    return {
        "hypothesis_type": "REGISTERED_FAULT",
        "target_fault_id": fault_id,
        "status": status,
        "supporting_evidence_fact_ids": [f"EF-{fault_id}-{index}" for index in range(support)],
        "contradicting_evidence_fact_ids": [] if status != "CONTRADICTED" else [f"EF-C-{fault_id}"],
        "missing_evidence_definition_ids": [],
    }


def test_adaptive_top_k_tracks_forced_candidates_with_bounds() -> None:
    candidates = [_signal(f"F-{index}", "UNDER_VERIFICATION") for index in range(5)]
    assert resolve_candidate_top_k(
        candidates,
        mode="adaptive_forced_count",
        configured_top_k=4,
        min_top_k=2,
        max_top_k=6,
    ) == 5
    assert resolve_candidate_top_k(
        candidates[:1],
        mode="adaptive_forced_count",
        configured_top_k=4,
        min_top_k=2,
        max_top_k=6,
    ) == 2
    assert resolve_candidate_top_k(
        candidates + [_signal("F-6", "SUPPORTED"), _signal("F-7", "SUPPORTED")],
        mode="adaptive_forced_count",
        configured_top_k=4,
        min_top_k=2,
        max_top_k=6,
    ) == 6


def test_fixed_top_k_remains_backward_compatible() -> None:
    candidates = [_signal("F-A", "SUPPORTED")]
    assert resolve_candidate_top_k(
        candidates,
        mode="fixed",
        configured_top_k=4,
        min_top_k=2,
        max_top_k=6,
    ) == 4


def test_adaptive_nominal_still_short_circuits() -> None:
    candidates = [_signal("F-A", "CONTRADICTED"), _signal("F-B", "CONTRADICTED")]
    effective_top_k = resolve_candidate_top_k(
        candidates,
        mode="adaptive_forced_count",
        configured_top_k=4,
        min_top_k=2,
        max_top_k=6,
    )
    result = select_candidates(CandidateSelectionInput(
        registered_fault_ids=["F-A", "F-B"],
        candidates=candidates,
        top_k=effective_top_k,
    ))
    assert effective_top_k == 2
    assert result.selected_fault_ids == []
    assert result.nominal_short_circuit is True


def test_adaptive_config_validates_bounds_and_records_effective_k() -> None:
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(
            model="local",
            candidate_top_k_mode="adaptive_forced_count",
            candidate_top_k_min=7,
            candidate_top_k_max=6,
        )

    config = OpenAICompatibleConfig(
        model="local",
        prompt_profile="evidence_protocol_v059_compact",
        candidate_top_k_mode="adaptive_forced_count",
        candidate_top_k_min=2,
        candidate_top_k_max=6,
    )
    backend = OpenAICompatibleStructuredBackend(config)
    payload = {
        "registered_fault_ids": ["F-A", "F-B", "F-C", "F-D"],
        "reference_hypothesis_set": {
            "hypotheses": [
                _hypothesis("F-A", "SUPPORTED", 1),
                _hypothesis("F-B", "UNDER_VERIFICATION", 1),
                _hypothesis("F-C", "CONTRADICTED"),
                _hypothesis("F-D", "CONTRADICTED"),
            ]
        },
    }
    assert backend._select_compact_fault_ids(payload) == ["F-A", "F-B"]
    assert backend._last_candidate_selection is not None
    assert backend._last_candidate_selection["top_k_mode"] == "adaptive_forced_count"
    assert backend._last_candidate_selection["effective_top_k"] == 2
    assert backend.config.public_snapshot()["candidate_top_k_max"] == 6


def test_adaptive_robustness_expands_only_when_needed() -> None:
    documents = [
        ("single", {"hypotheses": [
            _hypothesis("F-A", "SUPPORTED", 1),
            _hypothesis("F-B", "CONTRADICTED"),
            _hypothesis("F-C", "CONTRADICTED"),
            _hypothesis("F-D", "CONTRADICTED"),
            _hypothesis("F-E", "CONTRADICTED"),
            _hypothesis("F-F", "CONTRADICTED"),
        ]}),
        ("composite", {"hypotheses": [
            _hypothesis("F-A", "SUPPORTED", 1),
            _hypothesis("F-B", "SUPPORTED", 1),
            _hypothesis("F-C", "CONTRADICTED"),
            _hypothesis("F-D", "CONTRADICTED"),
            _hypothesis("F-E", "CONTRADICTED"),
            _hypothesis("F-F", "CONTRADICTED"),
        ]}),
        ("nominal", {"hypotheses": [
            _hypothesis("F-A", "CONTRADICTED"),
            _hypothesis("F-B", "CONTRADICTED"),
            _hypothesis("F-C", "CONTRADICTED"),
        ]}),
    ]
    report = run_adaptive_topk_robustness(documents, min_top_k=2, max_top_k=6)
    summary = report["summary"]
    assert summary["member_recall"] == 1.0
    assert summary["all_targets_case_recall"] == 1.0
    assert summary["safe_empty_unknown_proxy"] is True
    assert int(summary["effective_top_k_distribution"]["2"]) > 0
    assert any(int(key) > 2 for key in summary["effective_top_k_distribution"])
