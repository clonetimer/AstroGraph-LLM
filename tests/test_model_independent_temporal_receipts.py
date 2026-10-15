from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_development_policy_receipt_keeps_claim_boundary():
    path = ROOT / "configs/research/adcs_model_independent_temporal_policy_r001.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["development_groups"] == [3, 8]
    assert payload["excluded_groups"] == [4, 9]
    assert payload["historical_model_required"] is False
    assert payload["model_package_sha256"] is None
    assert payload["reference_guard_tuned_on_stress"] is False
    assert payload["new_confirmatory_evidence"] == "NONE"
    assert payload["selected_policy"]["trigger_threshold"] == 0.05
    assert payload["selected_policy"]["reference_prefix_max"] == 0.05


def test_trace_index_receipt_is_label_free_and_deterministic():
    path = ROOT / "research/model_independent_temporal_r001/trace_index_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["development_groups"] == [3, 8]
    assert payload["excluded_groups"] == [4, 9]
    assert payload["truth_access_during_generation"] is False
    assert payload["stress_outcomes_read_during_generation"] is False
    assert set(payload["trace_bundles"]) == {
        "EXACT_PAIRED",
        "REFERENCE_SHIFT_1S",
        "REFERENCE_SHIFT_2S",
        "CURRENT_NOISE_1PCT_SCALE",
        "CURRENT_NOISE_5PCT_SCALE",
        "CROSS_MC_REFERENCE",
    }


def test_trace_replay_is_byte_deterministic():
    path = ROOT / "research/model_independent_temporal_r001/trace_replay_verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exact_file_hash_match"] is True
    assert payload["different"] == []


def test_thesis_assessment_does_not_overclaim():
    path = ROOT / "research/model_independent_temporal_r001/thesis_contribution_assessment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["canonical_exact_pair_feasibility_strong"] is True
    assert payload["reference_mismatch_brittleness_observed"] is True
    assert payload["fault_identification_evaluated"] is False
    assert payload["standalone_sufficient_for_excellent_thesis"] is False
    assert payload["new_confirmatory_evidence"] == "NONE"
