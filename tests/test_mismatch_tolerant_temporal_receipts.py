from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_alignment_contract_is_prefix_only_and_invalidates_edges():
    payload = json.loads(
        (ROOT / "configs/research/adcs_mismatch_tolerant_alignment_contract_r001.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["development_groups"] == [3, 8]
    assert payload["excluded_groups"] == [4, 9]
    assert payload["prefix_points"] == 31
    assert payload["candidate_lags"] == [-3, -2, -1, 0, 1, 2, 3]
    assert payload["padded_edge_action"] == "INVALIDATE"
    assert payload["runtime_ground_truth_access"] is False


def test_candidate_grid_discloses_development_iteration():
    payload = json.loads(
        (ROOT / "configs/research/adcs_mismatch_tolerant_temporal_candidate_grid_r001.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["development_groups"] == [3, 8]
    assert payload["excluded_groups"] == [4, 9]
    assert payload["candidate_count"] == 40
    assert "prior groups-3/8" in payload["design_disclosure"]
    assert payload["new_confirmatory_evidence"] == "NONE"


def test_selected_policy_is_bounded_and_not_confirmatory():
    payload = json.loads(
        (ROOT / "configs/research/adcs_mismatch_tolerant_temporal_policy_r001.json").read_text(
            encoding="utf-8"
        )
    )
    policy = payload["selected_policy"]
    assert payload["project_status"] == "B009_WIP_DEVELOPMENT_EVIDENCE"
    assert payload["development_groups"] == [3, 8]
    assert payload["excluded_groups"] == [4, 9]
    assert payload["groups_4_9_read"] is False
    assert payload["runtime_ground_truth_access"] is False
    assert payload["new_confirmatory_evidence"] == "NONE"
    assert policy["trigger_threshold"] == 0.06
    assert policy["alignment_fit_max"] == 0.08
    assert policy["notification_buffer_points"] == 3


def test_development_result_reports_bounded_success_and_cross_reference_failure_boundary():
    payload = json.loads(
        (ROOT / "research/mismatch_tolerant_temporal_r001/development_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["development_groups"] == [3, 8]
    assert payload["excluded_groups"] == [4, 9]
    assert payload["groups_4_9_read"] is False
    metrics = payload["selected_policy_receipt"]["selected_metrics_by_variant"]
    assert metrics["REFERENCE_SHIFT_1S"]["notified_within_20s_rate"] == 1.0
    assert metrics["REFERENCE_SHIFT_2S"]["notified_within_20s_rate"] == 1.0
    assert metrics["CURRENT_NOISE_5PCT_SCALE"]["notified_within_20s_rate"] > 0.99
    assert metrics["CURRENT_NOISE_5PCT_SCALE"]["nominal_false_alarm_count"] == 0
    assert metrics["CROSS_MC_REFERENCE"]["fault_alignment_admission_rate"] == 0.25
    assert metrics["CROSS_MC_REFERENCE"]["minimum_class_notified_within_20s_rate"] == 0.0
    assessment = payload["thesis_contribution_assessment"]
    assert assessment["arbitrary_cross_reference_robustness_established"] is False
    assert assessment["standalone_sufficient_for_excellent_thesis"] is False
    assert assessment["new_confirmatory_evidence"] == "NONE"


def test_trace_index_is_label_free_and_not_full_trajectory_warped():
    payload = json.loads(
        (ROOT / "research/mismatch_tolerant_temporal_r001/trace_index_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["truth_access_during_generation"] is False
    assert payload["outcome_data_read_during_generation"] is False
    assert payload["full_trajectory_alignment_used"] is False
    assert payload["padded_edge_action"] == "INVALIDATE"
    assert payload["development_groups"] == [3, 8]
    assert payload["excluded_groups"] == [4, 9]


def test_prefix_truth_audit_is_offline_and_has_no_development_overlap():
    payload = json.loads(
        (ROOT / "research/mismatch_tolerant_temporal_r001/prefix_truth_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["audit_is_offline_after_trace_freeze"] is True
    assert payload["audit_used_for_policy_selection"] is False
    assert payload["groups_4_9_read"] is False
    assert payload["prefix_points"] == 31
    assert payload["prefix_end_time_s"] == pytest.approx(30.0, abs=1e-12)
    assert payload["minimum_fault_active_start_s"] == 36.0
    assert payload["fault_prefix_overlap_count"] == 0
    assert payload["nominal_prefix_observed_on_this_development_set"] is True


def test_candidate_selection_summary_discloses_search_and_ties():
    payload = json.loads(
        (
            ROOT
            / "research/mismatch_tolerant_temporal_r001/candidate_selection_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["candidate_count"] == 40
    assert payload["safe_candidate_count"] == 36
    assert payload["exact_noninferiority_candidate_count"] == 16
    assert payload["top_scientific_outcome_tie_count"] >= 2
    assert payload["selected_policy_id"] == "ADCS-MT-R001-F0.08-T0.06-P1"
    assert payload["selection_informed_by_prior_development_exploration"] is True
    assert payload["groups_4_9_read"] is False
    assert payload["new_confirmatory_evidence"] == "NONE"


def test_replay_and_label_isolation_receipts_are_pass():
    trace_replay = json.loads(
        (
            ROOT
            / "research/mismatch_tolerant_temporal_r001/trace_replay_verification.json"
        ).read_text(encoding="utf-8")
    )
    evaluation_replay = json.loads(
        (
            ROOT
            / "research/mismatch_tolerant_temporal_r001/evaluation_replay_verification.json"
        ).read_text(encoding="utf-8")
    )
    label_audit = json.loads(
        (
            ROOT
            / "research/mismatch_tolerant_temporal_r001/label_isolation_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert trace_replay["byte_identical"] is True
    assert evaluation_replay["byte_identical"] is True
    assert label_audit["label_isolation_pass"] is True
    assert label_audit["development_groups_observed"] == [3, 8]
    assert label_audit["groups_4_9_present"] is False
    assert label_audit["runtime_key_violations"] == []
    assert label_audit["telemetry_header_violations"] == []


def test_failure_analysis_keeps_the_noise_miss_visible():
    payload = json.loads(
        (ROOT / "research/mismatch_tolerant_temporal_r001/failure_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    noise = payload["variants"]["CURRENT_NOISE_5PCT_SCALE"]
    assert noise["fault_miss_count"] == 1
    miss = noise["fault_misses"][0]
    assert miss["opaque_trajectory_id"] == "TRJ-E11E2C25E138D71155EE"
    assert miss["diagnostic_class"] == "adcs_gyro_bias_step"
    assert miss["status"] == "NO_ALARM"
    cross = payload["variants"]["CROSS_MC_REFERENCE"]
    assert cross["fault_miss_count"] == 160
    assert cross["abstained_count"] == 136
