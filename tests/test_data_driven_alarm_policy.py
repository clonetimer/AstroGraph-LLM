from __future__ import annotations

import pytest

from astrograph.data_driven.alarm_policy import (
    AlarmPolicy, alarm_event_metrics,
    apply_alarm_policy,
    candidate_alarm_policies,
)


def test_point_policy_matches_independent_thresholding() -> None:
    scores = [0.2, 0.5, 0.7, 0.49]
    result = apply_alarm_policy(scores, ["a"] * 4, [0.0, 1.0, 2.0, 3.0], AlarmPolicy(0.5, 0.5))
    assert result == [0, 1, 1, 0]


def test_hysteresis_and_minimum_consecutive_are_causal_per_group() -> None:
    result = apply_alarm_policy(
        [0.7, 0.7, 0.45, 0.35, 0.8, 0.8],
        ["a", "a", "a", "a", "b", "b"],
        [0, 1, 2, 3, 0, 1],
        AlarmPolicy(0.6, 0.4, 2),
    )
    assert result == [0, 1, 1, 0, 0, 1]


def test_interleaved_groups_return_original_order_and_reject_time_drift() -> None:
    result = apply_alarm_policy(
        [0.8, 0.1, 0.2, 0.9], ["a", "b", "a", "b"], [1, 0, 0, 1],
        AlarmPolicy(0.5, 0.5),
    )
    assert result == [1, 0, 0, 1]
    with pytest.raises(ValueError, match="strictly increasing"):
        apply_alarm_policy([0.1, 0.2], ["a", "a"], [1, 1], AlarmPolicy(0.5, 0.5))


def test_alarm_policy_candidate_grid_is_frozen_and_validated() -> None:
    assert len(candidate_alarm_policies("point", [0.4, 0.5])) == 2
    assert len(candidate_alarm_policies("hysteresis", [0.4, 0.5])) == 6
    with pytest.raises(ValueError, match="0 <= off <= on"):
        AlarmPolicy(0.4, 0.5)


def test_event_metrics_separate_false_alarms_from_post_onset_detection() -> None:
    report = alarm_event_metrics(
        truth=[0, 0, 1, 1, 0, 0, 0],
        predicted=[1, 0, 0, 1, 0, 1, 1],
        group_ids=["fault"] * 4 + ["nominal"] * 3,
        times_s=[0, 1, 2, 3, 0, 1, 2],
        onset_times_s=[2, 2, 2, 2, float("nan"), float("nan"), float("nan")],
    )
    assert report["eligible_fault_events"] == 1
    assert report["fault_event_recall"] == 1.0
    assert report["median_detection_delay_s"] == 1.0
    assert report["healthy_windows"] == 5
    assert report["false_alarm_windows"] == 3
    assert report["false_alarm_episodes"] == 2
