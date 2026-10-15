from __future__ import annotations

import pytest

from astrograph.data_driven.evaluation_protocol import (
    FaultInterval,
    binary_aupr,
    event_detection_metrics,
    expected_calibration_error,
    multiclass_metrics,
    precision_recall_auc,
    three_task_evaluation_audit,
    wilson_interval,
)


def test_macro_f1_keeps_a_completely_missed_class_in_denominator() -> None:
    result = multiclass_metrics(
        truth=["A", "A", "B", "B"],
        predicted=["A", "A", "A", "A"],
        labels=["A", "B", "UNSEEN_ON_THIS_SPLIT"],
    )

    assert result["per_class"]["B"]["f1"] == 0.0
    assert result["per_class"]["UNSEEN_ON_THIS_SPLIT"]["support"] == 0
    assert result["evaluated_class_count"] == 2
    assert result["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.0) / 2.0)
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_event_detection_reports_delay_miss_false_and_duplicate_alarms() -> None:
    result = event_detection_metrics(
        fault_intervals=[FaultInterval(100.0, 200.0), FaultInterval(300.0, 400.0)],
        detection_times_s=[50.0, 120.0, 130.0],
        observation_duration_s=1800.0,
    )

    assert result["detected_event_count"] == 1
    assert result["missed_event_count"] == 1
    assert result["false_alarm_count"] == 1
    assert result["healthy_exposure_s"] == pytest.approx(1600.0)
    assert result["false_alarms_per_hour"] == pytest.approx(2.25)
    assert result["duplicate_alarm_count"] == 1
    assert result["event_precision"] == pytest.approx(1 / 3)
    assert result["detection_delay_s"]["median"] == 20.0


def test_event_detection_rejects_overlapping_truth_intervals() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        event_detection_metrics(
            fault_intervals=[FaultInterval(1.0, 3.0), FaultInterval(2.0, 4.0)],
            detection_times_s=[],
            observation_duration_s=10.0,
        )


def test_event_detection_rejects_truth_interval_beyond_observation() -> None:
    with pytest.raises(ValueError, match="outside the observation"):
        event_detection_metrics(
            fault_intervals=[FaultInterval(8.0, 11.0)],
            detection_times_s=[],
            observation_duration_s=10.0,
        )


def test_no_alarm_precision_is_undefined_but_pr_curve_starts_at_one() -> None:
    result = event_detection_metrics(
        fault_intervals=[FaultInterval(5.0, 8.0)],
        detection_times_s=[],
        observation_duration_s=10.0,
    )
    assert result["event_precision"] is None
    assert result["precision_recall_point"] == {
        "recall": 0.0,
        "precision": 1.0,
        "no_alarm_precision_convention_applied": True,
    }


def test_event_detection_freezes_alarm_merge_semantics() -> None:
    result = event_detection_metrics(
        fault_intervals=[FaultInterval(100.0, 200.0)],
        detection_times_s=[110.0, 111.0, 130.0],
        observation_duration_s=1000.0,
        alarm_merge_window_s=2.0,
    )
    assert result["raw_alarm_count"] == 3
    assert result["merged_alarm_count"] == 2
    assert result["merged_raw_alarm_count"] == 1
    assert result["duplicate_alarm_count"] == 1


def test_precision_recall_auc_uses_precision_envelope() -> None:
    assert precision_recall_auc([(1.0, 0.5), (0.5, 0.8), (0.0, 1.0)]) == pytest.approx(0.65)


def test_multiclass_metrics_rejects_ontology_contract_violations() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        multiclass_metrics(["A"], ["A"], labels=["A", "A"])
    with pytest.raises(ValueError, match="predicted"):
        multiclass_metrics(["A"], ["B"], labels=["A"])


def test_binary_aupr_ties_are_order_independent() -> None:
    expected = binary_aupr([True, False, True], [0.9, 0.8, 0.7])
    reordered = binary_aupr([True, True, False], [0.7, 0.9, 0.8])
    assert expected == pytest.approx(5 / 6)
    assert reordered == pytest.approx(expected)
    assert binary_aupr([False, False], [0.2, 0.1]) is None


def test_ece_uses_frozen_bins_and_wilson_is_bounded() -> None:
    result = expected_calibration_error(
        [True, False], [0.9, 0.8], bin_edges=[0.0, 0.5, 1.0]
    )
    assert result["ece"] == pytest.approx(0.35)
    assert result["bin_edges"] == [0.0, 0.5, 1.0]
    interval = wilson_interval(8, 10)
    assert 0.0 <= interval["lower"] < 0.8 < interval["upper"] <= 1.0


def test_three_task_audit_reports_frozen_metrics_and_group_slices() -> None:
    result = three_task_evaluation_audit(
        group_ids=["trace-a", "trace-a", "trace-b", "trace-b"],
        detection_truth=[0, 1, 0, 1],
        detection_predicted=[0, 1, 1, 1],
        detection_positive_scores=[0.1, 0.9, 0.8, 0.7],
        component_truth=[-1, 0, -1, 1],
        component_predicted=[0, 0, 0, 0],
        class_truth=[-1, 0, -1, 1],
        class_predicted=[0, 0, 0, 0],
        component_labels=["adcs", "eps"],
        class_labels=["gyro_bias", "battery_fault"],
        group_key="sample_id",
    )

    assert result["schema_version"] == "astrograph-three-task-evaluation-audit.v1"
    assert result["overall"]["detection"]["accuracy"] == pytest.approx(0.75)
    assert result["overall"]["detection"]["macro_recall"] == pytest.approx(0.75)
    assert result["overall"]["localization"]["macro_f1"] == pytest.approx(1 / 3)
    assert result["overall"]["classification"]["per_class"]["battery_fault"]["f1"] == 0.0
    assert result["overall"]["end_to_end"]["evaluated_class_count"] == 3
    assert [group["sample_id"] for group in result["groups"]] == ["trace-a", "trace-b"]
    assert result["groups"][0]["end_to_end_accuracy"] == 1.0
    assert result["groups"][1]["classification_accuracy"] == 0.0


def test_three_task_audit_rejects_missing_fault_truth() -> None:
    with pytest.raises(ValueError, match="require component and class truth"):
        three_task_evaluation_audit(
            group_ids=["trace"],
            detection_truth=[1],
            detection_predicted=[1],
            detection_positive_scores=[0.9],
            component_truth=[-1],
            component_predicted=[0],
            class_truth=[-1],
            class_predicted=[0],
            component_labels=["adcs"],
            class_labels=["gyro_bias"],
            group_key="sample_id",
        )
