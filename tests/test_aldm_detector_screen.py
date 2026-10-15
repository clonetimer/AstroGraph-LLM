from __future__ import annotations

import pytest

from scripts.data_driven.screen_aldm_detector import (
    _no_regression,
    promotion_decision,
)


def _result(
    *,
    macro_f1: float,
    recall: float,
    false_alarms_per_hour: float,
    p95_delay_s: float,
) -> dict[str, object]:
    return {
        "point": {"macro_f1": macro_f1},
        "event": {
            "fault_event_recall": recall,
            "false_alarm_episodes_per_hour": false_alarms_per_hour,
            "p95_detection_delay_s": p95_delay_s,
        },
    }


def test_promotion_requires_gain_and_all_event_non_regression_gates() -> None:
    baseline = _result(
        macro_f1=0.80,
        recall=0.90,
        false_alarms_per_hour=3.0,
        p95_delay_s=12.0,
    )
    candidate = _result(
        macro_f1=0.82,
        recall=0.91,
        false_alarms_per_hour=2.0,
        p95_delay_s=10.0,
    )

    decision = promotion_decision(baseline, candidate, minimum_gain=0.01)

    assert decision["status"] == "PASS_CONTINUE_TO_CONFIRMATION"
    assert decision["macro_f1_gain"] == pytest.approx(0.02)
    assert decision["test_authorized"] is False


@pytest.mark.parametrize(
    ("candidate", "failed_gate"),
    [
        (_result(macro_f1=0.805, recall=0.91, false_alarms_per_hour=2.0, p95_delay_s=10.0), "gain"),
        (_result(macro_f1=0.82, recall=0.89, false_alarms_per_hour=2.0, p95_delay_s=10.0), "recall"),
        (_result(macro_f1=0.82, recall=0.91, false_alarms_per_hour=4.0, p95_delay_s=10.0), "false_alarm"),
        (_result(macro_f1=0.82, recall=0.91, false_alarms_per_hour=2.0, p95_delay_s=13.0), "delay"),
    ],
)
def test_each_failed_gate_stops_before_test(
    candidate: dict[str, object], failed_gate: str,
) -> None:
    baseline = _result(
        macro_f1=0.80,
        recall=0.90,
        false_alarms_per_hour=3.0,
        p95_delay_s=12.0,
    )

    decision = promotion_decision(baseline, candidate, minimum_gain=0.01)

    assert decision["status"] == "STOP_NO_VALIDATION_GAIN"
    assert decision["test_authorized"] is False
    if failed_gate == "recall":
        assert decision["event_recall_no_regression"] is False
    elif failed_gate == "false_alarm":
        assert decision["false_alarm_rate_no_regression"] is False
    elif failed_gate == "delay":
        assert decision["p95_delay_no_regression"] is False


def test_missing_event_metric_fails_closed() -> None:
    assert _no_regression(None, 1.0, lower_is_better=True) is False
    assert _no_regression(None, None, lower_is_better=False) is True
