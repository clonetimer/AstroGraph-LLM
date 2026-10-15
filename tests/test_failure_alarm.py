"""E4: failure-threshold alarm tests."""
from __future__ import annotations

import pytest

from astrograph.data_driven.failure_alarm import FailureAlarmMonitor


def test_success_resets_counter():
    monitor = FailureAlarmMonitor(threshold=3)
    monitor.record_failure('deep-kg')
    monitor.record_success('deep-kg')
    assert monitor.state('deep-kg').consecutive_failures == 0
    assert not monitor.state('deep-kg').degraded


def test_threshold_triggers_degrade_and_alarm():
    monitor = FailureAlarmMonitor(threshold=3)
    monitor.record_failure('deep-kg')
    monitor.record_failure('deep-kg')
    monitor.record_failure('deep-kg')
    state = monitor.state('deep-kg')
    assert state.degraded is True
    assert state.alarm_raised is True
    assert state.consecutive_failures == 3


def test_below_threshold_not_degraded():
    monitor = FailureAlarmMonitor(threshold=3)
    monitor.record_failure('fast-rule')
    monitor.record_failure('fast-rule')
    assert not monitor.state('fast-rule').degraded


def test_independent_routes():
    monitor = FailureAlarmMonitor(threshold=2)
    monitor.record_failure('fast')
    monitor.record_failure('fast')
    monitor.record_failure('deep')
    assert monitor.state('fast').degraded
    assert not monitor.state('deep').degraded


def test_custom_threshold_invalid():
    with pytest.raises(ValueError):
        FailureAlarmMonitor(threshold=0)