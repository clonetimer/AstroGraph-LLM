from __future__ import annotations

import time
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class AlarmState(BaseModel):
    """E4: failure-threshold alarm state for a diagnosis route."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-failure-alarm.v1'
    route: str = Field(min_length=1)
    consecutive_failures: int = Field(default=0, ge=0)
    degraded: bool = False
    alarm_raised: bool = False
    threshold: int = Field(default=3, ge=1)
    updated_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class FailureAlarmMonitor:
    """Tracks consecutive failures per route; degrades + alarms past threshold (E4)."""

    schema_version = 'astrograph-failure-alarm-monitor.v1'

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError('threshold must be >= 1')
        self.threshold = threshold
        self._states: dict[str, AlarmState] = {}

    def record_failure(self, route: str) -> AlarmState:
        state = self._states.get(route, AlarmState(route=route, threshold=self.threshold))
        state.consecutive_failures += 1
        state.degraded = state.consecutive_failures >= self.threshold
        state.alarm_raised = state.degraded
        state.updated_at = time.strftime('%Y-%m-%dT%H:%M:%S+08:00')
        self._states[route] = state
        return state

    def record_success(self, route: str) -> AlarmState:
        state = self._states.get(route, AlarmState(route=route, threshold=self.threshold))
        state.consecutive_failures = 0
        state.degraded = False
        state.alarm_raised = False
        state.updated_at = time.strftime('%Y-%m-%dT%H:%M:%S+08:00')
        self._states[route] = state
        return state

    def state(self, route: str) -> AlarmState:
        return self._states.get(route, AlarmState(route=route, threshold=self.threshold))


__all__ = ['AlarmState', 'FailureAlarmMonitor']