"""Causal per-trajectory alarm threshold and hysteresis policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Hashable, Sequence


@dataclass(frozen=True)
class AlarmPolicy:
    on_threshold: float
    off_threshold: float
    minimum_consecutive: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.off_threshold <= self.on_threshold <= 1.0:
            raise ValueError("alarm thresholds must satisfy 0 <= off <= on <= 1")
        if self.minimum_consecutive < 1:
            raise ValueError("minimum_consecutive must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "on_threshold": self.on_threshold,
            "off_threshold": self.off_threshold,
            "minimum_consecutive": self.minimum_consecutive,
        }


def candidate_alarm_policies(mode: str, thresholds: Sequence[float]) -> list[AlarmPolicy]:
    if mode not in {"point", "hysteresis"}:
        raise ValueError(f"unknown alarm policy mode: {mode}")
    policies = [AlarmPolicy(float(value), float(value), 1) for value in thresholds]
    if mode == "hysteresis":
        policies.extend(
            AlarmPolicy(float(value), max(0.0, float(value) - 0.1), consecutive)
            for value in thresholds
            for consecutive in (1, 3)
        )
    return list(dict.fromkeys(policies))


def apply_alarm_policy(
    scores: Sequence[float],
    group_ids: Sequence[Hashable],
    times_s: Sequence[float],
    policy: AlarmPolicy,
) -> list[int]:
    """Apply a causal state machine independently to each trajectory."""

    if not (len(scores) == len(group_ids) == len(times_s)):
        raise ValueError("scores, group_ids and times_s must have equal length")
    if not scores:
        raise ValueError("at least one alarm score is required")
    values = [float(value) for value in scores]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("alarm scores must lie in [0, 1]")

    output = [0] * len(values)
    positions_by_group = {}
    for index, group_id in enumerate(group_ids):
        positions_by_group.setdefault(group_id, []).append(index)
    for group_id in sorted(positions_by_group, key=str):
        positions = positions_by_group[group_id]
        positions.sort(key=lambda index: (float(times_s[index]), index))
        ordered_times = [float(times_s[index]) for index in positions]
        if any(right <= left for left, right in zip(ordered_times, ordered_times[1:])):
            raise ValueError(f"timestamps must be strictly increasing within group {group_id}")
        active = False
        consecutive = 0
        for index in positions:
            score = values[index]
            if active:
                if score < policy.off_threshold:
                    active = False
                    consecutive = 0
            elif score >= policy.on_threshold:
                consecutive += 1
                if consecutive >= policy.minimum_consecutive:
                    active = True
            else:
                consecutive = 0
            output[index] = int(active)
    return output


def alarm_event_metrics(
    truth: Sequence[int],
    predicted: Sequence[int],
    group_ids: Sequence[Hashable],
    times_s: Sequence[float],
    onset_times_s: Sequence[float],
) -> dict[str, float | int | None]:
    """Summarize causal alarms without using event metadata for prediction."""

    size = len(truth)
    if not all(len(values) == size for values in (predicted, group_ids, times_s, onset_times_s)):
        raise ValueError("event metric inputs must have equal length")
    if not set(truth).issubset({0, 1}) or not set(predicted).issubset({0, 1}):
        raise ValueError("event metric truth and predictions must be binary")
    false_alarm_windows = 0
    healthy_windows = 0
    false_alarm_episodes = 0
    healthy_exposure_s = 0.0
    delays: list[float] = []
    eligible = 0
    positions_by_group = {}
    for index, group_id in enumerate(group_ids):
        positions_by_group.setdefault(group_id, []).append(index)
    for group_id in sorted(positions_by_group, key=str):
        positions = positions_by_group[group_id]
        positions.sort(key=lambda index: (float(times_s[index]), index))
        ordered_times = [float(times_s[index]) for index in positions]
        if any(right <= left for left, right in zip(ordered_times, ordered_times[1:])):
            raise ValueError(f"timestamps must be strictly increasing within group {group_id}")
        group_onsets = {
            float(onset_times_s[index]) for index in positions
            if not math.isnan(float(onset_times_s[index]))
        }
        if len(group_onsets) > 1:
            raise ValueError(f"inconsistent onset metadata within group {group_id}")
        onset = next(iter(group_onsets), None)
        if onset is not None:
            eligible += 1
        first_detection = None
        previous_healthy_alarm = False
        for offset, index in enumerate(positions):
            time_s = float(times_s[index])
            healthy = int(truth[index]) == 0
            alarm = int(predicted[index]) == 1
            if healthy:
                healthy_windows += 1
                false_alarm_windows += int(alarm)
                if alarm and not previous_healthy_alarm:
                    false_alarm_episodes += 1
                previous_healthy_alarm = alarm
                if offset + 1 < len(positions):
                    next_time = float(times_s[positions[offset + 1]])
                    healthy_exposure_s += next_time - time_s
            else:
                previous_healthy_alarm = False
            if onset is not None and time_s >= onset and alarm and first_detection is None:
                first_detection = time_s
        if onset is not None and first_detection is not None:
            delays.append(first_detection - onset)
    exposure_hours = healthy_exposure_s / 3600.0
    return {
        "eligible_fault_events": eligible,
        "detected_fault_events": len(delays),
        "fault_event_recall": len(delays) / eligible if eligible else 0.0,
        "mean_detection_delay_s": statistics.fmean(delays) if delays else None,
        "median_detection_delay_s": statistics.median(delays) if delays else None,
        "p95_detection_delay_s": (
            sorted(delays)[min(len(delays) - 1, math.ceil(0.95 * len(delays)) - 1)]
            if delays else None
        ),
        "healthy_windows": healthy_windows,
        "false_alarm_windows": false_alarm_windows,
        "false_alarm_window_rate": false_alarm_windows / healthy_windows if healthy_windows else 0.0,
        "healthy_exposure_hours": exposure_hours,
        "false_alarm_episodes": false_alarm_episodes,
        "false_alarm_episodes_per_hour": (
            false_alarm_episodes / exposure_hours if exposure_hours > 0.0 else None
        ),
    }
