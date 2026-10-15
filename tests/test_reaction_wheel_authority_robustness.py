from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from astrograph.cli import _dataset_detector_selection
from astrograph.diagnosis import (
    RW4_AUTHORITY_ROBUST_DETECTOR_ID,
    RW4_SINGLE_FAULT_DETECTOR_ID,
    detect_rw4_authority_robust,
    detect_rw4_single_fault,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _channels(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) != "time_s"
        and not str(column).lower().startswith(
            ("label.", "label_", "truth.", "truth_", "ground_truth")
        )
    ]


def _authority_frame(
    fault_wheel: int | None,
    retention: float = 1.0,
    *,
    sample_rate_hz: float = 1.0,
    shared_bias_a: float = 0.03,
    noise_scale_a: float = 0.0005,
) -> pd.DataFrame:
    time_s = np.arange(0.0, 60.0 + 1.0e-9, 1.0 / sample_rate_hz)
    payload: dict[str, np.ndarray] = {"time_s": time_s}
    for wheel in range(1, 5):
        command = (
            0.18 * np.sin(0.37 * time_s + 0.6 * wheel)
            + 0.12 * np.cos(0.19 * time_s - 0.2 * wheel)
        )
        gain = np.ones(len(time_s))
        if fault_wheel == wheel:
            gain[time_s >= 20.0] = retention
        actual = (
            shared_bias_a
            + gain * command
            + noise_scale_a * np.sin(1.7 * time_s + wheel)
        )
        payload[f"rw{wheel}_speed_rad_s"] = 50.0 + wheel + np.cumsum(actual) * 0.01
        payload[f"rw{wheel}_command_current_a"] = command
        payload[f"rw{wheel}_current_a"] = actual
    return pd.DataFrame(payload)


def _legacy_collapse_frame(fault_wheel: int = 2) -> pd.DataFrame:
    payload: dict[str, list[float]] = {"time_s": [float(index) for index in range(50)]}
    for wheel in range(1, 5):
        current: list[float] = []
        speed: list[float] = []
        for index in range(50):
            if index < 15:
                current_value = -(0.040 + 0.004 * index) + 0.002 * (wheel - 1)
                speed_value = 50.0 + wheel + 0.03 * index
            elif fault_wheel == wheel:
                current_value = 0.002 if index % 2 == 0 else -0.002
                speed_value = 50.0 + wheel + 0.03 * 14 + 0.005 * (index - 14)
            else:
                current_value = -(0.100 - 0.001 * (index - 15)) + 0.002 * (wheel - 1)
                speed_value = 50.0 + wheel + 0.03 * index
            current.append(current_value)
            speed.append(speed_value)
        payload[f"rw{wheel}_speed_rad_s"] = speed
        payload[f"rw{wheel}_current_a"] = current
    return pd.DataFrame(payload)


@pytest.mark.parametrize("fault_wheel", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("retention", "expected_severity"),
    [
        (0.00, "complete"),
        (0.25, "critical"),
        (0.50, "severe"),
        (0.75, "moderate"),
        (0.85, "mild"),
    ],
)
@pytest.mark.parametrize("sample_rate_hz", [0.5, 1.0, 2.0])
def test_v3_detects_all_declared_wheels_severities_and_sample_rates(
    fault_wheel: int,
    retention: float,
    expected_severity: str,
    sample_rate_hz: float,
) -> None:
    frame = _authority_frame(
        fault_wheel,
        retention,
        sample_rate_hz=sample_rate_hz,
    )
    observations, detections, anomalies = detect_rw4_authority_robust(
        frame,
        timestamp_column="time_s",
        channels=_channels(frame),
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )

    assert observations
    assert len(detections) == 1
    assert len(anomalies) == 1
    detection = detections[0]
    assert detection.detector_id == RW4_AUTHORITY_ROBUST_DETECTOR_ID
    assert detection.metadata["wheel_ranking"][0]["wheel_id"] == f"RW{fault_wheel}"
    assert detection.metadata["estimated_severity"] == expected_severity
    assert detection.metadata["labels_used"] is False
    assert detection.metadata["truth_used"] is False
    assert all(detection.metadata["gate_checks"].values())
    assert abs(float(detection.observed_at) - 20.0) <= 1.0


@pytest.mark.parametrize("shared_bias_a", [-0.08, 0.0, 0.08])
@pytest.mark.parametrize("sample_rate_hz", [0.5, 1.0, 2.0])
def test_v3_shared_bias_normal_controls_fail_closed(
    shared_bias_a: float,
    sample_rate_hz: float,
) -> None:
    frame = _authority_frame(
        None,
        sample_rate_hz=sample_rate_hz,
        shared_bias_a=shared_bias_a,
        noise_scale_a=0.002,
    )
    observations, detections, anomalies = detect_rw4_authority_robust(
        frame,
        timestamp_column="time_s",
        channels=_channels(frame),
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )
    assert observations
    assert detections == []
    assert anomalies == []


def test_v3_stable_timestamp_sort_and_truth_exclusion_are_invariant() -> None:
    frame = _authority_frame(4, 0.50, sample_rate_hz=1.0)
    frame["label.fault_active"] = frame["time_s"] >= 20.0
    frame["truth.fault_wheel"] = 4
    shuffled = frame.sample(frac=1.0, random_state=20260725).reset_index(drop=True)

    ordered = detect_rw4_authority_robust(
        frame,
        timestamp_column="time_s",
        channels=list(frame.columns),
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )[1]
    unordered = detect_rw4_authority_robust(
        shuffled,
        timestamp_column="time_s",
        channels=list(shuffled.columns),
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )[1]

    assert len(ordered) == len(unordered) == 1
    assert ordered[0].anomaly_id == unordered[0].anomaly_id
    assert ordered[0].observed_at == unordered[0].observed_at
    assert ordered[0].metadata["estimated_severity"] == unordered[0].metadata["estimated_severity"]
    assert unordered[0].metadata["timestamp_sort_applied"] is True
    assert unordered[0].metadata["raw_out_of_order_pair_count"] > 0


def test_v3_falls_back_to_frozen_v2_for_legacy_nine_field_data() -> None:
    frame = _legacy_collapse_frame(2)
    observations, detections, anomalies = detect_rw4_authority_robust(
        frame,
        timestamp_column="time_s",
        channels=_channels(frame),
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )
    assert observations
    assert len(detections) == 1
    assert len(anomalies) == 1
    assert detections[0].detector_id == RW4_AUTHORITY_ROBUST_DETECTOR_ID
    assert detections[0].metadata["fallback_detector_id"] == RW4_SINGLE_FAULT_DETECTOR_ID
    assert detections[0].metadata["detection_mode"] == "current_collapse_v2_fallback"
    assert anomalies[0].metadata["wheel_id"] == "RW2"


def test_v3_rejects_partially_enhanced_command_response_schema() -> None:
    frame = _authority_frame(1, 0.5).drop(columns=["rw4_command_current_a"])
    observations, detections, anomalies = detect_rw4_authority_robust(
        frame,
        timestamp_column="time_s",
        channels=_channels(frame),
        baseline_fraction=0.2,
        z_threshold=6.0,
        minimum_baseline_points=10,
    )
    assert detections == []
    assert anomalies == []
    assert observations[0].metadata["status"] == "UNSUPPORTED_SCHEMA"
    assert observations[0].metadata["missing_command_response_pairs"] == ["RW4"]
    assert observations[0].metadata["fail_closed"] is True


def test_cli_rw4_selector_uses_v3_by_default_and_keeps_explicit_v2() -> None:
    robust_detector, robust_id = _dataset_detector_selection(
        SimpleNamespace(detector="rw4-single-fault")
    )
    frozen_detector, frozen_id = _dataset_detector_selection(
        SimpleNamespace(detector="rw4-single-fault-v2")
    )
    assert robust_detector is detect_rw4_authority_robust
    assert robust_id == RW4_AUTHORITY_ROBUST_DETECTOR_ID
    assert frozen_detector is detect_rw4_single_fault
    assert frozen_id == RW4_SINGLE_FAULT_DETECTOR_ID
