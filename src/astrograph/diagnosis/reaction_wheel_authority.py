from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from astrograph.audit.store import sha256_json
from astrograph.schemas import AnomalyEvidence

from .dataset_models import DetectionRecord, ObservationFact
from .reaction_wheel_single_fault import (
    RW4_SINGLE_FAULT_DETECTOR_ID as RW4_V2_DETECTOR_ID,
    detect_rw4_single_fault as detect_rw4_single_fault_v2,
    resolve_rw4_signal_columns,
    rw4_anomaly_id,
    rw4_anomaly_name,
    rw4_loss_severity,
)


RW4_AUTHORITY_ROBUST_DETECTOR_ID = (
    "astrograph-rw4-command-response-robust-change-point.v3"
)
RW4_AUTHORITY_ROBUST_SCHEMA_VERSION = (
    "astrograph-rw4-authority-robust-change-point.v3"
)
RW4_AUTHORITY_ROBUST_LEGACY_DETECTOR_IDS = (
    RW4_V2_DETECTOR_ID,
    "astrograph-rw4-current-collapse-change-point.v1",
)


@dataclass(frozen=True)
class RW4AuthorityRobustDetectorConfig:
    """Engineering gates for persistent wheel command/response gain changes.

    The v3 detector estimates one shared additive sensor bias and independent
    pre/post command-response gains at each candidate change point.  Time-based
    windows make the evidence comparable across 0.5--2 Hz telemetry.  The
    change-strength gate is intentionally conservative because the detector
    scans many wheel/time hypotheses; it is not a calibrated probability.
    """

    minimum_healthy_baseline_s: float = 8.0
    pre_window_s: float = 12.0
    post_window_s: float = 30.0
    minimum_post_span_s: float = 12.0
    minimum_pre_points: int = 3
    minimum_post_points: int = 5
    minimum_current_command_activity_a: float = 0.02
    minimum_torque_command_activity_nm: float = 0.0008
    current_residual_noise_floor_a: float = 0.002
    torque_residual_noise_floor_nm: float = 0.00008
    minimum_healthy_gain: float = 0.75
    maximum_healthy_gain: float = 1.25
    minimum_delivery_loss: float = 0.12
    minimum_gain_step: float = 0.10
    minimum_change_strength_z: float = 7.0
    minimum_persistent_loss: float = 0.09
    minimum_model_improvement: float = 0.15
    minimum_wheel_dominance_ratio: float = 1.35
    minimum_wheel_dominance_margin: float = 0.06

    def __post_init__(self) -> None:
        positive = (
            "minimum_healthy_baseline_s",
            "pre_window_s",
            "post_window_s",
            "minimum_post_span_s",
            "minimum_current_command_activity_a",
            "minimum_torque_command_activity_nm",
            "current_residual_noise_floor_a",
            "torque_residual_noise_floor_nm",
            "minimum_healthy_gain",
            "maximum_healthy_gain",
            "minimum_delivery_loss",
            "minimum_gain_step",
            "minimum_change_strength_z",
            "minimum_persistent_loss",
            "minimum_model_improvement",
            "minimum_wheel_dominance_ratio",
            "minimum_wheel_dominance_margin",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_pre_points < 3 or self.minimum_post_points < 3:
            raise ValueError("minimum pre/post points must each be at least three")
        if self.maximum_healthy_gain <= self.minimum_healthy_gain:
            raise ValueError("maximum_healthy_gain must exceed minimum_healthy_gain")
        if self.minimum_healthy_baseline_s > self.pre_window_s:
            raise ValueError("minimum healthy baseline cannot exceed pre_window_s")
        if self.minimum_post_span_s > self.post_window_s:
            raise ValueError("minimum post span cannot exceed post_window_s")


DEFAULT_RW4_AUTHORITY_ROBUST_CONFIG = RW4AuthorityRobustDetectorConfig()


def _stable_id(prefix: str, payload: Any, length: int = 24) -> str:
    return f"{prefix}-{sha256_json(payload).upper()[:length]}"


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _observed_time(frame: pd.DataFrame, timestamp_column: str | None, index: int) -> Any:
    if timestamp_column and timestamp_column in frame.columns:
        value = frame.iloc[index][timestamp_column]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float)):
            return value
        return str(value)
    return int(index)


def _bounded_ratio(value: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(reference)))


def _stable_time_sort(
    frame: pd.DataFrame,
    timestamp_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not timestamp_column or timestamp_column not in frame.columns:
        return frame.reset_index(drop=True), {
            "timestamp_sort_applied": False,
            "raw_out_of_order_pair_count": 0,
            "timestamp_numeric": False,
        }
    numeric = pd.to_numeric(frame[timestamp_column], errors="coerce")
    if numeric.isna().any():
        return frame.reset_index(drop=True), {
            "timestamp_sort_applied": False,
            "raw_out_of_order_pair_count": 0,
            "timestamp_numeric": False,
        }
    disorder = int((numeric.diff().fillna(0.0) < 0.0).sum())
    working = frame.assign(__rw4_arrival_order=range(len(frame)))
    working = working.sort_values(
        [timestamp_column, "__rw4_arrival_order"], kind="mergesort"
    ).drop(columns="__rw4_arrival_order")
    return working.reset_index(drop=True), {
        "timestamp_sort_applied": True,
        "raw_out_of_order_pair_count": disorder,
        "timestamp_numeric": True,
    }


def _fit_origin_with_bias_removed(
    command: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    *,
    bias: float,
    minimum_points: int,
) -> float | None:
    x = command[mask]
    y = actual[mask] - float(bias)
    if len(x) < minimum_points:
        return None
    energy = float(np.sum(x * x))
    if energy <= 1.0e-8:
        return None
    return float(np.sum(x * y) / energy)


def _piecewise_shared_bias_candidate(
    frame: pd.DataFrame,
    *,
    wheel: int,
    command_signal: str,
    actual_signal: str,
    speed_signal: str | None,
    timestamp_column: str,
    candidate_index: int,
    command_activity: float,
    residual_noise_floor: float,
    config: RW4AuthorityRobustDetectorConfig,
) -> dict[str, Any] | None:
    time = pd.to_numeric(frame[timestamp_column], errors="coerce").to_numpy(dtype=float)
    command = pd.to_numeric(frame[command_signal], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(frame[actual_signal], errors="coerce").to_numpy(dtype=float)
    candidate_time = float(time[candidate_index])
    finite = np.isfinite(time) & np.isfinite(command) & np.isfinite(actual)
    active = np.abs(command) >= float(command_activity)
    in_window = (
        (time >= candidate_time - float(config.pre_window_s))
        & (time <= candidate_time + float(config.post_window_s))
    )
    eligible = finite & active & in_window
    pre = eligible & (time < candidate_time)
    post = eligible & (time >= candidate_time)
    if int(pre.sum()) < config.minimum_pre_points:
        return None
    if int(post.sum()) < config.minimum_post_points:
        return None
    pre_times = time[pre]
    post_times = time[post]
    pre_span = candidate_time - float(np.min(pre_times))
    post_span = float(np.max(post_times)) - candidate_time
    if pre_span < config.minimum_healthy_baseline_s:
        return None
    if post_span < config.minimum_post_span_s:
        return None

    x = command[eligible]
    y = actual[eligible]
    is_pre = pre[eligible].astype(float)
    is_post = post[eligible].astype(float)
    design = np.column_stack((x * is_pre, x * is_post, np.ones(len(x))))
    if np.linalg.matrix_rank(design) < 3:
        return None
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ beta
    residual = y - predicted
    residual_median = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_median)))
    residual_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    residual_scale = max(
        1.4826 * residual_mad,
        0.5 * residual_std,
        float(residual_noise_floor),
    )
    covariance = np.linalg.pinv(design.T @ design) * residual_scale**2
    gain_difference = float(beta[0] - beta[1])
    difference_variance = max(
        0.0,
        float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]),
    )
    difference_se = math.sqrt(difference_variance)
    pre_gain = float(beta[0])
    post_gain = float(beta[1])
    shared_bias = float(beta[2])
    normalized_loss = max(
        0.0,
        min(1.5, 1.0 - post_gain / max(pre_gain, 1.0e-6)),
    )
    positive_gain_step = max(0.0, gain_difference)
    change_strength_z = max(0.0, gain_difference / max(difference_se, 1.0e-9))

    no_change_design = np.column_stack((x, np.ones(len(x))))
    no_change_beta, *_ = np.linalg.lstsq(no_change_design, y, rcond=None)
    no_change_residual = y - no_change_design @ no_change_beta
    no_change_sse = float(np.sum(no_change_residual * no_change_residual))
    change_sse = float(np.sum(residual * residual))
    model_improvement = max(
        0.0,
        min(1.0, (no_change_sse - change_sse) / max(no_change_sse, 1.0e-12)),
    )

    post_midpoint = candidate_time + float(config.post_window_s) / 2.0
    early_post = post & (time < post_midpoint)
    late_post = post & (time >= post_midpoint)
    early_gain = _fit_origin_with_bias_removed(
        command,
        actual,
        early_post,
        bias=shared_bias,
        minimum_points=max(2, config.minimum_post_points // 2),
    )
    late_gain = _fit_origin_with_bias_removed(
        command,
        actual,
        late_post,
        bias=shared_bias,
        minimum_points=max(2, config.minimum_post_points // 2),
    )
    persistent_losses = [
        max(0.0, min(1.5, 1.0 - gain / max(pre_gain, 1.0e-6)))
        for gain in (early_gain, late_gain)
        if gain is not None
    ]
    persistent_loss = min(persistent_losses) if len(persistent_losses) == 2 else 0.0

    speed_support = 0.5
    if speed_signal and speed_signal in frame.columns:
        speed = pd.to_numeric(frame[speed_signal], errors="coerce")
        speed_step = speed.diff().abs()
        speed_window = speed_step.iloc[
            candidate_index : min(len(frame), candidate_index + config.minimum_post_points)
        ].dropna()
        if len(speed_window):
            median_step = float(speed_window.median())
            speed_support = 1.0 / (1.0 + median_step / 0.02)

    signature_score = (
        0.27 * _bounded_ratio(normalized_loss, 0.75)
        + 0.25 * _bounded_ratio(change_strength_z, 20.0)
        + 0.20 * _bounded_ratio(persistent_loss, 0.75)
        + 0.18 * _bounded_ratio(model_improvement, 0.80)
        + 0.10 * speed_support
    )
    objective = (
        normalized_loss
        * min(change_strength_z, 20.0)
        * persistent_loss
        * max(model_improvement, 0.01)
    )
    return {
        "wheel_id": f"RW{wheel}",
        "wheel_index_one_based": wheel,
        "command_signal": command_signal,
        "actual_signal": actual_signal,
        "current_signal": actual_signal,
        "speed_signal": speed_signal,
        "data_index": int(candidate_index),
        "observed_time": candidate_time,
        "pre_command_response_gain": pre_gain,
        "post_command_response_gain": post_gain,
        "shared_additive_bias": shared_bias,
        "estimated_retention_fraction": max(0.0, min(1.0, post_gain / max(pre_gain, 1.0e-6))),
        "estimated_loss_fraction": max(0.0, min(1.0, normalized_loss)),
        "estimated_severity": rw4_loss_severity(normalized_loss),
        "gain_step": positive_gain_step,
        "gain_step_standard_error": difference_se,
        "change_strength_z": change_strength_z,
        "persistent_loss_fraction": persistent_loss,
        "model_improvement_fraction": model_improvement,
        "pre_eligible_points": int(pre.sum()),
        "post_eligible_points": int(post.sum()),
        "pre_time_span_s": pre_span,
        "post_time_span_s": post_span,
        "command_activity_threshold": command_activity,
        "residual_scale": residual_scale,
        "speed_quiet_support": speed_support,
        "signature_score": max(0.0, min(0.999, signature_score)),
        "selection_objective": objective,
        "observable_window": True,
    }


def _unsupported_schema_observation(
    *,
    frame: pd.DataFrame,
    resolved: Mapping[int, Mapping[str, str]],
    duplicates: list[dict[str, Any]],
    missing_pairs: list[str],
) -> ObservationFact:
    payload = {
        "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
        "schema_version": RW4_AUTHORITY_ROBUST_SCHEMA_VERSION,
        "resolved_signals": resolved,
        "duplicate_aliases": duplicates,
        "missing_command_response_pairs": missing_pairs,
    }
    return ObservationFact(
        observation_id=_stable_id("OBS", payload),
        signal="rw4_schema",
        sample_count=len(frame),
        missing_count=0,
        metadata={**payload, "status": "UNSUPPORTED_SCHEMA", "fail_closed": True},
    )


def _rewrite_v2_fallback(
    observations: list[ObservationFact],
    detections: list[DetectionRecord],
    anomalies: list[AnomalyEvidence],
) -> tuple[list[ObservationFact], list[DetectionRecord], list[AnomalyEvidence]]:
    rewritten_observations = []
    for item in observations:
        metadata = dict(item.metadata)
        metadata.update(
            {
                "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
                "fallback_detector_id": RW4_V2_DETECTOR_ID,
                "fallback_reason": "enhanced command/response pairs unavailable",
            }
        )
        rewritten_observations.append(item.model_copy(update={"metadata": metadata}))
    rewritten_detections = []
    for item in detections:
        metadata = dict(item.metadata)
        metadata.update(
            {
                "schema_version": RW4_AUTHORITY_ROBUST_SCHEMA_VERSION,
                "fallback_detector_id": RW4_V2_DETECTOR_ID,
                "detection_mode": "current_collapse_v2_fallback",
            }
        )
        rewritten_detections.append(
            item.model_copy(
                update={
                    "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
                    "metadata": metadata,
                }
            )
        )
    rewritten_anomalies = []
    for item in anomalies:
        metadata = dict(item.metadata)
        metadata.update(
            {
                "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
                "fallback_detector_id": RW4_V2_DETECTOR_ID,
            }
        )
        rewritten_anomalies.append(item.model_copy(update={"metadata": metadata}))
    return rewritten_observations, rewritten_detections, rewritten_anomalies


def detect_rw4_authority_robust(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    channels: list[str],
    baseline_fraction: float,
    z_threshold: float,
    minimum_baseline_points: int,
    signal_mapping: dict[str, dict[str, str]] | None = None,
    detector_config: RW4AuthorityRobustDetectorConfig | None = None,
) -> tuple[list[ObservationFact], list[DetectionRecord], list[AnomalyEvidence]]:
    """Detect one persistent RW authority loss with sample-rate-aware evidence.

    ``baseline_fraction``, ``z_threshold`` and ``minimum_baseline_points`` remain
    in the shared detector protocol.  The v3 detector records them but uses its
    explicit time-domain gates.  Labels and truth channels are excluded before
    signal resolution and are never inspected.
    """

    config = detector_config or DEFAULT_RW4_AUTHORITY_ROBUST_CONFIG
    mapping = signal_mapping or {}
    if frame.empty:
        return [], [], []
    usable_channels = [
        str(item)
        for item in channels
        if item in frame.columns
        and not str(item).lower().startswith(
            ("label.", "label_", "truth.", "truth_", "ground_truth")
        )
    ]
    resolved, duplicates = resolve_rw4_signal_columns(usable_channels)
    current_pairs_ready = all(
        "command_current_a" in resolved[wheel] and "current_a" in resolved[wheel]
        for wheel in range(1, 5)
    )
    torque_pairs_ready = all(
        "command_torque_nm" in resolved[wheel]
        and "actual_torque_nm" in resolved[wheel]
        for wheel in range(1, 5)
    )
    enhanced_slots_present = any(
        any(
            kind in resolved[wheel]
            for kind in (
                "command_current_a",
                "command_torque_nm",
                "actual_torque_nm",
            )
        )
        for wheel in range(1, 5)
    )
    if not enhanced_slots_present:
        return _rewrite_v2_fallback(
            *detect_rw4_single_fault_v2(
                frame,
                timestamp_column=timestamp_column,
                channels=channels,
                baseline_fraction=baseline_fraction,
                z_threshold=z_threshold,
                minimum_baseline_points=minimum_baseline_points,
                signal_mapping=signal_mapping,
            )
        )
    if duplicates or not (current_pairs_ready or torque_pairs_ready):
        missing_pairs = []
        for wheel in range(1, 5):
            if not (
                (
                    "command_current_a" in resolved[wheel]
                    and "current_a" in resolved[wheel]
                )
                or (
                    "command_torque_nm" in resolved[wheel]
                    and "actual_torque_nm" in resolved[wheel]
                )
            ):
                missing_pairs.append(f"RW{wheel}")
        return [
            _unsupported_schema_observation(
                frame=frame,
                resolved=resolved,
                duplicates=duplicates,
                missing_pairs=missing_pairs,
            )
        ], [], []
    if not timestamp_column or timestamp_column not in frame.columns:
        payload = {
            "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
            "schema_version": RW4_AUTHORITY_ROBUST_SCHEMA_VERSION,
            "status": "NUMERIC_TIMESTAMP_REQUIRED",
            "timestamp_column": timestamp_column or "",
            "fail_closed": True,
        }
        return [
            ObservationFact(
                observation_id=_stable_id("OBS", payload),
                signal="rw4_schema",
                sample_count=len(frame),
                missing_count=0,
                metadata=payload,
            )
        ], [], []

    working, time_sort = _stable_time_sort(frame, timestamp_column)
    numeric_time = pd.to_numeric(working[timestamp_column], errors="coerce")
    if numeric_time.isna().any() or len(working) < 8:
        payload = {
            "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
            "schema_version": RW4_AUTHORITY_ROBUST_SCHEMA_VERSION,
            "status": "INSUFFICIENT_NUMERIC_TIME_SERIES",
            "fail_closed": True,
            **time_sort,
        }
        return [
            ObservationFact(
                observation_id=_stable_id("OBS", payload),
                signal="rw4_schema",
                sample_count=len(working),
                missing_count=int(numeric_time.isna().sum()),
                metadata=payload,
            )
        ], [], []

    pair_kind = "current" if current_pairs_ready else "torque"
    command_kind = "command_current_a" if pair_kind == "current" else "command_torque_nm"
    actual_kind = "current_a" if pair_kind == "current" else "actual_torque_nm"
    command_activity = (
        config.minimum_current_command_activity_a
        if pair_kind == "current"
        else config.minimum_torque_command_activity_nm
    )
    residual_noise_floor = (
        config.current_residual_noise_floor_a
        if pair_kind == "current"
        else config.torque_residual_noise_floor_nm
    )
    time_values = numeric_time.to_numpy(dtype=float)
    first_time = float(time_values[0])
    last_time = float(time_values[-1])
    scans: list[dict[str, Any]] = []
    for candidate_index, candidate_time in enumerate(time_values):
        if candidate_time - first_time < config.minimum_healthy_baseline_s:
            continue
        if last_time - candidate_time < config.minimum_post_span_s:
            continue
        candidates: list[dict[str, Any]] = []
        for wheel in range(1, 5):
            candidate = _piecewise_shared_bias_candidate(
                working,
                wheel=wheel,
                command_signal=resolved[wheel][command_kind],
                actual_signal=resolved[wheel][actual_kind],
                speed_signal=resolved[wheel].get("speed_rad_s"),
                timestamp_column=timestamp_column,
                candidate_index=candidate_index,
                command_activity=command_activity,
                residual_noise_floor=residual_noise_floor,
                config=config,
            )
            if candidate is not None:
                candidates.append(candidate)
        if len(candidates) != 4:
            continue
        candidates.sort(
            key=lambda item: (
                -float(item["estimated_loss_fraction"]),
                -float(item["change_strength_z"]),
                str(item["wheel_id"]),
            )
        )
        top = candidates[0]
        second_loss = float(candidates[1]["estimated_loss_fraction"])
        top_loss = float(top["estimated_loss_fraction"])
        dominance_ratio = top_loss / max(second_loss, 1.0e-9)
        dominance_margin = top_loss - second_loss
        top["wheel_dominance_ratio"] = dominance_ratio
        top["wheel_dominance_margin"] = dominance_margin
        gate_checks = {
            "observable_time_window": bool(top["observable_window"]),
            "healthy_pre_gain": (
                config.minimum_healthy_gain
                <= float(top["pre_command_response_gain"])
                <= config.maximum_healthy_gain
            ),
            "delivery_loss": top_loss >= config.minimum_delivery_loss,
            "gain_step": float(top["gain_step"]) >= config.minimum_gain_step,
            "multiple_scan_change_strength": (
                float(top["change_strength_z"])
                >= config.minimum_change_strength_z
            ),
            "persistent_delivery_loss": (
                float(top["persistent_loss_fraction"])
                >= config.minimum_persistent_loss
            ),
            "piecewise_model_improvement": (
                float(top["model_improvement_fraction"])
                >= config.minimum_model_improvement
            ),
            "cross_wheel_dominance_ratio": (
                dominance_ratio >= config.minimum_wheel_dominance_ratio
            ),
            "cross_wheel_dominance_margin": (
                dominance_margin >= config.minimum_wheel_dominance_margin
            ),
        }
        scans.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_time": float(candidate_time),
                "candidates": candidates,
                "top": top,
                "gate_checks": gate_checks,
                "passed": all(gate_checks.values()),
            }
        )

    if not scans:
        payload = {
            "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
            "schema_version": RW4_AUTHORITY_ROBUST_SCHEMA_VERSION,
            "status": "NO_OBSERVABLE_CANDIDATE_WINDOW",
            "fail_closed": True,
            "detector_config": config.__dict__,
            "requested_baseline_fraction": baseline_fraction,
            "requested_z_threshold": z_threshold,
            "requested_minimum_baseline_points": minimum_baseline_points,
            **time_sort,
        }
        return [
            ObservationFact(
                observation_id=_stable_id("OBS", payload),
                signal="rw4_schema",
                sample_count=len(working),
                missing_count=0,
                metadata=payload,
            )
        ], [], []

    passed = [item for item in scans if bool(item["passed"])]
    selectable = passed or scans
    selected = max(
        selectable,
        key=lambda item: (
            float(item["top"]["model_improvement_fraction"]),
            float(item["top"]["change_strength_z"]),
            float(item["top"]["selection_objective"]),
            -float(item["candidate_time"]),
        ),
    )
    candidates = [dict(item) for item in selected["candidates"]]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        if rank > 1:
            candidate.setdefault("wheel_dominance_ratio", 0.0)
            candidate.setdefault("wheel_dominance_margin", 0.0)
    top = candidates[0]
    gate_checks = dict(selected["gate_checks"])
    observations: list[ObservationFact] = []
    for candidate in candidates:
        actual_signal = str(candidate["actual_signal"])
        numeric = pd.to_numeric(working[actual_signal], errors="coerce")
        observations.append(
            ObservationFact(
                observation_id=_stable_id(
                    "OBS",
                    {
                        "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
                        "candidate": candidate,
                        "gate_checks": gate_checks,
                    },
                ),
                signal=actual_signal,
                sample_count=len(working),
                missing_count=int(numeric.isna().sum()),
                maximum_robust_score=float(candidate["change_strength_z"]),
                metadata={
                    "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
                    "schema_version": RW4_AUTHORITY_ROBUST_SCHEMA_VERSION,
                    "status": "CANDIDATE_RANKED",
                    "detection_mode": "piecewise_shared_bias_command_response",
                    "pair_kind": pair_kind,
                    "rank": int(candidate["rank"]),
                    "candidate": candidate,
                    "top_gate_checks": gate_checks,
                    "detection_emitted": bool(selected["passed"]),
                    "scanned_candidate_count": len(scans),
                    "detector_config": config.__dict__,
                    "score_is_calibrated_probability": False,
                    "labels_used": False,
                    "truth_used": False,
                    **time_sort,
                },
            )
        )
    if not bool(selected["passed"]):
        return observations, [], []

    wheel = int(top["wheel_index_one_based"])
    actual_signal = str(top["actual_signal"])
    configured = mapping.get(actual_signal, {})
    anomaly_id = str(configured.get("anomaly_id") or rw4_anomaly_id(wheel))
    anomaly_name = str(
        configured.get("name")
        or configured.get("anomaly_name")
        or rw4_anomaly_name(wheel)
    )
    event_index = int(selected["candidate_index"])
    observed_at = _observed_time(working, timestamp_column, event_index)
    dominance_ratio = float(top["wheel_dominance_ratio"])
    dominance_margin = float(top["wheel_dominance_margin"])
    confidence = max(
        0.0,
        min(
            0.999,
            0.84 * float(top["signature_score"])
            + 0.09 * _bounded_ratio(dominance_ratio, 3.0)
            + 0.07 * _bounded_ratio(dominance_margin, 0.40),
        ),
    )
    payload = {
        "anomaly_id": anomaly_id,
        "signal": actual_signal,
        "data_index": event_index,
        "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
        "ranking": candidates,
    }
    detection = DetectionRecord(
        detection_id=_stable_id("DET", payload),
        anomaly_id=anomaly_id,
        anomaly_name=anomaly_name,
        signal=actual_signal,
        detector_id=RW4_AUTHORITY_ROBUST_DETECTOR_ID,
        score=confidence,
        confidence=confidence,
        observed_at=observed_at,
        missing_fraction=float(
            pd.to_numeric(working[actual_signal], errors="coerce").isna().mean()
        ),
        mapped=True,
        metadata={
            "schema_version": RW4_AUTHORITY_ROBUST_SCHEMA_VERSION,
            "detection_mode": "piecewise_shared_bias_command_response",
            "pair_kind": pair_kind,
            "score_is_calibrated_probability": False,
            "score_semantics": (
                "E0 persistent wheel command/response gain-change evidence strength; "
                "not a calibrated probability"
            ),
            "gate_checks": gate_checks,
            "detector_config": config.__dict__,
            "wheel_ranking": candidates,
            "estimated_retention_fraction": top["estimated_retention_fraction"],
            "estimated_loss_fraction": top["estimated_loss_fraction"],
            "estimated_severity": top["estimated_severity"],
            "joint_evidence": {
                "command_signal": top["command_signal"],
                "actual_signal": top["actual_signal"],
                "pre_command_response_gain": top["pre_command_response_gain"],
                "post_command_response_gain": top["post_command_response_gain"],
                "shared_additive_bias": top["shared_additive_bias"],
                "gain_step": top["gain_step"],
                "gain_step_standard_error": top["gain_step_standard_error"],
                "change_strength_z": top["change_strength_z"],
                "persistent_loss_fraction": top["persistent_loss_fraction"],
                "piecewise_model_improvement_fraction": top[
                    "model_improvement_fraction"
                ],
                "cross_wheel_dominance_ratio": dominance_ratio,
                "cross_wheel_dominance_margin": dominance_margin,
                "pre_time_span_s": top["pre_time_span_s"],
                "post_time_span_s": top["post_time_span_s"],
            },
            "requested_common_detector_fields": {
                "baseline_fraction": baseline_fraction,
                "z_threshold": z_threshold,
                "minimum_baseline_points": minimum_baseline_points,
            },
            "labels_used": False,
            "truth_used": False,
            "claim_boundary": (
                "E0 finite engineering-synthetic persistent RW authority-loss "
                "signature only; requires an observable healthy baseline and "
                "persistent command excitation; not calibrated, flight-qualified, "
                "or a probability of fault."
            ),
            **time_sort,
        },
    )
    observed_at_s = _safe_float(observed_at)
    anomaly = AnomalyEvidence(
        anomaly_id=anomaly_id,
        name=anomaly_name,
        signal=actual_signal,
        confidence=confidence,
        confidence_type="rw4_e0_robust_gain_change_strength_not_calibrated",
        observed_at_s=(
            observed_at_s
            if observed_at_s is not None and observed_at_s >= 0.0
            else None
        ),
        metadata={
            "detection_id": detection.detection_id,
            "detector_id": RW4_AUTHORITY_ROBUST_DETECTOR_ID,
            "mapped_to_registered_semantics": True,
            "wheel_id": f"RW{wheel}",
            "estimated_loss_fraction": top["estimated_loss_fraction"],
            "estimated_severity": top["estimated_severity"],
            "wheel_ranking": candidates,
            "labels_used": False,
            "truth_used": False,
        },
    )
    return observations, [detection], [anomaly]


__all__ = [
    "RW4_AUTHORITY_ROBUST_DETECTOR_ID",
    "RW4_AUTHORITY_ROBUST_SCHEMA_VERSION",
    "RW4_AUTHORITY_ROBUST_LEGACY_DETECTOR_IDS",
    "RW4AuthorityRobustDetectorConfig",
    "DEFAULT_RW4_AUTHORITY_ROBUST_CONFIG",
    "detect_rw4_authority_robust",
]
