from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from astrograph.audit.store import sha256_json
from astrograph.schemas import AnomalyEvidence

from .dataset_models import DetectionRecord, ObservationFact


RW4_SINGLE_FAULT_DETECTOR_ID = "astrograph-rw4-command-response-change-point.v2"
RW4_SINGLE_FAULT_LEGACY_DETECTOR_IDS = (
    "astrograph-rw4-current-collapse-change-point.v1",
)
RW4_SINGLE_FAULT_SCHEMA_VERSION = "astrograph-rw4-single-fault-detector.v2"

_SHORT_SIGNAL = re.compile(
    r"^rw(?P<wheel>[1-4])_(?P<kind>speed_rad_s|current_a|command_current_a|command_torque_nm|actual_torque_nm)$",
    re.IGNORECASE,
)
_CANONICAL_SIGNAL = re.compile(
    r"^(?:(?:sensor|estimate)\.)?adcs\.rw\.(?P<kind>speed_rad_s|motor_current_a|command_current_a|command_torque_nm|actual_torque_nm|applied_torque_nm)_(?P<index>[0-3])$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RW4SingleFaultDetectorConfig:
    """E0 detector settings for the four-wheel single-authority-loss schema.

    These thresholds are engineering gates for the deterministic RW4 generator,
    not calibrated flight limits or probabilities. The detector is deliberately
    separate from the generic robust-baseline detector.
    """

    maximum_warmup_fraction: float = 0.10
    pre_window_points: int = 5
    post_window_points: int = 8
    minimum_current_step_a: float = 0.05
    minimum_current_collapse_ratio: float = 0.80
    minimum_step_to_pre_activity_ratio: float = 0.75
    minimum_wheel_dominance_ratio: float = 2.40
    speed_quiet_reference_rad_s_per_sample: float = 0.02
    minimum_command_activity: float = 0.08
    minimum_pre_eligible_points: int = 3
    minimum_post_eligible_points: int = 5
    maximum_pre_delivery_loss: float = 0.12
    minimum_delivery_loss: float = 0.12
    minimum_delivery_loss_step: float = 0.10
    minimum_delivery_persistence: float = 0.75
    minimum_delivery_dominance_ratio: float = 1.35
    minimum_delivery_dominance_margin: float = 0.06

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_warmup_fraction < 0.5:
            raise ValueError("maximum_warmup_fraction must be in (0, 0.5)")
        if self.pre_window_points < 3 or self.post_window_points < 3:
            raise ValueError("pre/post windows must each contain at least three points")
        if not 1 <= self.minimum_pre_eligible_points <= self.pre_window_points:
            raise ValueError("minimum_pre_eligible_points must fit inside pre_window_points")
        if not 1 <= self.minimum_post_eligible_points <= self.post_window_points:
            raise ValueError("minimum_post_eligible_points must fit inside post_window_points")
        for name in (
            "minimum_current_step_a",
            "minimum_current_collapse_ratio",
            "minimum_step_to_pre_activity_ratio",
            "minimum_wheel_dominance_ratio",
            "speed_quiet_reference_rad_s_per_sample",
            "minimum_command_activity",
            "maximum_pre_delivery_loss",
            "minimum_delivery_loss",
            "minimum_delivery_loss_step",
            "minimum_delivery_persistence",
            "minimum_delivery_dominance_ratio",
            "minimum_delivery_dominance_margin",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


DEFAULT_RW4_SINGLE_FAULT_CONFIG = RW4SingleFaultDetectorConfig()


def _stable_id(prefix: str, payload: Any, length: int = 24) -> str:
    digest = sha256_json(payload).upper()
    return f"{prefix}-{digest[:length]}"


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


def _signal_semantics(signal: str) -> tuple[int, str] | None:
    short = _SHORT_SIGNAL.fullmatch(str(signal))
    if short:
        return int(short.group("wheel")), short.group("kind").lower()
    canonical = _CANONICAL_SIGNAL.fullmatch(str(signal))
    if canonical:
        kind = canonical.group("kind").lower()
        normalized_kind = {
            "motor_current_a": "current_a",
            "applied_torque_nm": "actual_torque_nm",
        }.get(kind, kind)
        return int(canonical.group("index")) + 1, normalized_kind
    return None


def resolve_rw4_signal_columns(
    columns: Iterable[str],
) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]]]:
    """Resolve compact one-based and canonical zero-based RW field names.

    The first occurrence for one semantic slot is retained deterministically.
    Duplicate aliases are reported so callers can fail closed or surface an
    auditable warning instead of silently mixing two channels.
    """

    resolved: dict[int, dict[str, str]] = {wheel: {} for wheel in range(1, 5)}
    duplicates: list[dict[str, Any]] = []
    for raw in columns:
        signal = str(raw)
        semantics = _signal_semantics(signal)
        if semantics is None:
            continue
        wheel, kind = semantics
        previous = resolved[wheel].get(kind)
        if previous is None:
            resolved[wheel][kind] = signal
        elif previous != signal:
            duplicates.append(
                {
                    "wheel_id": f"RW{wheel}",
                    "kind": kind,
                    "selected": previous,
                    "ignored": signal,
                }
            )
    return resolved, duplicates


def rw4_anomaly_id(wheel: int) -> str:
    if wheel not in {1, 2, 3, 4}:
        raise ValueError("wheel must be in 1..4")
    return f"ANOM_RW{wheel}_TORQUE_AUTHORITY_LOSS_SIGNATURE"


def rw4_anomaly_name(wheel: int) -> str:
    return f"Reaction wheel {wheel} current-collapse torque-authority-loss signature"


def _robust_location_scale(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0, 1e-9
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        std = float(clean.std(ddof=0))
        scale = std if np.isfinite(std) and std > 1e-12 else max(abs(median) * 1e-3, 1e-9)
    return median, scale


def _bounded_ratio(value: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(reference)))


def _candidate_for_wheel(
    frame: pd.DataFrame,
    *,
    wheel: int,
    current_signal: str,
    speed_signal: str | None,
    warmup_points: int,
    config: RW4SingleFaultDetectorConfig,
) -> dict[str, Any]:
    current = pd.to_numeric(frame[current_signal], errors="coerce")
    current_step = current.diff().abs()
    baseline_step = current_step.iloc[1:warmup_points]
    baseline_median, baseline_scale = _robust_location_scale(baseline_step)

    last_candidate_index = len(frame) - config.post_window_points
    candidate_indices = range(warmup_points, max(warmup_points, last_candidate_index + 1))
    scored: list[dict[str, Any]] = []
    for index in candidate_indices:
        step = _safe_float(current_step.iloc[index])
        if step is None:
            continue
        pre = current.iloc[max(0, index - config.pre_window_points) : index].dropna().abs()
        post = current.iloc[index : index + config.post_window_points].dropna().abs()
        if len(pre) < 3 or len(post) < 3:
            continue
        pre_activity = float(pre.median())
        post_activity = float(post.median())
        current_collapse = max(
            0.0,
            min(1.0, 1.0 - post_activity / max(pre_activity, 1e-9)),
        )
        step_to_pre = step / max(pre_activity, 1e-9)
        robust_step_strength = max(0.0, (step - baseline_median) / max(baseline_scale, 1e-9))

        speed_pre_step = None
        speed_post_step = None
        speed_quiet_support = 0.5
        if speed_signal and speed_signal in frame.columns:
            speed = pd.to_numeric(frame[speed_signal], errors="coerce")
            speed_step = speed.diff().abs()
            pre_speed = speed_step.iloc[max(1, index - config.pre_window_points) : index].dropna()
            post_speed = speed_step.iloc[index : index + config.post_window_points].dropna()
            if len(pre_speed) >= 2 and len(post_speed) >= 2:
                speed_pre_step = float(pre_speed.median())
                speed_post_step = float(post_speed.median())
                speed_quiet_support = 1.0 / (
                    1.0
                    + speed_post_step
                    / max(config.speed_quiet_reference_rad_s_per_sample, 1e-9)
                )

        signature_score = (
            0.35 * _bounded_ratio(step, 0.08)
            + 0.30 * current_collapse
            + 0.20 * _bounded_ratio(step_to_pre, 1.0)
            + 0.15 * speed_quiet_support
        )
        scored.append(
            {
                "wheel_id": f"RW{wheel}",
                "wheel_index_one_based": wheel,
                "current_signal": current_signal,
                "speed_signal": speed_signal,
                "data_index": int(index),
                "current_step_a": float(step),
                "pre_current_abs_median_a": pre_activity,
                "post_current_abs_median_a": post_activity,
                "current_collapse_ratio": current_collapse,
                "step_to_pre_activity_ratio": step_to_pre,
                "baseline_step_median_a": baseline_median,
                "baseline_step_robust_scale_a": baseline_scale,
                "robust_step_strength": robust_step_strength,
                "speed_pre_abs_step_median_rad_s_per_sample": speed_pre_step,
                "speed_post_abs_step_median_rad_s_per_sample": speed_post_step,
                "speed_quiet_support": speed_quiet_support,
                "signature_score": max(0.0, min(0.999, signature_score)),
            }
        )

    if not scored:
        return {
            "wheel_id": f"RW{wheel}",
            "wheel_index_one_based": wheel,
            "current_signal": current_signal,
            "speed_signal": speed_signal,
            "status": "INSUFFICIENT_NUMERIC_WINDOW",
            "signature_score": 0.0,
            "current_step_a": 0.0,
            "current_collapse_ratio": 0.0,
            "step_to_pre_activity_ratio": 0.0,
        }
    return max(
        scored,
        key=lambda item: (
            item["signature_score"],
            item["current_step_a"],
            -item["data_index"],
        ),
    )



def rw4_loss_severity(loss_fraction: float) -> str:
    """Return an auditable engineering severity band for observed delivery loss."""

    value = max(0.0, min(1.0, float(loss_fraction)))
    if value >= 0.95:
        return "complete"
    if value >= 0.70:
        return "critical"
    if value >= 0.45:
        return "severe"
    if value >= 0.20:
        return "moderate"
    if value >= 0.10:
        return "mild"
    return "nominal_or_below_gate"


def _delivery_deficit_series(
    command: pd.Series,
    actual: pd.Series,
    *,
    minimum_command_activity: float,
) -> tuple[pd.Series, pd.Series]:
    command_numeric = pd.to_numeric(command, errors="coerce")
    actual_numeric = pd.to_numeric(actual, errors="coerce")
    activity = command_numeric.abs()
    eligible = (activity >= minimum_command_activity) & command_numeric.notna() & actual_numeric.notna()
    direction = np.sign(command_numeric)
    delivered_along_command = direction * actual_numeric
    retention = (delivered_along_command / activity.replace(0.0, np.nan)).clip(lower=0.0, upper=1.5)
    deficit = (1.0 - retention).clip(lower=0.0, upper=1.0).where(eligible)
    return deficit, eligible


def _delivery_window_candidate(
    frame: pd.DataFrame,
    *,
    wheel: int,
    command_signal: str,
    actual_signal: str,
    speed_signal: str | None,
    index: int,
    config: RW4SingleFaultDetectorConfig,
) -> dict[str, Any]:
    command = pd.to_numeric(frame[command_signal], errors="coerce")
    actual = pd.to_numeric(frame[actual_signal], errors="coerce")
    deficit, eligible = _delivery_deficit_series(
        command, actual, minimum_command_activity=config.minimum_command_activity
    )
    pre = deficit.iloc[max(0, index - config.pre_window_points) : index].dropna()
    post = deficit.iloc[index : index + config.post_window_points].dropna()
    post_commands = command.iloc[index : index + config.post_window_points].abs()
    pre_loss = float(pre.median()) if len(pre) else 1.0
    post_loss = float(post.median()) if len(post) else 0.0
    loss_step = max(0.0, post_loss - pre_loss)
    point_loss_value = _safe_float(deficit.iloc[index])
    point_loss = float(point_loss_value) if point_loss_value is not None else 0.0
    instant_loss_step = max(0.0, point_loss - pre_loss)
    persistence_threshold = max(
        config.minimum_delivery_loss * 0.75,
        pre_loss + config.minimum_delivery_loss_step * 0.75,
    )
    persistence = (
        float((post >= persistence_threshold).mean()) if len(post) else 0.0
    )
    post_eligible = eligible.iloc[index : index + config.post_window_points]
    command_activity = (
        float(post_commands[post_eligible].median())
        if int(post_eligible.sum())
        else 0.0
    )
    observable_window = (
        len(pre) >= config.minimum_pre_eligible_points
        and len(post) >= config.minimum_post_eligible_points
        and point_loss_value is not None
        and command_activity >= config.minimum_command_activity
    )
    signature_score = (
        0.32 * _bounded_ratio(post_loss, 0.75)
        + 0.28 * _bounded_ratio(loss_step, 0.60)
        + 0.22 * _bounded_ratio(instant_loss_step, 0.60)
        + 0.12 * persistence
        + 0.06 * _bounded_ratio(command_activity, 0.10)
    )
    speed_support = 0.5
    if speed_signal and speed_signal in frame.columns:
        speed = pd.to_numeric(frame[speed_signal], errors="coerce")
        speed_step = speed.diff().abs().iloc[index : index + config.post_window_points].dropna()
        if len(speed_step):
            speed_support = 1.0 / (1.0 + float(speed_step.median()) / max(config.speed_quiet_reference_rad_s_per_sample, 1e-9))
    return {
        "wheel_id": f"RW{wheel}",
        "wheel_index_one_based": wheel,
        "command_signal": command_signal,
        "actual_signal": actual_signal,
        "current_signal": actual_signal,
        "speed_signal": speed_signal,
        "data_index": int(index),
        "pre_delivery_loss_median": pre_loss,
        "post_delivery_loss_median": post_loss,
        "point_delivery_loss": point_loss,
        "delivery_loss_step": loss_step,
        "instant_delivery_loss_step": instant_loss_step,
        "delivery_persistence": persistence,
        "pre_eligible_points": int(len(pre)),
        "post_eligible_points": int(len(post)),
        "post_command_activity_median": command_activity,
        "speed_quiet_support": speed_support,
        "estimated_retention_fraction": max(0.0, min(1.0, 1.0 - post_loss)),
        "estimated_loss_fraction": max(0.0, min(1.0, post_loss)),
        "estimated_severity": rw4_loss_severity(post_loss),
        "observable_window": observable_window,
        "signature_score": (
            max(0.0, min(0.999, 0.90 * signature_score + 0.10 * speed_support))
            if observable_window
            else 0.0
        ),
    }


def _best_delivery_candidate(
    frame: pd.DataFrame,
    *,
    wheel: int,
    command_signal: str,
    actual_signal: str,
    speed_signal: str | None,
    warmup_points: int,
    config: RW4SingleFaultDetectorConfig,
) -> dict[str, Any]:
    last_candidate_index = len(frame) - config.post_window_points
    candidates = [
        _delivery_window_candidate(
            frame,
            wheel=wheel,
            command_signal=command_signal,
            actual_signal=actual_signal,
            speed_signal=speed_signal,
            index=index,
            config=config,
        )
        for index in range(warmup_points, max(warmup_points, last_candidate_index + 1))
    ]
    if not candidates:
        return {
            "wheel_id": f"RW{wheel}",
            "wheel_index_one_based": wheel,
            "command_signal": command_signal,
            "actual_signal": actual_signal,
            "current_signal": actual_signal,
            "speed_signal": speed_signal,
            "status": "INSUFFICIENT_NUMERIC_WINDOW",
            "signature_score": 0.0,
            "estimated_loss_fraction": 0.0,
            "delivery_loss_step": 0.0,
        }
    locally_gated = [
        item
        for item in candidates
        if bool(item.get("observable_window"))
        and float(item["pre_delivery_loss_median"]) <= config.maximum_pre_delivery_loss
        and float(item["post_delivery_loss_median"]) >= config.minimum_delivery_loss
        and float(item["delivery_loss_step"]) >= config.minimum_delivery_loss_step
        and float(item["instant_delivery_loss_step"]) >= config.minimum_delivery_loss_step
        and float(item["delivery_persistence"]) >= config.minimum_delivery_persistence
    ]
    if locally_gated:
        # Once a persistent, observable wheel-local loss signature is present,
        # the earliest qualifying sample is the auditable change point. This
        # avoids moving the reported onset deeper into a stable degradation only
        # because a later window has a marginally larger score.
        return min(
            locally_gated,
            key=lambda item: (
                int(item["data_index"]),
                -float(item["instant_delivery_loss_step"]),
                -float(item["signature_score"]),
            ),
        )
    return max(
        candidates,
        key=lambda item: (
            float(item["signature_score"]),
            float(item["instant_delivery_loss_step"]),
            float(item["delivery_loss_step"]),
            -int(item["data_index"]),
        ),
    )


def _detect_rw4_command_response(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    resolved: dict[int, dict[str, str]],
    pair_kind: str,
    baseline_fraction: float,
    minimum_baseline_points: int,
    mapping: dict[str, dict[str, str]],
    config: RW4SingleFaultDetectorConfig,
) -> tuple[list[ObservationFact], list[DetectionRecord], list[AnomalyEvidence]]:
    observations: list[ObservationFact] = []
    detections: list[DetectionRecord] = []
    anomalies: list[AnomalyEvidence] = []
    command_kind = "command_current_a" if pair_kind == "current" else "command_torque_nm"
    actual_kind = "current_a" if pair_kind == "current" else "actual_torque_nm"
    maximum_warmup = max(3, int(math.floor(len(frame) * config.maximum_warmup_fraction)))
    maximum_allowed = len(frame) - config.post_window_points - 1
    if maximum_allowed < 3:
        observations.append(
            ObservationFact(
                observation_id=_stable_id("OBS", {"mode": "command_response", "row_count": len(frame)}),
                signal="rw4_schema",
                sample_count=len(frame),
                missing_count=0,
                metadata={
                    "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
                    "schema_version": RW4_SINGLE_FAULT_SCHEMA_VERSION,
                    "status": "INSUFFICIENT_ROWS",
                    "minimum_required_rows": config.post_window_points + 4,
                    "fail_closed": True,
                },
            )
        )
        return observations, detections, anomalies
    preferred_minimum = min(max(3, minimum_baseline_points), maximum_warmup)
    requested_warmup = max(preferred_minimum, int(round(len(frame) * baseline_fraction)))
    warmup_points = min(requested_warmup, maximum_warmup, maximum_allowed)
    best = [
        _best_delivery_candidate(
            frame,
            wheel=wheel,
            command_signal=resolved[wheel][command_kind],
            actual_signal=resolved[wheel][actual_kind],
            speed_signal=resolved[wheel].get("speed_rad_s"),
            warmup_points=warmup_points,
            config=config,
        )
        for wheel in range(1, 5)
    ]
    event = max(
        best,
        key=lambda item: (
            float(item.get("signature_score", 0.0)),
            float(item.get("delivery_loss_step", 0.0)),
            -int(item.get("data_index", len(frame))),
        ),
    )
    event_index = int(event.get("data_index", warmup_points))
    candidates = [
        _delivery_window_candidate(
            frame,
            wheel=wheel,
            command_signal=resolved[wheel][command_kind],
            actual_signal=resolved[wheel][actual_kind],
            speed_signal=resolved[wheel].get("speed_rad_s"),
            index=event_index,
            config=config,
        )
        for wheel in range(1, 5)
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["signature_score"]),
            -float(item["estimated_loss_fraction"]),
            str(item["wheel_id"]),
        )
    )
    top = candidates[0]
    second_loss = max(float(item["estimated_loss_fraction"]) for item in candidates[1:])
    top_loss = float(top["estimated_loss_fraction"])
    dominance_ratio = top_loss / max(second_loss, 1e-9)
    dominance_margin = top_loss - second_loss
    top["wheel_dominance_ratio"] = dominance_ratio
    top["wheel_dominance_margin"] = dominance_margin
    top["signature_score"] = max(
        0.0,
        min(
            0.999,
            0.82 * float(top["signature_score"])
            + 0.10 * _bounded_ratio(dominance_ratio, 3.0)
            + 0.08 * _bounded_ratio(dominance_margin, 0.40),
        ),
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate.setdefault("wheel_dominance_ratio", dominance_ratio if rank == 1 else 0.0)
        candidate.setdefault("wheel_dominance_margin", dominance_margin if rank == 1 else 0.0)
        actual_signal = str(candidate["actual_signal"])
        numeric = pd.to_numeric(frame[actual_signal], errors="coerce")
        observations.append(
            ObservationFact(
                observation_id=_stable_id(
                    "OBS",
                    {
                        "mode": "command_response",
                        "wheel_id": candidate["wheel_id"],
                        "candidate": candidate,
                        "warmup_points": warmup_points,
                    },
                ),
                signal=actual_signal,
                sample_count=len(frame),
                missing_count=int(numeric.isna().sum()),
                maximum_robust_score=float(candidate["signature_score"]),
                metadata={
                    "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
                    "schema_version": RW4_SINGLE_FAULT_SCHEMA_VERSION,
                    "status": "CANDIDATE_RANKED",
                    "detection_mode": "command_response",
                    "pair_kind": pair_kind,
                    "rank": rank,
                    "candidate": candidate,
                    "warmup_points": warmup_points,
                    "maximum_warmup_points": maximum_warmup,
                    "score_is_calibrated_probability": False,
                },
            )
        )
    gate_checks = {
        "pre_observable_points": int(top["pre_eligible_points"]) >= config.minimum_pre_eligible_points,
        "post_observable_points": int(top["post_eligible_points"]) >= config.minimum_post_eligible_points,
        "command_activity": float(top["post_command_activity_median"]) >= config.minimum_command_activity,
        "pre_delivery_healthy": float(top["pre_delivery_loss_median"]) <= config.maximum_pre_delivery_loss,
        "delivery_loss": top_loss >= config.minimum_delivery_loss,
        "delivery_loss_step": float(top["delivery_loss_step"]) >= config.minimum_delivery_loss_step,
        "instant_delivery_loss_step": float(top["instant_delivery_loss_step"])
        >= config.minimum_delivery_loss_step,
        "delivery_persistence": float(top["delivery_persistence"]) >= config.minimum_delivery_persistence,
        "cross_wheel_dominance_ratio": dominance_ratio >= config.minimum_delivery_dominance_ratio,
        "cross_wheel_dominance_margin": dominance_margin >= config.minimum_delivery_dominance_margin,
    }
    if not all(gate_checks.values()):
        observations[0].metadata["top_gate_checks"] = gate_checks
        observations[0].metadata["detection_emitted"] = False
        return observations, detections, anomalies
    wheel = int(top["wheel_index_one_based"])
    actual_signal = str(top["actual_signal"])
    configured = mapping.get(actual_signal, {})
    anomaly_id = str(configured.get("anomaly_id") or rw4_anomaly_id(wheel))
    anomaly_name = str(configured.get("name") or configured.get("anomaly_name") or rw4_anomaly_name(wheel))
    observed_at = _observed_time(frame, timestamp_column, event_index)
    confidence = max(0.0, min(0.999, float(top["signature_score"])))
    ranking = [dict(item) for item in candidates]
    payload = {
        "anomaly_id": anomaly_id,
        "signal": actual_signal,
        "data_index": event_index,
        "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
        "mode": "command_response",
        "ranking": ranking,
    }
    detection = DetectionRecord(
        detection_id=_stable_id("DET", payload),
        anomaly_id=anomaly_id,
        anomaly_name=anomaly_name,
        signal=actual_signal,
        detector_id=RW4_SINGLE_FAULT_DETECTOR_ID,
        score=confidence,
        confidence=confidence,
        observed_at=observed_at,
        missing_fraction=float(pd.to_numeric(frame[actual_signal], errors="coerce").isna().mean()),
        mapped=True,
        metadata={
            "schema_version": RW4_SINGLE_FAULT_SCHEMA_VERSION,
            "detection_mode": "command_response",
            "pair_kind": pair_kind,
            "score_is_calibrated_probability": False,
            "score_semantics": "E0 wheel command/response delivery-loss change-point signature strength",
            "gate_checks": gate_checks,
            "detector_config": config.__dict__,
            "wheel_ranking": ranking,
            "estimated_retention_fraction": top["estimated_retention_fraction"],
            "estimated_loss_fraction": top["estimated_loss_fraction"],
            "estimated_severity": top["estimated_severity"],
            "joint_evidence": {
                "command_signal": top["command_signal"],
                "actual_signal": top["actual_signal"],
                "pre_delivery_loss_median": top["pre_delivery_loss_median"],
                "post_delivery_loss_median": top["post_delivery_loss_median"],
                "delivery_loss_step": top["delivery_loss_step"],
                "instant_delivery_loss_step": top["instant_delivery_loss_step"],
                "delivery_persistence": top["delivery_persistence"],
                "cross_wheel_dominance_ratio": dominance_ratio,
                "cross_wheel_dominance_margin": dominance_margin,
            },
            "claim_boundary": (
                "E0 deterministic RW4 command/response signature and engineering severity band only; "
                "not calibrated, not flight-qualified, and not a probability of fault."
            ),
        },
    )
    detections.append(detection)
    observed_at_s = _safe_float(observed_at)
    anomalies.append(
        AnomalyEvidence(
            anomaly_id=anomaly_id,
            name=anomaly_name,
            signal=actual_signal,
            confidence=confidence,
            confidence_type="rw4_e0_command_response_signature_strength_not_calibrated",
            observed_at_s=observed_at_s if observed_at_s is not None and observed_at_s >= 0 else None,
            metadata={
                "detection_id": detection.detection_id,
                "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
                "mapped_to_registered_semantics": True,
                "wheel_id": f"RW{wheel}",
                "estimated_loss_fraction": top["estimated_loss_fraction"],
                "estimated_severity": top["estimated_severity"],
                "wheel_ranking": ranking,
                "labels_used": False,
                "truth_used": False,
            },
        )
    )
    return observations, detections, anomalies

def detect_rw4_single_fault(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    channels: list[str],
    baseline_fraction: float,
    z_threshold: float,
    minimum_baseline_points: int,
    signal_mapping: dict[str, dict[str, str]] | None = None,
    detector_config: RW4SingleFaultDetectorConfig | None = None,
) -> tuple[list[ObservationFact], list[DetectionRecord], list[AnomalyEvidence]]:
    """Detect and localize one RW torque-authority loss without truth inputs.

    The detector looks for a wheel-local motor-current discontinuity followed by
    a persistent current collapse. Wheel speed is used as corroborating context,
    while cross-wheel peak dominance prevents ordinary control reallocation from
    becoming a fault conclusion. Only the winning wheel is emitted as anomaly
    evidence; the complete four-wheel ranking is retained in metadata.

    ``baseline_fraction`` is capped at 10% for warm-up because this detector is a
    change-point detector, not the generic long-baseline offset detector. The
    label/ground-truth columns are not accepted through ``channels`` and are never
    inspected.
    """

    del z_threshold  # retained in the common detector protocol; gates are explicit E0 settings.
    config = detector_config or DEFAULT_RW4_SINGLE_FAULT_CONFIG
    mapping = signal_mapping or {}
    observations: list[ObservationFact] = []
    detections: list[DetectionRecord] = []
    anomalies: list[AnomalyEvidence] = []
    if frame.empty:
        return observations, detections, anomalies

    usable_channels = [
        str(item)
        for item in channels
        if item in frame.columns
        and not str(item).lower().startswith(("label.", "label_", "truth.", "truth_", "ground_truth"))
    ]
    resolved, duplicates = resolve_rw4_signal_columns(usable_channels)
    enhanced_slots_present = any(
        any(kind in resolved[wheel] for kind in ("command_current_a", "command_torque_nm", "actual_torque_nm"))
        for wheel in range(1, 5)
    )
    current_pairs_ready = all(
        "command_current_a" in resolved[wheel] and "current_a" in resolved[wheel]
        for wheel in range(1, 5)
    )
    torque_pairs_ready = all(
        "command_torque_nm" in resolved[wheel] and "actual_torque_nm" in resolved[wheel]
        for wheel in range(1, 5)
    )
    if enhanced_slots_present:
        if duplicates or not (current_pairs_ready or torque_pairs_ready):
            missing_pairs = []
            for wheel in range(1, 5):
                if not (
                    ("command_current_a" in resolved[wheel] and "current_a" in resolved[wheel])
                    or ("command_torque_nm" in resolved[wheel] and "actual_torque_nm" in resolved[wheel])
                ):
                    missing_pairs.append(f"RW{wheel}")
            payload = {
                "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
                "schema_version": RW4_SINGLE_FAULT_SCHEMA_VERSION,
                "resolved_signals": resolved,
                "duplicate_aliases": duplicates,
                "missing_command_response_pairs": missing_pairs,
            }
            observations.append(
                ObservationFact(
                    observation_id=_stable_id("OBS", payload),
                    signal="rw4_schema",
                    sample_count=len(frame),
                    missing_count=0,
                    metadata={**payload, "status": "UNSUPPORTED_SCHEMA", "fail_closed": True},
                )
            )
            return observations, detections, anomalies
        pair_kind = "current" if current_pairs_ready else "torque"
        return _detect_rw4_command_response(
            frame,
            timestamp_column=timestamp_column,
            resolved=resolved,
            pair_kind=pair_kind,
            baseline_fraction=baseline_fraction,
            minimum_baseline_points=minimum_baseline_points,
            mapping=mapping,
            config=config,
        )

    missing_current = [f"RW{wheel}" for wheel in range(1, 5) if "current_a" not in resolved[wheel]]

    schema_payload = {
        "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
        "schema_version": RW4_SINGLE_FAULT_SCHEMA_VERSION,
        "missing_current_wheels": missing_current,
        "duplicate_aliases": duplicates,
        "resolved_signals": resolved,
    }
    if missing_current or duplicates:
        observations.append(
            ObservationFact(
                observation_id=_stable_id("OBS", schema_payload),
                signal="rw4_schema",
                sample_count=len(frame),
                missing_count=0,
                metadata={
                    **schema_payload,
                    "status": "UNSUPPORTED_SCHEMA",
                    "fail_closed": True,
                },
            )
        )
        return observations, detections, anomalies

    maximum_warmup = max(3, int(math.floor(len(frame) * config.maximum_warmup_fraction)))
    # The shared request field is treated as a preferred lower bound, not as
    # permission to consume a known change point. The dedicated detector always
    # honors its warm-up cap and needs only three points for robust step context.
    preferred_minimum = min(max(3, minimum_baseline_points), maximum_warmup)
    requested_warmup = max(preferred_minimum, int(round(len(frame) * baseline_fraction)))
    warmup_points = min(requested_warmup, maximum_warmup)
    maximum_allowed = len(frame) - config.post_window_points - 1
    if maximum_allowed < 3:
        observations.append(
            ObservationFact(
                observation_id=_stable_id("OBS", {**schema_payload, "row_count": len(frame)}),
                signal="rw4_schema",
                sample_count=len(frame),
                missing_count=0,
                metadata={
                    **schema_payload,
                    "status": "INSUFFICIENT_ROWS",
                    "minimum_required_rows": config.post_window_points + 4,
                    "fail_closed": True,
                },
            )
        )
        return observations, detections, anomalies
    warmup_points = min(warmup_points, maximum_allowed)

    candidates = [
        _candidate_for_wheel(
            frame,
            wheel=wheel,
            current_signal=resolved[wheel]["current_a"],
            speed_signal=resolved[wheel].get("speed_rad_s"),
            warmup_points=warmup_points,
            config=config,
        )
        for wheel in range(1, 5)
    ]
    candidates.sort(
        key=lambda item: (
            -float(item.get("signature_score", 0.0)),
            -float(item.get("current_step_a", 0.0)),
            str(item["wheel_id"]),
        )
    )
    top = candidates[0]
    second_step = max(float(item.get("current_step_a", 0.0)) for item in candidates[1:])
    dominance = float(top.get("current_step_a", 0.0)) / max(second_step, 1e-9)
    top["wheel_dominance_ratio"] = dominance
    top["signature_score"] = max(
        0.0,
        min(
            0.999,
            0.85 * float(top.get("signature_score", 0.0))
            + 0.15 * _bounded_ratio(dominance, 3.0),
        ),
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate.setdefault("wheel_dominance_ratio", dominance if rank == 1 else 0.0)
        current_signal = str(candidate["current_signal"])
        numeric = pd.to_numeric(frame[current_signal], errors="coerce")
        observation_payload = {
            "wheel_id": candidate["wheel_id"],
            "current_signal": current_signal,
            "candidate": candidate,
            "warmup_points": warmup_points,
        }
        observations.append(
            ObservationFact(
                observation_id=_stable_id("OBS", observation_payload),
                signal=current_signal,
                sample_count=len(frame),
                missing_count=int(numeric.isna().sum()),
                baseline_median=float(candidate.get("baseline_step_median_a", 0.0)),
                baseline_mad=float(candidate.get("baseline_step_robust_scale_a", 0.0)) / 1.4826,
                maximum_robust_score=float(candidate.get("robust_step_strength", 0.0)),
                metadata={
                    "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
                    "schema_version": RW4_SINGLE_FAULT_SCHEMA_VERSION,
                    "status": "CANDIDATE_RANKED",
                    "detection_mode": "current_collapse_fallback",
                    "rank": rank,
                    "candidate": candidate,
                    "warmup_points": warmup_points,
                    "requested_baseline_fraction": baseline_fraction,
                    "requested_minimum_baseline_points": minimum_baseline_points,
                    "maximum_warmup_points": maximum_warmup,
                    "effective_warmup_fraction": warmup_points / len(frame),
                    "score_is_calibrated_probability": False,
                },
            )
        )

    gate_checks = {
        "current_step": float(top["current_step_a"]) >= config.minimum_current_step_a,
        "current_collapse": float(top["current_collapse_ratio"])
        >= config.minimum_current_collapse_ratio,
        "step_to_pre_activity": float(top["step_to_pre_activity_ratio"])
        >= config.minimum_step_to_pre_activity_ratio,
        "cross_wheel_dominance": dominance >= config.minimum_wheel_dominance_ratio,
    }
    if not all(gate_checks.values()):
        observations[0].metadata.setdefault("top_gate_checks", gate_checks)
        observations[0].metadata.setdefault("detection_emitted", False)
        return observations, detections, anomalies

    wheel = int(top["wheel_index_one_based"])
    current_signal = str(top["current_signal"])
    configured = mapping.get(current_signal, {})
    anomaly_id = str(configured.get("anomaly_id") or rw4_anomaly_id(wheel))
    anomaly_name = str(
        configured.get("name")
        or configured.get("anomaly_name")
        or rw4_anomaly_name(wheel)
    )
    observed_at = _observed_time(frame, timestamp_column, int(top["data_index"]))
    confidence = max(0.0, min(0.999, float(top["signature_score"])))
    ranking = [dict(item) for item in candidates]
    payload = {
        "anomaly_id": anomaly_id,
        "signal": current_signal,
        "data_index": int(top["data_index"]),
        "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
        "ranking": ranking,
    }
    detection = DetectionRecord(
        detection_id=_stable_id("DET", payload),
        anomaly_id=anomaly_id,
        anomaly_name=anomaly_name,
        signal=current_signal,
        detector_id=RW4_SINGLE_FAULT_DETECTOR_ID,
        score=confidence,
        confidence=confidence,
        observed_at=observed_at,
        missing_fraction=float(pd.to_numeric(frame[current_signal], errors="coerce").isna().mean()),
        mapped=True,
        metadata={
            "schema_version": RW4_SINGLE_FAULT_SCHEMA_VERSION,
            "score_is_calibrated_probability": False,
            "detection_mode": "current_collapse_fallback",
            "score_semantics": "E0 current-collapse/change-point signature strength",
            "estimated_retention_fraction": max(0.0, min(1.0, 1.0 - float(top["current_collapse_ratio"]))),
            "estimated_loss_fraction": max(0.0, min(1.0, float(top["current_collapse_ratio"]))),
            "estimated_severity": rw4_loss_severity(float(top["current_collapse_ratio"])),
            "gate_checks": gate_checks,
            "detector_config": config.__dict__,
            "wheel_ranking": ranking,
            "joint_evidence": {
                "current_step_a": top["current_step_a"],
                "current_collapse_ratio": top["current_collapse_ratio"],
                "speed_signal": top.get("speed_signal"),
                "speed_post_abs_step_median_rad_s_per_sample": top.get(
                    "speed_post_abs_step_median_rad_s_per_sample"
                ),
                "speed_quiet_support": top.get("speed_quiet_support"),
                "cross_wheel_dominance_ratio": dominance,
            },
            "claim_boundary": (
                "E0 deterministic RW4 engineering signature only; not calibrated, "
                "not flight-qualified, and not a probability of fault."
            ),
        },
    )
    detections.append(detection)
    observed_at_s = _safe_float(observed_at)
    anomalies.append(
        AnomalyEvidence(
            anomaly_id=anomaly_id,
            name=anomaly_name,
            signal=current_signal,
            confidence=confidence,
            confidence_type="rw4_e0_signature_strength_not_calibrated",
            observed_at_s=observed_at_s if observed_at_s is not None and observed_at_s >= 0 else None,
            metadata={
                "detection_id": detection.detection_id,
                "detector_id": RW4_SINGLE_FAULT_DETECTOR_ID,
                "mapped_to_registered_semantics": True,
                "wheel_id": f"RW{wheel}",
                "wheel_ranking": ranking,
                "labels_used": False,
                "truth_used": False,
            },
        )
    )
    return observations, detections, anomalies


__all__ = [
    "RW4_SINGLE_FAULT_DETECTOR_ID",
    "RW4_SINGLE_FAULT_SCHEMA_VERSION",
    "RW4_SINGLE_FAULT_LEGACY_DETECTOR_IDS",
    "RW4SingleFaultDetectorConfig",
    "DEFAULT_RW4_SINGLE_FAULT_CONFIG",
    "detect_rw4_single_fault",
    "resolve_rw4_signal_columns",
    "rw4_anomaly_id",
    "rw4_anomaly_name",
    "rw4_loss_severity",
]
