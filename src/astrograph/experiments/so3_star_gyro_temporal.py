"""SO(3)-aware star-tracker/gyro temporal evidence for ADCS development.

This module adds a label-free attitude-consistency branch to the frozen B011
hybrid temporal route.  It compares the observed star-tracker quaternion with a
strictly causal gyro-propagated attitude over several short horizons.  The
rotation error is measured with the sign-invariant SO(3) geodesic distance
``2*acos(abs(q1 dot q2))``.

Nominal anchors are fitted only on groups 0/1/2/5/6/7. Candidate policies may
be selected only by the offline evaluator on groups 3/8. Groups 4/9 remain
permanently post-hoc. Runtime objects contain no fault class, onset, active
interval, prediction correctness, or evaluator outcome.
"""
from __future__ import annotations

from functools import cached_property
import math
from typing import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from astrograph.assurance import canonical_sha256

_SHA256 = r"^[0-9a-f]{64}$"

QUATERNION_CHANNELS: tuple[str, ...] = (
    "adcs.star_tracker.q0",
    "adcs.star_tracker.q1",
    "adcs.star_tracker.q2",
    "adcs.star_tracker.q3",
)
GYRO_CHANNELS: tuple[str, ...] = (
    "adcs.sensor.gyro_measured_rad_s_x",
    "adcs.sensor.gyro_measured_rad_s_y",
    "adcs.sensor.gyro_measured_rad_s_z",
)
VALIDITY_CHANNELS: tuple[str, ...] = (
    "adcs.star_tracker.valid",
    "adcs.star_tracker.solution_fresh",
    "adcs.sensor.imu.valid",
    "adcs.sensor.gyro_axis_valid_x",
    "adcs.sensor.gyro_axis_valid_y",
    "adcs.sensor.gyro_axis_valid_z",
)
SO3_STAR_GYRO_CHANNEL_ORDER: tuple[str, ...] = QUATERNION_CHANNELS + GYRO_CHANNELS + VALIDITY_CHANNELS
FORMAL_HORIZON_SETS: tuple[tuple[int, ...], ...] = (
    (1,), (2,), (4,), (8,), (1, 2), (1, 2, 4), (1, 2, 4, 8), (2, 4, 8)
)
FORMAL_THRESHOLDS: tuple[float, ...] = (1.05, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0)
FORMAL_PERSISTENCE: tuple[int, ...] = (1, 2, 3, 4)


class SO3StarGyroContract(BaseModel):
    """Frozen score semantics for the B012 attitude-consistency branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-so3-star-gyro-contract.v1"
    contract_id: str = "ADCS-SO3-STAR-GYRO-R001"
    horizons_points: tuple[int, ...] = (1, 2, 4, 8)
    warmup_points: int = Field(default=31, ge=9)
    notification_buffer_points: int = Field(default=3, ge=0)
    sample_period_s: float = Field(default=1.0, gt=0.0)
    quaternion_order: str = "SCALAR_FIRST_Q0_Q1_Q2_Q3"
    propagation_convention: str = "RIGHT_MULTIPLY_POSITIVE_BODY_RATE"
    integration_rule: str = "SEQUENTIAL_TRAPEZOIDAL_RATE"
    distance_metric: str = "SO3_GEODESIC_DEG_2_ACOS_ABS_DOT"
    anchor_estimator: str = "MAX_VALID_NOMINAL_AFTER_WARMUP"
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    calibration_groups: tuple[int, ...] = (0, 1, 2, 5, 6, 7)
    runtime_ground_truth_access: bool = False
    claim_boundary: str = (
        "Development-only label-free attitude-consistency score. Nominal anchors "
        "are empirical maxima, not calibrated probabilities or conformal bounds."
    )

    @model_validator(mode="after")
    def exact_contract(self) -> "SO3StarGyroContract":
        if self.horizons_points != (1, 2, 4, 8):
            raise ValueError("B012 formal horizons must remain 1/2/4/8 points")
        if set(self.development_groups) != {3, 8} or set(self.excluded_groups) != {4, 9}:
            raise ValueError("development/post-hoc boundary differs")
        if set(self.calibration_groups) & {3, 4, 8, 9}:
            raise ValueError("calibration groups overlap development or post-hoc groups")
        if self.propagation_convention != "RIGHT_MULTIPLY_POSITIVE_BODY_RATE":
            raise ValueError("only the audited right-multiply convention is allowed")
        if self.integration_rule != "SEQUENTIAL_TRAPEZOIDAL_RATE":
            raise ValueError("only the audited trapezoidal integration rule is allowed")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        return self

    @computed_field
    @cached_property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"contract_sha256"}))


class SO3StarGyroPolicy(BaseModel):
    """One candidate trigger policy over normalized SO(3) residuals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-so3-star-gyro-policy.v1"
    policy_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=_SHA256)
    horizon_points: tuple[int, ...] = Field(min_length=1)
    threshold_multiplier: float = Field(ge=1.05)
    persistence_points: int = Field(ge=1, le=8)
    warmup_points: int = Field(default=31, ge=9)
    notification_buffer_points: int = Field(default=3, ge=0)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    calibration_groups: tuple[int, ...] = (0, 1, 2, 5, 6, 7)
    runtime_ground_truth_access: bool = False

    @field_validator("threshold_multiplier")
    @classmethod
    def finite_threshold(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("threshold must be finite")
        return value

    @model_validator(mode="after")
    def exact_policy(self) -> "SO3StarGyroPolicy":
        if tuple(self.horizon_points) not in FORMAL_HORIZON_SETS:
            raise ValueError("horizon set is outside the frozen candidate family")
        if set(self.development_groups) != {3, 8} or set(self.excluded_groups) != {4, 9}:
            raise ValueError("development/post-hoc boundary differs")
        if set(self.calibration_groups) & {3, 4, 8, 9}:
            raise ValueError("calibration groups overlap development or post-hoc groups")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        return self

    @computed_field
    @cached_property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"policy_sha256"}))


def candidate_policies(contract: SO3StarGyroContract) -> tuple[SO3StarGyroPolicy, ...]:
    output: list[SO3StarGyroPolicy] = []
    for horizons in FORMAL_HORIZON_SETS:
        for threshold in FORMAL_THRESHOLDS:
            for persistence in FORMAL_PERSISTENCE:
                seed = {
                    "contract_sha256": contract.contract_sha256,
                    "horizon_points": horizons,
                    "threshold_multiplier": threshold,
                    "persistence_points": persistence,
                }
                output.append(
                    SO3StarGyroPolicy(
                        policy_id="SO3P-" + canonical_sha256(seed)[:20].upper(),
                        contract_sha256=contract.contract_sha256,
                        horizon_points=horizons,
                        threshold_multiplier=threshold,
                        persistence_points=persistence,
                        warmup_points=contract.warmup_points,
                        notification_buffer_points=contract.notification_buffer_points,
                    )
                )
    return tuple(output)


def normalize_quaternions(quaternions: Sequence[Sequence[float | None]]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(quaternions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("quaternion array must have shape [time,4]")
    norm = np.linalg.norm(array, axis=1)
    valid = np.isfinite(array).all(axis=1) & np.isfinite(norm) & (norm > 1e-12)
    output = np.full_like(array, np.nan)
    output[valid] = array[valid] / norm[valid, None]
    return output, valid


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lv = left[..., 0], left[..., 1:]
    rw, rv = right[..., 0], right[..., 1:]
    scalar = lw * rw - np.sum(lv * rv, axis=-1)
    vector = lw[..., None] * rv + rw[..., None] * lv + np.cross(lv, rv)
    return np.concatenate((scalar[..., None], vector), axis=-1)


def rotation_vector_to_quaternion(rotation_vectors: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vectors, dtype=np.float64)
    if vector.shape[-1] != 3:
        raise ValueError("rotation vectors must have final dimension 3")
    angle = np.linalg.norm(vector, axis=-1)
    output = np.zeros(vector.shape[:-1] + (4,), dtype=np.float64)
    output[..., 0] = np.cos(angle / 2.0)
    scale = np.empty_like(angle)
    small = angle < 1e-12
    scale[small] = 0.5
    scale[~small] = np.sin(angle[~small] / 2.0) / angle[~small]
    output[..., 1:] = vector * scale[..., None]
    return output


def quaternion_geodesic_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("quaternion geodesic inputs must have matching [...,4] shape")
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    valid = (
        np.isfinite(left).all(axis=-1)
        & np.isfinite(right).all(axis=-1)
        & (left_norm > 1e-12)
        & (right_norm > 1e-12)
    )
    output = np.full(left.shape[:-1], np.nan, dtype=np.float64)
    normalized_left = left[valid] / left_norm[valid, None]
    normalized_right = right[valid] / right_norm[valid, None]
    dot = np.abs(np.sum(normalized_left * normalized_right, axis=-1))
    output[valid] = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    return output


def generate_so3_residuals(
    time_s: Sequence[float],
    quaternions: Sequence[Sequence[float | None]],
    gyro_rad_s: Sequence[Sequence[float | None]],
    validity: Sequence[Sequence[float | bool | None]],
    contract: SO3StarGyroContract,
) -> dict[int, np.ndarray]:
    """Generate strictly causal gyro-propagation residuals in degrees."""

    time = np.asarray(time_s, dtype=np.float64)
    gyro = np.asarray(gyro_rad_s, dtype=np.float64)
    validity_array = np.asarray(validity, dtype=np.float64)
    quaternion, quaternion_valid = normalize_quaternions(quaternions)
    if time.ndim != 1 or gyro.shape != (len(time), 3):
        raise ValueError("time/gyro shapes differ")
    if validity_array.ndim != 2 or validity_array.shape[0] != len(time):
        raise ValueError("validity matrix shape differs")
    if len(time) < max(contract.horizons_points) + 1:
        raise ValueError("trajectory is too short")
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be finite and strictly increasing")
    if not np.allclose(np.diff(time), contract.sample_period_s, rtol=0.0, atol=1e-9):
        raise ValueError("runtime sample period differs from the frozen contract")

    sample_valid = (
        quaternion_valid
        & np.isfinite(gyro).all(axis=1)
        & np.isfinite(validity_array).all(axis=1)
        & (validity_array > 0.5).all(axis=1)
    )
    dt = np.diff(time)
    midpoint_rate = 0.5 * (gyro[:-1] + gyro[1:])
    increment = rotation_vector_to_quaternion(midpoint_rate * dt[:, None])

    output: dict[int, np.ndarray] = {}
    count = len(time)
    for horizon in contract.horizons_points:
        usable = count - horizon
        product = increment[:usable].copy()
        valid = sample_valid[:usable] & sample_valid[horizon:]
        for step in range(1, horizon):
            product = quaternion_multiply(product, increment[step : step + usable])
            valid &= sample_valid[step : step + usable]
        prediction = quaternion_multiply(quaternion[:usable], product)
        residual = quaternion_geodesic_deg(prediction, quaternion[horizon:])
        trace = np.full(count, np.nan, dtype=np.float64)
        trace[horizon:] = residual
        trace[horizon:][~valid] = np.nan
        trace[: contract.warmup_points] = np.nan
        output[horizon] = trace
    return output


def calibrate_nominal_anchors(
    nominal_residuals: Sequence[Mapping[int, Sequence[float]]],
    contract: SO3StarGyroContract,
) -> dict[int, float]:
    if len(nominal_residuals) < 2:
        raise ValueError("at least two nominal calibration trajectories are required")
    output: dict[int, float] = {}
    for horizon in contract.horizons_points:
        maxima: list[float] = []
        for residual in nominal_residuals:
            values = np.asarray(residual[horizon], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if len(finite):
                maxima.append(float(np.max(finite)))
        if not maxima or max(maxima) <= 0.0:
            raise ValueError(f"no positive nominal anchor for horizon {horizon}")
        output[horizon] = max(maxima)
    return output


def normalized_policy_score(
    residuals: Mapping[int, Sequence[float]],
    anchors_deg: Mapping[int, float],
    policy: SO3StarGyroPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    if set(policy.horizon_points) - set(residuals):
        raise ValueError("policy horizon is absent from residuals")
    if set(policy.horizon_points) - set(anchors_deg):
        raise ValueError("policy horizon is absent from anchors")
    length = len(np.asarray(residuals[policy.horizon_points[0]]))
    score = np.zeros(length, dtype=np.float64)
    valid = np.zeros(length, dtype=bool)
    for horizon in policy.horizon_points:
        anchor = float(anchors_deg[horizon])
        if not math.isfinite(anchor) or anchor <= 0.0:
            raise ValueError("anchor must be finite and positive")
        values = np.asarray(residuals[horizon], dtype=np.float64)
        if len(values) != length:
            raise ValueError("residual lengths differ")
        finite = np.isfinite(values)
        score[finite] = np.maximum(score[finite], values[finite] / anchor)
        valid |= finite
    return score, valid


def select_so3_star_gyro(
    residuals: Mapping[int, Sequence[float]],
    anchors_deg: Mapping[int, float],
    policy: SO3StarGyroPolicy,
) -> dict:
    score, valid = normalized_policy_score(residuals, anchors_deg, policy)
    persistence = 0
    for index in range(policy.warmup_points, len(score)):
        if not valid[index]:
            persistence = 0
            continue
        persistence = persistence + 1 if score[index] >= policy.threshold_multiplier else 0
        if persistence >= policy.persistence_points:
            return {
                "status": "SELECTED",
                "selected_index": index,
                "selected_time_s": float(index),
                "notification_available_time_s": float(index + policy.notification_buffer_points),
                "selected_score": float(score[index]),
                "policy_sha256": policy.policy_sha256,
            }
    return {
        "status": "NO_ALARM",
        "selected_index": None,
        "selected_time_s": None,
        "notification_available_time_s": None,
        "selected_score": None,
        "policy_sha256": policy.policy_sha256,
    }
