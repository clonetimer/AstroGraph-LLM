"""Label-free actuator command / control-authority temporal evidence.

B013 adds an activity-gated actuator effectiveness branch to the frozen B012
temporal stack.  It compares commanded reaction-wheel torque and current with
their executed/applied counterparts.  The score is dimensionless loss of
delivery, evaluated only while the command is observably active.

Nominal activity gates use the existing audited engineering values from the
reaction-wheel authority detector. Candidate policies are selected only by the
offline evaluator on groups 3/8. Groups 4/9 remain permanently post-hoc.
"""
from __future__ import annotations

from collections import deque
from functools import cached_property
import math
from typing import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from astrograph.assurance import canonical_sha256

_SHA256 = r"^[0-9a-f]{64}$"

COMMAND_TORQUE_CHANNELS: tuple[str, ...] = tuple(
    f"adcs.rw.command_torque_nm_{index}" for index in range(4)
)
COMMAND_EXECUTION_RESIDUAL_CHANNELS: tuple[str, ...] = tuple(
    f"adcs.rw.command_execution_residual_nm_{index}" for index in range(4)
)
COMMANDED_CURRENT_CHANNELS: tuple[str, ...] = tuple(
    f"adcs.rw.commanded_current_a_{index}" for index in range(4)
)
APPLIED_CURRENT_CHANNELS: tuple[str, ...] = tuple(
    f"adcs.rw.applied_current_a_{index}" for index in range(4)
)
CONTROL_AUTHORITY_CHANNEL_ORDER: tuple[str, ...] = (
    COMMAND_TORQUE_CHANNELS
    + COMMAND_EXECUTION_RESIDUAL_CHANNELS
    + COMMANDED_CURRENT_CHANNELS
    + APPLIED_CURRENT_CHANNELS
)

FORMAL_THRESHOLDS: tuple[float, ...] = (0.50, 0.75, 0.90, 1.00)
FORMAL_EVIDENCE_WINDOWS: tuple[int, ...] = (3, 5, 10, 20)
FORMAL_MINIMUM_HITS: tuple[int, ...] = (1, 2, 3, 4)


class ControlAuthorityContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-control-authority-contract.v1"
    contract_id: str = "ADCS-CONTROL-AUTHORITY-R001"
    torque_command_activity_nm: float = Field(default=0.0008, gt=0.0)
    current_command_activity_a: float = Field(default=0.02, gt=0.0)
    score_cap: float = Field(default=2.0, ge=1.0)
    warmup_points: int = Field(default=31, ge=5)
    notification_buffer_points: int = Field(default=3, ge=0)
    score_semantics: str = "MAX_ACTIVITY_GATED_TORQUE_OR_CURRENT_DELIVERY_LOSS"
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    calibration_groups: tuple[int, ...] = (0, 1, 2, 5, 6, 7)
    runtime_ground_truth_access: bool = False
    claim_boundary: str = (
        "Development-only digital-twin actuator-delivery evidence. Applied torque/current "
        "observability is assumed by this dataset and is not established for flight hardware."
    )

    @model_validator(mode="after")
    def exact_contract(self) -> "ControlAuthorityContract":
        if set(self.development_groups) != {3, 8} or set(self.excluded_groups) != {4, 9}:
            raise ValueError("development/post-hoc boundary differs")
        if set(self.calibration_groups) & {3, 4, 8, 9}:
            raise ValueError("calibration groups overlap development or post-hoc groups")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        return self

    @computed_field
    @cached_property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"contract_sha256"}))


class ControlAuthorityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-control-authority-policy.v1"
    policy_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=_SHA256)
    threshold: float
    evidence_window_points: int
    minimum_hits: int
    warmup_points: int = Field(default=31, ge=5)
    notification_buffer_points: int = Field(default=3, ge=0)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    calibration_groups: tuple[int, ...] = (0, 1, 2, 5, 6, 7)
    runtime_ground_truth_access: bool = False

    @field_validator("threshold")
    @classmethod
    def finite_threshold(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("threshold must be finite")
        return value

    @model_validator(mode="after")
    def exact_policy(self) -> "ControlAuthorityPolicy":
        if self.threshold not in FORMAL_THRESHOLDS:
            raise ValueError("threshold is outside the formal candidate family")
        if self.evidence_window_points not in FORMAL_EVIDENCE_WINDOWS:
            raise ValueError("evidence window is outside the formal candidate family")
        if self.minimum_hits not in FORMAL_MINIMUM_HITS:
            raise ValueError("minimum hits is outside the formal candidate family")
        if self.minimum_hits > self.evidence_window_points:
            raise ValueError("minimum hits cannot exceed the evidence window")
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


def candidate_policies(contract: ControlAuthorityContract) -> tuple[ControlAuthorityPolicy, ...]:
    output: list[ControlAuthorityPolicy] = []
    for threshold in FORMAL_THRESHOLDS:
        for window in FORMAL_EVIDENCE_WINDOWS:
            for minimum_hits in FORMAL_MINIMUM_HITS:
                if minimum_hits > window:
                    continue
                seed = {
                    "contract_sha256": contract.contract_sha256,
                    "threshold": threshold,
                    "evidence_window_points": window,
                    "minimum_hits": minimum_hits,
                }
                output.append(
                    ControlAuthorityPolicy(
                        policy_id="CAP-" + canonical_sha256(seed)[:20].upper(),
                        contract_sha256=contract.contract_sha256,
                        threshold=threshold,
                        evidence_window_points=window,
                        minimum_hits=minimum_hits,
                        warmup_points=contract.warmup_points,
                        notification_buffer_points=contract.notification_buffer_points,
                    )
                )
    return tuple(output)


def generate_control_authority_score(
    command_torque_nm: Sequence[Sequence[float | None]],
    command_execution_residual_nm: Sequence[Sequence[float | None]],
    commanded_current_a: Sequence[Sequence[float | None]],
    applied_current_a: Sequence[Sequence[float | None]],
    contract: ControlAuthorityContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return score, validity and dominant evidence source.

    Dominant-source values are 0 for unavailable, 1 for torque-delivery loss,
    and 2 for current-delivery loss.
    """

    command = np.asarray(command_torque_nm, dtype=np.float64)
    residual = np.asarray(command_execution_residual_nm, dtype=np.float64)
    commanded_current = np.asarray(commanded_current_a, dtype=np.float64)
    applied_current = np.asarray(applied_current_a, dtype=np.float64)
    shapes = {command.shape, residual.shape, commanded_current.shape, applied_current.shape}
    if len(shapes) != 1 or command.ndim != 2 or command.shape[1] != 4:
        raise ValueError("all actuator arrays must have matching [time,4] shape")

    command_abs = np.abs(command)
    current_abs = np.abs(commanded_current)
    torque_valid = (
        np.isfinite(command)
        & np.isfinite(residual)
        & (command_abs >= contract.torque_command_activity_nm)
    )
    current_valid = (
        np.isfinite(commanded_current)
        & np.isfinite(applied_current)
        & (current_abs >= contract.current_command_activity_a)
    )

    torque_loss = np.full_like(command, np.nan)
    current_loss = np.full_like(command, np.nan)
    torque_loss[torque_valid] = (
        np.abs(residual[torque_valid])
        / np.maximum(command_abs[torque_valid], contract.torque_command_activity_nm)
    )
    current_loss[current_valid] = (
        np.abs(commanded_current[current_valid] - applied_current[current_valid])
        / np.maximum(current_abs[current_valid], contract.current_command_activity_a)
    )

    torque_score = np.full(len(command), np.nan)
    current_score = np.full(len(command), np.nan)
    torque_rows = np.isfinite(torque_loss).any(axis=1)
    current_rows = np.isfinite(current_loss).any(axis=1)
    torque_score[torque_rows] = np.nanmax(torque_loss[torque_rows], axis=1)
    current_score[current_rows] = np.nanmax(current_loss[current_rows], axis=1)

    valid = torque_rows | current_rows
    score = np.zeros(len(command), dtype=np.float64)
    source = np.zeros(len(command), dtype=np.uint8)
    for index in np.flatnonzero(valid):
        torque_value = torque_score[index] if np.isfinite(torque_score[index]) else -np.inf
        current_value = current_score[index] if np.isfinite(current_score[index]) else -np.inf
        if current_value > torque_value:
            score[index] = current_value
            source[index] = 2
        else:
            score[index] = torque_value
            source[index] = 1
    score = np.clip(score, 0.0, contract.score_cap)
    score[: contract.warmup_points] = 0.0
    valid[: contract.warmup_points] = False
    source[: contract.warmup_points] = 0
    return score, valid, source


def select_control_authority(
    time_s: Sequence[float],
    score: Sequence[float],
    valid: Sequence[bool],
    source: Sequence[int],
    policy: ControlAuthorityPolicy,
) -> dict:
    time = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(score, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    sources = np.asarray(source, dtype=np.uint8)
    if time.ndim != 1 or values.shape != time.shape or validity.shape != time.shape or sources.shape != time.shape:
        raise ValueError("time/score/valid/source shapes differ")
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be finite and strictly increasing")

    hits: deque[int] = deque()
    for index in range(policy.warmup_points, len(time)):
        while hits and hits[0] < index - policy.evidence_window_points + 1:
            hits.popleft()
        if validity[index] and values[index] >= policy.threshold:
            hits.append(index)
        if len(hits) >= policy.minimum_hits:
            return {
                "status": "SELECTED",
                "selected_index": index,
                "selected_time_s": float(time[index]),
                "notification_available_time_s": float(
                    time[index] + policy.notification_buffer_points
                ),
                "selected_score": float(values[index]),
                "dominant_source": (
                    "CURRENT_DELIVERY_LOSS" if int(sources[index]) == 2
                    else "TORQUE_DELIVERY_LOSS"
                ),
                "policy_sha256": policy.policy_sha256,
            }
    return {
        "status": "NO_ALARM",
        "selected_index": None,
        "selected_time_s": None,
        "notification_available_time_s": None,
        "selected_score": None,
        "dominant_source": None,
        "policy_sha256": policy.policy_sha256,
    }
