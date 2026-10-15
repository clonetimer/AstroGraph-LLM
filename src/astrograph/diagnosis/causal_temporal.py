"""Causal, label-free selection of a diagnostic window from an online score stream.

The selector consumes only model/evidence scores available at or before each
candidate endpoint.  It never receives fault labels, injection times, or future
points.  Ground truth may be joined later by an offline evaluator, but is not
part of any model in this module.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from astrograph.assurance import canonical_sha256




class PairedResidualScoringContract(BaseModel):
    """Frozen trigger-score definition for a time-aligned nominal counterpart.

    A score is the mean of the largest normalized absolute residual elements in
    the current causal window.  Values missing on both sides contribute zero; a
    one-sided missing value contributes the frozen mismatch score.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-paired-residual-scoring-contract.v1"
    contract_id: str = Field(min_length=1)
    channel_order: tuple[str, ...] = Field(min_length=1)
    channel_scale: tuple[float, ...] = Field(min_length=1)
    top_k_elements: int = Field(default=8, ge=1)
    minimum_scale: float = Field(default=1e-6, gt=0.0)
    one_sided_missing_score: float = Field(default=10.0, ge=0.0)
    claim_boundary: str = (
        "Paired, time-aligned synthetic/digital-twin residual trigger. Exact pairing is a task assumption; "
        "this contract does not establish robustness to model mismatch or external flight telemetry."
    )

    @field_validator("channel_order")
    @classmethod
    def unique_channels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("paired residual channel order must contain unique nonempty names")
        return value

    @field_validator("channel_scale")
    @classmethod
    def finite_positive_scale(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        converted = tuple(float(item) for item in value)
        if any(not math.isfinite(item) or item <= 0.0 for item in converted):
            raise ValueError("paired residual scales must be finite and positive")
        return converted

    @model_validator(mode="after")
    def matching_channels(self) -> "PairedResidualScoringContract":
        if len(self.channel_order) != len(self.channel_scale):
            raise ValueError("paired residual scale count must equal channel count")
        return self

    @computed_field
    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"contract_sha256"}))


def paired_residual_trigger_score(
    current_window: Sequence[Sequence[float | None]],
    nominal_window: Sequence[Sequence[float | None]],
    contract: PairedResidualScoringContract,
) -> float:
    """Compute one deterministic, causal paired-residual trigger score."""

    current = np.asarray(current_window, dtype=np.float64)
    nominal = np.asarray(nominal_window, dtype=np.float64)
    if current.ndim != 2 or nominal.ndim != 2 or current.shape != nominal.shape:
        raise ValueError("paired residual windows must be equal-rank, equal-shape matrices")
    if current.shape[1] != len(contract.channel_order):
        raise ValueError("paired residual window channel count differs from contract")
    scale = np.maximum(np.asarray(contract.channel_scale, dtype=np.float64), contract.minimum_scale)
    current_finite = np.isfinite(current)
    nominal_finite = np.isfinite(nominal)
    both = current_finite & nominal_finite
    one_sided = current_finite ^ nominal_finite
    residual = np.zeros(current.shape, dtype=np.float64)
    residual[both] = np.abs(current[both] - nominal[both]) / np.broadcast_to(scale, current.shape)[both]
    residual[one_sided] = contract.one_sided_missing_score
    flattened = residual.reshape(-1)
    count = min(contract.top_k_elements, flattened.size)
    if count <= 0:
        raise ValueError("paired residual window cannot be empty")
    top = np.partition(flattened, flattened.size - count)[-count:]
    value = float(top.mean())
    if not math.isfinite(value):
        raise ValueError("paired residual trigger score is non-finite")
    return value


class TemporalTriggerSource(str, Enum):
    MODEL_DETECTION_SCORE = "MODEL_DETECTION_SCORE"
    PAIRED_RESIDUAL_SCORE = "PAIRED_RESIDUAL_SCORE"
    EXTERNAL_CAUSAL_SCORE = "EXTERNAL_CAUSAL_SCORE"


class TemporalSelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    NO_ALARM = "NO_ALARM"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"


class TemporalScorePoint(BaseModel):
    """One model-visible, causal endpoint summary.

    ``trigger_score`` may be a detector score or a paired residual score.  Its
    semantics are frozen by the policy/receipt rather than inferred from the
    numeric range.  Truth and event timing fields are intentionally absent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-causal-temporal-score-point.v1"
    sequence_index: int = Field(ge=0)
    endpoint_index: int = Field(ge=0)
    endpoint_time_s: float
    trigger_score: float = Field(ge=0.0)
    detection_score: float = Field(ge=0.0, le=1.0)
    top_fault_id: str = Field(min_length=1)
    top_fault_score: float = Field(ge=0.0, le=1.0)
    runner_up_fault_score: float = Field(ge=0.0, le=1.0)
    prediction_id: str = Field(min_length=1)
    input_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("endpoint_time_s", "trigger_score", "detection_score", "top_fault_score", "runner_up_fault_score", "source_missing_fraction")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("temporal score values must be finite")
        return value

    @model_validator(mode="after")
    def ordered_class_scores(self) -> "TemporalScorePoint":
        if self.runner_up_fault_score > self.top_fault_score:
            raise ValueError("runner-up score cannot exceed top-fault score")
        return self

    @computed_field
    @property
    def attribution_margin(self) -> float:
        return self.top_fault_score - self.runner_up_fault_score


class CausalTemporalPolicy(BaseModel):
    """Frozen state-machine parameters for first stable causal selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-causal-temporal-policy.v1"
    policy_id: str = Field(min_length=1)
    trigger_source: TemporalTriggerSource
    window_length_samples: int = Field(ge=1)
    stride_samples: int = Field(ge=1)
    warmup_points: int = Field(default=0, ge=0)
    trigger_threshold: float = Field(ge=0.0)
    detection_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_top_fault_score: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_attribution_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    trigger_persistence_points: int = Field(default=1, ge=1)
    attribution_persistence_points: int = Field(default=1, ge=1)
    selection_delay_points: int = Field(default=0, ge=0)
    maximum_endpoint_gap_samples: int | None = Field(default=None, ge=1)
    maximum_source_missing_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    no_alarm_action: str = "ABSTAIN"
    claim_boundary: str = (
        "Label-free causal selection policy. It selects the first endpoint satisfying the frozen "
        "score/persistence contract and does not establish score calibration, causal fault identity, "
        "external validity, or flight qualification."
    )

    @field_validator("trigger_threshold")
    @classmethod
    def finite_threshold(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("trigger threshold must be finite")
        return value

    @field_validator("no_alarm_action")
    @classmethod
    def abstain_only(cls, value: str) -> str:
        if value != "ABSTAIN":
            raise ValueError("causal temporal policies must abstain when no alarm is selected")
        return value

    @model_validator(mode="after")
    def valid_gap(self) -> "CausalTemporalPolicy":
        gap = self.maximum_endpoint_gap_samples
        if gap is not None and gap < self.stride_samples:
            raise ValueError("maximum endpoint gap cannot be smaller than stride")
        return self

    @computed_field
    @property
    def policy_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"policy_sha256"})
        return canonical_sha256(payload)


class TemporalSelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-causal-temporal-selection.v1"
    status: TemporalSelectionStatus
    policy_id: str
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_point: TemporalScorePoint | None = None
    first_trigger_point: TemporalScorePoint | None = None
    points_consumed: int = Field(ge=0)
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reset_count: int = Field(ge=0)
    reason: str = Field(min_length=1)
    claim_boundary: str

    @model_validator(mode="after")
    def consistent_status(self) -> "TemporalSelectionResult":
        if self.status == TemporalSelectionStatus.SELECTED and self.selected_point is None:
            raise ValueError("SELECTED result requires selected_point")
        if self.status != TemporalSelectionStatus.SELECTED and self.selected_point is not None:
            raise ValueError("non-selected result cannot contain selected_point")
        return self


class CausalTemporalSelector:
    """Streaming first-stable-alarm selector.

    Once selected, the endpoint is immutable.  Appending future score points
    therefore cannot change the decision, which is the key prefix-invariance
    property required for causal offline replay.
    """

    def __init__(self, policy: CausalTemporalPolicy) -> None:
        self.policy = policy
        self._points: list[TemporalScorePoint] = []
        self._selected: TemporalScorePoint | None = None
        self._first_trigger: TemporalScorePoint | None = None
        self._trigger_run = 0
        self._fault_run = 0
        self._stable_fault: str | None = None
        self._pending_fault: str | None = None
        self._pending_delay = 0
        self._reset_count = 0

    @property
    def selected_point(self) -> TemporalScorePoint | None:
        return self._selected

    def _reset_runs(self) -> None:
        if self._trigger_run or self._fault_run or self._pending_fault is not None:
            self._reset_count += 1
        self._trigger_run = 0
        self._fault_run = 0
        self._stable_fault = None
        self._pending_fault = None
        self._pending_delay = 0

    def _validate_order(self, point: TemporalScorePoint) -> None:
        if not self._points:
            return
        previous = self._points[-1]
        if point.sequence_index != previous.sequence_index + 1:
            raise ValueError("temporal score sequence_index must be contiguous and strictly increasing")
        if point.endpoint_index <= previous.endpoint_index:
            raise ValueError("temporal score endpoint_index must be strictly increasing")
        if point.endpoint_time_s <= previous.endpoint_time_s:
            raise ValueError("temporal score endpoint_time_s must be strictly increasing")

    def update(self, point: TemporalScorePoint) -> TemporalScorePoint | None:
        self._validate_order(point)
        if self._selected is not None:
            # Preserve an exact audit trace without allowing future points to alter
            # the already-issued decision.
            self._points.append(point)
            return self._selected

        if self._points:
            previous = self._points[-1]
            allowed_gap = self.policy.maximum_endpoint_gap_samples
            if allowed_gap is not None and point.endpoint_index - previous.endpoint_index > allowed_gap:
                self._reset_runs()
        self._points.append(point)

        if len(self._points) <= self.policy.warmup_points:
            return None

        gate = (
            point.trigger_score >= self.policy.trigger_threshold
            and (
                self.policy.detection_threshold is None
                or point.detection_score >= self.policy.detection_threshold
            )
            and point.top_fault_score >= self.policy.minimum_top_fault_score
            and point.attribution_margin >= self.policy.minimum_attribution_margin
            and point.source_missing_fraction <= self.policy.maximum_source_missing_fraction
        )
        if not gate:
            self._reset_runs()
            return None

        self._trigger_run += 1
        if self._first_trigger is None:
            self._first_trigger = point
        if self._stable_fault == point.top_fault_id:
            self._fault_run += 1
        else:
            self._stable_fault = point.top_fault_id
            self._fault_run = 1
            self._pending_fault = None
            self._pending_delay = 0

        ready = (
            self._trigger_run >= self.policy.trigger_persistence_points
            and self._fault_run >= self.policy.attribution_persistence_points
        )
        if not ready:
            return None

        if self._pending_fault is None:
            self._pending_fault = point.top_fault_id
            self._pending_delay = self.policy.selection_delay_points
        elif self._pending_fault != point.top_fault_id:
            self._pending_fault = point.top_fault_id
            self._pending_delay = self.policy.selection_delay_points

        if self._pending_delay > 0:
            self._pending_delay -= 1
            return None

        self._selected = point
        return self._selected

    def result(self) -> TemporalSelectionResult:
        if self._selected is not None:
            status = TemporalSelectionStatus.SELECTED
            reason = "first stable causal alarm selected"
        elif len(self._points) <= self.policy.warmup_points:
            status = TemporalSelectionStatus.INSUFFICIENT_OBSERVATIONS
            reason = "score stream ended before the frozen warmup contract was satisfied"
        else:
            status = TemporalSelectionStatus.NO_ALARM
            reason = "no endpoint satisfied the frozen trigger and attribution persistence contract"
        return TemporalSelectionResult(
            status=status,
            policy_id=self.policy.policy_id,
            policy_sha256=self.policy.policy_sha256,
            selected_point=self._selected,
            first_trigger_point=self._first_trigger,
            points_consumed=len(self._points),
            trace_sha256=canonical_sha256([item.model_dump(mode="json") for item in self._points]),
            reset_count=self._reset_count,
            reason=reason,
            claim_boundary=self.policy.claim_boundary,
        )

    @classmethod
    def select(
        cls, policy: CausalTemporalPolicy, points: Iterable[TemporalScorePoint]
    ) -> TemporalSelectionResult:
        selector = cls(policy)
        for point in points:
            selector.update(point)
            if selector.selected_point is not None:
                # Stop at the first decision.  This keeps points_consumed and the
                # trace hash independent of any future suffix.
                break
        return selector.result()


__all__ = [
    "CausalTemporalPolicy",
    "PairedResidualScoringContract",
    "CausalTemporalSelector",
    "TemporalScorePoint",
    "TemporalSelectionResult",
    "TemporalSelectionStatus",
    "TemporalTriggerSource",
    "paired_residual_trigger_score",
]
