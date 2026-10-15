"""Bounded mismatch-tolerant, label-isolated temporal localization.

This module extends the B008 classifier-independent paired-reference route with
an explicitly bounded prefix-only alignment and a selective reference-fit gate.
The runtime objects deliberately contain no fault labels, event times, or
prediction-correctness fields.  Ground truth is joined only by the offline
evaluator after trace bundles are content-addressed and frozen.

The alignment is not unconstrained DTW.  It searches a small, predeclared set
of integer lags using only an initial reference-admission prefix.  Padded edge
samples are invalidated rather than silently repeated.  A conservative online
availability time includes the maximum alignment buffer, so bounded reference
look-ahead is not reported as zero-latency causal notification.
"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from functools import cached_property
import math
from statistics import median
from typing import Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from astrograph.assurance import canonical_sha256
from astrograph.experiments.model_independent_temporal import (
    ModelIndependentScaleContract,
    ModelIndependentTruthTable,
    TraceVariant,
)

_SHA256 = r"^[0-9a-f]{64}$"
_OPAQUE_TRAJECTORY = r"^TRJ-[0-9A-F]{20}$"
_OPAQUE_GROUP = r"^GRP-[0-9A-F]{16}$"


class AlignmentDecision(str, Enum):
    ADMITTED = "ADMITTED"
    ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE = "ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE"


class MismatchLocalizationStatus(str, Enum):
    SELECTED = "SELECTED"
    NO_ALARM = "NO_ALARM"
    ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE = "ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE"


class MismatchTolerantAlignmentContract(BaseModel):
    """Content-addressed runtime alignment and score semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-mismatch-tolerant-alignment-contract.v1"
    contract_id: str = Field(min_length=1)
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    prefix_points: int = Field(default=31, ge=3)
    maximum_absolute_lag_points: int = Field(default=3, ge=0, le=30)
    sample_period_s: float = Field(default=1.0, gt=0.0)
    alignment_channel_quantile: float = Field(default=0.75, gt=0.5, lt=1.0)
    alignment_time_aggregation: str = "MEDIAN"
    score_channel_quantile: float = Field(default=0.99, gt=0.5, lt=1.0)
    prefix_baseline_quantile: float = Field(default=0.99, gt=0.5, lt=1.0)
    padded_edge_action: str = "INVALIDATE"
    lag_tie_break: str = "LOWEST_FIT_THEN_MIN_ABS_LAG_THEN_SIGNED_LAG"
    online_reference_mode: str = "PRECOMPUTED_REFERENCE_OR_BOUNDED_BUFFER"
    notification_buffer_points: int = Field(default=3, ge=0)
    runtime_ground_truth_access: bool = False
    claim_boundary: str = (
        "Development-only bounded integer-lag alignment using the initial label-free prefix. "
        "It is not unconstrained warping and does not establish arbitrary cross-Monte-Carlo, "
        "external, HIL, safety, flight, fault-identification, or confirmatory validity."
    )

    @field_validator("development_groups", "excluded_groups")
    @classmethod
    def unique_groups(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        converted = tuple(int(item) for item in value)
        if len(converted) != len(set(converted)):
            raise ValueError("group lists must be unique")
        return converted

    @field_validator(
        "sample_period_s",
        "alignment_channel_quantile",
        "score_channel_quantile",
        "prefix_baseline_quantile",
    )
    @classmethod
    def finite_float(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("alignment contract values must be finite")
        return value

    @model_validator(mode="after")
    def scientific_contract(self) -> "MismatchTolerantAlignmentContract":
        if set(self.development_groups) != {3, 8}:
            raise ValueError("mismatch-tolerant development groups must be exactly 3/8")
        if set(self.excluded_groups) != {4, 9}:
            raise ValueError("groups 4/9 must remain permanently excluded")
        if set(self.development_groups) & set(self.excluded_groups):
            raise ValueError("development and excluded groups overlap")
        if self.alignment_time_aggregation != "MEDIAN":
            raise ValueError("only the audited MEDIAN alignment aggregation is supported")
        if self.padded_edge_action != "INVALIDATE":
            raise ValueError("alignment padding must be invalidated")
        if self.prefix_points - self.maximum_absolute_lag_points < 3:
            raise ValueError("alignment prefix must retain at least three valid points at the maximum lag")
        if self.notification_buffer_points < self.maximum_absolute_lag_points:
            raise ValueError("notification buffer must cover the maximum absolute lag")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        return self

    @computed_field
    @cached_property
    def candidate_lags(self) -> tuple[int, ...]:
        maximum = self.maximum_absolute_lag_points
        return tuple(range(-maximum, maximum + 1))

    @computed_field
    @cached_property
    def contract_sha256(self) -> str:
        return canonical_sha256(
            self.model_dump(mode="json", exclude={"candidate_lags", "contract_sha256"})
        )


class PrefixAlignmentResult(BaseModel):
    """Label-free alignment diagnostics frozen into each runtime trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-prefix-alignment-result.v1"
    selected_lag_points: int
    fit_score: float = Field(ge=0.0)
    second_best_margin: float = Field(ge=0.0)
    candidate_lags: tuple[int, ...] = Field(min_length=1)
    candidate_fit_scores: tuple[float, ...] = Field(min_length=1)
    invalid_prefix_points: int = Field(ge=0)
    invalid_suffix_points: int = Field(ge=0)
    fit_prefix_valid_points: int = Field(ge=1)
    runtime_ground_truth_access: bool = False

    @field_validator("fit_score", "second_best_margin", "candidate_fit_scores")
    @classmethod
    def finite_scores(cls, value):
        if isinstance(value, tuple):
            converted = tuple(float(item) for item in value)
            if any(not math.isfinite(item) or item < 0.0 for item in converted):
                raise ValueError("alignment fit scores must be finite and nonnegative")
            return converted
        converted = float(value)
        if not math.isfinite(converted) or converted < 0.0:
            raise ValueError("alignment fit scores must be finite and nonnegative")
        return converted

    @model_validator(mode="after")
    def consistent_result(self) -> "PrefixAlignmentResult":
        if len(self.candidate_lags) != len(self.candidate_fit_scores):
            raise ValueError("candidate lag and score lengths differ")
        if len(set(self.candidate_lags)) != len(self.candidate_lags):
            raise ValueError("candidate lags must be unique")
        if self.selected_lag_points not in self.candidate_lags:
            raise ValueError("selected lag is absent from candidate set")
        selected_index = self.candidate_lags.index(self.selected_lag_points)
        if not math.isclose(
            self.fit_score,
            self.candidate_fit_scores[selected_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selected fit score differs from candidate score")
        if self.invalid_prefix_points and self.invalid_suffix_points:
            raise ValueError("a single integer lag cannot invalidate both edges")
        if self.runtime_ground_truth_access:
            raise ValueError("alignment cannot access ground truth")
        return self


class FrozenMismatchTolerantTrace(BaseModel):
    """Frozen runtime trace with alignment diagnostics and no truth fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-mismatch-tolerant-temporal-trace.v1"
    opaque_trajectory_id: str = Field(pattern=_OPAQUE_TRAJECTORY)
    independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    development_group: int
    variant: TraceVariant
    current_telemetry_sha256: str = Field(pattern=_SHA256)
    reference_telemetry_sha256: str = Field(pattern=_SHA256)
    alternate_reference_sha256: str | None = Field(default=None, pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    alignment_contract_sha256: str = Field(pattern=_SHA256)
    time_s: tuple[float, ...] = Field(min_length=2)
    trigger_score: tuple[float, ...] = Field(min_length=2)
    source_missing_fraction: tuple[float, ...] = Field(min_length=2)
    alignment_valid: tuple[bool, ...] = Field(min_length=2)
    prefix_baseline_score: float = Field(ge=0.0)
    alignment: PrefixAlignmentResult
    runtime_ground_truth_access: bool = False
    generator_version: str = "b009.mismatch-tolerant-temporal.r001"
    claim_boundary: str = (
        "Label-free bounded-alignment runtime trace. The alignment and score generator did not "
        "receive a fault label, onset, active interval, or prediction outcome."
    )

    @field_validator("time_s", "trigger_score", "source_missing_fraction")
    @classmethod
    def finite_series(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        converted = tuple(float(item) for item in value)
        if any(not math.isfinite(item) for item in converted):
            raise ValueError("trace arrays must be finite")
        return converted

    @field_validator("prefix_baseline_score")
    @classmethod
    def finite_baseline(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("prefix baseline must be finite and nonnegative")
        return value

    @model_validator(mode="after")
    def consistent_trace(self) -> "FrozenMismatchTolerantTrace":
        lengths = {
            len(self.time_s),
            len(self.trigger_score),
            len(self.source_missing_fraction),
            len(self.alignment_valid),
        }
        if len(lengths) != 1:
            raise ValueError("trace arrays must have equal lengths")
        if any(next_time <= time for time, next_time in zip(self.time_s, self.time_s[1:])):
            raise ValueError("trace times must be strictly increasing")
        if any(score < 0.0 for score in self.trigger_score):
            raise ValueError("trigger scores must be nonnegative")
        if any(value < 0.0 or value > 1.0 for value in self.source_missing_fraction):
            raise ValueError("missing fractions must lie in [0,1]")
        for valid, score, missing in zip(
            self.alignment_valid,
            self.trigger_score,
            self.source_missing_fraction,
        ):
            if not valid and (score != 0.0 or missing != 1.0):
                raise ValueError("invalid alignment edge points must have zero score and full missingness")
        if self.development_group in {4, 9}:
            raise ValueError("groups 4/9 are permanently post-hoc")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        return self

    @computed_field
    @cached_property
    def trace_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"trace_sha256"}))


class FrozenMismatchTraceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-mismatch-tolerant-trace-bundle.v1"
    bundle_id: str = Field(min_length=1)
    variant: TraceVariant
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    runtime_manifest_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    alignment_contract_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    traces: tuple[FrozenMismatchTolerantTrace, ...] = Field(min_length=1)
    predictions_frozen_before_truth_join: bool = True
    runtime_ground_truth_access: bool = False

    @model_validator(mode="after")
    def consistent_bundle(self) -> "FrozenMismatchTraceBundle":
        if len({trace.opaque_trajectory_id for trace in self.traces}) != len(self.traces):
            raise ValueError("bundle contains duplicate trajectories")
        if any(trace.variant != self.variant for trace in self.traces):
            raise ValueError("trace variant differs from bundle")
        if any(trace.scale_contract_sha256 != self.scale_contract_sha256 for trace in self.traces):
            raise ValueError("trace scale contract differs from bundle")
        if any(
            trace.alignment_contract_sha256 != self.alignment_contract_sha256
            for trace in self.traces
        ):
            raise ValueError("trace alignment contract differs from bundle")
        groups = {trace.development_group for trace in self.traces}
        if groups != set(self.development_groups):
            raise ValueError("bundle must contain exactly development groups 3/8")
        if groups & set(self.excluded_groups):
            raise ValueError("post-hoc groups cannot enter a development bundle")
        if not self.predictions_frozen_before_truth_join or self.runtime_ground_truth_access:
            raise ValueError("bundle violates label-isolation contract")
        return self

    @computed_field
    @cached_property
    def bundle_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"bundle_sha256"}))


class MismatchTolerantTemporalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-mismatch-tolerant-temporal-policy.v1"
    policy_id: str = Field(min_length=1)
    trigger_threshold: float = Field(gt=0.0)
    trigger_persistence_points: int = Field(default=1, ge=1)
    warmup_points: int = Field(default=31, ge=0)
    maximum_source_missing_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    alignment_fit_max: float = Field(gt=0.0)
    alignment_out_of_envelope_action: str = "ABSTAIN"
    no_alarm_action: str = "ABSTAIN"
    notification_buffer_points: int = Field(default=3, ge=0)
    claim_boundary: str = (
        "Development-selected bounded mismatch policy. Selection uses groups 3/8 and predeclared "
        "development stress variants only; it is not confirmatory and does not authorize groups 4/9 tuning."
    )

    @field_validator("trigger_threshold", "alignment_fit_max")
    @classmethod
    def finite_threshold(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("policy thresholds must be finite")
        return value

    @model_validator(mode="after")
    def abstention_contract(self) -> "MismatchTolerantTemporalPolicy":
        if self.alignment_out_of_envelope_action != "ABSTAIN":
            raise ValueError("out-of-envelope alignment must abstain")
        if self.no_alarm_action != "ABSTAIN":
            raise ValueError("no-alarm result must abstain")
        return self

    @computed_field
    @cached_property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"policy_sha256"}))


class MismatchTolerantSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-mismatch-tolerant-selection.v1"
    opaque_trajectory_id: str = Field(pattern=_OPAQUE_TRAJECTORY)
    independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    development_group: int
    trace_sha256: str = Field(pattern=_SHA256)
    policy_sha256: str = Field(pattern=_SHA256)
    alignment_decision: AlignmentDecision
    status: MismatchLocalizationStatus
    selected_index: int | None = Field(default=None, ge=0)
    selected_time_s: float | None = None
    notification_available_time_s: float | None = None
    selected_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def selected_fields(self) -> "MismatchTolerantSelection":
        selected = self.status == MismatchLocalizationStatus.SELECTED
        fields_present = all(
            value is not None
            for value in (
                self.selected_index,
                self.selected_time_s,
                self.notification_available_time_s,
                self.selected_score,
            )
        )
        if selected != fields_present:
            raise ValueError("selection fields must be present exactly for SELECTED status")
        if (
            self.status == MismatchLocalizationStatus.ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE
            and self.alignment_decision
            != AlignmentDecision.ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE
        ):
            raise ValueError("alignment abstention status/decision mismatch")
        if selected and self.notification_available_time_s < self.selected_time_s:
            raise ValueError("notification cannot precede the selected event time")
        return self


class MismatchTolerantTemporalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-mismatch-tolerant-temporal-metrics.v1"
    variant: TraceVariant
    policy_sha256: str = Field(pattern=_SHA256)
    trajectory_count: int = Field(ge=1)
    independent_group_count: int = Field(ge=1)
    fault_trajectory_count: int = Field(ge=0)
    nominal_trajectory_count: int = Field(ge=0)
    alignment_admitted_count: int = Field(ge=0)
    alignment_abstention_count: int = Field(ge=0)
    fault_alignment_admitted_count: int = Field(ge=0)
    nominal_alignment_admitted_count: int = Field(ge=0)
    selected_fault_count: int = Field(ge=0)
    selected_within_active_count: int = Field(ge=0)
    localized_within_5s_count: int = Field(ge=0)
    localized_within_20s_count: int = Field(ge=0)
    notified_within_5s_count: int = Field(ge=0)
    notified_within_10s_count: int = Field(ge=0)
    notified_within_20s_count: int = Field(ge=0)
    pre_onset_selection_count: int = Field(ge=0)
    pre_onset_selection_group_count: int = Field(ge=0)
    nominal_false_alarm_count: int = Field(ge=0)
    nominal_false_alarm_group_count: int = Field(ge=0)
    alignment_admission_rate: float = Field(ge=0.0, le=1.0)
    fault_alignment_admission_rate: float = Field(ge=0.0, le=1.0)
    nominal_alignment_admission_rate: float = Field(ge=0.0, le=1.0)
    event_selection_recall: float = Field(ge=0.0, le=1.0)
    active_window_selection_rate: float = Field(ge=0.0, le=1.0)
    localized_within_5s_rate: float = Field(ge=0.0, le=1.0)
    localized_within_20s_rate: float = Field(ge=0.0, le=1.0)
    notified_within_5s_rate: float = Field(ge=0.0, le=1.0)
    notified_within_10s_rate: float = Field(ge=0.0, le=1.0)
    notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    conditional_notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    nominal_false_alarm_rate: float = Field(ge=0.0, le=1.0)
    group_macro_notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    minimum_class_notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    median_localization_delay_s: float | None = None
    maximum_localization_delay_s: float | None = None
    median_notification_delay_s: float | None = None
    maximum_notification_delay_s: float | None = None
    per_group_notified_within_20s_rate: dict[str, float]
    per_group_fault_alignment_admission_rate: dict[str, float]
    per_class_notified_within_20s_rate: dict[str, float]
    selected_lag_distribution: dict[str, int]
    alignment_fit_quantiles: dict[str, float]
    claim_boundary: str = (
        "Development-only bounded mismatch temporal metrics over independent trajectory-pair groups. "
        "Notification metrics include the conservative alignment buffer."
    )


class MismatchPolicyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: MismatchTolerantTemporalPolicy
    metrics_by_variant: dict[str, MismatchTolerantTemporalMetrics]
    safe_across_development_variants: bool
    exact_noninferiority_pass: bool
    selection_key: tuple[float, ...]


class FrozenMismatchPolicyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-frozen-mismatch-tolerant-policy-receipt.v1"
    receipt_id: str = Field(min_length=1)
    selected_policy: MismatchTolerantTemporalPolicy
    selected_metrics_by_variant: dict[str, MismatchTolerantTemporalMetrics]
    trace_bundle_sha256s: dict[str, str]
    truth_table_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    alignment_contract_sha256: str = Field(pattern=_SHA256)
    candidate_grid_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    candidate_count: int = Field(ge=1)
    development_variants: tuple[str, ...]
    selection_informed_by_prior_development_exploration: bool = True
    groups_4_9_read: bool = False
    new_confirmatory_evidence: str = "NONE"
    claim_boundary: str = (
        "Frozen using groups 3/8 and declared development stress variants. The selection is development "
        "evidence, not untouched confirmation. Groups 4/9 were not read."
    )

    @model_validator(mode="after")
    def boundaries(self) -> "FrozenMismatchPolicyReceipt":
        if set(self.development_groups) != {3, 8} or set(self.excluded_groups) != {4, 9}:
            raise ValueError("development/post-hoc group boundary differs")
        if self.groups_4_9_read:
            raise ValueError("groups 4/9 cannot be read during development selection")
        if self.new_confirmatory_evidence != "NONE":
            raise ValueError("this receipt cannot create confirmatory evidence")
        return self

    @computed_field
    @cached_property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"receipt_sha256"}))


def _validate_matrices(
    current: Sequence[Sequence[float | None]],
    reference: Sequence[Sequence[float | None]],
    scale_contract: ModelIndependentScaleContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_array = np.asarray(current, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if current_array.ndim != 2 or current_array.shape != reference_array.shape:
        raise ValueError("current/reference matrices must have equal two-dimensional shape")
    if current_array.shape[1] != len(scale_contract.channel_order):
        raise ValueError("matrix channel count differs from scale contract")
    scale = np.asarray(scale_contract.channel_scale, dtype=np.float64)
    return current_array, reference_array, scale


def shift_reference_with_validity(reference: np.ndarray, lag_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift reference by integer points and explicitly invalidate padded edges.

    Positive lag delays the reference (invalid prefix).  Negative lag advances
    the reference (invalid suffix).  No wrapped or repeated edge is considered
    valid evidence.
    """

    array = np.asarray(reference, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("reference must be a two-dimensional matrix")
    length = array.shape[0]
    if abs(lag_points) >= length:
        raise ValueError("absolute lag must be smaller than the trace length")
    shifted = np.empty_like(array)
    valid = np.ones(length, dtype=bool)
    if lag_points == 0:
        shifted[:] = array
    elif lag_points > 0:
        shifted[:lag_points] = array[0]
        shifted[lag_points:] = array[:-lag_points]
        valid[:lag_points] = False
    else:
        points = -lag_points
        shifted[:-points] = array[points:]
        shifted[-points:] = array[-1]
        valid[-points:] = False
    return shifted, valid


def normalized_residual_matrix(
    current: Sequence[Sequence[float | None]],
    reference: Sequence[Sequence[float | None]],
    scale_contract: ModelIndependentScaleContract,
) -> tuple[np.ndarray, np.ndarray]:
    """Return channel-wise normalized residuals and source missing fraction."""

    current_array, reference_array, scale = _validate_matrices(
        current, reference, scale_contract
    )
    finite_current = np.isfinite(current_array)
    finite_reference = np.isfinite(reference_array)
    both = finite_current & finite_reference
    one_sided = finite_current ^ finite_reference
    residual = np.zeros_like(current_array)
    broadcast_scale = np.broadcast_to(scale, current_array.shape)
    residual[both] = (
        np.abs(current_array[both] - reference_array[both]) / broadcast_scale[both]
    )
    residual[one_sided] = scale_contract.one_sided_missing_score
    missing = 1.0 - np.mean(both, axis=1)
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(missing)):
        raise ValueError("non-finite residual or missing fraction")
    return residual, missing


def bounded_prefix_alignment(
    current: Sequence[Sequence[float | None]],
    reference: Sequence[Sequence[float | None]],
    scale_contract: ModelIndependentScaleContract,
    alignment_contract: MismatchTolerantAlignmentContract,
) -> tuple[np.ndarray, np.ndarray, PrefixAlignmentResult]:
    """Select a bounded integer lag using only the initial label-free prefix."""

    current_array, reference_array, _ = _validate_matrices(
        current, reference, scale_contract
    )
    if alignment_contract.scale_contract_sha256 != scale_contract.contract_sha256:
        raise ValueError("alignment and scale contracts differ")
    if current_array.shape[0] < alignment_contract.prefix_points:
        raise ValueError("trace is shorter than the alignment prefix")

    candidates: list[tuple[float, int, int, np.ndarray, np.ndarray, int]] = []
    for lag in alignment_contract.candidate_lags:
        shifted, valid = shift_reference_with_validity(reference_array, lag)
        residual, _ = normalized_residual_matrix(current_array, shifted, scale_contract)
        prefix_valid = valid[: alignment_contract.prefix_points]
        valid_count = int(np.sum(prefix_valid))
        if valid_count < 3:
            raise ValueError("too few valid prefix points for alignment")
        row_score = np.quantile(
            residual[: alignment_contract.prefix_points],
            alignment_contract.alignment_channel_quantile,
            axis=1,
        )
        fit = float(np.median(row_score[prefix_valid]))
        if not math.isfinite(fit) or fit < 0.0:
            raise ValueError("invalid alignment fit")
        candidates.append((fit, abs(lag), lag, shifted, valid, valid_count))

    ranked = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))
    fit, _, selected_lag, shifted, valid, valid_count = ranked[0]
    second_best_margin = float(ranked[1][0] - fit) if len(ranked) > 1 else 0.0
    fit_by_lag = {item[2]: item[0] for item in candidates}
    invalid_prefix = selected_lag if selected_lag > 0 else 0
    invalid_suffix = -selected_lag if selected_lag < 0 else 0
    result = PrefixAlignmentResult(
        selected_lag_points=selected_lag,
        fit_score=fit,
        second_best_margin=max(0.0, second_best_margin),
        candidate_lags=alignment_contract.candidate_lags,
        candidate_fit_scores=tuple(fit_by_lag[lag] for lag in alignment_contract.candidate_lags),
        invalid_prefix_points=invalid_prefix,
        invalid_suffix_points=invalid_suffix,
        fit_prefix_valid_points=valid_count,
    )
    return shifted, valid, result


def generate_mismatch_tolerant_scores(
    current: Sequence[Sequence[float | None]],
    reference: Sequence[Sequence[float | None]],
    scale_contract: ModelIndependentScaleContract,
    alignment_contract: MismatchTolerantAlignmentContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, PrefixAlignmentResult]:
    """Generate a prefix-calibrated score after bounded alignment."""

    current_array, _, _ = _validate_matrices(current, reference, scale_contract)
    aligned_reference, alignment_valid, alignment = bounded_prefix_alignment(
        current_array,
        reference,
        scale_contract,
        alignment_contract,
    )
    residual, missing = normalized_residual_matrix(
        current_array,
        aligned_reference,
        scale_contract,
    )
    raw_score = np.quantile(
        residual,
        alignment_contract.score_channel_quantile,
        axis=1,
    )
    valid_prefix = alignment_valid[: alignment_contract.prefix_points]
    baseline = float(
        np.quantile(
            raw_score[: alignment_contract.prefix_points][valid_prefix],
            alignment_contract.prefix_baseline_quantile,
        )
    )
    score = np.maximum(0.0, raw_score - baseline)
    score[: alignment_contract.prefix_points] = 0.0
    score[~alignment_valid] = 0.0
    missing = np.asarray(missing, dtype=np.float64)
    missing[~alignment_valid] = 1.0
    if not np.all(np.isfinite(score)) or not math.isfinite(baseline):
        raise ValueError("non-finite mismatch-tolerant score")
    return score, missing, alignment_valid, baseline, alignment


def select_mismatch_tolerant_trace(
    trace: FrozenMismatchTolerantTrace,
    policy: MismatchTolerantTemporalPolicy,
    alignment_contract: MismatchTolerantAlignmentContract,
) -> MismatchTolerantSelection:
    if policy.notification_buffer_points < alignment_contract.maximum_absolute_lag_points:
        raise ValueError("policy notification buffer is smaller than alignment bound")
    if trace.alignment.fit_score > policy.alignment_fit_max:
        return MismatchTolerantSelection(
            opaque_trajectory_id=trace.opaque_trajectory_id,
            independent_group_id=trace.independent_group_id,
            development_group=trace.development_group,
            trace_sha256=trace.trace_sha256,
            policy_sha256=policy.policy_sha256,
            alignment_decision=AlignmentDecision.ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE,
            status=MismatchLocalizationStatus.ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE,
        )

    persistence = 0
    for index, (time_s, score, missing, valid) in enumerate(
        zip(
            trace.time_s,
            trace.trigger_score,
            trace.source_missing_fraction,
            trace.alignment_valid,
        )
    ):
        if (
            index < policy.warmup_points
            or not valid
            or missing > policy.maximum_source_missing_fraction
        ):
            persistence = 0
            continue
        persistence = persistence + 1 if score >= policy.trigger_threshold else 0
        if persistence >= policy.trigger_persistence_points:
            availability = float(time_s) + (
                policy.notification_buffer_points * alignment_contract.sample_period_s
            )
            return MismatchTolerantSelection(
                opaque_trajectory_id=trace.opaque_trajectory_id,
                independent_group_id=trace.independent_group_id,
                development_group=trace.development_group,
                trace_sha256=trace.trace_sha256,
                policy_sha256=policy.policy_sha256,
                alignment_decision=AlignmentDecision.ADMITTED,
                status=MismatchLocalizationStatus.SELECTED,
                selected_index=index,
                selected_time_s=float(time_s),
                notification_available_time_s=availability,
                selected_score=float(score),
            )
    return MismatchTolerantSelection(
        opaque_trajectory_id=trace.opaque_trajectory_id,
        independent_group_id=trace.independent_group_id,
        development_group=trace.development_group,
        trace_sha256=trace.trace_sha256,
        policy_sha256=policy.policy_sha256,
        alignment_decision=AlignmentDecision.ADMITTED,
        status=MismatchLocalizationStatus.NO_ALARM,
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _macro(values: Mapping[str, float]) -> float:
    return float(sum(values.values()) / len(values)) if values else 0.0


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "q00": float(np.quantile(array, 0.0)),
        "q50": float(np.quantile(array, 0.5)),
        "q90": float(np.quantile(array, 0.9)),
        "q99": float(np.quantile(array, 0.99)),
        "q100": float(np.quantile(array, 1.0)),
    }


def evaluate_mismatch_tolerant_policy(
    policy: MismatchTolerantTemporalPolicy,
    bundle: FrozenMismatchTraceBundle,
    truth_table: ModelIndependentTruthTable,
    alignment_contract: MismatchTolerantAlignmentContract,
) -> MismatchTolerantTemporalMetrics:
    truth = {row.opaque_trajectory_id: row for row in truth_table.rows}
    traces = {trace.opaque_trajectory_id: trace for trace in bundle.traces}
    if set(traces) != set(truth):
        raise ValueError("trace bundle and truth table trajectory sets differ")
    if bundle.dataset_archive_sha256 != truth_table.dataset_archive_sha256:
        raise ValueError("trace/truth dataset identity differs")
    if bundle.alignment_contract_sha256 != alignment_contract.contract_sha256:
        raise ValueError("bundle/alignment contract identity differs")

    fault_total = nominal_total = admitted = abstained = 0
    fault_admitted = nominal_admitted = selected_fault = selected_active = 0
    localized5 = localized20 = notified5 = notified10 = notified20 = 0
    pre_onset = nominal_false = 0
    pre_groups: set[str] = set()
    false_groups: set[str] = set()
    localization_delays: list[float] = []
    notification_delays: list[float] = []
    group_total: dict[str, int] = defaultdict(int)
    group_admitted: dict[str, int] = defaultdict(int)
    group_timely: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)
    class_timely: dict[str, int] = defaultdict(int)
    lag_distribution: dict[str, int] = defaultdict(int)
    fit_scores: list[float] = []

    for trajectory_id, trace in traces.items():
        row = truth[trajectory_id]
        if (
            trace.independent_group_id != row.independent_group_id
            or trace.development_group != row.development_group
        ):
            raise ValueError("trace/truth group binding differs")
        selection = select_mismatch_tolerant_trace(trace, policy, alignment_contract)
        fit_scores.append(trace.alignment.fit_score)
        lag_distribution[str(trace.alignment.selected_lag_points)] += 1
        if selection.alignment_decision == AlignmentDecision.ADMITTED:
            admitted += 1
        else:
            abstained += 1
        if row.is_nominal:
            nominal_total += 1
            if selection.alignment_decision == AlignmentDecision.ADMITTED:
                nominal_admitted += 1
            if selection.status == MismatchLocalizationStatus.SELECTED:
                nominal_false += 1
                false_groups.add(row.independent_group_id)
            continue

        fault_total += 1
        group_total[row.independent_group_id] += 1
        class_total[row.diagnostic_class] += 1
        if selection.alignment_decision == AlignmentDecision.ADMITTED:
            fault_admitted += 1
            group_admitted[row.independent_group_id] += 1
        if (
            selection.status != MismatchLocalizationStatus.SELECTED
            or selection.selected_time_s is None
            or selection.notification_available_time_s is None
        ):
            continue
        selected_fault += 1
        start = float(row.active_start_s)
        end = float(row.active_end_s)
        selected_time = float(selection.selected_time_s)
        notification_time = float(selection.notification_available_time_s)
        if selected_time < start:
            pre_onset += 1
            pre_groups.add(row.independent_group_id)
            continue
        if selected_time > end:
            continue
        selected_active += 1
        localization_delay = selected_time - start
        notification_delay = notification_time - start
        localization_delays.append(localization_delay)
        notification_delays.append(notification_delay)
        if localization_delay <= 5.0:
            localized5 += 1
        if localization_delay <= 20.0:
            localized20 += 1
        if notification_delay <= 5.0:
            notified5 += 1
        if notification_delay <= 10.0:
            notified10 += 1
        if notification_delay <= 20.0:
            notified20 += 1
            group_timely[row.independent_group_id] += 1
            class_timely[row.diagnostic_class] += 1

    per_group = {
        group: _fraction(group_timely[group], total)
        for group, total in sorted(group_total.items())
    }
    per_group_admission = {
        group: _fraction(group_admitted[group], total)
        for group, total in sorted(group_total.items())
    }
    per_class = {
        fault: _fraction(class_timely[fault], total)
        for fault, total in sorted(class_total.items())
    }
    return MismatchTolerantTemporalMetrics(
        variant=bundle.variant,
        policy_sha256=policy.policy_sha256,
        trajectory_count=len(traces),
        independent_group_count=len({row.independent_group_id for row in truth.values()}),
        fault_trajectory_count=fault_total,
        nominal_trajectory_count=nominal_total,
        alignment_admitted_count=admitted,
        alignment_abstention_count=abstained,
        fault_alignment_admitted_count=fault_admitted,
        nominal_alignment_admitted_count=nominal_admitted,
        selected_fault_count=selected_fault,
        selected_within_active_count=selected_active,
        localized_within_5s_count=localized5,
        localized_within_20s_count=localized20,
        notified_within_5s_count=notified5,
        notified_within_10s_count=notified10,
        notified_within_20s_count=notified20,
        pre_onset_selection_count=pre_onset,
        pre_onset_selection_group_count=len(pre_groups),
        nominal_false_alarm_count=nominal_false,
        nominal_false_alarm_group_count=len(false_groups),
        alignment_admission_rate=_fraction(admitted, len(traces)),
        fault_alignment_admission_rate=_fraction(fault_admitted, fault_total),
        nominal_alignment_admission_rate=_fraction(nominal_admitted, nominal_total),
        event_selection_recall=_fraction(selected_fault, fault_total),
        active_window_selection_rate=_fraction(selected_active, fault_total),
        localized_within_5s_rate=_fraction(localized5, fault_total),
        localized_within_20s_rate=_fraction(localized20, fault_total),
        notified_within_5s_rate=_fraction(notified5, fault_total),
        notified_within_10s_rate=_fraction(notified10, fault_total),
        notified_within_20s_rate=_fraction(notified20, fault_total),
        conditional_notified_within_20s_rate=_fraction(notified20, fault_admitted),
        nominal_false_alarm_rate=_fraction(nominal_false, nominal_total),
        group_macro_notified_within_20s_rate=_macro(per_group),
        minimum_class_notified_within_20s_rate=min(per_class.values()) if per_class else 0.0,
        median_localization_delay_s=median(localization_delays) if localization_delays else None,
        maximum_localization_delay_s=max(localization_delays) if localization_delays else None,
        median_notification_delay_s=median(notification_delays) if notification_delays else None,
        maximum_notification_delay_s=max(notification_delays) if notification_delays else None,
        per_group_notified_within_20s_rate=per_group,
        per_group_fault_alignment_admission_rate=per_group_admission,
        per_class_notified_within_20s_rate=per_class,
        selected_lag_distribution=dict(sorted(lag_distribution.items(), key=lambda item: int(item[0]))),
        alignment_fit_quantiles=_quantiles(fit_scores),
    )


IN_ENVELOPE_VARIANTS: tuple[TraceVariant, ...] = (
    TraceVariant.EXACT_PAIRED,
    TraceVariant.REFERENCE_SHIFT_1S,
    TraceVariant.REFERENCE_SHIFT_2S,
    TraceVariant.CURRENT_NOISE_1PCT_SCALE,
    TraceVariant.CURRENT_NOISE_5PCT_SCALE,
)
REQUIRED_DEVELOPMENT_VARIANTS: tuple[TraceVariant, ...] = (*IN_ENVELOPE_VARIANTS, TraceVariant.CROSS_MC_REFERENCE)


def mismatch_policy_selection_key(
    metrics_by_variant: Mapping[str, MismatchTolerantTemporalMetrics],
    policy: MismatchTolerantTemporalPolicy,
) -> tuple[float, ...]:
    required = {variant.value for variant in REQUIRED_DEVELOPMENT_VARIANTS}
    if set(metrics_by_variant) != required:
        raise ValueError("policy evaluation must contain exactly the declared development variants")
    metrics = {TraceVariant(key): value for key, value in metrics_by_variant.items()}
    safety_violations = sum(
        value.pre_onset_selection_group_count + value.nominal_false_alarm_group_count
        for value in metrics.values()
    )
    safe = safety_violations == 0
    exact = metrics[TraceVariant.EXACT_PAIRED]
    exact_noninferiority = (
        exact.group_macro_notified_within_20s_rate >= 0.95
        and exact.minimum_class_notified_within_20s_rate >= 0.95
        and exact.fault_alignment_admission_rate >= 0.95
    )
    in_envelope = [metrics[variant] for variant in IN_ENVELOPE_VARIANTS]
    worst_group = min(item.group_macro_notified_within_20s_rate for item in in_envelope)
    worst_admission = min(item.fault_alignment_admission_rate for item in in_envelope)
    mean_group = float(
        sum(item.group_macro_notified_within_20s_rate for item in in_envelope)
        / len(in_envelope)
    )
    minimum_class = min(item.minimum_class_notified_within_20s_rate for item in in_envelope)
    cross = metrics[TraceVariant.CROSS_MC_REFERENCE]
    notification_delay = (
        exact.median_notification_delay_s
        if exact.median_notification_delay_s is not None
        else 1e12
    )
    return (
        1.0 if safe else 0.0,
        1.0 if exact_noninferiority else 0.0,
        -float(safety_violations),
        worst_group,
        worst_admission,
        mean_group,
        minimum_class,
        cross.notified_within_20s_rate,
        cross.fault_alignment_admission_rate,
        -float(notification_delay),
        -float(policy.alignment_fit_max),
        float(policy.trigger_threshold),
        -float(policy.trigger_persistence_points),
        -int(policy.policy_sha256, 16),
    )


def rank_mismatch_tolerant_policies(
    policies: Iterable[MismatchTolerantTemporalPolicy],
    bundles: Mapping[TraceVariant, FrozenMismatchTraceBundle],
    truth_table: ModelIndependentTruthTable,
    alignment_contract: MismatchTolerantAlignmentContract,
) -> list[MismatchPolicyCandidate]:
    if set(bundles) != set(REQUIRED_DEVELOPMENT_VARIANTS):
        raise ValueError("development selection requires exactly the frozen six variants")
    if set(truth_table.development_groups) != {3, 8} or set(truth_table.excluded_groups) != {4, 9}:
        raise ValueError("truth table development/post-hoc boundary differs")
    results: list[MismatchPolicyCandidate] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.policy_sha256 in seen:
            raise ValueError("duplicate policy candidate")
        seen.add(policy.policy_sha256)
        metrics = {
            variant.value: evaluate_mismatch_tolerant_policy(
                policy,
                bundles[variant],
                truth_table,
                alignment_contract,
            )
            for variant in REQUIRED_DEVELOPMENT_VARIANTS
        }
        safety_violations = sum(
            item.pre_onset_selection_group_count + item.nominal_false_alarm_group_count
            for item in metrics.values()
        )
        exact = metrics[TraceVariant.EXACT_PAIRED.value]
        exact_noninferiority = (
            exact.group_macro_notified_within_20s_rate >= 0.95
            and exact.minimum_class_notified_within_20s_rate >= 0.95
            and exact.fault_alignment_admission_rate >= 0.95
        )
        results.append(
            MismatchPolicyCandidate(
                policy=policy,
                metrics_by_variant=metrics,
                safe_across_development_variants=safety_violations == 0,
                exact_noninferiority_pass=exact_noninferiority,
                selection_key=mismatch_policy_selection_key(metrics, policy),
            )
        )
    if not results:
        raise ValueError("candidate grid is empty")
    return sorted(results, key=lambda item: item.selection_key, reverse=True)


def freeze_mismatch_tolerant_policy(
    ranked_candidates: Sequence[MismatchPolicyCandidate],
    bundles: Mapping[TraceVariant, FrozenMismatchTraceBundle],
    truth_table: ModelIndependentTruthTable,
    scale_contract: ModelIndependentScaleContract,
    alignment_contract: MismatchTolerantAlignmentContract,
    candidate_grid_sha256: str,
) -> FrozenMismatchPolicyReceipt:
    if not ranked_candidates:
        raise ValueError("cannot freeze an empty policy ranking")
    ranked = sorted(ranked_candidates, key=lambda item: item.selection_key, reverse=True)
    if list(ranked_candidates) != ranked:
        raise ValueError("candidate list must be supplied in ranked order")
    selected = ranked[0]
    if not selected.safe_across_development_variants or not selected.exact_noninferiority_pass:
        raise ValueError("no scientifically admissible mismatch-tolerant policy was selected")
    bundle_hashes = {
        variant.value: bundles[variant].bundle_sha256
        for variant in REQUIRED_DEVELOPMENT_VARIANTS
    }
    payload = {
        "policy_sha256": selected.policy.policy_sha256,
        "trace_bundle_sha256s": bundle_hashes,
        "truth_table_sha256": truth_table.truth_table_sha256,
        "scale_contract_sha256": scale_contract.contract_sha256,
        "alignment_contract_sha256": alignment_contract.contract_sha256,
        "candidate_grid_sha256": candidate_grid_sha256,
    }
    return FrozenMismatchPolicyReceipt(
        receipt_id="MTTPR-" + canonical_sha256(payload)[:32].upper(),
        selected_policy=selected.policy,
        selected_metrics_by_variant=selected.metrics_by_variant,
        trace_bundle_sha256s=bundle_hashes,
        truth_table_sha256=truth_table.truth_table_sha256,
        scale_contract_sha256=scale_contract.contract_sha256,
        alignment_contract_sha256=alignment_contract.contract_sha256,
        candidate_grid_sha256=candidate_grid_sha256,
        candidate_count=len(ranked),
        development_variants=tuple(variant.value for variant in REQUIRED_DEVELOPMENT_VARIANTS),
    )


__all__ = [
    "AlignmentDecision",
    "FrozenMismatchPolicyReceipt",
    "FrozenMismatchTolerantTrace",
    "FrozenMismatchTraceBundle",
    "IN_ENVELOPE_VARIANTS",
    "MismatchLocalizationStatus",
    "MismatchPolicyCandidate",
    "MismatchTolerantAlignmentContract",
    "MismatchTolerantSelection",
    "MismatchTolerantTemporalMetrics",
    "MismatchTolerantTemporalPolicy",
    "PrefixAlignmentResult",
    "REQUIRED_DEVELOPMENT_VARIANTS",
    "bounded_prefix_alignment",
    "evaluate_mismatch_tolerant_policy",
    "freeze_mismatch_tolerant_policy",
    "generate_mismatch_tolerant_scores",
    "mismatch_policy_selection_key",
    "normalized_residual_matrix",
    "rank_mismatch_tolerant_policies",
    "select_mismatch_tolerant_trace",
    "shift_reference_with_validity",
]
