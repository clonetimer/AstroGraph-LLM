"""Label-free, classifier-independent temporal localization over paired telemetry.

The runtime objects in this module contain no fault labels, event times, or
prediction-correctness fields.  A trace is frozen before a separate offline
evaluator joins a truth table.  The current implementation is intentionally
bounded to paired synthetic/digital-twin telemetry: it does not imply that an
exact nominal counterpart is available in flight.
"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
import math
from statistics import median
from typing import Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from astrograph.assurance import canonical_sha256


_SHA256 = r"^[0-9a-f]{64}$"
_OPAQUE_TRAJECTORY = r"^TRJ-[0-9A-F]{20}$"
_OPAQUE_GROUP = r"^GRP-[0-9A-F]{16}$"


class TraceVariant(str, Enum):
    EXACT_PAIRED = "EXACT_PAIRED"
    REFERENCE_SHIFT_1S = "REFERENCE_SHIFT_1S"
    REFERENCE_SHIFT_2S = "REFERENCE_SHIFT_2S"
    CURRENT_NOISE_1PCT_SCALE = "CURRENT_NOISE_1PCT_SCALE"
    CURRENT_NOISE_5PCT_SCALE = "CURRENT_NOISE_5PCT_SCALE"
    CROSS_MC_REFERENCE = "CROSS_MC_REFERENCE"


class ReferenceDecision(str, Enum):
    ADMITTED = "ADMITTED"
    ABSTAIN_REFERENCE_OUT_OF_ENVELOPE = "ABSTAIN_REFERENCE_OUT_OF_ENVELOPE"


class LocalizationStatus(str, Enum):
    SELECTED = "SELECTED"
    NO_ALARM = "NO_ALARM"
    ABSTAIN_REFERENCE_OUT_OF_ENVELOPE = "ABSTAIN_REFERENCE_OUT_OF_ENVELOPE"


class ModelIndependentScaleContract(BaseModel):
    """Content-addressed score semantics, fitted without fault labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-scale-contract.v1"
    contract_id: str = Field(min_length=1)
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    scale_fit_manifest_sha256: str = Field(pattern=_SHA256)
    scale_fit_groups: tuple[int, ...] = Field(min_length=1)
    excluded_groups: tuple[int, ...] = (4, 9)
    channel_order: tuple[str, ...] = Field(min_length=1)
    channel_scale: tuple[float, ...] = Field(min_length=1)
    score_aggregation: str = "MAX_NORMALIZED_ABSOLUTE_RESIDUAL"
    scale_quantile: float = Field(default=0.99, gt=0.5, lt=1.0)
    continuous_minimum_scale: float = Field(default=1e-6, gt=0.0)
    discrete_minimum_scale: float = Field(default=1.0, gt=0.0)
    one_sided_missing_score: float = Field(default=10.0, ge=0.0)
    runtime_ground_truth_access: bool = False
    claim_boundary: str = (
        "Classifier-independent paired residual score fitted from nominal reference telemetry. "
        "It assumes a time-aligned reference and does not establish robustness to reference mismatch, "
        "external validity, fault identity, causal mechanism, HIL, safety, or flight qualification."
    )

    @field_validator("scale_fit_groups", "excluded_groups")
    @classmethod
    def unique_groups(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            raise ValueError("group lists must not contain duplicates")
        return tuple(int(item) for item in value)

    @field_validator("channel_order")
    @classmethod
    def valid_channels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("channel order must be unique")
        if any(not item.startswith("adcs.") for item in value):
            raise ValueError("model-independent ADCS contract accepts only adcs.* channels")
        forbidden = ("fault", "label", "truth", "event", "condition", "target")
        if any(any(token in item.lower() for token in forbidden) for item in value):
            raise ValueError("truth-like channel names are forbidden")
        return value

    @field_validator("channel_scale")
    @classmethod
    def finite_scales(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        converted = tuple(float(item) for item in value)
        if any(not math.isfinite(item) or item <= 0.0 for item in converted):
            raise ValueError("channel scales must be finite and positive")
        return converted

    @model_validator(mode="after")
    def aligned_contract(self) -> "ModelIndependentScaleContract":
        if len(self.channel_order) != len(self.channel_scale):
            raise ValueError("channel order and scale lengths differ")
        if set(self.scale_fit_groups) & set(self.excluded_groups):
            raise ValueError("scale fitting cannot use excluded groups")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        return self

    @computed_field
    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"contract_sha256"}))


class FrozenModelIndependentTrace(BaseModel):
    """A frozen runtime trace.  Truth fields are structurally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-temporal-trace.v1"
    opaque_trajectory_id: str = Field(pattern=_OPAQUE_TRAJECTORY)
    independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    development_group: int
    variant: TraceVariant
    current_telemetry_sha256: str = Field(pattern=_SHA256)
    reference_telemetry_sha256: str = Field(pattern=_SHA256)
    alternate_reference_sha256: str | None = Field(default=None, pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    time_s: tuple[float, ...] = Field(min_length=2)
    trigger_score: tuple[float, ...] = Field(min_length=2)
    source_missing_fraction: tuple[float, ...] = Field(min_length=2)
    reference_prefix_points: int = Field(ge=1)
    reference_prefix_quantile: float = Field(gt=0.0, lt=1.0)
    reference_prefix_statistic: float = Field(ge=0.0)
    runtime_ground_truth_access: bool = False
    generator_version: str = "b008.model-independent-temporal.r001"
    claim_boundary: str = (
        "Label-free runtime score trace. No fault label, onset, active interval, or prediction outcome "
        "was available to the generator. Truth may be joined only after this trace is frozen."
    )

    @field_validator("time_s", "trigger_score", "source_missing_fraction")
    @classmethod
    def finite_series(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        converted = tuple(float(item) for item in value)
        if any(not math.isfinite(item) for item in converted):
            raise ValueError("trace arrays must contain finite numbers")
        return converted

    @model_validator(mode="after")
    def consistent_trace(self) -> "FrozenModelIndependentTrace":
        lengths = {len(self.time_s), len(self.trigger_score), len(self.source_missing_fraction)}
        if len(lengths) != 1:
            raise ValueError("trace arrays must have equal lengths")
        if any(next_time <= time for time, next_time in zip(self.time_s, self.time_s[1:])):
            raise ValueError("trace times must be strictly increasing")
        if any(score < 0.0 for score in self.trigger_score):
            raise ValueError("trigger scores must be nonnegative")
        if any(value < 0.0 or value > 1.0 for value in self.source_missing_fraction):
            raise ValueError("missing fractions must lie in [0,1]")
        if self.reference_prefix_points > len(self.time_s):
            raise ValueError("reference prefix exceeds trace length")
        if self.development_group in {4, 9}:
            raise ValueError("groups 4/9 are permanently post-hoc")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        return self

    @computed_field
    @property
    def trace_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"trace_sha256"}))


class FrozenTraceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-trace-bundle.v1"
    bundle_id: str = Field(min_length=1)
    variant: TraceVariant
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    runtime_manifest_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    traces: tuple[FrozenModelIndependentTrace, ...] = Field(min_length=1)
    predictions_frozen_before_truth_join: bool = True
    runtime_ground_truth_access: bool = False

    @model_validator(mode="after")
    def consistent_bundle(self) -> "FrozenTraceBundle":
        if len({trace.opaque_trajectory_id for trace in self.traces}) != len(self.traces):
            raise ValueError("bundle contains duplicate trajectory IDs")
        if any(trace.variant != self.variant for trace in self.traces):
            raise ValueError("trace variant differs from bundle variant")
        if any(trace.scale_contract_sha256 != self.scale_contract_sha256 for trace in self.traces):
            raise ValueError("trace scale contract differs from bundle")
        groups = {trace.development_group for trace in self.traces}
        if groups != set(self.development_groups):
            raise ValueError("trace bundle must contain exactly the frozen development groups")
        if groups & set(self.excluded_groups):
            raise ValueError("post-hoc groups cannot enter a development trace bundle")
        if not self.predictions_frozen_before_truth_join or self.runtime_ground_truth_access:
            raise ValueError("trace bundle violates label-isolation contract")
        return self

    @computed_field
    @property
    def bundle_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"bundle_sha256"}))


class ModelIndependentTemporalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-temporal-policy.v1"
    policy_id: str = Field(min_length=1)
    trigger_threshold: float = Field(ge=0.0)
    trigger_persistence_points: int = Field(ge=1)
    warmup_points: int = Field(default=31, ge=0)
    maximum_source_missing_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    reference_prefix_max: float | None = Field(default=0.05, ge=0.0)
    reference_out_of_envelope_action: str = "ABSTAIN"
    no_alarm_action: str = "ABSTAIN"
    claim_boundary: str = (
        "Development-selected temporal localization policy for classifier-independent paired traces. "
        "The reference-admissibility limit is predeclared and is not tuned on stress outcomes."
    )

    @field_validator("trigger_threshold", "reference_prefix_max")
    @classmethod
    def finite_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("thresholds must be finite")
        return value

    @model_validator(mode="after")
    def abstention_contract(self) -> "ModelIndependentTemporalPolicy":
        if self.reference_out_of_envelope_action != "ABSTAIN" or self.no_alarm_action != "ABSTAIN":
            raise ValueError("model-independent policies must abstain on invalid reference or no alarm")
        return self

    @computed_field
    @property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"policy_sha256"}))


class ModelIndependentSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-selection.v1"
    opaque_trajectory_id: str = Field(pattern=_OPAQUE_TRAJECTORY)
    independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    development_group: int
    trace_sha256: str = Field(pattern=_SHA256)
    policy_sha256: str = Field(pattern=_SHA256)
    reference_decision: ReferenceDecision
    status: LocalizationStatus
    selected_index: int | None = Field(default=None, ge=0)
    selected_time_s: float | None = None
    selected_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def status_fields(self) -> "ModelIndependentSelection":
        selected = self.status == LocalizationStatus.SELECTED
        fields_present = self.selected_index is not None and self.selected_time_s is not None and self.selected_score is not None
        if selected != fields_present:
            raise ValueError("selection fields must be present exactly when status is SELECTED")
        if self.status == LocalizationStatus.ABSTAIN_REFERENCE_OUT_OF_ENVELOPE and self.reference_decision != ReferenceDecision.ABSTAIN_REFERENCE_OUT_OF_ENVELOPE:
            raise ValueError("reference abstention status/decision mismatch")
        return self


class ModelIndependentTruthRow(BaseModel):
    """Evaluator-only truth joined after trace freezing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-truth-row.v1"
    opaque_trajectory_id: str = Field(pattern=_OPAQUE_TRAJECTORY)
    independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    development_group: int
    is_nominal: bool
    diagnostic_class: str = Field(min_length=1)
    active_start_s: float | None = None
    active_end_s: float | None = None
    source_metadata_sha256: str = Field(pattern=_SHA256)
    source_labels_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def truth_semantics(self) -> "ModelIndependentTruthRow":
        if self.development_group in {4, 9}:
            raise ValueError("groups 4/9 are permanently post-hoc")
        if self.is_nominal:
            if self.active_start_s is not None or self.active_end_s is not None:
                raise ValueError("nominal truth cannot contain an active interval")
        else:
            if self.active_start_s is None or self.active_end_s is None:
                raise ValueError("fault truth requires an active interval")
            if self.active_end_s < self.active_start_s:
                raise ValueError("active interval is reversed")
        return self


class ModelIndependentTruthTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-truth-table.v1"
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    rows: tuple[ModelIndependentTruthRow, ...] = Field(min_length=1)
    joined_after_trace_freeze: bool = True

    @model_validator(mode="after")
    def exact_rows(self) -> "ModelIndependentTruthTable":
        if len({row.opaque_trajectory_id for row in self.rows}) != len(self.rows):
            raise ValueError("truth table contains duplicate trajectories")
        if {row.development_group for row in self.rows} != set(self.development_groups):
            raise ValueError("truth table must contain exactly development groups 3/8")
        if not self.joined_after_trace_freeze:
            raise ValueError("truth table must be joined only after trace freeze")
        return self

    @computed_field
    @property
    def truth_table_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"truth_table_sha256"}))


class ModelIndependentTemporalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-independent-temporal-metrics.v1"
    variant: TraceVariant
    policy_sha256: str = Field(pattern=_SHA256)
    trajectory_count: int = Field(ge=1)
    independent_group_count: int = Field(ge=1)
    fault_trajectory_count: int = Field(ge=0)
    nominal_trajectory_count: int = Field(ge=0)
    reference_admitted_count: int = Field(ge=0)
    reference_abstention_count: int = Field(ge=0)
    selected_fault_count: int = Field(ge=0)
    selected_within_active_count: int = Field(ge=0)
    detected_within_5s_count: int = Field(ge=0)
    detected_within_10s_count: int = Field(ge=0)
    detected_within_20s_count: int = Field(ge=0)
    pre_onset_selection_count: int = Field(ge=0)
    pre_onset_selection_group_count: int = Field(ge=0)
    nominal_false_alarm_count: int = Field(ge=0)
    nominal_false_alarm_group_count: int = Field(ge=0)
    reference_admission_rate: float = Field(ge=0.0, le=1.0)
    fault_reference_admission_rate: float = Field(ge=0.0, le=1.0)
    event_selection_recall: float = Field(ge=0.0, le=1.0)
    active_window_selection_rate: float = Field(ge=0.0, le=1.0)
    detected_within_5s_rate: float = Field(ge=0.0, le=1.0)
    detected_within_10s_rate: float = Field(ge=0.0, le=1.0)
    detected_within_20s_rate: float = Field(ge=0.0, le=1.0)
    conditional_detected_within_20s_rate: float = Field(ge=0.0, le=1.0)
    nominal_false_alarm_rate: float = Field(ge=0.0, le=1.0)
    group_macro_detected_within_20s_rate: float = Field(ge=0.0, le=1.0)
    minimum_class_detected_within_20s_rate: float = Field(ge=0.0, le=1.0)
    median_detection_delay_s: float | None = None
    maximum_detection_delay_s: float | None = None
    per_group_detected_within_20s_rate: dict[str, float]
    per_class_detected_within_20s_rate: dict[str, float]
    claim_boundary: str = (
        "Development-only temporal localization metrics over independent trajectory/Monte-Carlo groups. "
        "They do not establish fault identification, external validity, confirmation, HIL, safety, or flight qualification."
    )


class ModelIndependentPolicyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: ModelIndependentTemporalPolicy
    metrics: ModelIndependentTemporalMetrics
    selection_key: tuple[float, ...]


class FrozenModelIndependentPolicyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-frozen-model-independent-temporal-policy-receipt.v1"
    receipt_id: str = Field(min_length=1)
    selected_policy: ModelIndependentTemporalPolicy
    selected_metrics: ModelIndependentTemporalMetrics
    canonical_trace_bundle_sha256: str = Field(pattern=_SHA256)
    truth_table_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    candidate_grid_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    candidate_count: int = Field(ge=1)
    model_package_sha256: None = None
    historical_model_required: bool = False
    reference_guard_tuned_on_stress: bool = False
    new_confirmatory_evidence: str = "NONE"
    claim_boundary: str = (
        "Frozen on groups 3/8 development traces only. Stress variants were replayed after selection and "
        "did not influence the policy. Groups 4/9 were not read. This receipt is not confirmatory."
    )

    @computed_field
    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"receipt_sha256"}))


def normalized_residual_scores(
    current: Sequence[Sequence[float | None]],
    reference: Sequence[Sequence[float | None]],
    contract: ModelIndependentScaleContract,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-endpoint score and source-missing fraction."""

    current_array = np.asarray(current, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if current_array.ndim != 2 or current_array.shape != reference_array.shape:
        raise ValueError("current/reference matrices must have equal two-dimensional shape")
    if current_array.shape[1] != len(contract.channel_order):
        raise ValueError("matrix channel count differs from scale contract")
    scale = np.asarray(contract.channel_scale, dtype=np.float64)
    finite_current = np.isfinite(current_array)
    finite_reference = np.isfinite(reference_array)
    both = finite_current & finite_reference
    one_sided = finite_current ^ finite_reference
    residual = np.zeros_like(current_array)
    broadcast_scale = np.broadcast_to(scale, current_array.shape)
    residual[both] = np.abs(current_array[both] - reference_array[both]) / broadcast_scale[both]
    residual[one_sided] = contract.one_sided_missing_score
    score = np.max(residual, axis=1)
    missing = 1.0 - np.mean(both, axis=1)
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(missing)):
        raise ValueError("non-finite score or missing fraction")
    return score, missing


def reference_prefix_statistic(scores: Sequence[float], *, prefix_points: int, quantile: float) -> float:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or len(array) < prefix_points:
        raise ValueError("insufficient score points for reference prefix")
    value = float(np.quantile(array[:prefix_points], quantile))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("invalid reference-prefix statistic")
    return value


def select_model_independent_trace(
    trace: FrozenModelIndependentTrace,
    policy: ModelIndependentTemporalPolicy,
) -> ModelIndependentSelection:
    if policy.reference_prefix_max is not None and trace.reference_prefix_statistic > policy.reference_prefix_max:
        return ModelIndependentSelection(
            opaque_trajectory_id=trace.opaque_trajectory_id,
            independent_group_id=trace.independent_group_id,
            development_group=trace.development_group,
            trace_sha256=trace.trace_sha256,
            policy_sha256=policy.policy_sha256,
            reference_decision=ReferenceDecision.ABSTAIN_REFERENCE_OUT_OF_ENVELOPE,
            status=LocalizationStatus.ABSTAIN_REFERENCE_OUT_OF_ENVELOPE,
        )

    persistence = 0
    for index, (time_s, score, missing) in enumerate(
        zip(trace.time_s, trace.trigger_score, trace.source_missing_fraction)
    ):
        if index < policy.warmup_points or missing > policy.maximum_source_missing_fraction:
            persistence = 0
            continue
        persistence = persistence + 1 if score >= policy.trigger_threshold else 0
        if persistence >= policy.trigger_persistence_points:
            selected_index = index
            return ModelIndependentSelection(
                opaque_trajectory_id=trace.opaque_trajectory_id,
                independent_group_id=trace.independent_group_id,
                development_group=trace.development_group,
                trace_sha256=trace.trace_sha256,
                policy_sha256=policy.policy_sha256,
                reference_decision=ReferenceDecision.ADMITTED,
                status=LocalizationStatus.SELECTED,
                selected_index=selected_index,
                selected_time_s=time_s,
                selected_score=score,
            )
    return ModelIndependentSelection(
        opaque_trajectory_id=trace.opaque_trajectory_id,
        independent_group_id=trace.independent_group_id,
        development_group=trace.development_group,
        trace_sha256=trace.trace_sha256,
        policy_sha256=policy.policy_sha256,
        reference_decision=ReferenceDecision.ADMITTED,
        status=LocalizationStatus.NO_ALARM,
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _macro(values: Mapping[str, float]) -> float:
    return float(sum(values.values()) / len(values)) if values else 0.0


def evaluate_model_independent_policy(
    policy: ModelIndependentTemporalPolicy,
    bundle: FrozenTraceBundle,
    truth_table: ModelIndependentTruthTable,
) -> ModelIndependentTemporalMetrics:
    truth = {row.opaque_trajectory_id: row for row in truth_table.rows}
    traces = {trace.opaque_trajectory_id: trace for trace in bundle.traces}
    if set(traces) != set(truth):
        raise ValueError("trace bundle and truth table trajectory sets differ")
    if bundle.dataset_archive_sha256 != truth_table.dataset_archive_sha256:
        raise ValueError("trace/truth dataset identity differs")

    fault_total = nominal_total = admitted = abstained = 0
    fault_admitted = selected_fault = selected_active = timely5 = timely10 = timely20 = 0
    pre_onset = nominal_false = 0
    pre_groups: set[str] = set()
    false_groups: set[str] = set()
    delays: list[float] = []
    group_total: dict[str, int] = defaultdict(int)
    group_timely: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)
    class_timely: dict[str, int] = defaultdict(int)

    for trajectory_id, trace in traces.items():
        row = truth[trajectory_id]
        if trace.independent_group_id != row.independent_group_id or trace.development_group != row.development_group:
            raise ValueError("trace/truth group binding differs")
        selection = select_model_independent_trace(trace, policy)
        if selection.reference_decision == ReferenceDecision.ADMITTED:
            admitted += 1
        else:
            abstained += 1
        if row.is_nominal:
            nominal_total += 1
            if selection.status == LocalizationStatus.SELECTED:
                nominal_false += 1
                false_groups.add(row.independent_group_id)
            continue

        fault_total += 1
        group_total[row.independent_group_id] += 1
        class_total[row.diagnostic_class] += 1
        if selection.reference_decision == ReferenceDecision.ADMITTED:
            fault_admitted += 1
        if selection.status != LocalizationStatus.SELECTED or selection.selected_time_s is None:
            continue
        selected_fault += 1
        start = float(row.active_start_s)
        end = float(row.active_end_s)
        selected_time = float(selection.selected_time_s)
        if selected_time < start:
            pre_onset += 1
            pre_groups.add(row.independent_group_id)
            continue
        if selected_time > end:
            continue
        selected_active += 1
        delay = selected_time - start
        delays.append(delay)
        if delay <= 5.0:
            timely5 += 1
        if delay <= 10.0:
            timely10 += 1
        if delay <= 20.0:
            timely20 += 1
            group_timely[row.independent_group_id] += 1
            class_timely[row.diagnostic_class] += 1

    per_group = {group: _fraction(group_timely[group], total) for group, total in sorted(group_total.items())}
    per_class = {fault: _fraction(class_timely[fault], total) for fault, total in sorted(class_total.items())}
    return ModelIndependentTemporalMetrics(
        variant=bundle.variant,
        policy_sha256=policy.policy_sha256,
        trajectory_count=len(traces),
        independent_group_count=len({row.independent_group_id for row in truth.values()}),
        fault_trajectory_count=fault_total,
        nominal_trajectory_count=nominal_total,
        reference_admitted_count=admitted,
        reference_abstention_count=abstained,
        selected_fault_count=selected_fault,
        selected_within_active_count=selected_active,
        detected_within_5s_count=timely5,
        detected_within_10s_count=timely10,
        detected_within_20s_count=timely20,
        pre_onset_selection_count=pre_onset,
        pre_onset_selection_group_count=len(pre_groups),
        nominal_false_alarm_count=nominal_false,
        nominal_false_alarm_group_count=len(false_groups),
        reference_admission_rate=_fraction(admitted, len(traces)),
        fault_reference_admission_rate=_fraction(fault_admitted, fault_total),
        event_selection_recall=_fraction(selected_fault, fault_total),
        active_window_selection_rate=_fraction(selected_active, fault_total),
        detected_within_5s_rate=_fraction(timely5, fault_total),
        detected_within_10s_rate=_fraction(timely10, fault_total),
        detected_within_20s_rate=_fraction(timely20, fault_total),
        conditional_detected_within_20s_rate=_fraction(timely20, fault_admitted),
        nominal_false_alarm_rate=_fraction(nominal_false, nominal_total),
        group_macro_detected_within_20s_rate=_macro(per_group),
        minimum_class_detected_within_20s_rate=min(per_class.values()) if per_class else 0.0,
        median_detection_delay_s=median(delays) if delays else None,
        maximum_detection_delay_s=max(delays) if delays else None,
        per_group_detected_within_20s_rate=per_group,
        per_class_detected_within_20s_rate=per_class,
    )


def policy_selection_key(
    metrics: ModelIndependentTemporalMetrics,
    policy: ModelIndependentTemporalPolicy,
) -> tuple[float, ...]:
    delay = metrics.median_detection_delay_s if metrics.median_detection_delay_s is not None else 1e12
    return (
        -float(metrics.nominal_false_alarm_group_count),
        -float(metrics.pre_onset_selection_group_count),
        metrics.group_macro_detected_within_20s_rate,
        metrics.minimum_class_detected_within_20s_rate,
        metrics.detected_within_5s_rate,
        metrics.detected_within_20s_rate,
        -float(delay),
        float(policy.trigger_threshold),
        -float(policy.trigger_persistence_points),
        -int(policy.policy_sha256, 16),
    )


def rank_model_independent_policies(
    policies: Iterable[ModelIndependentTemporalPolicy],
    canonical_bundle: FrozenTraceBundle,
    truth_table: ModelIndependentTruthTable,
) -> list[ModelIndependentPolicyCandidate]:
    if canonical_bundle.variant != TraceVariant.EXACT_PAIRED:
        raise ValueError("policy selection must use the canonical exact-paired development bundle")
    results: list[ModelIndependentPolicyCandidate] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.policy_sha256 in seen:
            raise ValueError("duplicate policy candidate")
        seen.add(policy.policy_sha256)
        metrics = evaluate_model_independent_policy(policy, canonical_bundle, truth_table)
        results.append(ModelIndependentPolicyCandidate(policy=policy, metrics=metrics, selection_key=policy_selection_key(metrics, policy)))
    if not results:
        raise ValueError("candidate grid is empty")
    return sorted(results, key=lambda item: item.selection_key, reverse=True)


def freeze_model_independent_policy(
    ranked_candidates: Sequence[ModelIndependentPolicyCandidate],
    canonical_bundle: FrozenTraceBundle,
    truth_table: ModelIndependentTruthTable,
    scale_contract: ModelIndependentScaleContract,
    candidate_grid_sha256: str,
) -> FrozenModelIndependentPolicyReceipt:
    if not ranked_candidates:
        raise ValueError("cannot freeze an empty policy ranking")
    ranked = sorted(ranked_candidates, key=lambda item: item.selection_key, reverse=True)
    if list(ranked_candidates) != ranked:
        raise ValueError("candidate list must be supplied in ranked order")
    selected = ranked[0]
    payload = {
        "policy_sha256": selected.policy.policy_sha256,
        "canonical_trace_bundle_sha256": canonical_bundle.bundle_sha256,
        "truth_table_sha256": truth_table.truth_table_sha256,
        "scale_contract_sha256": scale_contract.contract_sha256,
        "candidate_grid_sha256": candidate_grid_sha256,
    }
    return FrozenModelIndependentPolicyReceipt(
        receipt_id="MITPR-" + canonical_sha256(payload)[:32].upper(),
        selected_policy=selected.policy,
        selected_metrics=selected.metrics,
        canonical_trace_bundle_sha256=canonical_bundle.bundle_sha256,
        truth_table_sha256=truth_table.truth_table_sha256,
        scale_contract_sha256=scale_contract.contract_sha256,
        candidate_grid_sha256=candidate_grid_sha256,
        candidate_count=len(ranked),
    )


__all__ = [
    "FrozenModelIndependentPolicyReceipt",
    "FrozenModelIndependentTrace",
    "FrozenTraceBundle",
    "LocalizationStatus",
    "ModelIndependentPolicyCandidate",
    "ModelIndependentScaleContract",
    "ModelIndependentSelection",
    "ModelIndependentTemporalMetrics",
    "ModelIndependentTemporalPolicy",
    "ModelIndependentTruthRow",
    "ModelIndependentTruthTable",
    "ReferenceDecision",
    "TraceVariant",
    "evaluate_model_independent_policy",
    "freeze_model_independent_policy",
    "normalized_residual_scores",
    "policy_selection_key",
    "rank_model_independent_policies",
    "reference_prefix_statistic",
    "select_model_independent_trace",
]
