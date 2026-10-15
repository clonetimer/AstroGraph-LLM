"""Reference-set, label-isolated temporal localization for ADCS development.

The B009 route used one nominal reference at a time.  This module adds a
content-addressed nominal reference library and a leave-one-provenance-group-out
runtime audit.  Each candidate reference is aligned using only the initial
label-free prefix, corrected by a prefix-only per-channel additive offset, and
then combined into an uncertainty envelope.  The envelope is deliberately
bounded and selective; it is not a learned probabilistic digital twin and it
provides no conformal or external-validity guarantee.

Runtime objects contain no diagnostic class, event onset, active interval, or
prediction-outcome field.  Ground truth is accepted only by the offline
evaluator after trace bundles have been frozen.
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
from astrograph.experiments.mismatch_tolerant_temporal import shift_reference_with_validity
from astrograph.experiments.model_independent_temporal import (
    ModelIndependentScaleContract,
    ModelIndependentTruthTable,
)

_SHA256 = r"^[0-9a-f]{64}$"
_OPAQUE_TRAJECTORY = r"^TRJ-[0-9A-F]{20}$"
_OPAQUE_GROUP = r"^GRP-[0-9A-F]{16}$"
_REFERENCE_ID = r"^REF-[0-9A-F]{20}$"
_FAMILY_ID = r"^FAM-[0-9A-F]{16}$"


class ReferenceSetVariant(str, Enum):
    LEAVE_ONE_GROUP_OUT = "LEAVE_ONE_GROUP_OUT"
    REFERENCE_LIBRARY_SHIFT_1S = "REFERENCE_LIBRARY_SHIFT_1S"
    REFERENCE_LIBRARY_SHIFT_2S = "REFERENCE_LIBRARY_SHIFT_2S"
    CURRENT_NOISE_1PCT_SCALE = "CURRENT_NOISE_1PCT_SCALE"
    CURRENT_NOISE_5PCT_SCALE = "CURRENT_NOISE_5PCT_SCALE"


class ReferenceSetDecision(str, Enum):
    ADMITTED = "ADMITTED"
    ABSTAIN_REFERENCE_SET_OUT_OF_ENVELOPE = "ABSTAIN_REFERENCE_SET_OUT_OF_ENVELOPE"


class ReferenceSetLocalizationStatus(str, Enum):
    SELECTED = "SELECTED"
    NO_ALARM = "NO_ALARM"
    ABSTAIN_REFERENCE_SET_OUT_OF_ENVELOPE = "ABSTAIN_REFERENCE_SET_OUT_OF_ENVELOPE"


class ReferenceLibraryEntry(BaseModel):
    """One nominal reference with an opaque provenance-group binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-library-entry.v1"
    reference_id: str = Field(pattern=_REFERENCE_ID)
    reference_family_id: str = Field(pattern=_FAMILY_ID)
    associated_independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    telemetry_relative_path: str = Field(min_length=1)
    telemetry_sha256: str = Field(pattern=_SHA256)
    row_count: int = Field(ge=2)
    channel_count: int = Field(ge=1)
    nominal_source_attested: bool = True
    runtime_ground_truth_access: bool = False

    @model_validator(mode="after")
    def safe_entry(self) -> "ReferenceLibraryEntry":
        if self.telemetry_relative_path.startswith(("/", "\\")) or ".." in self.telemetry_relative_path.split("/"):
            raise ValueError("reference path must be relative and cannot traverse")
        if not self.nominal_source_attested:
            raise ValueError("reference library entries require an attested nominal source")
        if self.runtime_ground_truth_access:
            raise ValueError("reference entries cannot expose runtime ground truth")
        return self


class FrozenReferenceLibrary(BaseModel):
    """Content-addressed reference library used by runtime trace generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-frozen-reference-library.v1"
    library_id: str = Field(min_length=1)
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    reference_source_manifest_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    channel_order: tuple[str, ...] = Field(min_length=1)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    entries: tuple[ReferenceLibraryEntry, ...] = Field(min_length=2)
    target_provenance_group_exclusion_required: bool = True
    runtime_ground_truth_access: bool = False
    claim_boundary: str = (
        "Nominal development reference library. The leave-one-provenance-group-out audit "
        "tests reference substitution on groups 3/8 only and does not establish deployment, "
        "external, HIL, safety, flight, or confirmatory validity."
    )

    @model_validator(mode="after")
    def exact_library(self) -> "FrozenReferenceLibrary":
        if set(self.development_groups) != {3, 8} or set(self.excluded_groups) != {4, 9}:
            raise ValueError("reference library boundary must remain development 3/8 and excluded 4/9")
        if len({entry.reference_id for entry in self.entries}) != len(self.entries):
            raise ValueError("duplicate reference IDs")
        if len({entry.telemetry_sha256 for entry in self.entries}) != len(self.entries):
            raise ValueError("duplicate reference telemetry payloads")
        if len({entry.associated_independent_group_id for entry in self.entries}) != len(self.entries):
            raise ValueError("each reference must bind to a unique provenance group")
        if any(entry.channel_count != len(self.channel_order) for entry in self.entries):
            raise ValueError("reference channel count differs from library channel order")
        if not self.target_provenance_group_exclusion_required:
            raise ValueError("leave-one-provenance-group-out exclusion must be required")
        if self.runtime_ground_truth_access:
            raise ValueError("reference library cannot access runtime ground truth")
        return self

    @computed_field
    @cached_property
    def library_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"library_sha256"}))


class ReferenceSetTemporalContract(BaseModel):
    """Frozen score semantics selected during groups-3/8 method development."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-set-temporal-contract.v1"
    contract_id: str = Field(min_length=1)
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    runtime_manifest_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    reference_library_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    prefix_points: int = Field(default=31, ge=5)
    maximum_absolute_lag_points: int = Field(default=3, ge=0, le=30)
    sample_period_s: float = Field(default=1.0, gt=0.0)
    alignment_channel_quantile: float = Field(default=0.75, gt=0.5, lt=1.0)
    alignment_time_aggregation: str = "MEDIAN"
    offset_estimator: str = "PER_CHANNEL_PREFIX_MEDIAN"
    minimum_candidate_references: int = Field(default=5, ge=2)
    level_half_width_multiplier: float = Field(default=9.0, ge=1.0)
    gradient_half_width_multiplier: float = Field(default=5.0, ge=1.0)
    score_channel_quantile: float = Field(default=0.99, gt=0.5, lt=1.0)
    prefix_baseline_quantile: float = Field(default=0.99, gt=0.5, lt=1.0)
    padded_edge_action: str = "INVALIDATE"
    envelope_support_rule: str = "ALL_SELECTED_REFERENCES"
    notification_buffer_points: int = Field(default=3, ge=0)
    runtime_ground_truth_access: bool = False
    method_selection_disclosure: str = (
        "Envelope widths and score components were selected after exploratory groups-3/8 development. "
        "They are development choices, not untouched preregistration. Groups 4/9 were not read."
    )
    claim_boundary: str = (
        "Development-only leave-one-provenance-group-out reference-set score. It is an empirical "
        "uncertainty envelope, not a calibrated probability distribution or conformal guarantee."
    )

    @field_validator(
        "sample_period_s",
        "alignment_channel_quantile",
        "level_half_width_multiplier",
        "gradient_half_width_multiplier",
        "score_channel_quantile",
        "prefix_baseline_quantile",
    )
    @classmethod
    def finite_values(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("reference-set contract values must be finite")
        return value

    @model_validator(mode="after")
    def scientific_contract(self) -> "ReferenceSetTemporalContract":
        if set(self.development_groups) != {3, 8} or set(self.excluded_groups) != {4, 9}:
            raise ValueError("reference-set development boundary must remain groups 3/8 vs 4/9")
        if self.alignment_time_aggregation != "MEDIAN":
            raise ValueError("only MEDIAN alignment aggregation is audited")
        if self.offset_estimator != "PER_CHANNEL_PREFIX_MEDIAN":
            raise ValueError("only prefix-median offset correction is audited")
        if self.padded_edge_action != "INVALIDATE":
            raise ValueError("alignment padding must be invalidated")
        if self.envelope_support_rule != "ALL_SELECTED_REFERENCES":
            raise ValueError("all selected references must support each valid envelope point")
        if self.prefix_points - self.maximum_absolute_lag_points < 3:
            raise ValueError("prefix is too short for the declared lag range")
        if self.notification_buffer_points < self.maximum_absolute_lag_points:
            raise ValueError("notification buffer must cover the lag range")
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


class ReferenceCandidateAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-candidate-alignment.v1"
    reference_id: str = Field(pattern=_REFERENCE_ID)
    associated_independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    reference_telemetry_sha256: str = Field(pattern=_SHA256)
    selected_lag_points: int
    fit_score: float = Field(ge=0.0)
    second_best_margin: float = Field(ge=0.0)
    offset_vector_sha256: str = Field(pattern=_SHA256)
    invalid_prefix_points: int = Field(ge=0)
    invalid_suffix_points: int = Field(ge=0)
    fit_prefix_valid_points: int = Field(ge=1)
    runtime_ground_truth_access: bool = False

    @field_validator("fit_score", "second_best_margin")
    @classmethod
    def finite_score(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("candidate alignment scores must be finite")
        return value

    @model_validator(mode="after")
    def safe_alignment(self) -> "ReferenceCandidateAlignment":
        if self.invalid_prefix_points and self.invalid_suffix_points:
            raise ValueError("a single lag cannot invalidate both edges")
        if self.runtime_ground_truth_access:
            raise ValueError("candidate alignment cannot access truth")
        return self


class FrozenReferenceSetTrace(BaseModel):
    """Frozen runtime score trace with no evaluator truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-set-temporal-trace.v1"
    opaque_trajectory_id: str = Field(pattern=_OPAQUE_TRAJECTORY)
    independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    development_group: int
    variant: ReferenceSetVariant
    current_telemetry_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    reference_library_sha256: str = Field(pattern=_SHA256)
    temporal_contract_sha256: str = Field(pattern=_SHA256)
    time_s: tuple[float, ...] = Field(min_length=2)
    level_score: tuple[float, ...] = Field(min_length=2)
    gradient_score: tuple[float, ...] = Field(min_length=2)
    source_missing_fraction: tuple[float, ...] = Field(min_length=2)
    score_valid: tuple[bool, ...] = Field(min_length=2)
    level_prefix_baseline: float = Field(ge=0.0)
    gradient_prefix_baseline: float = Field(ge=0.0)
    best_reference_fit_score: float = Field(ge=0.0)
    median_reference_fit_score: float = Field(ge=0.0)
    maximum_reference_fit_score: float = Field(ge=0.0)
    candidate_alignments: tuple[ReferenceCandidateAlignment, ...] = Field(min_length=2)
    target_reference_excluded: bool = True
    runtime_ground_truth_access: bool = False
    generator_version: str = "b010.reference-set-temporal.r001"
    claim_boundary: str = (
        "Label-free leave-one-provenance-group-out runtime score trace. No fault label, onset, "
        "active interval, prediction correctness, or evaluator outcome was available to generation."
    )

    @field_validator("time_s", "level_score", "gradient_score", "source_missing_fraction")
    @classmethod
    def finite_series(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        converted = tuple(float(item) for item in value)
        if any(not math.isfinite(item) for item in converted):
            raise ValueError("trace arrays must be finite")
        return converted

    @model_validator(mode="after")
    def exact_trace(self) -> "FrozenReferenceSetTrace":
        lengths = {
            len(self.time_s),
            len(self.level_score),
            len(self.gradient_score),
            len(self.source_missing_fraction),
            len(self.score_valid),
        }
        if len(lengths) != 1:
            raise ValueError("trace arrays must have equal lengths")
        if any(b <= a for a, b in zip(self.time_s, self.time_s[1:])):
            raise ValueError("trace times must be strictly increasing")
        if any(value < 0.0 for value in (*self.level_score, *self.gradient_score)):
            raise ValueError("scores must be nonnegative")
        if any(value < 0.0 or value > 1.0 for value in self.source_missing_fraction):
            raise ValueError("missing fractions must lie in [0,1]")
        if self.development_group in {4, 9}:
            raise ValueError("groups 4/9 are permanently post-hoc")
        if any(item.associated_independent_group_id == self.independent_group_id for item in self.candidate_alignments):
            raise ValueError("target provenance group leaked into candidate reference set")
        if len({item.reference_id for item in self.candidate_alignments}) != len(self.candidate_alignments):
            raise ValueError("duplicate candidate references")
        if not self.target_reference_excluded:
            raise ValueError("target reference must be excluded")
        if self.runtime_ground_truth_access:
            raise ValueError("runtime ground-truth access must remain false")
        if not math.isclose(self.best_reference_fit_score, min(item.fit_score for item in self.candidate_alignments), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("best reference fit is inconsistent")
        ordered = sorted(item.fit_score for item in self.candidate_alignments)
        if not math.isclose(self.median_reference_fit_score, float(np.median(ordered)), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("median reference fit is inconsistent")
        if not math.isclose(self.maximum_reference_fit_score, max(ordered), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("maximum reference fit is inconsistent")
        return self

    @computed_field
    @cached_property
    def trace_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"trace_sha256"}))


class FrozenReferenceSetTraceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-set-temporal-trace-bundle.v1"
    bundle_id: str = Field(min_length=1)
    variant: ReferenceSetVariant
    dataset_archive_sha256: str = Field(pattern=_SHA256)
    runtime_manifest_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    reference_library_sha256: str = Field(pattern=_SHA256)
    temporal_contract_sha256: str = Field(pattern=_SHA256)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    traces: tuple[FrozenReferenceSetTrace, ...] = Field(min_length=1)
    predictions_frozen_before_truth_join: bool = True
    runtime_ground_truth_access: bool = False

    @model_validator(mode="after")
    def exact_bundle(self) -> "FrozenReferenceSetTraceBundle":
        if {trace.development_group for trace in self.traces} != {3, 8}:
            raise ValueError("bundle must contain exactly development groups 3/8")
        if any(trace.variant != self.variant for trace in self.traces):
            raise ValueError("trace variant differs from bundle")
        if len({trace.opaque_trajectory_id for trace in self.traces}) != len(self.traces):
            raise ValueError("duplicate trajectory in bundle")
        if any(trace.reference_library_sha256 != self.reference_library_sha256 for trace in self.traces):
            raise ValueError("trace/library identity differs")
        if any(trace.temporal_contract_sha256 != self.temporal_contract_sha256 for trace in self.traces):
            raise ValueError("trace/contract identity differs")
        if not self.predictions_frozen_before_truth_join or self.runtime_ground_truth_access:
            raise ValueError("bundle violates label-isolation contract")
        return self

    @computed_field
    @cached_property
    def bundle_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"bundle_sha256"}))


class ReferenceSetTemporalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-set-temporal-policy.v1"
    policy_id: str = Field(min_length=1)
    reference_fit_max: float = Field(gt=0.0)
    trigger_threshold: float = Field(gt=0.0)
    trigger_persistence_points: int = Field(default=3, ge=1)
    warmup_points: int = Field(default=31, ge=0)
    gradient_weight: float = Field(default=0.0, ge=0.0)
    score_combination: str = "MAX"
    maximum_source_missing_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    notification_buffer_points: int = Field(default=3, ge=0)
    out_of_envelope_action: str = "ABSTAIN"
    no_alarm_action: str = "ABSTAIN"
    claim_boundary: str = (
        "Development-selected reference-set temporal policy using groups 3/8 only. "
        "It is not confirmatory and cannot be tuned on groups 4/9."
    )

    @field_validator("reference_fit_max", "trigger_threshold", "gradient_weight")
    @classmethod
    def finite_policy_value(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("policy values must be finite")
        return value

    @model_validator(mode="after")
    def safe_policy(self) -> "ReferenceSetTemporalPolicy":
        if self.score_combination != "MAX":
            raise ValueError("only MAX level/gradient combination is audited")
        if self.out_of_envelope_action != "ABSTAIN" or self.no_alarm_action != "ABSTAIN":
            raise ValueError("reference-set uncertainty and no-alarm outcomes must abstain")
        return self

    @computed_field
    @cached_property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"policy_sha256"}))


class ReferenceSetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-set-temporal-selection.v1"
    opaque_trajectory_id: str = Field(pattern=_OPAQUE_TRAJECTORY)
    independent_group_id: str = Field(pattern=_OPAQUE_GROUP)
    development_group: int
    trace_sha256: str = Field(pattern=_SHA256)
    policy_sha256: str = Field(pattern=_SHA256)
    reference_set_decision: ReferenceSetDecision
    status: ReferenceSetLocalizationStatus
    selected_index: int | None = Field(default=None, ge=0)
    selected_time_s: float | None = None
    notification_available_time_s: float | None = None
    selected_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def exact_selection(self) -> "ReferenceSetSelection":
        selected = self.status == ReferenceSetLocalizationStatus.SELECTED
        present = all(
            value is not None
            for value in (
                self.selected_index,
                self.selected_time_s,
                self.notification_available_time_s,
                self.selected_score,
            )
        )
        if selected != present:
            raise ValueError("selection fields must be present exactly for SELECTED status")
        if selected and self.notification_available_time_s < self.selected_time_s:
            raise ValueError("notification cannot precede event localization")
        return self


class ReferenceSetTemporalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-reference-set-temporal-metrics.v1"
    variant: ReferenceSetVariant
    policy_sha256: str = Field(pattern=_SHA256)
    trajectory_count: int = Field(ge=1)
    independent_group_count: int = Field(ge=1)
    fault_trajectory_count: int = Field(ge=0)
    nominal_trajectory_count: int = Field(ge=0)
    reference_set_admitted_count: int = Field(ge=0)
    reference_set_abstention_count: int = Field(ge=0)
    fault_reference_set_admitted_count: int = Field(ge=0)
    nominal_reference_set_admitted_count: int = Field(ge=0)
    selected_fault_count: int = Field(ge=0)
    selected_within_active_count: int = Field(ge=0)
    notified_within_5s_count: int = Field(ge=0)
    notified_within_10s_count: int = Field(ge=0)
    notified_within_20s_count: int = Field(ge=0)
    pre_onset_selection_count: int = Field(ge=0)
    pre_onset_selection_group_count: int = Field(ge=0)
    nominal_false_alarm_count: int = Field(ge=0)
    nominal_false_alarm_group_count: int = Field(ge=0)
    reference_set_admission_rate: float = Field(ge=0.0, le=1.0)
    fault_reference_set_admission_rate: float = Field(ge=0.0, le=1.0)
    nominal_reference_set_admission_rate: float = Field(ge=0.0, le=1.0)
    event_selection_recall: float = Field(ge=0.0, le=1.0)
    active_window_selection_rate: float = Field(ge=0.0, le=1.0)
    notified_within_5s_rate: float = Field(ge=0.0, le=1.0)
    notified_within_10s_rate: float = Field(ge=0.0, le=1.0)
    notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    conditional_notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    nominal_false_alarm_rate: float = Field(ge=0.0, le=1.0)
    group_macro_notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    minimum_group_notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    minimum_class_notified_within_20s_rate: float = Field(ge=0.0, le=1.0)
    median_notification_delay_s: float | None = None
    maximum_notification_delay_s: float | None = None
    per_group_notified_within_20s_rate: dict[str, float]
    per_group_fault_reference_set_admission_rate: dict[str, float]
    per_class_notified_within_20s_rate: dict[str, float]
    best_fit_quantiles: dict[str, float]
    claim_boundary: str = (
        "Development-only leave-one-provenance-group-out metrics. The statistical unit is the "
        "independent trajectory-pair/Monte-Carlo group, not a window or telemetry sample."
    )


class ReferenceSetPolicyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: ReferenceSetTemporalPolicy
    metrics_by_variant: dict[str, ReferenceSetTemporalMetrics]
    safe_across_development_variants: bool
    canonical_admission_pass: bool
    selection_key: tuple[float, ...]


class FrozenReferenceSetPolicyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-frozen-reference-set-policy-receipt.v1"
    receipt_id: str = Field(min_length=1)
    selected_policy: ReferenceSetTemporalPolicy
    selected_metrics_by_variant: dict[str, ReferenceSetTemporalMetrics]
    trace_bundle_sha256s: dict[str, str]
    truth_table_sha256: str = Field(pattern=_SHA256)
    scale_contract_sha256: str = Field(pattern=_SHA256)
    reference_library_sha256: str = Field(pattern=_SHA256)
    temporal_contract_sha256: str = Field(pattern=_SHA256)
    candidate_grid_sha256: str = Field(pattern=_SHA256)
    candidate_count: int = Field(ge=1)
    development_groups: tuple[int, ...] = (3, 8)
    excluded_groups: tuple[int, ...] = (4, 9)
    groups_4_9_read: bool = False
    new_confirmatory_evidence: str = "NONE"
    claim_boundary: str = (
        "Frozen from groups 3/8 development traces only. The reference-set method and candidate grid "
        "were informed by groups-3/8 exploration; this receipt is not preregistered confirmation."
    )

    @model_validator(mode="after")
    def boundary(self) -> "FrozenReferenceSetPolicyReceipt":
        if set(self.development_groups) != {3, 8} or set(self.excluded_groups) != {4, 9}:
            raise ValueError("receipt split boundary differs")
        if self.groups_4_9_read:
            raise ValueError("groups 4/9 cannot enter reference-set policy selection")
        if self.new_confirmatory_evidence != "NONE":
            raise ValueError("development receipt cannot create confirmatory evidence")
        return self

    @computed_field
    @cached_property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"receipt_sha256"}))


def _validate_matrix(matrix: Sequence[Sequence[float | None]], contract: ModelIndependentScaleContract) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(contract.channel_order):
        raise ValueError("telemetry matrix shape differs from scale contract")
    return array


def _normalized_absolute_residual(
    current: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    valid_time: np.ndarray,
    one_sided_missing_score: float,
) -> np.ndarray:
    finite_current = np.isfinite(current)
    finite_reference = np.isfinite(reference)
    both = finite_current & finite_reference & valid_time[:, None]
    one = (finite_current ^ finite_reference) & valid_time[:, None]
    output = np.zeros_like(current, dtype=np.float64)
    broadcast = np.broadcast_to(scale, current.shape)
    output[both] = np.abs(current[both] - reference[both]) / broadcast[both]
    output[one] = one_sided_missing_score
    return output


def align_reference_candidate(
    current: Sequence[Sequence[float | None]],
    reference: Sequence[Sequence[float | None]],
    scale_contract: ModelIndependentScaleContract,
    temporal_contract: ReferenceSetTemporalContract,
    entry: ReferenceLibraryEntry,
) -> tuple[np.ndarray, np.ndarray, ReferenceCandidateAlignment]:
    """Prefix-align one candidate and apply a prefix-only additive offset."""

    current_array = _validate_matrix(current, scale_contract)
    reference_array = _validate_matrix(reference, scale_contract)
    if current_array.shape != reference_array.shape:
        raise ValueError("current/reference shape differs")
    scale = np.asarray(scale_contract.channel_scale, dtype=np.float64)
    prefix = temporal_contract.prefix_points
    candidates: list[tuple[tuple[float, int, int], np.ndarray, np.ndarray, np.ndarray, int]] = []
    for lag in temporal_contract.candidate_lags:
        shifted, valid = shift_reference_with_validity(reference_array, lag)
        shifted = shifted.copy()
        shifted[~valid] = np.nan
        both = np.isfinite(current_array[:prefix]) & np.isfinite(shifted[:prefix]) & valid[:prefix, None]
        offset = np.zeros(current_array.shape[1], dtype=np.float64)
        for channel in range(current_array.shape[1]):
            values = current_array[:prefix, channel][both[:, channel]] - shifted[:prefix, channel][both[:, channel]]
            offset[channel] = float(np.median(values)) if len(values) else 0.0
        adjusted = shifted + offset[None, :]
        residual = _normalized_absolute_residual(
            current_array[:prefix],
            adjusted[:prefix],
            scale,
            valid[:prefix],
            scale_contract.one_sided_missing_score,
        )
        per_time = np.quantile(residual, temporal_contract.alignment_channel_quantile, axis=1)
        fit_values = per_time[valid[:prefix]]
        if len(fit_values) < 3:
            fit = math.inf
        else:
            fit = float(np.median(fit_values))
        candidates.append(((fit, abs(lag), lag), adjusted, valid, offset, lag))
    candidates.sort(key=lambda item: item[0])
    selected_key, adjusted, valid, offset, lag = candidates[0]
    if not math.isfinite(selected_key[0]):
        raise ValueError("no finite prefix alignment candidate")
    second = candidates[1][0][0] if len(candidates) > 1 else selected_key[0]
    diagnostic = ReferenceCandidateAlignment(
        reference_id=entry.reference_id,
        associated_independent_group_id=entry.associated_independent_group_id,
        reference_telemetry_sha256=entry.telemetry_sha256,
        selected_lag_points=lag,
        fit_score=selected_key[0],
        second_best_margin=max(0.0, float(second - selected_key[0])),
        offset_vector_sha256=canonical_sha256(tuple(float(value) for value in offset)),
        invalid_prefix_points=max(0, lag),
        invalid_suffix_points=max(0, -lag),
        fit_prefix_valid_points=int(np.sum(valid[:prefix])),
    )
    return adjusted, valid, diagnostic


def _envelope_component(
    current: np.ndarray,
    references: np.ndarray,
    candidate_validity: np.ndarray,
    scale: np.ndarray,
    *,
    half_width_multiplier: float,
    score_quantile: float,
    prefix_points: int,
    prefix_baseline_quantile: float,
    gradient: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if gradient:
        current_component = np.full_like(current, np.nan)
        reference_component = np.full_like(references, np.nan)
        current_component[1:] = current[1:] - current[:-1]
        reference_component[:, 1:] = references[:, 1:] - references[:, :-1]
        candidate_valid = candidate_validity.copy()
        candidate_valid[:, 1:] &= candidate_validity[:, :-1]
        candidate_valid[:, 0] = False
    else:
        current_component = current
        reference_component = references
        candidate_valid = candidate_validity

    all_time_valid = np.all(candidate_valid, axis=0)
    finite_references = np.isfinite(reference_component)
    support = np.sum(finite_references, axis=0)
    required = references.shape[0]
    channel_valid = np.isfinite(current_component) & (support == required) & all_time_valid[:, None]

    finite_for_envelope = np.isfinite(reference_component)
    lower_raw = np.min(np.where(finite_for_envelope, reference_component, np.inf), axis=0)
    upper_raw = np.max(np.where(finite_for_envelope, reference_component, -np.inf), axis=0)
    unsupported = np.sum(finite_for_envelope, axis=0) == 0
    lower_raw[unsupported] = np.nan
    upper_raw[unsupported] = np.nan
    center = (lower_raw + upper_raw) / 2.0
    half_width = (upper_raw - lower_raw) / 2.0
    lower = center - half_width_multiplier * half_width
    upper = center + half_width_multiplier * half_width
    exceedance = np.maximum(np.maximum(lower - current_component, current_component - upper), 0.0)
    normalized = exceedance / scale[None, :]
    normalized[~channel_valid] = np.nan

    raw = np.zeros(current.shape[0], dtype=np.float64)
    missing = np.ones(current.shape[0], dtype=np.float64)
    valid_time = np.zeros(current.shape[0], dtype=bool)
    for index in range(current.shape[0]):
        values = normalized[index][np.isfinite(normalized[index])]
        missing[index] = 1.0 - len(values) / current.shape[1]
        if all_time_valid[index] and len(values):
            raw[index] = float(np.quantile(values, score_quantile))
            valid_time[index] = True
    prefix_values = raw[:prefix_points][valid_time[:prefix_points]]
    if len(prefix_values) < 3:
        raise ValueError("insufficient valid prefix points for envelope baseline")
    baseline = float(np.quantile(prefix_values, prefix_baseline_quantile))
    score = np.maximum(0.0, raw - baseline)
    score[:prefix_points] = 0.0
    score[~valid_time] = 0.0
    return score, valid_time, missing, baseline


def prepare_reference_set_candidates(
    current_prefix_source: Sequence[Sequence[float | None]],
    candidate_references: Mapping[str, Sequence[Sequence[float | None]]],
    reference_library: FrozenReferenceLibrary,
    scale_contract: ModelIndependentScaleContract,
    temporal_contract: ReferenceSetTemporalContract,
    *,
    target_independent_group_id: str,
) -> tuple[np.ndarray, np.ndarray, tuple[ReferenceCandidateAlignment, ...]]:
    """Prepare an aligned leave-one-provenance-group-out reference ensemble.

    The returned arrays depend only on the label-free prefix of
    ``current_prefix_source`` and the frozen nominal library.
    """

    current_array = _validate_matrix(current_prefix_source, scale_contract)
    entries = [
        entry
        for entry in reference_library.entries
        if entry.associated_independent_group_id != target_independent_group_id
    ]
    if len(entries) < temporal_contract.minimum_candidate_references:
        raise ValueError("insufficient independent candidate references")
    if any(entry.reference_id not in candidate_references for entry in entries):
        raise ValueError("candidate reference matrix is missing")
    adjusted: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    diagnostics: list[ReferenceCandidateAlignment] = []
    for entry in sorted(entries, key=lambda item: item.reference_id):
        prediction, valid, diagnostic = align_reference_candidate(
            current_array,
            candidate_references[entry.reference_id],
            scale_contract,
            temporal_contract,
            entry,
        )
        adjusted.append(prediction)
        validities.append(valid)
        diagnostics.append(diagnostic)
    return (
        np.asarray(adjusted, dtype=np.float64),
        np.asarray(validities, dtype=bool),
        tuple(diagnostics),
    )


def generate_reference_set_scores_from_prepared(
    current: Sequence[Sequence[float | None]],
    prepared_references: np.ndarray,
    prepared_validity: np.ndarray,
    diagnostics: tuple[ReferenceCandidateAlignment, ...],
    scale_contract: ModelIndependentScaleContract,
    temporal_contract: ReferenceSetTemporalContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, tuple[ReferenceCandidateAlignment, ...]]:
    current_array = _validate_matrix(current, scale_contract)
    if prepared_references.ndim != 3 or prepared_references.shape[1:] != current_array.shape:
        raise ValueError("prepared reference ensemble shape differs from current telemetry")
    if prepared_validity.shape != prepared_references.shape[:2]:
        raise ValueError("prepared reference validity shape differs")
    if len(diagnostics) != prepared_references.shape[0]:
        raise ValueError("prepared diagnostics/reference count differs")
    scale = np.asarray(scale_contract.channel_scale, dtype=np.float64)
    level, level_valid, level_missing, level_baseline = _envelope_component(
        current_array,
        prepared_references,
        prepared_validity,
        scale,
        half_width_multiplier=temporal_contract.level_half_width_multiplier,
        score_quantile=temporal_contract.score_channel_quantile,
        prefix_points=temporal_contract.prefix_points,
        prefix_baseline_quantile=temporal_contract.prefix_baseline_quantile,
        gradient=False,
    )
    gradient, gradient_valid, gradient_missing, gradient_baseline = _envelope_component(
        current_array,
        prepared_references,
        prepared_validity,
        scale,
        half_width_multiplier=temporal_contract.gradient_half_width_multiplier,
        score_quantile=temporal_contract.score_channel_quantile,
        prefix_points=temporal_contract.prefix_points,
        prefix_baseline_quantile=temporal_contract.prefix_baseline_quantile,
        gradient=True,
    )
    score_valid = level_valid & gradient_valid
    missing = np.maximum(level_missing, gradient_missing)
    level[~score_valid] = 0.0
    gradient[~score_valid] = 0.0
    return level, gradient, missing, score_valid, level_baseline, gradient_baseline, diagnostics


def generate_reference_set_scores(
    current: Sequence[Sequence[float | None]],
    candidate_references: Mapping[str, Sequence[Sequence[float | None]]],
    reference_library: FrozenReferenceLibrary,
    scale_contract: ModelIndependentScaleContract,
    temporal_contract: ReferenceSetTemporalContract,
    *,
    target_independent_group_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, tuple[ReferenceCandidateAlignment, ...]]:
    prepared, validity, diagnostics = prepare_reference_set_candidates(
        current,
        candidate_references,
        reference_library,
        scale_contract,
        temporal_contract,
        target_independent_group_id=target_independent_group_id,
    )
    return generate_reference_set_scores_from_prepared(
        current, prepared, validity, diagnostics, scale_contract, temporal_contract
    )


def select_reference_set_trace(
    trace: FrozenReferenceSetTrace,
    policy: ReferenceSetTemporalPolicy,
) -> ReferenceSetSelection:
    if trace.best_reference_fit_score > policy.reference_fit_max:
        return ReferenceSetSelection(
            opaque_trajectory_id=trace.opaque_trajectory_id,
            independent_group_id=trace.independent_group_id,
            development_group=trace.development_group,
            trace_sha256=trace.trace_sha256,
            policy_sha256=policy.policy_sha256,
            reference_set_decision=ReferenceSetDecision.ABSTAIN_REFERENCE_SET_OUT_OF_ENVELOPE,
            status=ReferenceSetLocalizationStatus.ABSTAIN_REFERENCE_SET_OUT_OF_ENVELOPE,
        )
    level = np.asarray(trace.level_score, dtype=np.float64)
    gradient = np.asarray(trace.gradient_score, dtype=np.float64)
    combined = np.maximum(level, policy.gradient_weight * gradient)
    persistence = 0
    for index, (time_s, score, missing, valid) in enumerate(
        zip(trace.time_s, combined, trace.source_missing_fraction, trace.score_valid)
    ):
        if index < policy.warmup_points or not valid or missing > policy.maximum_source_missing_fraction:
            persistence = 0
            continue
        persistence = persistence + 1 if score >= policy.trigger_threshold else 0
        if persistence >= policy.trigger_persistence_points:
            selected_index = index
            selected_time = float(time_s)
            return ReferenceSetSelection(
                opaque_trajectory_id=trace.opaque_trajectory_id,
                independent_group_id=trace.independent_group_id,
                development_group=trace.development_group,
                trace_sha256=trace.trace_sha256,
                policy_sha256=policy.policy_sha256,
                reference_set_decision=ReferenceSetDecision.ADMITTED,
                status=ReferenceSetLocalizationStatus.SELECTED,
                selected_index=selected_index,
                selected_time_s=selected_time,
                notification_available_time_s=selected_time + policy.notification_buffer_points,
                selected_score=float(score),
            )
    return ReferenceSetSelection(
        opaque_trajectory_id=trace.opaque_trajectory_id,
        independent_group_id=trace.independent_group_id,
        development_group=trace.development_group,
        trace_sha256=trace.trace_sha256,
        policy_sha256=policy.policy_sha256,
        reference_set_decision=ReferenceSetDecision.ADMITTED,
        status=ReferenceSetLocalizationStatus.NO_ALARM,
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"q00": 0.0, "q50": 0.0, "q90": 0.0, "q99": 0.0, "q100": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "q00": float(np.quantile(array, 0.0)),
        "q50": float(np.quantile(array, 0.5)),
        "q90": float(np.quantile(array, 0.9)),
        "q99": float(np.quantile(array, 0.99)),
        "q100": float(np.quantile(array, 1.0)),
    }


def evaluate_reference_set_policy(
    policy: ReferenceSetTemporalPolicy,
    bundle: FrozenReferenceSetTraceBundle,
    truth_table: ModelIndependentTruthTable,
) -> ReferenceSetTemporalMetrics:
    if bundle.dataset_archive_sha256 != truth_table.dataset_archive_sha256:
        raise ValueError("trace/truth dataset identity differs")
    truth = {row.opaque_trajectory_id: row for row in truth_table.rows}
    traces = {trace.opaque_trajectory_id: trace for trace in bundle.traces}
    if set(traces) != set(truth):
        raise ValueError("trace and truth trajectory sets differ")
    if any(trace.development_group in {4, 9} for trace in traces.values()):
        raise ValueError("groups 4/9 cannot enter development evaluation")

    fault_total = nominal_total = admitted = abstained = fault_admitted = nominal_admitted = 0
    selected_fault = selected_active = notified5 = notified10 = notified20 = 0
    pre_onset = nominal_false = 0
    pre_groups: set[str] = set()
    false_groups: set[str] = set()
    delays: list[float] = []
    fit_scores: list[float] = []
    group_fault_total: defaultdict[str, int] = defaultdict(int)
    group_fault_admitted: defaultdict[str, int] = defaultdict(int)
    group_timely: defaultdict[str, int] = defaultdict(int)
    class_total: defaultdict[str, int] = defaultdict(int)
    class_timely: defaultdict[str, int] = defaultdict(int)

    for trajectory_id in sorted(traces):
        trace = traces[trajectory_id]
        row = truth[trajectory_id]
        if trace.independent_group_id != row.independent_group_id or trace.development_group != row.development_group:
            raise ValueError("trace/truth group binding differs")
        selection = select_reference_set_trace(trace, policy)
        fit_scores.append(trace.best_reference_fit_score)
        is_admitted = selection.reference_set_decision == ReferenceSetDecision.ADMITTED
        if is_admitted:
            admitted += 1
        else:
            abstained += 1
        if row.is_nominal:
            nominal_total += 1
            if is_admitted:
                nominal_admitted += 1
            if selection.status == ReferenceSetLocalizationStatus.SELECTED:
                nominal_false += 1
                false_groups.add(row.independent_group_id)
            continue

        fault_total += 1
        group_fault_total[row.independent_group_id] += 1
        class_total[row.diagnostic_class] += 1
        if is_admitted:
            fault_admitted += 1
            group_fault_admitted[row.independent_group_id] += 1
        if selection.status != ReferenceSetLocalizationStatus.SELECTED:
            continue
        selected_fault += 1
        selected_time = float(selection.selected_time_s)
        notification_time = float(selection.notification_available_time_s)
        if selected_time < row.active_start_s:
            pre_onset += 1
            pre_groups.add(row.independent_group_id)
            continue
        if selected_time <= row.active_end_s:
            selected_active += 1
        delay = notification_time - row.active_start_s
        if selected_time <= row.active_end_s and delay <= 20.0:
            notified20 += 1
            group_timely[row.independent_group_id] += 1
            class_timely[row.diagnostic_class] += 1
            delays.append(delay)
            if delay <= 10.0:
                notified10 += 1
            if delay <= 5.0:
                notified5 += 1

    all_groups = sorted({row.independent_group_id for row in truth.values()})
    groups = sorted(group_fault_total)
    per_group = {
        group: _fraction(group_timely[group], group_fault_total[group]) for group in groups
    }
    per_group_admission = {
        group: _fraction(group_fault_admitted[group], group_fault_total[group]) for group in groups
    }
    per_class = {
        diagnostic_class: _fraction(class_timely[diagnostic_class], class_total[diagnostic_class])
        for diagnostic_class in sorted(class_total)
    }
    return ReferenceSetTemporalMetrics(
        variant=bundle.variant,
        policy_sha256=policy.policy_sha256,
        trajectory_count=len(traces),
        independent_group_count=len(all_groups),
        fault_trajectory_count=fault_total,
        nominal_trajectory_count=nominal_total,
        reference_set_admitted_count=admitted,
        reference_set_abstention_count=abstained,
        fault_reference_set_admitted_count=fault_admitted,
        nominal_reference_set_admitted_count=nominal_admitted,
        selected_fault_count=selected_fault,
        selected_within_active_count=selected_active,
        notified_within_5s_count=notified5,
        notified_within_10s_count=notified10,
        notified_within_20s_count=notified20,
        pre_onset_selection_count=pre_onset,
        pre_onset_selection_group_count=len(pre_groups),
        nominal_false_alarm_count=nominal_false,
        nominal_false_alarm_group_count=len(false_groups),
        reference_set_admission_rate=_fraction(admitted, len(traces)),
        fault_reference_set_admission_rate=_fraction(fault_admitted, fault_total),
        nominal_reference_set_admission_rate=_fraction(nominal_admitted, nominal_total),
        event_selection_recall=_fraction(selected_fault, fault_total),
        active_window_selection_rate=_fraction(selected_active, fault_total),
        notified_within_5s_rate=_fraction(notified5, fault_total),
        notified_within_10s_rate=_fraction(notified10, fault_total),
        notified_within_20s_rate=_fraction(notified20, fault_total),
        conditional_notified_within_20s_rate=_fraction(notified20, fault_admitted),
        nominal_false_alarm_rate=_fraction(nominal_false, nominal_total),
        group_macro_notified_within_20s_rate=float(np.mean(list(per_group.values()))) if per_group else 0.0,
        minimum_group_notified_within_20s_rate=min(per_group.values()) if per_group else 0.0,
        minimum_class_notified_within_20s_rate=min(per_class.values()) if per_class else 0.0,
        median_notification_delay_s=median(delays) if delays else None,
        maximum_notification_delay_s=max(delays) if delays else None,
        per_group_notified_within_20s_rate=per_group,
        per_group_fault_reference_set_admission_rate=per_group_admission,
        per_class_notified_within_20s_rate=per_class,
        best_fit_quantiles=_quantiles(fit_scores),
    )


REQUIRED_REFERENCE_SET_VARIANTS: tuple[ReferenceSetVariant, ...] = (
    ReferenceSetVariant.LEAVE_ONE_GROUP_OUT,
    ReferenceSetVariant.REFERENCE_LIBRARY_SHIFT_1S,
    ReferenceSetVariant.REFERENCE_LIBRARY_SHIFT_2S,
)


def reference_set_policy_selection_key(
    metrics_by_variant: Mapping[str, ReferenceSetTemporalMetrics],
    policy: ReferenceSetTemporalPolicy,
) -> tuple[float, ...]:
    required = {variant.value for variant in REQUIRED_REFERENCE_SET_VARIANTS}
    if set(metrics_by_variant) != required:
        raise ValueError("reference-set evaluation must contain exactly the declared variants")
    metrics = {ReferenceSetVariant(key): value for key, value in metrics_by_variant.items()}
    safety_violations = sum(
        metric.pre_onset_selection_group_count + metric.nominal_false_alarm_group_count
        for metric in metrics.values()
    )
    canonical = metrics[ReferenceSetVariant.LEAVE_ONE_GROUP_OUT]
    canonical_admission = canonical.fault_reference_set_admission_rate >= 0.95
    worst_group_macro = min(metric.group_macro_notified_within_20s_rate for metric in metrics.values())
    worst_fault_admission = min(metric.fault_reference_set_admission_rate for metric in metrics.values())
    mean_group_macro = float(np.mean([metric.group_macro_notified_within_20s_rate for metric in metrics.values()]))
    return (
        1.0 if safety_violations == 0 else 0.0,
        1.0 if canonical_admission else 0.0,
        -float(safety_violations),
        worst_group_macro,
        worst_fault_admission,
        canonical.group_macro_notified_within_20s_rate,
        canonical.minimum_group_notified_within_20s_rate,
        canonical.notified_within_20s_rate,
        canonical.minimum_class_notified_within_20s_rate,
        mean_group_macro,
        -float(policy.reference_fit_max),
        float(policy.trigger_threshold),
        -float(policy.trigger_persistence_points),
        -float(policy.gradient_weight),
        -int(policy.policy_sha256, 16),
    )


def rank_reference_set_policies(
    policies: Iterable[ReferenceSetTemporalPolicy],
    bundles: Mapping[ReferenceSetVariant, FrozenReferenceSetTraceBundle],
    truth_table: ModelIndependentTruthTable,
) -> list[ReferenceSetPolicyCandidate]:
    if set(bundles) != set(REQUIRED_REFERENCE_SET_VARIANTS):
        raise ValueError("reference-set selection requires all declared development variants")
    if set(truth_table.development_groups) != {3, 8} or set(truth_table.excluded_groups) != {4, 9}:
        raise ValueError("truth table split boundary differs")
    output: list[ReferenceSetPolicyCandidate] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.policy_sha256 in seen:
            raise ValueError("duplicate reference-set policy")
        seen.add(policy.policy_sha256)
        metrics = {
            variant.value: evaluate_reference_set_policy(policy, bundles[variant], truth_table)
            for variant in REQUIRED_REFERENCE_SET_VARIANTS
        }
        violations = sum(
            metric.pre_onset_selection_group_count + metric.nominal_false_alarm_group_count
            for metric in metrics.values()
        )
        canonical = metrics[ReferenceSetVariant.LEAVE_ONE_GROUP_OUT.value]
        output.append(
            ReferenceSetPolicyCandidate(
                policy=policy,
                metrics_by_variant=metrics,
                safe_across_development_variants=violations == 0,
                canonical_admission_pass=canonical.fault_reference_set_admission_rate >= 0.95,
                selection_key=reference_set_policy_selection_key(metrics, policy),
            )
        )
    if not output:
        raise ValueError("reference-set policy grid is empty")
    return sorted(output, key=lambda item: item.selection_key, reverse=True)


def freeze_reference_set_policy(
    ranked_candidates: Sequence[ReferenceSetPolicyCandidate],
    bundles: Mapping[ReferenceSetVariant, FrozenReferenceSetTraceBundle],
    truth_table: ModelIndependentTruthTable,
    scale_contract: ModelIndependentScaleContract,
    reference_library: FrozenReferenceLibrary,
    temporal_contract: ReferenceSetTemporalContract,
    candidate_grid_sha256: str,
) -> FrozenReferenceSetPolicyReceipt:
    if not ranked_candidates:
        raise ValueError("cannot freeze an empty ranking")
    ranked = sorted(ranked_candidates, key=lambda item: item.selection_key, reverse=True)
    if list(ranked_candidates) != ranked:
        raise ValueError("candidate list must be supplied in ranked order")
    selected = ranked[0]
    if not selected.safe_across_development_variants or not selected.canonical_admission_pass:
        raise ValueError("no scientifically admissible reference-set policy was selected")
    bundle_hashes = {variant.value: bundles[variant].bundle_sha256 for variant in REQUIRED_REFERENCE_SET_VARIANTS}
    payload = {
        "policy_sha256": selected.policy.policy_sha256,
        "trace_bundle_sha256s": bundle_hashes,
        "truth_table_sha256": truth_table.truth_table_sha256,
        "scale_contract_sha256": scale_contract.contract_sha256,
        "reference_library_sha256": reference_library.library_sha256,
        "temporal_contract_sha256": temporal_contract.contract_sha256,
        "candidate_grid_sha256": candidate_grid_sha256,
    }
    return FrozenReferenceSetPolicyReceipt(
        receipt_id="RSPR-" + canonical_sha256(payload)[:32].upper(),
        selected_policy=selected.policy,
        selected_metrics_by_variant=selected.metrics_by_variant,
        trace_bundle_sha256s=bundle_hashes,
        truth_table_sha256=truth_table.truth_table_sha256,
        scale_contract_sha256=scale_contract.contract_sha256,
        reference_library_sha256=reference_library.library_sha256,
        temporal_contract_sha256=temporal_contract.contract_sha256,
        candidate_grid_sha256=candidate_grid_sha256,
        candidate_count=len(ranked),
    )


__all__ = [
    "FrozenReferenceLibrary",
    "FrozenReferenceSetPolicyReceipt",
    "FrozenReferenceSetTrace",
    "FrozenReferenceSetTraceBundle",
    "REQUIRED_REFERENCE_SET_VARIANTS",
    "ReferenceCandidateAlignment",
    "ReferenceLibraryEntry",
    "ReferenceSetDecision",
    "ReferenceSetLocalizationStatus",
    "ReferenceSetPolicyCandidate",
    "ReferenceSetSelection",
    "ReferenceSetTemporalContract",
    "ReferenceSetTemporalMetrics",
    "ReferenceSetTemporalPolicy",
    "ReferenceSetVariant",
    "align_reference_candidate",
    "evaluate_reference_set_policy",
    "freeze_reference_set_policy",
    "generate_reference_set_scores",
    "generate_reference_set_scores_from_prepared",
    "prepare_reference_set_candidates",
    "rank_reference_set_policies",
    "reference_set_policy_selection_key",
    "select_reference_set_trace",
]
