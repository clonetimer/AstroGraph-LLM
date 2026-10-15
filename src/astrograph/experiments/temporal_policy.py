"""Development-only evaluation and deterministic freezing of causal temporal policies.

The selector itself is label-free.  This module is deliberately evaluator-side:
truth is joined only after score traces have been generated, and every metric is
reported at both trajectory and independent-group level where applicable.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean, median
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from astrograph.assurance import canonical_sha256
from astrograph.diagnosis.causal_temporal import (
    CausalTemporalPolicy,
    CausalTemporalSelector,
    TemporalScorePoint,
    TemporalSelectionStatus,
)


class TemporalDevelopmentTrajectory(BaseModel):
    """Evaluator-side trajectory with truth joined after score generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-temporal-development-trajectory.v2"
    trajectory_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    truth_fault_id: str | None = None
    active_start_s: float | None = None
    active_end_s: float | None = None
    points: tuple[TemporalScorePoint, ...]

    @model_validator(mode="after")
    def valid_truth_window(self) -> "TemporalDevelopmentTrajectory":
        values = (self.active_start_s, self.active_end_s)
        if self.truth_fault_id is None:
            if any(value is not None for value in values):
                raise ValueError("nominal trajectory cannot declare a fault-active interval")
        else:
            if not self.truth_fault_id.strip():
                raise ValueError("fault trajectory truth_fault_id must be nonempty")
            if any(value is None for value in values):
                raise ValueError("fault trajectory requires a complete active interval")
            start = float(self.active_start_s)
            end = float(self.active_end_s)
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("fault-active interval must be finite")
            if end < start:
                raise ValueError("fault-active interval end must be >= start")
        return self


class TemporalPolicyMetrics(BaseModel):
    """Trajectory and group-level metrics for one frozen temporal policy.

    ``correct_identification_count`` is intentionally strict: a trajectory is
    correct only when the selected endpoint lies inside the declared active
    interval *and* the selected fault identity is correct.  Class-only accuracy
    among selected trajectories is reported separately so timing mistakes cannot
    be hidden by a correct label.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-temporal-policy-metrics.v2"
    policy_id: str
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_count: int = Field(ge=0)
    independent_group_count: int = Field(ge=0)
    fault_trajectory_count: int = Field(ge=0)
    nominal_trajectory_count: int = Field(ge=0)
    fault_independent_group_count: int = Field(ge=0)
    nominal_independent_group_count: int = Field(ge=0)
    selected_fault_trajectory_count: int = Field(ge=0)
    selected_within_active_count: int = Field(ge=0)
    pre_onset_selection_count: int = Field(ge=0)
    pre_onset_selection_group_count: int = Field(ge=0)
    post_active_selection_count: int = Field(ge=0)
    correct_identification_count: int = Field(ge=0)
    selected_classification_correct_count: int = Field(ge=0)
    nominal_false_alarm_count: int = Field(ge=0)
    nominal_false_alarm_group_count: int = Field(ge=0)
    event_selection_recall: float = Field(ge=0.0, le=1.0)
    active_window_selection_rate: float = Field(ge=0.0, le=1.0)
    end_to_end_identification_accuracy: float = Field(ge=0.0, le=1.0)
    selected_identification_accuracy: float = Field(ge=0.0, le=1.0)
    selected_classification_accuracy: float = Field(ge=0.0, le=1.0)
    nominal_false_alarm_rate: float = Field(ge=0.0, le=1.0)
    group_macro_event_selection_recall: float = Field(ge=0.0, le=1.0)
    group_macro_active_window_selection_rate: float = Field(ge=0.0, le=1.0)
    group_macro_end_to_end_accuracy: float = Field(ge=0.0, le=1.0)
    median_detection_delay_s: float | None = None
    maximum_detection_delay_s: float | None = None
    per_group_event_selection_recall: dict[str, float] = Field(default_factory=dict)
    per_group_active_window_selection_rate: dict[str, float] = Field(default_factory=dict)
    per_group_end_to_end_accuracy: dict[str, float] = Field(default_factory=dict)
    claim_boundary: str

    @model_validator(mode="after")
    def internally_consistent(self) -> "TemporalPolicyMetrics":
        if self.correct_identification_count > self.selected_within_active_count:
            raise ValueError("correct end-to-end count cannot exceed active-window selections")
        if self.selected_within_active_count > self.selected_fault_trajectory_count:
            raise ValueError("active-window selections cannot exceed selected fault trajectories")
        if self.selected_classification_correct_count > self.selected_fault_trajectory_count:
            raise ValueError("class-correct selections cannot exceed selected fault trajectories")
        if self.nominal_false_alarm_group_count > self.nominal_independent_group_count:
            raise ValueError("nominal false-alarm groups exceed nominal groups")
        return self


class TemporalPolicyCandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: CausalTemporalPolicy
    metrics: TemporalPolicyMetrics
    selection_key: tuple[float, ...]


class FrozenTemporalPolicyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-frozen-temporal-policy-receipt.v2"
    receipt_id: str = Field(min_length=1)
    policy: CausalTemporalPolicy
    model_manifest_id: str = Field(min_length=1)
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_dataset_id: str = Field(min_length=1)
    development_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_split_id: str = Field(min_length=1)
    development_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_grid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trigger_contract_id: str = Field(min_length=1)
    trigger_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_rule_id: str = Field(min_length=1)
    selection_rule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_rank: int = Field(ge=1)
    selected_metrics: TemporalPolicyMetrics
    candidate_count: int = Field(ge=1)
    excluded_evaluation_splits: tuple[str, ...] = ()
    evaluation_set_status: str = Field(min_length=1)
    claim_boundary: str

    @computed_field
    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"receipt_sha256"}))

    @model_validator(mode="after")
    def policy_matches_metrics(self) -> "FrozenTemporalPolicyReceipt":
        if self.policy.policy_id != self.selected_metrics.policy_id:
            raise ValueError("receipt policy and selected metrics use different policy IDs")
        if self.policy.policy_sha256 != self.selected_metrics.policy_sha256:
            raise ValueError("receipt policy hash and selected metrics hash differ")
        if self.development_split_id in self.excluded_evaluation_splits:
            raise ValueError("development split cannot also be an excluded evaluation split")
        return self


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _macro(values: dict[str, float]) -> float:
    return fmean(values.values()) if values else 0.0


def evaluate_temporal_policy(
    policy: CausalTemporalPolicy,
    trajectories: Iterable[TemporalDevelopmentTrajectory],
) -> TemporalPolicyMetrics:
    items = tuple(trajectories)
    trajectory_ids = [item.trajectory_id for item in items]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("temporal evaluation contains duplicate trajectory_id values")

    selected_fault = 0
    selected_active = 0
    pre_onset = 0
    post_active = 0
    correct_e2e = 0
    class_correct_selected = 0
    nominal_false = 0
    delays: list[float] = []

    group_fault_total: dict[str, int] = defaultdict(int)
    group_selected: dict[str, int] = defaultdict(int)
    group_active: dict[str, int] = defaultdict(int)
    group_correct: dict[str, int] = defaultdict(int)
    nominal_groups: set[str] = set()
    nominal_false_groups: set[str] = set()
    pre_onset_groups: set[str] = set()
    fault_groups: set[str] = set()
    fault_count = 0
    nominal_count = 0

    for item in items:
        selection = CausalTemporalSelector.select(policy, item.points)
        selected = selection.status == TemporalSelectionStatus.SELECTED
        if item.truth_fault_id is None:
            nominal_count += 1
            nominal_groups.add(item.group_id)
            if selected:
                nominal_false += 1
                nominal_false_groups.add(item.group_id)
            continue

        fault_count += 1
        fault_groups.add(item.group_id)
        group_fault_total[item.group_id] += 1
        if not selected or selection.selected_point is None:
            continue

        selected_fault += 1
        group_selected[item.group_id] += 1
        endpoint = selection.selected_point.endpoint_time_s
        active_start = float(item.active_start_s)
        active_end = float(item.active_end_s)
        within_active = active_start <= endpoint <= active_end
        class_correct = selection.selected_point.top_fault_id == item.truth_fault_id
        if class_correct:
            class_correct_selected += 1
        if endpoint < active_start:
            pre_onset += 1
            pre_onset_groups.add(item.group_id)
        elif endpoint > active_end:
            post_active += 1
        else:
            selected_active += 1
            group_active[item.group_id] += 1
            delays.append(endpoint - active_start)
            if class_correct:
                correct_e2e += 1
                group_correct[item.group_id] += 1

    per_group_selected = {
        group_id: _fraction(group_selected[group_id], total)
        for group_id, total in sorted(group_fault_total.items())
    }
    per_group_active = {
        group_id: _fraction(group_active[group_id], total)
        for group_id, total in sorted(group_fault_total.items())
    }
    per_group_correct = {
        group_id: _fraction(group_correct[group_id], total)
        for group_id, total in sorted(group_fault_total.items())
    }

    return TemporalPolicyMetrics(
        policy_id=policy.policy_id,
        policy_sha256=policy.policy_sha256,
        trajectory_count=len(items),
        independent_group_count=len({item.group_id for item in items}),
        fault_trajectory_count=fault_count,
        nominal_trajectory_count=nominal_count,
        fault_independent_group_count=len(fault_groups),
        nominal_independent_group_count=len(nominal_groups),
        selected_fault_trajectory_count=selected_fault,
        selected_within_active_count=selected_active,
        pre_onset_selection_count=pre_onset,
        pre_onset_selection_group_count=len(pre_onset_groups),
        post_active_selection_count=post_active,
        correct_identification_count=correct_e2e,
        selected_classification_correct_count=class_correct_selected,
        nominal_false_alarm_count=nominal_false,
        nominal_false_alarm_group_count=len(nominal_false_groups),
        event_selection_recall=_fraction(selected_fault, fault_count),
        active_window_selection_rate=_fraction(selected_active, fault_count),
        end_to_end_identification_accuracy=_fraction(correct_e2e, fault_count),
        selected_identification_accuracy=_fraction(correct_e2e, selected_fault),
        selected_classification_accuracy=_fraction(class_correct_selected, selected_fault),
        nominal_false_alarm_rate=_fraction(nominal_false, nominal_count),
        group_macro_event_selection_recall=_macro(per_group_selected),
        group_macro_active_window_selection_rate=_macro(per_group_active),
        group_macro_end_to_end_accuracy=_macro(per_group_correct),
        median_detection_delay_s=median(delays) if delays else None,
        maximum_detection_delay_s=max(delays) if delays else None,
        per_group_event_selection_recall=per_group_selected,
        per_group_active_window_selection_rate=per_group_active,
        per_group_end_to_end_accuracy=per_group_correct,
        claim_boundary=(
            "Development-only evaluator metrics. A correct end-to-end identification requires an "
            "in-active-interval selection and the correct top fault. Ground truth is joined after "
            "score traces are generated. These metrics do not establish confirmatory, external, "
            "calibration, safety, or flight validity."
        ),
    )


def temporal_policy_selection_key(
    metrics: TemporalPolicyMetrics,
    policy: CausalTemporalPolicy,
) -> tuple[float, ...]:
    """Predeclared deterministic ordering; larger tuples are preferred.

    Group-level safety and effectiveness terms precede trajectory-level terms
    so a large number of correlated windows cannot dominate the policy choice.
    """

    median_delay = metrics.median_detection_delay_s
    return (
        -float(metrics.nominal_false_alarm_group_count),
        -metrics.nominal_false_alarm_rate,
        -float(metrics.pre_onset_selection_group_count),
        -float(metrics.pre_onset_selection_count),
        metrics.group_macro_end_to_end_accuracy,
        metrics.group_macro_active_window_selection_rate,
        metrics.end_to_end_identification_accuracy,
        metrics.active_window_selection_rate,
        metrics.selected_identification_accuracy,
        -(median_delay if median_delay is not None else 1e12),
        float(policy.trigger_persistence_points + policy.attribution_persistence_points),
        -float(policy.selection_delay_points),
        -int(policy.policy_sha256, 16),
    )


def rank_temporal_policies(
    policies: Iterable[CausalTemporalPolicy],
    trajectories: Iterable[TemporalDevelopmentTrajectory],
) -> list[TemporalPolicyCandidateResult]:
    trajectory_items = tuple(trajectories)
    results: list[TemporalPolicyCandidateResult] = []
    seen_hashes: set[str] = set()
    for policy in policies:
        if policy.policy_sha256 in seen_hashes:
            raise ValueError("candidate grid contains duplicate temporal policies")
        seen_hashes.add(policy.policy_sha256)
        metrics = evaluate_temporal_policy(policy, trajectory_items)
        results.append(
            TemporalPolicyCandidateResult(
                policy=policy,
                metrics=metrics,
                selection_key=temporal_policy_selection_key(metrics, policy),
            )
        )
    if not results:
        raise ValueError("candidate grid must contain at least one temporal policy")
    return sorted(results, key=lambda item: item.selection_key, reverse=True)


def freeze_temporal_policy(
    *,
    ranked_candidates: Iterable[TemporalPolicyCandidateResult],
    model_manifest_id: str,
    model_artifact_sha256: str,
    development_dataset_id: str,
    development_dataset_sha256: str,
    development_split_id: str,
    development_split_sha256: str,
    candidate_grid_sha256: str,
    trigger_contract_id: str,
    trigger_contract_sha256: str,
    excluded_evaluation_splits: Iterable[str],
    evaluation_set_status: str,
) -> FrozenTemporalPolicyReceipt:
    supplied = list(ranked_candidates)
    if not supplied:
        raise ValueError("cannot freeze an empty temporal-policy ranking")
    ranked = sorted(supplied, key=lambda item: item.selection_key, reverse=True)
    if supplied != ranked:
        raise ValueError("temporal-policy candidates must be supplied in ranked order")
    if len({item.policy.policy_sha256 for item in ranked}) != len(ranked):
        raise ValueError("ranked temporal-policy candidates contain duplicate policies")
    selected = ranked[0]
    selection_rule = {
        "rule_id": "temporal-policy-group-first-no-false-alarm-v2",
        "ordering": [
            "min nominal false-alarm group count",
            "min nominal false-alarm trajectory rate",
            "min pre-onset group count",
            "min pre-onset trajectory count",
            "max group-macro end-to-end identification accuracy",
            "max group-macro active-window selection rate",
            "max trajectory end-to-end identification accuracy",
            "max trajectory active-window selection rate",
            "max selected end-to-end identification accuracy",
            "min median delay",
            "max total persistence",
            "min selection delay",
            "deterministic policy hash tie-break",
        ],
    }
    payload = {
        "policy_sha256": selected.policy.policy_sha256,
        "model_manifest_id": model_manifest_id,
        "model_artifact_sha256": model_artifact_sha256,
        "development_dataset_sha256": development_dataset_sha256,
        "development_split_sha256": development_split_sha256,
        "candidate_grid_sha256": candidate_grid_sha256,
        "trigger_contract_sha256": trigger_contract_sha256,
        "selection_rule_sha256": canonical_sha256(selection_rule),
    }
    return FrozenTemporalPolicyReceipt(
        receipt_id="TPR-" + canonical_sha256(payload)[:32].upper(),
        policy=selected.policy,
        model_manifest_id=model_manifest_id,
        model_artifact_sha256=model_artifact_sha256,
        development_dataset_id=development_dataset_id,
        development_dataset_sha256=development_dataset_sha256,
        development_split_id=development_split_id,
        development_split_sha256=development_split_sha256,
        candidate_grid_sha256=candidate_grid_sha256,
        trigger_contract_id=trigger_contract_id,
        trigger_contract_sha256=trigger_contract_sha256,
        selection_rule_id=selection_rule["rule_id"],
        selection_rule_sha256=canonical_sha256(selection_rule),
        selected_candidate_rank=1,
        selected_metrics=selected.metrics,
        candidate_count=len(ranked),
        excluded_evaluation_splits=tuple(sorted(set(excluded_evaluation_splits))),
        evaluation_set_status=evaluation_set_status,
        claim_boundary=(
            "Frozen from the named development split only. Excluded evaluation splits were not "
            "used by the selection function. The receipt freezes model/data/split/grid/policy "
            "identity but does not make a reused or previously inspected evaluation set confirmatory."
        ),
    )


__all__ = [
    "FrozenTemporalPolicyReceipt",
    "TemporalDevelopmentTrajectory",
    "TemporalPolicyCandidateResult",
    "TemporalPolicyMetrics",
    "evaluate_temporal_policy",
    "freeze_temporal_policy",
    "rank_temporal_policies",
    "temporal_policy_selection_key",
]
