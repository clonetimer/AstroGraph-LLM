from __future__ import annotations

from collections import defaultdict
from random import Random
from statistics import fmean
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GroupedPrediction(BaseModel):
    """One held-out prediction with its independent resampling group.

    ``truth`` is evaluation-only metadata.  The model/fusion path must already
    have completed before this record is constructed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    truth: str = Field(min_length=1)
    prediction: str | None = None


class GroupedClassificationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-grouped-classification-summary.v1"
    observation_count: int = Field(ge=0)
    independent_group_count: int = Field(ge=0)
    label_count: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    abstention_rate: float = Field(ge=0.0, le=1.0)
    group_macro_accuracy: float = Field(ge=0.0, le=1.0)
    worst_group_accuracy: float = Field(ge=0.0, le=1.0)
    best_group_accuracy: float = Field(ge=0.0, le=1.0)
    group_accuracy: dict[str, float]
    cluster_bootstrap_accuracy_ci95: tuple[float, float]
    cluster_bootstrap_group_macro_accuracy_ci95: tuple[float, float]
    bootstrap_iterations: int = Field(ge=0)
    bootstrap_seed: int
    claim_boundary: str = (
        "Cluster bootstrap resamples independent groups, not individual rows. "
        "Precision remains limited when the number of independent groups is small."
    )

    @field_validator(
        "cluster_bootstrap_accuracy_ci95",
        "cluster_bootstrap_group_macro_accuracy_ci95",
    )
    @classmethod
    def ordered_interval(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not (0.0 <= value[0] <= value[1] <= 1.0):
            raise ValueError("bootstrap confidence interval must be ordered within [0, 1]")
        return value


class PairedGroupDeltaSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-paired-group-delta-summary.v1"
    reference_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    independent_group_count: int = Field(ge=1)
    mean_group_accuracy_delta: float
    group_accuracy_delta: dict[str, float]
    bootstrap_mean_group_delta_ci95: tuple[float, float]
    bootstrap_iterations: int = Field(ge=0)
    bootstrap_seed: int
    claim_boundary: str = (
        "Descriptive paired group effect only.  A bootstrap interval based on a small number "
        "of independent groups is not evidence of external validity or flight qualification."
    )


def _accuracy(rows: list[GroupedPrediction]) -> float:
    if not rows:
        return 0.0
    return sum(item.prediction == item.truth for item in rows) / len(rows)


def _macro_f1(rows: list[GroupedPrediction], labels: tuple[str, ...]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(item.truth == label and item.prediction == label for item in rows)
        fp = sum(item.truth != label and item.prediction == label for item in rows)
        fn = sum(item.truth == label and item.prediction != label for item in rows)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return fmean(scores) if scores else 0.0


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _ci95(values: list[float]) -> tuple[float, float]:
    return (_percentile(values, 0.025), _percentile(values, 0.975))


def summarize_grouped_classification(
    observations: Iterable[GroupedPrediction],
    *,
    label_universe: Iterable[str],
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260829,
) -> GroupedClassificationSummary:
    rows = list(observations)
    labels = tuple(label_universe)
    if not labels or len(labels) != len(set(labels)) or any(not item for item in labels):
        raise ValueError("label_universe must contain unique nonempty labels")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be nonnegative")
    unknown_truth = {item.truth for item in rows} - set(labels)
    if unknown_truth:
        raise ValueError(f"truth labels outside label_universe: {sorted(unknown_truth)}")
    unknown_predictions = {item.prediction for item in rows if item.prediction is not None} - set(labels)
    if unknown_predictions:
        raise ValueError(
            f"prediction labels outside label_universe: {sorted(unknown_predictions)}"
        )
    ids = [item.observation_id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("observation_id values must be unique")

    grouped: dict[str, list[GroupedPrediction]] = defaultdict(list)
    for item in rows:
        grouped[item.group_id].append(item)
    if rows and not grouped:
        raise ValueError("nonempty observations require at least one group")

    group_accuracy = {group_id: _accuracy(items) for group_id, items in sorted(grouped.items())}
    group_values = list(group_accuracy.values())
    group_macro = fmean(group_values) if group_values else 0.0

    boot_accuracy: list[float] = []
    boot_group_macro: list[float] = []
    group_ids = sorted(grouped)
    if bootstrap_iterations and group_ids:
        rng = Random(bootstrap_seed)
        for _ in range(bootstrap_iterations):
            selected = [rng.choice(group_ids) for _ in group_ids]
            sampled_rows: list[GroupedPrediction] = []
            sampled_group_accuracies: list[float] = []
            for group_id in selected:
                sampled_rows.extend(grouped[group_id])
                sampled_group_accuracies.append(group_accuracy[group_id])
            boot_accuracy.append(_accuracy(sampled_rows))
            boot_group_macro.append(fmean(sampled_group_accuracies))
    else:
        boot_accuracy = [_accuracy(rows)]
        boot_group_macro = [group_macro]

    return GroupedClassificationSummary(
        observation_count=len(rows),
        independent_group_count=len(grouped),
        label_count=len(labels),
        accuracy=_accuracy(rows),
        macro_f1=_macro_f1(rows, labels),
        abstention_rate=(
            sum(item.prediction is None for item in rows) / len(rows) if rows else 0.0
        ),
        group_macro_accuracy=group_macro,
        worst_group_accuracy=min(group_values) if group_values else 0.0,
        best_group_accuracy=max(group_values) if group_values else 0.0,
        group_accuracy=group_accuracy,
        cluster_bootstrap_accuracy_ci95=_ci95(boot_accuracy),
        cluster_bootstrap_group_macro_accuracy_ci95=_ci95(boot_group_macro),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )


def paired_group_accuracy_delta(
    reference: Iterable[GroupedPrediction],
    candidate: Iterable[GroupedPrediction],
    *,
    reference_id: str,
    candidate_id: str,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260829,
) -> PairedGroupDeltaSummary:
    left = {item.observation_id: item for item in reference}
    right = {item.observation_id: item for item in candidate}
    if set(left) != set(right):
        raise ValueError("paired comparisons require the same observation IDs")
    if not left:
        raise ValueError("paired comparisons require at least one observation")
    for observation_id in left:
        if left[observation_id].truth != right[observation_id].truth:
            raise ValueError(f"truth mismatch for paired observation {observation_id}")
        if left[observation_id].group_id != right[observation_id].group_id:
            raise ValueError(f"group mismatch for paired observation {observation_id}")

    left_groups: dict[str, list[GroupedPrediction]] = defaultdict(list)
    right_groups: dict[str, list[GroupedPrediction]] = defaultdict(list)
    for observation_id in sorted(left):
        left_groups[left[observation_id].group_id].append(left[observation_id])
        right_groups[right[observation_id].group_id].append(right[observation_id])
    group_ids = sorted(left_groups)
    deltas = {
        group_id: _accuracy(right_groups[group_id]) - _accuracy(left_groups[group_id])
        for group_id in group_ids
    }
    values = list(deltas.values())

    boot: list[float] = []
    if bootstrap_iterations:
        rng = Random(bootstrap_seed)
        for _ in range(bootstrap_iterations):
            selected = [rng.choice(group_ids) for _ in group_ids]
            boot.append(fmean(deltas[group_id] for group_id in selected))
    else:
        boot = [fmean(values)]

    return PairedGroupDeltaSummary(
        reference_id=reference_id,
        candidate_id=candidate_id,
        independent_group_count=len(group_ids),
        mean_group_accuracy_delta=fmean(values),
        group_accuracy_delta=deltas,
        bootstrap_mean_group_delta_ci95=_ci95(boot),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )


__all__ = [
    "GroupedClassificationSummary",
    "GroupedPrediction",
    "PairedGroupDeltaSummary",
    "paired_group_accuracy_delta",
    "summarize_grouped_classification",
]
