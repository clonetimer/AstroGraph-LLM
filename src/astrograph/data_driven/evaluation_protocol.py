"""Leakage-resistant metrics for the three-stage diagnosis experiments.

The legacy research scripts mainly score labelled fault windows.  Those scores
are useful representation probes, but they are not operational detection
metrics because the fault onset is already known.  This module keeps the two
tasks separate and provides one implementation of multiclass macro metrics so
that a completely missed class is not silently removed from the denominator.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil, sqrt
from statistics import NormalDist, median
from typing import Any


def multiclass_metrics(
    truth: Sequence[Hashable],
    predicted: Sequence[Hashable],
    *,
    labels: Iterable[Hashable] | None = None,
) -> dict[str, Any]:
    """Return accuracy, balanced accuracy and honest macro-F1.

    Macro metrics are averaged over every class present in ``truth``.  A class
    with non-zero support and zero true positives therefore contributes 0.0.
    ``labels`` may provide a frozen ontology order, but labels absent from the
    evaluated truth are reported and are not inserted into the macro
    denominator.
    """

    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal length")
    if not truth:
        raise ValueError("at least one evaluated item is required")

    observed = set(truth)
    if labels is None:
        label_order = sorted(observed, key=str)
    else:
        label_order = list(labels)
        if len(set(label_order)) != len(label_order):
            raise ValueError("frozen ontology labels must not contain duplicates")
        missing = observed.difference(label_order)
        if missing:
            raise ValueError(f"truth contains labels outside the frozen ontology: {sorted(missing, key=str)}")
        predicted_outside = set(predicted).difference(label_order)
        if predicted_outside:
            raise ValueError(
                "predicted contains labels outside the frozen ontology: "
                f"{sorted(predicted_outside, key=str)}"
            )

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for label in label_order:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted, strict=True))
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[str(label)] = {
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support:
            f1_values.append(f1)
            recall_values.append(recall)

    return {
        "n_items": len(truth),
        "accuracy": sum(t == p for t, p in zip(truth, predicted, strict=True)) / len(truth),
        "macro_f1": sum(f1_values) / len(f1_values),
        # Balanced accuracy is macro recall for single-label multiclass tasks.
        # Keep both names so experiment reports state the requested metric
        # explicitly without changing the established compatibility field.
        "macro_recall": sum(recall_values) / len(recall_values),
        "balanced_accuracy": sum(recall_values) / len(recall_values),
        "evaluated_class_count": len(f1_values),
        "frozen_ontology_class_count": len(label_order),
        "per_class": per_class,
    }


def three_task_evaluation_audit(
    *,
    group_ids: Sequence[Hashable],
    detection_truth: Sequence[int],
    detection_predicted: Sequence[int],
    detection_positive_scores: Sequence[float],
    component_truth: Sequence[int],
    component_predicted: Sequence[int],
    class_truth: Sequence[int],
    class_predicted: Sequence[int],
    component_labels: Sequence[str],
    class_labels: Sequence[str],
    group_key: str,
) -> dict[str, Any]:
    """Build the frozen evaluator-side audit for a three-task cascade.

    Component and class truth use ``-1`` outside fault-active windows.  The
    function deliberately receives predictions rather than model objects so
    it cannot influence training or checkpoint selection.  Group identifiers
    are report-only provenance and must never enter model features.
    """

    sequences = (
        detection_truth,
        detection_predicted,
        detection_positive_scores,
        component_truth,
        component_predicted,
        class_truth,
        class_predicted,
    )
    n_items = len(group_ids)
    if not group_key.strip():
        raise ValueError("group_key must be non-empty")
    if n_items == 0:
        raise ValueError("at least one evaluated item is required")
    if any(len(values) != n_items for values in sequences):
        raise ValueError("all three-task audit sequences must have equal length")
    if any(value not in (0, 1) for value in detection_truth):
        raise ValueError("detection truth must contain only 0 and 1")
    if any(value not in (0, 1) for value in detection_predicted):
        raise ValueError("detection predictions must contain only 0 and 1")

    active = [index for index, value in enumerate(detection_truth) if value == 1]
    if not active:
        raise ValueError("three-task audit requires at least one fault-active item")
    if any(component_truth[index] < 0 or class_truth[index] < 0 for index in active):
        raise ValueError("fault-active items require component and class truth")
    if any(component_truth[index] >= len(component_labels) for index in active):
        raise ValueError("component truth lies outside the frozen ontology")
    if any(class_truth[index] >= len(class_labels) for index in active):
        raise ValueError("class truth lies outside the frozen ontology")
    if any(component_predicted[index] not in range(len(component_labels)) for index in active):
        raise ValueError("component prediction lies outside the frozen ontology")
    if any(class_predicted[index] not in range(len(class_labels)) for index in active):
        raise ValueError("class prediction lies outside the frozen ontology")
    detected = [index for index, value in enumerate(detection_predicted) if value == 1]
    if any(class_predicted[index] not in range(len(class_labels)) for index in detected):
        raise ValueError("detected class prediction lies outside the frozen ontology")

    component_truth_active = [component_labels[component_truth[index]] for index in active]
    component_pred_active = [component_labels[component_predicted[index]] for index in active]
    class_truth_active = [class_labels[class_truth[index]] for index in active]
    class_pred_active = [class_labels[class_predicted[index]] for index in active]
    cascade_labels = ["normal", *class_labels]
    cascade_truth = [
        "normal" if value == 0 else class_labels[class_truth[index]]
        for index, value in enumerate(detection_truth)
    ]
    cascade_predicted = [
        "normal" if value == 0 else class_labels[class_predicted[index]]
        for index, value in enumerate(detection_predicted)
    ]

    groups: list[dict[str, Any]] = []
    positions_by_group: dict[Hashable, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        positions_by_group.setdefault(group_id, []).append(index)
    for group_id in sorted(positions_by_group, key=str):
        positions = positions_by_group[group_id]
        fault_positions = [index for index in positions if detection_truth[index] == 1]
        entry: dict[str, Any] = {
            group_key: str(group_id),
            "window_count": len(positions),
            "fault_window_count": len(fault_positions),
            "detection_accuracy": sum(
                detection_truth[index] == detection_predicted[index] for index in positions
            ) / len(positions),
            "end_to_end_accuracy": sum(
                cascade_truth[index] == cascade_predicted[index] for index in positions
            ) / len(positions),
            "localization_accuracy": None,
            "classification_accuracy": None,
        }
        if fault_positions:
            entry["localization_accuracy"] = sum(
                component_truth[index] == component_predicted[index] for index in fault_positions
            ) / len(fault_positions)
            entry["classification_accuracy"] = sum(
                class_truth[index] == class_predicted[index] for index in fault_positions
            ) / len(fault_positions)
        groups.append(entry)

    scores = [float(value) for value in detection_positive_scores]
    confidences = [score if prediction == 1 else 1.0 - score for score, prediction in zip(
        scores, detection_predicted, strict=True
    )]
    correctness = [expected == actual for expected, actual in zip(
        detection_truth, detection_predicted, strict=True
    )]
    return {
        "schema_version": "astrograph-three-task-evaluation-audit.v1",
        "evaluation_unit": "causal_window",
        "group_key": group_key,
        "selection_boundary": "report_only; test truth and group identifiers are evaluator-side only",
        "overall": {
            "detection": multiclass_metrics(
                detection_truth, detection_predicted, labels=[0, 1]
            ),
            "localization": multiclass_metrics(
                component_truth_active, component_pred_active, labels=component_labels
            ),
            "classification": multiclass_metrics(
                class_truth_active, class_pred_active, labels=class_labels
            ),
            "end_to_end": multiclass_metrics(
                cascade_truth, cascade_predicted, labels=cascade_labels
            ),
            "detection_aupr": binary_aupr(
                [value == 1 for value in detection_truth], scores
            ),
            "detection_calibration": expected_calibration_error(
                correctness,
                confidences,
                bin_edges=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            ),
        },
        "groups": groups,
    }


@dataclass(frozen=True)
class FaultInterval:
    """One evaluator-only fault interval in seconds."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("fault interval must satisfy 0 <= start_s < end_s")


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def event_detection_metrics(
    *,
    fault_intervals: Sequence[FaultInterval],
    detection_times_s: Sequence[float],
    observation_duration_s: float,
    alarm_merge_window_s: float = 0.0,
    healthy_exposure_s: float | None = None,
) -> dict[str, Any]:
    """Score causal detection events without exposing fault windows to inference.

    Consecutive raw alarms separated by at most ``alarm_merge_window_s`` are
    merged and represented by the first timestamp.  The first merged alarm in
    a fault interval detects that event; later merged alarms in the same event
    are duplicates.  Alarms outside every fault interval are false alarms.
    Duplicate alarms reduce event precision but do not count as healthy-time
    false alarms.  Truth intervals are evaluator-only inputs.

    False alarms/hour uses healthy exposure rather than total observation
    duration.  By default healthy exposure is observation duration minus all
    non-overlapping truth intervals; callers may provide an audited value.
    """

    if observation_duration_s <= 0:
        raise ValueError("observation_duration_s must be positive")
    if alarm_merge_window_s < 0:
        raise ValueError("alarm_merge_window_s must be non-negative")
    intervals = sorted(fault_intervals, key=lambda item: item.start_s)
    if any(interval.end_s > observation_duration_s for interval in intervals):
        raise ValueError("fault interval lies outside the observation duration")
    for left, right in zip(intervals, intervals[1:]):
        if right.start_s < left.end_s:
            raise ValueError("fault intervals must not overlap for single-fault event scoring")
    raw_alarms = sorted(float(value) for value in detection_times_s)
    if any(value < 0 or value > observation_duration_s for value in raw_alarms):
        raise ValueError("detection time lies outside the observation interval")
    alarms: list[float] = []
    merged_raw_alarm_count = 0
    last_raw_alarm: float | None = None
    for alarm in raw_alarms:
        if last_raw_alarm is not None and alarm - last_raw_alarm <= alarm_merge_window_s:
            merged_raw_alarm_count += 1
            last_raw_alarm = alarm
            continue
        alarms.append(alarm)
        last_raw_alarm = alarm

    inferred_healthy_exposure_s = observation_duration_s - sum(
        interval.end_s - interval.start_s for interval in intervals
    )
    exposure_s = inferred_healthy_exposure_s if healthy_exposure_s is None else healthy_exposure_s
    if exposure_s < 0 or exposure_s > observation_duration_s:
        raise ValueError("healthy_exposure_s must lie within the observation duration")

    matched: set[int] = set()
    delays: list[float] = []
    false_alarm_count = 0
    duplicate_alarm_count = 0
    for alarm in alarms:
        containing = next(
            (index for index, interval in enumerate(intervals) if interval.start_s <= alarm < interval.end_s),
            None,
        )
        if containing is None:
            false_alarm_count += 1
        elif containing in matched:
            duplicate_alarm_count += 1
        else:
            matched.add(containing)
            delays.append(alarm - intervals[containing].start_s)

    fault_count = len(intervals)
    detected_count = len(matched)
    scored_alarm_count = detected_count + false_alarm_count + duplicate_alarm_count
    event_precision = detected_count / scored_alarm_count if scored_alarm_count else None
    return {
        "fault_event_count": fault_count,
        "detected_event_count": detected_count,
        "missed_event_count": fault_count - detected_count,
        "event_recall": detected_count / fault_count if fault_count else None,
        "event_precision": event_precision,
        # PR curves conventionally start at recall=0, precision=1 when a
        # threshold emits no alarms.  Keep operational event_precision=None
        # (undefined) while exposing the explicit plotting/integration point.
        "precision_recall_point": {
            "recall": detected_count / fault_count if fault_count else None,
            "precision": event_precision if event_precision is not None else 1.0,
            "no_alarm_precision_convention_applied": scored_alarm_count == 0,
        },
        "raw_alarm_count": len(raw_alarms),
        "merged_alarm_count": len(alarms),
        "merged_raw_alarm_count": merged_raw_alarm_count,
        "alarm_merge_window_s": alarm_merge_window_s,
        "false_alarm_count": false_alarm_count,
        "healthy_exposure_s": exposure_s,
        "false_alarms_per_hour": (
            false_alarm_count / (exposure_s / 3600.0) if exposure_s > 0 else None
        ),
        "duplicate_alarm_count": duplicate_alarm_count,
        "detection_delay_s": {
            "median": float(median(delays)) if delays else None,
            "p95_nearest_rank": _percentile_nearest_rank(delays, 0.95),
            "maximum": max(delays) if delays else None,
        },
    }


def precision_recall_auc(points: Sequence[tuple[float, float]]) -> float:
    """Integrate a frozen threshold sweep's ``(recall, precision)`` points.

    Every point must come from the complete event policy, including hysteresis
    and alarm merging.  The monotone precision envelope and step integration
    make the result independent of threshold iteration order.  Test AUPRC is a
    report-only metric and must never select a threshold.
    """

    if not points:
        raise ValueError("at least one precision-recall point is required")
    by_recall: dict[float, float] = {}
    for recall, precision in points:
        recall_f = float(recall)
        precision_f = float(precision)
        if not (0.0 <= recall_f <= 1.0 and 0.0 <= precision_f <= 1.0):
            raise ValueError("precision and recall must lie in [0, 1]")
        by_recall[recall_f] = max(precision_f, by_recall.get(recall_f, 0.0))
    ordered = sorted(by_recall.items())
    envelope = [0.0] * len(ordered)
    best = 0.0
    for index in range(len(ordered) - 1, -1, -1):
        best = max(best, ordered[index][1])
        envelope[index] = best
    area = 0.0
    previous_recall = 0.0
    for (recall, _), precision in zip(ordered, envelope, strict=True):
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def binary_aupr(truth: Sequence[bool], positive_scores: Sequence[float]) -> float | None:
    """Return threshold-free AUPR for a frozen binary score.

    Tied scores are evaluated as one threshold, so the result does not depend
    on row order.  ``None`` means the evaluated slice contains no positive
    item and therefore cannot estimate positive-class recall.
    """

    if len(truth) != len(positive_scores):
        raise ValueError("truth and positive_scores must have equal length")
    if not truth:
        raise ValueError("at least one scored item is required")
    scores = [float(value) for value in positive_scores]
    if any(not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("positive scores must lie in [0, 1]")
    positive_count = sum(bool(value) for value in truth)
    if positive_count == 0:
        return None
    # Evaluate tied score blocks in one descending pass.  This is exactly the
    # same threshold set as ``score >= threshold`` above, but avoids rescanning
    # every sample once per unique score (O(U*N)).
    ordered = sorted(zip(scores, map(bool, truth), strict=True), key=lambda item: item[0], reverse=True)
    points: list[tuple[float, float]] = [(0.0, 1.0)]
    tp = 0
    fp = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        end = index
        while end < len(ordered) and ordered[end][0] == threshold:
            if ordered[end][1]:
                tp += 1
            else:
                fp += 1
            end += 1
        points.append((tp / positive_count, tp / (tp + fp) if tp + fp else 1.0))
        index = end
    return precision_recall_auc(points)


def expected_calibration_error(
    correctness: Sequence[bool],
    confidences: Sequence[float],
    *,
    bin_edges: Sequence[float],
) -> dict[str, Any]:
    """ECE using caller-supplied, frozen probability bins.

    Bin edges must start at 0 and end at 1.  Bins are ``[left, right)`` except
    the final bin, which includes confidence 1.  Bins are never inferred from
    evaluated data.
    """

    if len(correctness) != len(confidences):
        raise ValueError("correctness and confidences must have equal length")
    if not correctness:
        raise ValueError("at least one calibrated item is required")
    edges = tuple(float(value) for value in bin_edges)
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("frozen bin edges must start at 0 and end at 1")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("frozen bin edges must be strictly increasing")
    values = [float(value) for value in confidences]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("confidences must lie in [0, 1]")

    bins: list[dict[str, float | int]] = []
    weighted_gap = 0.0
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        members = [
            item_index
            for item_index, value in enumerate(values)
            if left <= value < right or (index == len(edges) - 2 and value == right)
        ]
        count = len(members)
        accuracy = sum(bool(correctness[item]) for item in members) / count if count else 0.0
        confidence = sum(values[item] for item in members) / count if count else 0.0
        gap = abs(accuracy - confidence) if count else 0.0
        weighted_gap += count / len(values) * gap
        bins.append(
            {
                "left": left,
                "right": right,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": confidence,
                "absolute_gap": gap,
            }
        )
    return {"ece": weighted_gap, "bin_edges": list(edges), "bins": bins}


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Wilson score interval for a binomial promotion metric."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "confidence_level": confidence_level,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


__all__ = [
    "FaultInterval",
    "binary_aupr",
    "event_detection_metrics",
    "expected_calibration_error",
    "multiclass_metrics",
    "precision_recall_auc",
    "wilson_interval",
]
