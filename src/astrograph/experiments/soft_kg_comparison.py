"""Frozen development comparison for hard, soft and data-driven diagnosis branches.

The comparison is intentionally outside Runtime activation.  It evaluates
content-addressed observations after prediction generation, reports selective
metrics, and offers PSL/DeepProbLog/Semantic-Loss *style* mathematical controls
without claiming those external frameworks are installed or qualified.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from statistics import fmean
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ComparisonObservation(BaseModel):
    """Truth-joined evaluator row; never accepted by online Runtime APIs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str = Field(min_length=1)
    domain_id: Literal["ADCS_V5_2_2", "PHM_V6_0"]
    group_id: str = Field(min_length=1)
    truth_fault_id: str = Field(min_length=1)
    fault_universe: tuple[str, ...] = Field(min_length=2)
    data_probabilities: dict[str, float]
    hard_predicate_scores: dict[str, float]
    soft_predicate_scores: dict[str, float]
    data_available: bool = True
    executable_coverage: float = Field(ge=0.0, le=1.0)
    contextual_reliability: float = Field(ge=0.0, le=1.0)
    branch_conflict: float = Field(default=0.0, ge=0.0, le=1.0)
    source_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_scores(self) -> "ComparisonObservation":
        universe = set(self.fault_universe)
        if len(universe) != len(self.fault_universe):
            raise ValueError("fault_universe must be unique")
        if self.truth_fault_id not in universe:
            raise ValueError("truth fault is outside comparison universe")
        for name, values in (
            ("data_probabilities", self.data_probabilities),
            ("hard_predicate_scores", self.hard_predicate_scores),
            ("soft_predicate_scores", self.soft_predicate_scores),
        ):
            if set(values) - universe:
                raise ValueError(f"{name} contains faults outside the universe")
            if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
                raise ValueError(f"{name} values must be finite and nonnegative")
        if self.data_available and sum(self.data_probabilities.values()) <= 0.0:
            raise ValueError("available data branch must contain positive probability mass")
        return self


class MethodPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: str
    probabilities: dict[str, float]
    predicted_fault_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool


METHOD_IDS = (
    "A0_DATA_ONLY",
    "HARD_RULE_ONLY",
    "FIXED_BIRRF",
    "BRANCH_SUPPRESSION",
    "COVERAGE_SCALING",
    "CONTEXTUAL_GATED_BIRRF",
    "PSL_STYLE_SOFT_KG_CHALLENGER",
    "DEEPPROBLOG_STYLE_CONTROL",
    "SEMANTIC_LOSS_STYLE_CONTROL",
)


def _normalize(values: dict[str, float], universe: Sequence[str]) -> dict[str, float]:
    raw = {item: max(float(values.get(item, 0.0)), 0.0) for item in universe}
    total = sum(raw.values())
    if total <= 0.0:
        return {item: 1.0 / len(universe) for item in universe}
    return {item: value / total for item, value in raw.items()}


def _ranked(values: dict[str, float], universe: Sequence[str]) -> list[str]:
    return [
        item
        for item in sorted(universe, key=lambda key: (-float(values.get(key, 0.0)), key))
        if float(values.get(item, 0.0)) > 0.0
    ]


def _rrf(
    universe: Sequence[str],
    branches: Sequence[tuple[dict[str, float], float]],
    *,
    rank_constant: float = 60.0,
) -> dict[str, float]:
    scores = {item: 0.0 for item in universe}
    for values, weight in branches:
        if weight <= 0.0:
            continue
        for rank, fault_id in enumerate(_ranked(values, universe), start=1):
            scores[fault_id] += weight / (rank_constant + rank)
    return _normalize(scores, universe)


def _prediction(
    method_id: str,
    probabilities: dict[str, float],
    *,
    confidence_threshold: float,
    force_abstain: bool = False,
) -> MethodPrediction:
    ordered = sorted(probabilities, key=lambda key: (-probabilities[key], key))
    predicted = ordered[0]
    confidence = float(probabilities[predicted])
    abstained = force_abstain or confidence < confidence_threshold
    return MethodPrediction(
        method_id=method_id,
        probabilities=probabilities,
        predicted_fault_id=None if abstained else predicted,
        confidence=confidence,
        abstained=abstained,
    )


def predict_comparison_methods(
    observation: ComparisonObservation,
    *,
    confidence_threshold: float = 0.35,
) -> dict[str, MethodPrediction]:
    """Evaluate the frozen nine-method comparison without changing Runtime."""

    universe = observation.fault_universe
    data = _normalize(observation.data_probabilities, universe)
    hard_available = max(observation.hard_predicate_scores.values(), default=0.0) > 0.0
    soft_available = bool(observation.soft_predicate_scores)
    hard = _normalize(observation.hard_predicate_scores, universe)
    soft = _normalize(observation.soft_predicate_scores, universe)
    fixed = _rrf(universe, ((data, 1.0), (hard, 1.0), (hard, 1.0)))
    suppressed = (
        data
        if not hard_available or observation.branch_conflict > 0.0
        else fixed
    )
    coverage = _rrf(
        universe,
        (
            (data, 1.0),
            (hard, observation.executable_coverage),
            (hard, observation.executable_coverage),
        ),
    )
    contextual = _rrf(
        universe,
        (
            (data, 1.0),
            (hard, observation.contextual_reliability),
            (hard, observation.contextual_reliability),
        ),
    )
    psl = _normalize(
        {
            item: data[item] + 2.0 * float(observation.soft_predicate_scores.get(item, 0.0))
            for item in universe
        },
        universe,
    )
    deepproblog = _normalize(
        {
            item: (
                1.0
                - (1.0 - data[item])
                * (1.0 - float(observation.soft_predicate_scores[item]))
                if item in observation.soft_predicate_scores
                else data[item]
            )
            for item in universe
        },
        universe,
    )
    semantic_loss = _normalize(
        {
            item: (
                data[item]
                * math.exp(
                    -2.0 * (1.0 - float(observation.soft_predicate_scores[item]))
                )
                if item in observation.soft_predicate_scores
                else data[item]
            )
            for item in universe
        },
        universe,
    )
    return {
        "A0_DATA_ONLY": _prediction(
            "A0_DATA_ONLY",
            data,
            confidence_threshold=confidence_threshold,
            force_abstain=not observation.data_available,
        ),
        "HARD_RULE_ONLY": _prediction(
            "HARD_RULE_ONLY",
            hard,
            confidence_threshold=confidence_threshold,
            force_abstain=not hard_available,
        ),
        "FIXED_BIRRF": _prediction(
            "FIXED_BIRRF", fixed, confidence_threshold=confidence_threshold
        ),
        "BRANCH_SUPPRESSION": _prediction(
            "BRANCH_SUPPRESSION", suppressed, confidence_threshold=confidence_threshold
        ),
        "COVERAGE_SCALING": _prediction(
            "COVERAGE_SCALING", coverage, confidence_threshold=confidence_threshold
        ),
        "CONTEXTUAL_GATED_BIRRF": _prediction(
            "CONTEXTUAL_GATED_BIRRF", contextual, confidence_threshold=confidence_threshold
        ),
        "PSL_STYLE_SOFT_KG_CHALLENGER": _prediction(
            "PSL_STYLE_SOFT_KG_CHALLENGER",
            psl,
            confidence_threshold=confidence_threshold,
            force_abstain=not soft_available and not observation.data_available,
        ),
        "DEEPPROBLOG_STYLE_CONTROL": _prediction(
            "DEEPPROBLOG_STYLE_CONTROL",
            deepproblog,
            confidence_threshold=confidence_threshold,
        ),
        "SEMANTIC_LOSS_STYLE_CONTROL": _prediction(
            "SEMANTIC_LOSS_STYLE_CONTROL",
            semantic_loss,
            confidence_threshold=confidence_threshold,
        ),
    }


def _ece(rows: Sequence[tuple[float, bool]], bins: int = 10) -> float:
    total = len(rows)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            row
            for row in rows
            if row[0] >= low and (row[0] < high or index == bins - 1)
        ]
        if not bucket:
            continue
        accuracy = sum(hit for _, hit in bucket) / len(bucket)
        confidence = fmean(item for item, _ in bucket)
        value += len(bucket) / total * abs(accuracy - confidence)
    return value


def _aurc(rows: Sequence[tuple[float, bool]]) -> tuple[float, list[dict[str, float]]]:
    ranked = sorted(rows, key=lambda item: item[0], reverse=True)
    errors = 0
    previous_coverage = previous_risk = area = 0.0
    curve: list[dict[str, float]] = []
    for index, (confidence, hit) in enumerate(ranked, start=1):
        errors += int(not hit)
        coverage = index / len(ranked)
        risk = errors / index
        area += (coverage - previous_coverage) * (risk + previous_risk) / 2.0
        curve.append({"coverage": coverage, "risk": risk, "confidence": confidence})
        previous_coverage, previous_risk = coverage, risk
    return area, curve


def _binary_correlation(left: Sequence[bool], right: Sequence[bool]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = [float(item) for item in left]
    y = [float(item) for item in right]
    mx, my = fmean(x), fmean(y)
    dx = math.sqrt(sum((item - mx) ** 2 for item in x))
    dy = math.sqrt(sum((item - my) ** 2 for item in y))
    if dx == 0.0 or dy == 0.0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (dx * dy)


def evaluate_comparison(
    observations: Sequence[ComparisonObservation],
    *,
    confidence_threshold: float = 0.35,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("comparison requires observations")
    predictions = {
        item.observation_id: predict_comparison_methods(
            item, confidence_threshold=confidence_threshold
        )
        for item in observations
    }
    truth_labels = sorted({item.truth_fault_id for item in observations})
    metrics: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        rows = [(item, predictions[item.observation_id][method_id]) for item in observations]
        recalls: list[float] = []
        f1s: list[float] = []
        for label in truth_labels:
            tp = sum(
                item.truth_fault_id == label and prediction.predicted_fault_id == label
                for item, prediction in rows
            )
            fn = sum(
                item.truth_fault_id == label and prediction.predicted_fault_id != label
                for item, prediction in rows
            )
            fp = sum(
                item.truth_fault_id != label and prediction.predicted_fault_id == label
                for item, prediction in rows
            )
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            recalls.append(recall)
            f1s.append(
                0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
            )
        correct = [prediction.predicted_fault_id == item.truth_fault_id for item, prediction in rows]
        accepted = [not prediction.abstained for _, prediction in rows]
        confidence_hits = [(prediction.confidence, hit) for (_, prediction), hit in zip(rows, correct, strict=True)]
        nll = fmean(
            -math.log(max(prediction.probabilities[item.truth_fault_id], 1e-15))
            for item, prediction in rows
        )
        brier = fmean(
            sum(
                (
                    prediction.probabilities[label]
                    - (1.0 if label == item.truth_fault_id else 0.0)
                ) ** 2
                for label in item.fault_universe
            )
            for item, prediction in rows
        )
        aurc, curve = _aurc(confidence_hits)
        by_group: dict[str, list[bool]] = defaultdict(list)
        for (item, _), hit in zip(rows, correct, strict=True):
            by_group[item.group_id].append(hit)
        group_accuracy = {
            group: sum(values) / len(values) for group, values in sorted(by_group.items())
        }
        wrong_count = sum(not item for item in correct)
        metrics[method_id] = {
            "observation_count": len(rows),
            "macro_f1": fmean(f1s),
            "balanced_accuracy": fmean(recalls),
            "accuracy": sum(correct) / len(correct),
            "nll": nll,
            "brier": brier,
            "ece": _ece(confidence_hits),
            "coverage": sum(accepted) / len(accepted),
            "selective_risk": (
                1.0
                if not any(accepted)
                else 1.0
                - sum(hit and take for hit, take in zip(correct, accepted, strict=True))
                / sum(accepted)
            ),
            "aurc": aurc,
            "worst_group_accuracy": min(group_accuracy.values()),
            "group_accuracy": group_accuracy,
            "failure_to_abstain": (
                0.0
                if wrong_count == 0
                else sum((not hit) and take for hit, take in zip(correct, accepted, strict=True))
                / wrong_count
            ),
            "risk_coverage_curve": curve,
        }

    data_errors: list[bool] = []
    hard_errors: list[bool] = []
    for item in observations:
        hard_available = max(item.hard_predicate_scores.values(), default=0.0) > 0.0
        if not hard_available:
            continue
        row = predictions[item.observation_id]
        data_errors.append(row["A0_DATA_ONLY"].predicted_fault_id != item.truth_fault_id)
        hard_errors.append(row["HARD_RULE_ONLY"].predicted_fault_id != item.truth_fault_id)
    branch_correlation = _binary_correlation(data_errors, hard_errors)
    payload = {
        "schema_version": "astrograph-soft-kg-fixed-comparison.v1",
        "method_ids": list(METHOD_IDS),
        "confidence_threshold": confidence_threshold,
        "metrics": metrics,
        "branch_error_correlation": {
            "data_vs_hard_phi": branch_correlation,
            "joint_observation_count": len(data_errors),
            "interpretation": "Evaluator-only binary error correlation; null means one branch error indicator was constant.",
        },
        "method_boundaries": {
            "PSL_STYLE_SOFT_KG_CHALLENGER": "Convex data-plus-soft-predicate score; not an external PSL runtime.",
            "DEEPPROBLOG_STYLE_CONTROL": "Noisy-OR probabilistic-logic control; not the DeepProbLog package.",
            "SEMANTIC_LOSS_STYLE_CONTROL": "Exponential semantic-constraint reweighting; not model retraining.",
        },
        "claim_boundary": (
            "Development comparison only. Soft truth is not a fault posterior and no method "
            "is activated in Runtime, Dataset Profiles, KG, BiRRF defaults, or Active KG."
        ),
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "ComparisonObservation",
    "METHOD_IDS",
    "MethodPrediction",
    "canonical_sha256",
    "evaluate_comparison",
    "predict_comparison_methods",
]
