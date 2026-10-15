"""Group-aware calibration and selective-risk evaluation.

Runtime predictions are truth-free.  Ground truth is joined only by this offline
module after predictions are frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FrozenPrediction:
    sample_id: str
    group_id: str
    probabilities: Mapping[str, float]
    predicted_id: str
    prediction_sha256: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id:
            raise ValueError("sample_id and group_id required")
        if len(self.prediction_sha256) != 64:
            raise ValueError("prediction_sha256 required")
        values = list(self.probabilities.values())
        if not values or any((not math.isfinite(v) or v < 0 or v > 1) for v in values):
            raise ValueError("probabilities must be finite in [0,1]")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to one")
        if self.predicted_id not in self.probabilities:
            raise ValueError("predicted_id absent from probabilities")


@dataclass(frozen=True)
class TruthRecord:
    sample_id: str
    group_id: str
    true_id: str


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float


def _join(predictions: Sequence[FrozenPrediction], truths: Sequence[TruthRecord]) -> list[tuple[FrozenPrediction, TruthRecord]]:
    truth_by_id = {t.sample_id: t for t in truths}
    if len(truth_by_id) != len(truths):
        raise ValueError("duplicate truth sample_id")
    joined = []
    seen = set()
    for p in predictions:
        if p.sample_id in seen:
            raise ValueError("duplicate prediction sample_id")
        seen.add(p.sample_id)
        t = truth_by_id.get(p.sample_id)
        if t is None:
            raise ValueError(f"missing truth for {p.sample_id}")
        if t.group_id != p.group_id:
            raise ValueError("prediction/truth group mismatch")
        joined.append((p, t))
    if set(truth_by_id) != seen:
        raise ValueError("truth contains samples without frozen predictions")
    return joined


def evaluate_probabilities(predictions: Sequence[FrozenPrediction], truths: Sequence[TruthRecord], *, ece_bins: int = 10) -> dict:
    joined = _join(predictions, truths)
    if ece_bins < 2:
        raise ValueError("ece_bins must be >=2")
    labels = sorted({k for p, _ in joined for k in p.probabilities})
    nll = 0.0
    brier = 0.0
    correct = []
    confidences = []
    per_group: dict[str, list[tuple[float, bool]]] = {}
    for p, t in joined:
        prob_true = max(float(p.probabilities.get(t.true_id, 0.0)), 1e-15)
        nll -= math.log(prob_true)
        brier += sum((float(p.probabilities.get(label, 0.0)) - (1.0 if label == t.true_id else 0.0)) ** 2 for label in labels)
        hit = p.predicted_id == t.true_id
        conf = max(float(v) for v in p.probabilities.values())
        correct.append(hit); confidences.append(conf)
        per_group.setdefault(p.group_id, []).append((conf, hit))
    n = len(joined)
    ece = 0.0
    for idx in range(ece_bins):
        lo, hi = idx / ece_bins, (idx + 1) / ece_bins
        ids = [i for i,c in enumerate(confidences) if (c >= lo and (c < hi or idx == ece_bins-1))]
        if ids:
            acc = sum(correct[i] for i in ids) / len(ids)
            avg = sum(confidences[i] for i in ids) / len(ids)
            ece += len(ids) / n * abs(acc - avg)
    group_accuracy = {g: sum(hit for _,hit in rows)/len(rows) for g, rows in per_group.items()}
    return {
        "schema_version": "astrograph-probabilistic-evaluation.v1",
        "sample_count": n,
        "group_count": len(per_group),
        "accuracy": sum(correct)/n,
        "group_macro_accuracy": sum(group_accuracy.values())/len(group_accuracy),
        "worst_group_accuracy": min(group_accuracy.values()),
        "nll": nll/n,
        "brier": brier/n,
        "ece": ece,
        "group_accuracy": dict(sorted(group_accuracy.items())),
    }


def risk_coverage_curve(predictions: Sequence[FrozenPrediction], truths: Sequence[TruthRecord]) -> list[RiskCoveragePoint]:
    joined = _join(predictions, truths)
    # One confidence threshold cannot split samples that have the same confidence.
    # Aggregate tied confidence values before computing coverage/risk so threshold
    # selection cannot exploit correctness ordering inside an equal-score block.
    by_confidence: dict[float, list[bool]] = {}
    for p, t in joined:
        confidence = float(max(p.probabilities.values()))
        by_confidence.setdefault(confidence, []).append(p.predicted_id == t.true_id)
    points = []
    errors = 0
    retained = 0
    n = len(joined)
    for confidence in sorted(by_confidence, reverse=True):
        hits = by_confidence[confidence]
        retained += len(hits)
        errors += sum(not hit for hit in hits)
        points.append(RiskCoveragePoint(confidence, retained / n, errors / retained))
    return points


def aurc(points: Sequence[RiskCoveragePoint]) -> float:
    if not points:
        raise ValueError("empty risk-coverage curve")
    area = 0.0
    prev_cov = 0.0
    prev_risk = points[0].risk
    for p in points:
        area += (p.coverage - prev_cov) * (prev_risk + p.risk) / 2
        prev_cov, prev_risk = p.coverage, p.risk
    return area



def verify_selective_threshold_receipt(receipt: Mapping[str, object]) -> dict:
    payload = dict(receipt)
    if payload.get("schema_version") != "astrograph-selective-threshold-receipt.v1":
        raise ValueError("unsupported selective-threshold receipt schema")
    declared = str(payload.get("receipt_sha256", ""))
    if len(declared) != 64:
        raise ValueError("selective-threshold receipt_sha256 required")
    canonical = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    actual = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != declared:
        raise ValueError("selective-threshold receipt SHA256 mismatch")
    groups = payload.get("selected_on_groups")
    forbidden = payload.get("forbidden_groups")
    if not isinstance(groups, list) or not isinstance(forbidden, list):
        raise ValueError("selective-threshold receipt group sets required")
    if set(map(str, groups)) & set(map(str, forbidden)):
        raise ValueError("selective-threshold receipt has overlapping selected/forbidden groups")
    threshold = payload.get("threshold")
    if threshold is not None and (not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1):
        raise ValueError("selective-threshold receipt threshold must be finite in [0,1]")
    return payload

def select_confidence_threshold(
    predictions: Sequence[FrozenPrediction], truths: Sequence[TruthRecord], *,
    allowed_groups: set[str], forbidden_groups: set[str], max_selective_risk: float,
) -> dict:
    from astrograph.evaluation.data_exposure import guard_in_memory_development
    provenance = guard_in_memory_development(
        'SELECT', group_ids=[p.group_id for p in predictions]
    )
    if allowed_groups & forbidden_groups:
        raise ValueError("allowed_groups overlaps forbidden_groups")
    if set(p.group_id for p in predictions) != allowed_groups:
        raise ValueError("threshold selection must use exactly the predeclared development groups")
    if set(p.group_id for p in predictions) & forbidden_groups:
        raise ValueError("forbidden post-hoc groups cannot select threshold")
    if not 0 <= max_selective_risk <= 1:
        raise ValueError("max_selective_risk must be in [0,1]")
    curve = risk_coverage_curve(predictions, truths)
    feasible = [p for p in curve if p.risk <= max_selective_risk]
    chosen = max(feasible, key=lambda p: p.coverage) if feasible else None
    payload = {
        "schema_version":"astrograph-selective-threshold-receipt.v1",
        "selected_on_groups": sorted(allowed_groups),
        "forbidden_groups": sorted(forbidden_groups),
        "max_selective_risk": max_selective_risk,
        "threshold": None if chosen is None else chosen.threshold,
        "coverage": 0.0 if chosen is None else chosen.coverage,
        "risk": None if chosen is None else chosen.risk,
        "selection_provenance": provenance,
    }
    payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return payload


__all__ = ["FrozenPrediction","TruthRecord","RiskCoveragePoint","evaluate_probabilities","risk_coverage_curve","aurc","select_confidence_threshold","verify_selective_threshold_receipt"]
