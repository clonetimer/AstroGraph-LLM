"""Group-aware offline calibration-transfer and selective-risk audit.

The module is deliberately offline: predictions are frozen before labels/context are
joined.  Calibration fitting, development threshold selection and confirmatory
assessment must use disjoint independent groups.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from .selective_risk import FrozenPrediction, TruthRecord, aurc, evaluate_probabilities, risk_coverage_curve, verify_selective_threshold_receipt

_ALLOWED_PARTITIONS = {
    "MODEL_TRAIN",
    "CALIBRATION_FIT",
    "DEVELOPMENT_SELECTION",
    "CONFIRMATORY_EVALUATION",
    "EXTERNAL_VALIDATION",
}


@dataclass(frozen=True)
class CalibrationContext:
    sample_id: str
    group_id: str
    partition: str
    strata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id:
            raise ValueError("sample_id and group_id required")
        if self.partition not in _ALLOWED_PARTITIONS:
            raise ValueError(f"unsupported calibration partition: {self.partition}")
        if not self.strata or any(not str(k).strip() or not str(v).strip() for k, v in self.strata.items()):
            raise ValueError("strata must contain nonempty names and values")


def _canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _join(
    predictions: Sequence[FrozenPrediction],
    truths: Sequence[TruthRecord],
    contexts: Sequence[CalibrationContext],
    *,
    required_dimensions: Sequence[str],
) -> list[tuple[FrozenPrediction, TruthRecord, CalibrationContext]]:
    pred = {p.sample_id: p for p in predictions}
    truth = {t.sample_id: t for t in truths}
    context = {c.sample_id: c for c in contexts}
    if len(pred) != len(predictions) or len(truth) != len(truths) or len(context) != len(contexts):
        raise ValueError("duplicate sample_id in calibration campaign input")
    if not pred or set(pred) != set(truth) or set(pred) != set(context):
        raise ValueError("prediction/truth/context sample sets must be nonempty and identical")
    required = set(required_dimensions)
    if not required or any(not str(name).strip() for name in required):
        raise ValueError("required calibration-transfer dimensions must be nonempty")
    label_universe = set(next(iter(pred.values())).probabilities)
    if not label_universe:
        raise ValueError("frozen probability universe must be nonempty")
    rows = []
    group_partition: dict[str, str] = {}
    group_strata: dict[tuple[str, str], str] = {}
    for sample_id in sorted(pred):
        p, t, c = pred[sample_id], truth[sample_id], context[sample_id]
        if p.group_id != t.group_id or p.group_id != c.group_id:
            raise ValueError(f"group mismatch for sample {sample_id}")
        if set(p.probabilities) != label_universe:
            raise ValueError(f"sample {sample_id} probability class universe drift")
        if t.true_id not in label_universe:
            raise ValueError(f"sample {sample_id} truth label absent from frozen probability universe")
        missing = sorted(required - set(c.strata))
        if missing:
            raise ValueError(f"sample {sample_id} missing required strata: {missing}")
        old_partition = group_partition.setdefault(p.group_id, c.partition)
        if old_partition != c.partition:
            raise ValueError(f"independent group {p.group_id} crosses calibration partitions")
        for dim in required:
            key = (p.group_id, dim)
            old = group_strata.setdefault(key, str(c.strata[dim]))
            if old != str(c.strata[dim]):
                raise ValueError(f"independent group {p.group_id} crosses stratum {dim}")
        rows.append((p, t, c))
    return rows


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def group_bootstrap_mean_ci(
    values_by_group: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict:
    if resamples < 200:
        raise ValueError("group bootstrap requires at least 200 resamples")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5,1)")
    groups = sorted(values_by_group)
    if not groups:
        raise ValueError("no independent groups")
    vals = [float(values_by_group[g]) for g in groups]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("group metric must be finite")
    estimate = fmean(vals)
    if len(groups) == 1:
        return {
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "group_count": 1,
            "method": "INSUFFICIENT_GROUPS_FOR_INTERVAL",
            "resamples": 0,
            "seed": seed,
        }
    rng = random.Random(seed)
    boots = []
    n = len(vals)
    for _ in range(resamples):
        boots.append(fmean(vals[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "lower": _percentile(boots, alpha),
        "upper": _percentile(boots, 1.0 - alpha),
        "confidence": confidence,
        "group_count": n,
        "method": "INDEPENDENT_GROUP_PERCENTILE_BOOTSTRAP_DESCRIPTIVE",
        "resamples": resamples,
        "seed": seed,
    }


def _binary_ece(probs: list[float], outcomes: list[int], bins: int) -> float:
    if len(probs) != len(outcomes) or not probs:
        raise ValueError("binary ECE input mismatch")
    result = 0.0
    n = len(probs)
    for idx in range(bins):
        lo, hi = idx / bins, (idx + 1) / bins
        ids = [i for i, p in enumerate(probs) if p >= lo and (p < hi or idx == bins - 1)]
        if ids:
            result += len(ids) / n * abs(fmean(outcomes[i] for i in ids) - fmean(probs[i] for i in ids))
    return result


def _group_metrics(rows: list[tuple[FrozenPrediction, TruthRecord, CalibrationContext]], ece_bins: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[FrozenPrediction, TruthRecord]]] = {}
    for p, t, _ in rows:
        grouped.setdefault(p.group_id, []).append((p, t))
    result: dict[str, dict[str, float]] = {}
    for group_id, items in grouped.items():
        metrics = evaluate_probabilities([p for p, _ in items], [t for _, t in items], ece_bins=ece_bins)
        result[group_id] = {
            "accuracy": float(metrics["accuracy"]),
            "nll": float(metrics["nll"]),
            "brier": float(metrics["brier"]),
            "ece": float(metrics["ece"]),
        }
    return result


def _summarize_rows(
    rows: list[tuple[FrozenPrediction, TruthRecord, CalibrationContext]],
    *,
    ece_bins: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    if not rows:
        return {"status": "NO_ROWS", "sample_count": 0, "group_count": 0}
    predictions = [p for p, _, _ in rows]
    truths = [t for _, t, _ in rows]
    global_metrics = evaluate_probabilities(predictions, truths, ece_bins=ece_bins)
    per_group = _group_metrics(rows, ece_bins)
    intervals = {}
    for offset, metric in enumerate(("accuracy", "nll", "brier", "ece")):
        intervals[metric] = group_bootstrap_mean_ci(
            {g: values[metric] for g, values in per_group.items()},
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + offset,
        )
    labels = sorted({label for p in predictions for label in p.probabilities})
    classwise = {}
    for label_idx, label in enumerate(labels):
        by_group: dict[str, dict[str, float]] = {}
        grouped: dict[str, list[tuple[FrozenPrediction, TruthRecord]]] = {}
        for p, t, _ in rows:
            grouped.setdefault(p.group_id, []).append((p, t))
        positive_groups = 0
        positive_samples = 0
        for group_id, items in grouped.items():
            probs = [float(p.probabilities.get(label, 0.0)) for p, _ in items]
            ys = [1 if t.true_id == label else 0 for _, t in items]
            positive_samples += sum(ys)
            positive_groups += int(any(ys))
            by_group[group_id] = {
                "brier_ovr": fmean((prob - y) ** 2 for prob, y in zip(probs, ys)),
                "ece_ovr": _binary_ece(probs, ys, ece_bins),
            }
        classwise[label] = {
            "positive_sample_count": positive_samples,
            "positive_group_count": positive_groups,
            "group_macro_brier_ovr": group_bootstrap_mean_ci(
                {g: v["brier_ovr"] for g, v in by_group.items()},
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + 100 + 2 * label_idx,
            ),
            "group_macro_ece_ovr": group_bootstrap_mean_ci(
                {g: v["ece_ovr"] for g, v in by_group.items()},
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + 101 + 2 * label_idx,
            ),
        }
    curve = risk_coverage_curve(predictions, truths)
    return {
        "status": "EVALUATED",
        "sample_count": len(rows),
        "group_count": len(per_group),
        "sample_global": {
            "accuracy": global_metrics["accuracy"],
            "nll": global_metrics["nll"],
            "brier": global_metrics["brier"],
            "ece": global_metrics["ece"],
            "ece_bins": ece_bins,
            "aurc": aurc(curve),
        },
        "equal_group_macro_with_interval": intervals,
        "classwise_one_vs_rest": classwise,
    }


def evaluate_calibration_transfer_campaign(
    predictions: Sequence[FrozenPrediction],
    truths: Sequence[TruthRecord],
    contexts: Sequence[CalibrationContext],
    *,
    required_dimensions: Sequence[str],
    baseline_values: Mapping[str, str],
    required_partitions: Sequence[str] = ("CALIBRATION_FIT", "DEVELOPMENT_SELECTION", "CONFIRMATORY_EVALUATION"),
    ece_bins: int = 10,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260918,
) -> dict:
    if ece_bins < 2:
        raise ValueError("ece_bins must be >=2")
    rows = _join(predictions, truths, contexts, required_dimensions=required_dimensions)
    required = list(dict.fromkeys(required_dimensions))
    if set(baseline_values) != set(required):
        raise ValueError("baseline_values must exactly cover required_dimensions")
    partitions: dict[str, list[tuple[FrozenPrediction, TruthRecord, CalibrationContext]]] = {}
    for row in rows:
        partitions.setdefault(row[2].partition, []).append(row)
    required_partition_set = set(required_partitions)
    if not required_partition_set <= _ALLOWED_PARTITIONS:
        raise ValueError("required_partitions contains unsupported partition")
    missing_partitions = sorted(required_partition_set - set(partitions))
    if missing_partitions:
        raise ValueError(f"calibration campaign missing required partitions: {missing_partitions}")
    group_sets = {name: {p.group_id for p, _, _ in part_rows} for name, part_rows in partitions.items()}
    names = sorted(group_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise ValueError(f"calibration partitions share independent groups: {sorted(overlap)}")

    partition_reports = {
        name: _summarize_rows(
            part_rows,
            ece_bins=ece_bins,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + idx * 1000,
        )
        for idx, (name, part_rows) in enumerate(sorted(partitions.items()))
    }

    confirm_rows = partitions.get("CONFIRMATORY_EVALUATION", [])
    strata_report: dict[str, dict[str, dict]] = {}
    for dim_idx, dim in enumerate(required):
        values = sorted({str(c.strata[dim]) for _, _, c in confirm_rows})
        if baseline_values[dim] not in values and confirm_rows:
            raise ValueError(f"declared baseline {dim}={baseline_values[dim]} absent from confirmatory evaluation")
        strata_report[dim] = {}
        for value_idx, value in enumerate(values):
            subset = [row for row in confirm_rows if str(row[2].strata[dim]) == value]
            strata_report[dim][value] = _summarize_rows(
                subset,
                ece_bins=ece_bins,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed + 10000 + dim_idx * 1000 + value_idx * 100,
            )
        baseline_value = baseline_values[dim]
        baseline_groups = {p.group_id for p, _, c in confirm_rows if str(c.strata[dim]) == baseline_value}
        nonbaseline_groups = {p.group_id for p, _, c in confirm_rows if str(c.strata[dim]) != baseline_value}
        confounded_with = []
        for other_dim in required:
            if other_dim == dim or not baseline_groups or not nonbaseline_groups:
                continue
            baseline_other = {str(c.strata[other_dim]) for p, _, c in confirm_rows if p.group_id in baseline_groups}
            shifted_other = {str(c.strata[other_dim]) for p, _, c in confirm_rows if p.group_id in nonbaseline_groups}
            if baseline_other != shifted_other:
                confounded_with.append(other_dim)
        strata_report[dim]["_interpretation"] = {
            "declared_baseline_value": baseline_value,
            "single_dimension_isolation_status": (
                "CONFOUNDED_WITH_OTHER_DECLARED_DIMENSIONS" if confounded_with else "NO_DECLARED_DIMENSION_CONFOUNDING_DETECTED"
            ),
            "confounded_with_dimensions": confounded_with,
            "claim_boundary": "Per-dimension deltas are descriptive strata contrasts, not causal effects of one shift axis."
        }
        base = strata_report[dim].get(baseline_value)
        if base and base.get("status") == "EVALUATED":
            base_metrics = base["equal_group_macro_with_interval"]
            for value, report in strata_report[dim].items():
                if value == "_interpretation" or report.get("status") != "EVALUATED":
                    continue
                report["delta_vs_declared_baseline_equal_group_macro"] = {
                    metric: report["equal_group_macro_with_interval"][metric]["estimate"] - base_metrics[metric]["estimate"]
                    for metric in ("accuracy", "nll", "brier", "ece")
                }

    prediction_identity = _canonical_sha({"records": [
        {"sample_id": p.sample_id, "group_id": p.group_id, "probabilities": dict(sorted((str(k), float(v)) for k, v in p.probabilities.items())), "predicted_id": p.predicted_id, "prediction_sha256": p.prediction_sha256}
        for p, _, _ in rows
    ]})
    truth_identity = _canonical_sha({"records": [
        {"sample_id": t.sample_id, "group_id": t.group_id, "true_id": t.true_id}
        for _, t, _ in rows
    ]})
    context_identity = _canonical_sha({"records": [
        {"sample_id": c.sample_id, "group_id": c.group_id, "partition": c.partition, "strata": dict(sorted((str(k), str(v)) for k, v in c.strata.items()))}
        for _, _, c in rows
    ]})
    payload = {
        "schema_version": "astrograph-calibration-transfer-campaign-report.v1",
        "input_identity": {
            "prediction_set_sha256": prediction_identity,
            "truth_set_sha256": truth_identity,
            "context_set_sha256": context_identity,
        },
        "sample_count": len(rows),
        "independent_group_count": len({p.group_id for p, _, _ in rows}),
        "required_dimensions": required,
        "declared_baseline_values": dict(sorted((str(k), str(v)) for k, v in baseline_values.items())),
        "partitions": partition_reports,
        "confirmatory_strata": strata_report,
        "ece_estimator": f"EQUAL_WIDTH_{ece_bins}_BIN_ECE",
        "uncertainty_method": "INDEPENDENT_GROUP_PERCENTILE_BOOTSTRAP_DESCRIPTIVE",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "calibration_qualification_status": "EVALUATION_ONLY_NOT_CALIBRATION_RECEIPT",
        "claim_boundary": (
            "Offline calibration-transfer evaluation on frozen predictions and independent groups. "
            "ECE is estimator- and binning-dependent; bootstrap intervals are descriptive. This report does not "
            "fit a calibrator, authorize a probability threshold, prove OOD calibration, or grant release qualification."
        ),
    }
    payload["report_sha256"] = _canonical_sha(payload)
    return payload


def evaluate_fixed_threshold_transfer(
    predictions: Sequence[FrozenPrediction],
    truths: Sequence[TruthRecord],
    contexts: Sequence[CalibrationContext],
    *,
    threshold_receipt: Mapping[str, object],
    required_dimensions: Sequence[str],
    target_partition: str = "CONFIRMATORY_EVALUATION",
    selected_predictions_are_calibrated: bool = True,
) -> dict:
    rows = _join(predictions, truths, contexts, required_dimensions=required_dimensions)
    verified_receipt = verify_selective_threshold_receipt(threshold_receipt)
    threshold = verified_receipt.get("threshold")
    selected_on = set(str(x) for x in verified_receipt.get("selected_on_groups", []))
    if target_partition not in {"CONFIRMATORY_EVALUATION", "EXTERNAL_VALIDATION"}:
        raise ValueError("target_partition must be CONFIRMATORY_EVALUATION or EXTERNAL_VALIDATION")
    confirm_rows = [row for row in rows if row[2].partition == target_partition]
    development_rows = [row for row in rows if row[2].partition == "DEVELOPMENT_SELECTION"]
    confirm_groups = {p.group_id for p, _, _ in confirm_rows}
    development_groups = {p.group_id for p, _, _ in development_rows}
    if selected_on != development_groups:
        raise ValueError("selective threshold receipt selected_on_groups does not match admitted development groups")
    if selected_on & confirm_groups:
        raise ValueError("selective threshold was selected using confirmatory evaluation groups")
    if threshold is None:
        return {
            "schema_version": "astrograph-selective-threshold-transfer-report.v1",
            "status": "NO_FEASIBLE_DEVELOPMENT_THRESHOLD",
            "target_partition": target_partition,
            "target_group_count": len(confirm_groups),
            "confirmatory_group_count": (len(confirm_groups) if target_partition == "CONFIRMATORY_EVALUATION" else None),
            "selected_predictions_are_calibrated": selected_predictions_are_calibrated,
            "claim_boundary": "No threshold transfer claim is possible because development selection produced no threshold.",
        }
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")

    def summarize(subset: list[tuple[FrozenPrediction, TruthRecord, CalibrationContext]]) -> dict:
        if not subset:
            return {"sample_count": 0, "group_count": 0, "coverage": None, "selective_risk": None}
        kept = [(p, t) for p, t, _ in subset if max(p.probabilities.values()) >= threshold]
        risk = None if not kept else sum(p.predicted_id != t.true_id for p, t in kept) / len(kept)
        grouped: dict[str, list[tuple[FrozenPrediction, TruthRecord]]] = {}
        for p, t, _ in subset:
            grouped.setdefault(p.group_id, []).append((p, t))
        group_coverages = {}
        group_risks = {}
        for group_id, items in grouped.items():
            group_kept = [(p, t) for p, t in items if max(p.probabilities.values()) >= threshold]
            group_coverages[group_id] = len(group_kept) / len(items)
            if group_kept:
                group_risks[group_id] = sum(p.predicted_id != t.true_id for p, t in group_kept) / len(group_kept)
        return {
            "sample_count": len(subset),
            "group_count": len(grouped),
            "coverage": len(kept) / len(subset),
            "selective_risk": risk,
            "equal_group_macro_coverage": fmean(group_coverages.values()),
            "equal_group_macro_selective_risk_retained_groups": (None if not group_risks else fmean(group_risks.values())),
            "retained_group_count": len(group_risks),
        }

    by_stratum = {}
    for dim in required_dimensions:
        by_stratum[dim] = {}
        for value in sorted({str(c.strata[dim]) for _, _, c in confirm_rows}):
            by_stratum[dim][value] = summarize([row for row in confirm_rows if str(row[2].strata[dim]) == value])
    payload = {
        "schema_version": "astrograph-selective-threshold-transfer-report.v1",
        "status": "EVALUATED_FIXED_DEVELOPMENT_THRESHOLD",
        "threshold": threshold,
        "selected_on_groups": sorted(selected_on),
        "target_partition": target_partition,
        "target_group_count": len(confirm_groups),
        "confirmatory_group_count": (len(confirm_groups) if target_partition == "CONFIRMATORY_EVALUATION" else None),
        "selected_predictions_are_calibrated": selected_predictions_are_calibrated,
        "overall": summarize(confirm_rows),
        "by_stratum": by_stratum,
        "claim_boundary": (
            f"Applies one development-selected threshold to disjoint {target_partition} groups. It does not retune the "
            "threshold on target data and does not establish OOD guarantees or release qualification."
        ),
    }
    payload["report_sha256"] = _canonical_sha(payload)
    return payload


__all__ = [
    "CalibrationContext",
    "evaluate_calibration_transfer_campaign",
    "evaluate_fixed_threshold_transfer",
    "group_bootstrap_mean_ci",
]
