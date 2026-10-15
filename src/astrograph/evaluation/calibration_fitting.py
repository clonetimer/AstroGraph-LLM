"""Governed post-hoc probability calibration fitting and external validation.

R19 deliberately separates four independent-group partitions:
CALIBRATION_FIT -> DEVELOPMENT_SELECTION -> CONFIRMATORY_EVALUATION -> EXTERNAL_VALIDATION.
Only CALIBRATION_FIT may fit the calibrator, only DEVELOPMENT_SELECTION may select
an abstention threshold, and neither confirmatory nor external groups may retune it.
"""
from __future__ import annotations

from .external_dataset_contract import TASK_FIELDS, task_identity, require_same_task

import hashlib
import json
import math
from statistics import fmean
from typing import Mapping, Sequence

from astrograph.data_driven.contracts import ProbabilityCalibrationQualificationReceipt

from .calibration_campaign import (
    CalibrationContext,
    evaluate_calibration_transfer_campaign,
    evaluate_fixed_threshold_transfer,
)
from .selective_risk import FrozenPrediction, TruthRecord, select_confidence_threshold


def _canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()



def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA256")


def _indexed_rows(
    predictions: Sequence[FrozenPrediction],
    truths: Sequence[TruthRecord],
    contexts: Sequence[CalibrationContext],
) -> list[tuple[FrozenPrediction, TruthRecord, CalibrationContext]]:
    pred = {p.sample_id: p for p in predictions}
    truth = {t.sample_id: t for t in truths}
    context = {c.sample_id: c for c in contexts}
    if len(pred) != len(predictions) or len(truth) != len(truths) or len(context) != len(contexts):
        raise ValueError("duplicate sample_id in calibration fitting input")
    if not pred or set(pred) != set(truth) or set(pred) != set(context):
        raise ValueError("prediction/truth/context sample sets must be nonempty and identical")
    rows = []
    class_universe = set(next(iter(pred.values())).probabilities)
    for sample_id in sorted(pred):
        p, t, c = pred[sample_id], truth[sample_id], context[sample_id]
        if p.group_id != t.group_id or p.group_id != c.group_id:
            raise ValueError(f"group mismatch for sample {sample_id}")
        if set(p.probabilities) != class_universe:
            raise ValueError(f"sample {sample_id} probability class universe drift")
        if t.true_id not in class_universe:
            raise ValueError(f"sample {sample_id} truth label absent from probability universe")
        rows.append((p, t, c))
    return rows


def temperature_scale_probability_map(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    labels = sorted(probabilities)
    if not labels:
        raise ValueError("probability map cannot be empty")
    logs = [math.log(max(float(probabilities[label]), 1e-15)) / temperature for label in labels]
    pivot = max(logs)
    exps = [math.exp(value - pivot) for value in logs]
    denom = sum(exps)
    return {label: value / denom for label, value in zip(labels, exps)}


def _equal_group_macro_nll(
    rows: Sequence[tuple[FrozenPrediction, TruthRecord, CalibrationContext]],
    temperature: float,
) -> float:
    grouped: dict[str, list[float]] = {}
    for p, t, _ in rows:
        calibrated = temperature_scale_probability_map(p.probabilities, temperature)
        grouped.setdefault(p.group_id, []).append(-math.log(max(calibrated[t.true_id], 1e-15)))
    if not grouped:
        raise ValueError("temperature-scaling objective has no independent groups")
    return fmean(fmean(values) for values in grouped.values())


def _fit_temperature(
    rows: Sequence[tuple[FrozenPrediction, TruthRecord, CalibrationContext]],
    *,
    lower: float,
    upper: float,
    coarse_points: int,
    refine_iterations: int,
) -> tuple[float, float]:
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('OPTIMIZE', group_ids=[p.group_id for p, t, c in rows])
    if not (0.0 < lower < upper) or coarse_points < 5 or refine_iterations < 10:
        raise ValueError("invalid deterministic temperature optimizer configuration")
    log_lo, log_hi = math.log(lower), math.log(upper)
    grid = [math.exp(log_lo + (log_hi - log_lo) * i / (coarse_points - 1)) for i in range(coarse_points)]
    objectives = [_equal_group_macro_nll(rows, temp) for temp in grid]
    best_idx = min(range(len(grid)), key=lambda idx: (objectives[idx], grid[idx]))
    left_idx = max(0, best_idx - 1)
    right_idx = min(len(grid) - 1, best_idx + 1)
    a, b = math.log(grid[left_idx]), math.log(grid[right_idx])
    if a == b:
        return grid[best_idx], objectives[best_idx]
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = _equal_group_macro_nll(rows, math.exp(c))
    fd = _equal_group_macro_nll(rows, math.exp(d))
    for _ in range(refine_iterations):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = _equal_group_macro_nll(rows, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = _equal_group_macro_nll(rows, math.exp(d))
    candidates = [(grid[best_idx], objectives[best_idx]), (math.exp(c), fc), (math.exp(d), fd)]
    if lower <= 1.0 <= upper:
        candidates.append((1.0, _equal_group_macro_nll(rows, 1.0)))
    return min(candidates, key=lambda item: (item[1], item[0]))


def fit_temperature_scaler_receipt(
    predictions: Sequence[FrozenPrediction],
    truths: Sequence[TruthRecord],
    contexts: Sequence[CalibrationContext],
    *,
    calibration_id: str,
    model_id: str,
    model_version: str,
    calibration_dataset_id: str,
    calibration_dataset_sha256: str,
    model_artifact_sha256: str,
    minimum_fit_groups: int = 8,
    minimum_positive_groups_per_class: int = 2,
    temperature_lower: float = 0.05,
    temperature_upper: float = 20.0,
    coarse_points: int = 121,
    refine_iterations: int = 80,
    evaluation_task_identity: Mapping[str, str] | None = None,
) -> dict:
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('CALIBRATE', group_ids=[c.group_id for c in contexts if c.partition == 'CALIBRATION_FIT'])
    _require_sha256("calibration_dataset_sha256", calibration_dataset_sha256)
    _require_sha256("model_artifact_sha256", model_artifact_sha256)
    rows = _indexed_rows(predictions, truths, contexts)
    fit_rows = [row for row in rows if row[2].partition == "CALIBRATION_FIT"]
    fit_groups = sorted({p.group_id for p, _, _ in fit_rows})
    if len(fit_groups) < minimum_fit_groups:
        raise ValueError(f"insufficient CALIBRATION_FIT independent groups: {len(fit_groups)}<{minimum_fit_groups}")
    labels = sorted(next(iter(fit_rows))[0].probabilities) if fit_rows else []
    positive_groups_by_class = {
        label: len({p.group_id for p, t, _ in fit_rows if t.true_id == label}) for label in labels
    }
    insufficient_classes = {label: count for label, count in positive_groups_by_class.items() if count < minimum_positive_groups_per_class}
    if insufficient_classes:
        raise ValueError(f"insufficient CALIBRATION_FIT positive independent groups per class: {insufficient_classes}")
    if any(row[2].partition != "CALIBRATION_FIT" for row in fit_rows):  # pragma: no cover - structural guard
        raise ValueError("calibrator fit consumed non-fit partition")
    temperature, after = _fit_temperature(
        fit_rows,
        lower=temperature_lower,
        upper=temperature_upper,
        coarse_points=coarse_points,
        refine_iterations=refine_iterations,
    )
    before = _equal_group_macro_nll(fit_rows, 1.0)
    payload = {
        "schema_version": "astrograph-probability-calibrator-fit-receipt.v1",
        "calibration_id": calibration_id,
        "model_id": model_id,
        "model_version": model_version,
        "calibration_dataset_id": calibration_dataset_id,
        "calibration_dataset_sha256": calibration_dataset_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "method": "TEMPERATURE_SCALING_ON_LOG_PROBABILITIES",
        "input_semantics": "UNCALIBRATED_CLASS_PROBABILITY",
        "output_semantics": "CALIBRATED_CLASS_PROBABILITY_CANDIDATE",
        "fit_partition": "CALIBRATION_FIT",
        "fit_objective": "EQUAL_INDEPENDENT_GROUP_MACRO_NLL",
        "fit_sample_count": len(fit_rows),
        "fit_group_count": len(fit_groups),
        "fit_group_ids": fit_groups,
        "fit_positive_group_count_by_class": positive_groups_by_class,
        "minimum_positive_groups_per_class": minimum_positive_groups_per_class,
        "fit_group_set_sha256": _canonical_sha({"group_ids": fit_groups}),
        "class_universe": labels,
        "class_universe_sha256": _canonical_sha({"labels": labels}),
        "fit_prediction_set_sha256": _canonical_sha({"predictions": [
            {"sample_id": p.sample_id, "group_id": p.group_id, "probabilities": dict(sorted((str(k), float(v)) for k, v in p.probabilities.items())), "prediction_sha256": p.prediction_sha256}
            for p, _, _ in fit_rows
        ]}),
        "fit_truth_set_sha256": _canonical_sha({"truths": [
            {"sample_id": t.sample_id, "group_id": t.group_id, "true_id": t.true_id} for _, t, _ in fit_rows
        ]}),
        "fit_context_set_sha256": _canonical_sha({"contexts": [
            {"sample_id": c.sample_id, "group_id": c.group_id, "partition": c.partition, "strata": dict(sorted((str(k), str(v)) for k, v in c.strata.items()))}
            for _, _, c in fit_rows
        ]}),
        "temperature": temperature,
        "temperature_bounds": [temperature_lower, temperature_upper],
        "optimizer": {
            "method": "LOG_SPACE_COARSE_GRID_PLUS_GOLDEN_SECTION",
            "coarse_points": coarse_points,
            "refine_iterations": refine_iterations,
            "deterministic": True,
        },
        "fit_equal_group_macro_nll_before": before,
        "fit_equal_group_macro_nll_after": after,
        "fit_improved_nll": after <= before + 1e-12,
        "claim_boundary": (
            "Fit receipt binds one deterministic post-hoc temperature scaler to CALIBRATION_FIT independent groups only. "
            "Fit improvement does not establish held-out calibration, shift transfer, selective-risk safety, or release qualification."
        ),
    }
    if evaluation_task_identity is not None:
        payload.update(task_identity(dict(evaluation_task_identity)))
    payload["receipt_sha256"] = _canonical_sha(payload)
    return payload


def verify_temperature_scaler_fit_receipt(receipt: Mapping[str, object]) -> dict:
    payload = dict(receipt)
    if any(k in payload for k in TASK_FIELDS):
        task_identity(payload)
    if payload.get("schema_version") != "astrograph-probability-calibrator-fit-receipt.v1":
        raise ValueError("unsupported calibrator-fit receipt schema")
    declared = str(payload.get("receipt_sha256", ""))
    if len(declared) != 64:
        raise ValueError("calibrator-fit receipt_sha256 required")
    canonical = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    if _canonical_sha(canonical) != declared:
        raise ValueError("calibrator-fit receipt SHA256 mismatch")
    if payload.get("method") != "TEMPERATURE_SCALING_ON_LOG_PROBABILITIES":
        raise ValueError("unsupported calibrator-fit method")
    if payload.get("fit_partition") != "CALIBRATION_FIT":
        raise ValueError("calibrator fit must be bound to CALIBRATION_FIT")
    temperature = payload.get("temperature")
    if not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("calibrator-fit temperature must be finite and positive")
    groups = payload.get("fit_group_ids")
    if not isinstance(groups, list) or not groups or len(set(map(str, groups))) != len(groups):
        raise ValueError("calibrator-fit receipt requires unique fit_group_ids")
    if int(payload.get("fit_group_count", 0)) != len(groups):
        raise ValueError("calibrator-fit group count mismatch")
    return payload


def apply_temperature_scaler_receipt(
    predictions: Sequence[FrozenPrediction],
    receipt: Mapping[str, object],
    *, evaluation_task_identity: Mapping[str, str] | None = None,
) -> list[FrozenPrediction]:
    verified = verify_temperature_scaler_fit_receipt(receipt)
    if evaluation_task_identity is not None or any(k in verified for k in TASK_FIELDS):
        require_same_task(verified, dict(evaluation_task_identity or {}))
    temperature = float(verified["temperature"])
    expected_labels = list(map(str, verified["class_universe"]))
    output = []
    for p in predictions:
        if sorted(map(str, p.probabilities)) != sorted(expected_labels):
            raise ValueError(f"prediction {p.sample_id} class universe does not match calibrator fit")
        probabilities = temperature_scale_probability_map(p.probabilities, temperature)
        maximum = max(probabilities.values())
        if probabilities[p.predicted_id] < maximum - 1e-12:
            raise ValueError(f"temperature scaling changed argmax for {p.sample_id}")
        predicted_id = p.predicted_id
        prediction_sha = _canonical_sha({
            "parent_prediction_sha256": p.prediction_sha256,
            "calibrator_fit_receipt_sha256": verified["receipt_sha256"],
            "probabilities": dict(sorted(probabilities.items())),
        })
        output.append(FrozenPrediction(p.sample_id, p.group_id, probabilities, predicted_id, prediction_sha))
    return output


def _metric_estimate(report: Mapping[str, object], partition: str, metric: str) -> float:
    return float(report["partitions"][partition]["equal_group_macro_with_interval"][metric]["estimate"])


def _classwise_estimates(report: Mapping[str, object], partition: str, label: str) -> tuple[float, float]:
    record = report["partitions"][partition]["classwise_one_vs_rest"][label]
    return float(record["group_macro_brier_ovr"]["estimate"]), float(record["group_macro_ece_ovr"]["estimate"])


def qualify_temperature_calibration(
    predictions: Sequence[FrozenPrediction],
    truths: Sequence[TruthRecord],
    contexts: Sequence[CalibrationContext],
    *,
    fit_receipt: Mapping[str, object],
    required_dimensions: Sequence[str],
    baseline_values: Mapping[str, str],
    model_id: str,
    model_version: str,
    external_validation_dataset_id: str,
    external_validation_dataset_sha256: str,
    model_artifact_sha256: str,
    design_sha256: str,
    admission_bundle_sha256: str,
    minimum_development_groups: int,
    minimum_confirmatory_groups: int,
    minimum_external_groups: int,
    minimum_positive_groups_per_class: int,
    ece_bins: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    max_development_selective_risk: float,
    max_confirmatory_nll_increase: float,
    max_confirmatory_brier_increase: float,
    max_confirmatory_ece_increase: float,
    max_external_nll_increase: float,
    max_external_brier_increase: float,
    max_external_ece_increase: float,
    max_classwise_brier_increase: float,
    max_classwise_ece_increase: float,
    max_external_selective_risk_increase: float,
    max_external_coverage_decrease: float,
    evaluation_task_identity: Mapping[str, str] | None = None,
) -> tuple[dict, ProbabilityCalibrationQualificationReceipt]:
    _require_sha256("external_validation_dataset_sha256", external_validation_dataset_sha256)
    _require_sha256("model_artifact_sha256", model_artifact_sha256)
    _require_sha256("design_sha256", design_sha256)
    _require_sha256("admission_bundle_sha256", admission_bundle_sha256)
    verified_fit = verify_temperature_scaler_fit_receipt(fit_receipt)
    bound_task = {}
    if evaluation_task_identity is not None or any(k in verified_fit for k in TASK_FIELDS):
        require_same_task(verified_fit, dict(evaluation_task_identity or {}))
        bound_task = task_identity(dict(evaluation_task_identity))
    if (verified_fit["model_id"], verified_fit["model_version"]) != (model_id, model_version):
        raise ValueError("fit receipt model identity mismatch")
    if str(verified_fit.get("model_artifact_sha256")) != model_artifact_sha256:
        raise ValueError("fit receipt model artifact identity mismatch")
    rows = _indexed_rows(predictions, truths, contexts)
    partition_groups: dict[str, set[str]] = {}
    for p, _, c in rows:
        partition_groups.setdefault(c.partition, set()).add(p.group_id)
    required_partitions = {"CALIBRATION_FIT", "DEVELOPMENT_SELECTION", "CONFIRMATORY_EVALUATION", "EXTERNAL_VALIDATION"}
    missing = sorted(required_partitions - set(partition_groups))
    if missing:
        raise ValueError(f"R19 calibration qualification missing required partitions: {missing}")
    names = sorted(required_partitions)
    for idx, left in enumerate(names):
        for right in names[idx + 1:]:
            overlap = partition_groups[left] & partition_groups[right]
            if overlap:
                raise ValueError(f"R19 calibration partitions share independent groups: {sorted(overlap)}")
    if set(map(str, verified_fit["fit_group_ids"])) != partition_groups["CALIBRATION_FIT"]:
        raise ValueError("calibrator fit receipt groups do not equal admitted CALIBRATION_FIT groups")
    fit_rows = [(p, t, c) for p, t, c in rows if c.partition == "CALIBRATION_FIT"]
    expected_fit_prediction_sha = _canonical_sha({"predictions": [
        {"sample_id": p.sample_id, "group_id": p.group_id, "probabilities": dict(sorted((str(k), float(v)) for k, v in p.probabilities.items())), "prediction_sha256": p.prediction_sha256}
        for p, _, _ in fit_rows
    ]})
    expected_fit_truth_sha = _canonical_sha({"truths": [
        {"sample_id": t.sample_id, "group_id": t.group_id, "true_id": t.true_id} for _, t, _ in fit_rows
    ]})
    expected_fit_context_sha = _canonical_sha({"contexts": [
        {"sample_id": c.sample_id, "group_id": c.group_id, "partition": c.partition, "strata": dict(sorted((str(k), str(v)) for k, v in c.strata.items()))}
        for _, _, c in fit_rows
    ]})
    expected_labels = sorted(next(iter(fit_rows))[0].probabilities)
    checks = {
        "fit_prediction_set_sha256": expected_fit_prediction_sha,
        "fit_truth_set_sha256": expected_fit_truth_sha,
        "fit_context_set_sha256": expected_fit_context_sha,
        "class_universe_sha256": _canonical_sha({"labels": expected_labels}),
    }
    for field, expected in checks.items():
        if str(verified_fit.get(field)) != expected:
            raise ValueError(f"calibrator fit receipt content identity mismatch: {field}")
    if len(partition_groups["DEVELOPMENT_SELECTION"]) < minimum_development_groups:
        raise ValueError("insufficient development independent groups for calibration qualification")
    if len(partition_groups["CONFIRMATORY_EVALUATION"]) < minimum_confirmatory_groups:
        raise ValueError("insufficient confirmatory independent groups for calibration qualification")
    if len(partition_groups["EXTERNAL_VALIDATION"]) < minimum_external_groups:
        raise ValueError("insufficient external-validation independent groups for calibration qualification")
    if external_validation_dataset_id == str(verified_fit["calibration_dataset_id"]):
        raise ValueError("external-validation dataset must be distinct from calibration-fit dataset")
    labels = sorted(next(iter(predictions)).probabilities)
    class_coverage_admission = {}
    for partition in ("CONFIRMATORY_EVALUATION", "EXTERNAL_VALIDATION"):
        counts = {
            label: len({p.group_id for p, t, c in rows if c.partition == partition and t.true_id == label})
            for label in labels
        }
        class_coverage_admission[partition] = counts
        insufficient = {label: count for label, count in counts.items() if count < minimum_positive_groups_per_class}
        if insufficient:
            raise ValueError(f"insufficient {partition} positive independent groups per class: {insufficient}")

    calibrated = apply_temperature_scaler_receipt(predictions, verified_fit, evaluation_task_identity=evaluation_task_identity)
    required_partition_list = ["CALIBRATION_FIT", "DEVELOPMENT_SELECTION", "CONFIRMATORY_EVALUATION", "EXTERNAL_VALIDATION"]
    raw_report = evaluate_calibration_transfer_campaign(
        predictions, truths, contexts,
        required_dimensions=required_dimensions,
        baseline_values=baseline_values,
        required_partitions=required_partition_list,
        ece_bins=ece_bins,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    calibrated_report = evaluate_calibration_transfer_campaign(
        calibrated, truths, contexts,
        required_dimensions=required_dimensions,
        baseline_values=baseline_values,
        required_partitions=required_partition_list,
        ece_bins=ece_bins,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 50000,
    )

    dev_groups = partition_groups["DEVELOPMENT_SELECTION"]
    forbidden = partition_groups["CALIBRATION_FIT"] | partition_groups["CONFIRMATORY_EVALUATION"] | partition_groups["EXTERNAL_VALIDATION"]
    dev_predictions = [p for p, _, c in _indexed_rows(calibrated, truths, contexts) if c.partition == "DEVELOPMENT_SELECTION"]
    dev_truths = [t for _, t, c in _indexed_rows(calibrated, truths, contexts) if c.partition == "DEVELOPMENT_SELECTION"]
    threshold_receipt = select_confidence_threshold(
        dev_predictions,
        dev_truths,
        allowed_groups=set(dev_groups),
        forbidden_groups=set(forbidden),
        max_selective_risk=max_development_selective_risk,
    )
    raw_dev_predictions = [p for p, _, c in rows if c.partition == "DEVELOPMENT_SELECTION"]
    raw_dev_truths = [t for _, t, c in rows if c.partition == "DEVELOPMENT_SELECTION"]
    raw_threshold_receipt = select_confidence_threshold(
        raw_dev_predictions,
        raw_dev_truths,
        allowed_groups=set(dev_groups),
        forbidden_groups=set(forbidden),
        max_selective_risk=max_development_selective_risk,
    )
    confirm_threshold = evaluate_fixed_threshold_transfer(
        calibrated, truths, contexts,
        threshold_receipt=threshold_receipt,
        required_dimensions=required_dimensions,
        target_partition="CONFIRMATORY_EVALUATION",
    )
    external_threshold = evaluate_fixed_threshold_transfer(
        calibrated, truths, contexts,
        threshold_receipt=threshold_receipt,
        required_dimensions=required_dimensions,
        target_partition="EXTERNAL_VALIDATION",
    )
    raw_external_threshold = evaluate_fixed_threshold_transfer(
        predictions, truths, contexts,
        threshold_receipt=raw_threshold_receipt,
        required_dimensions=required_dimensions,
        target_partition="EXTERNAL_VALIDATION",
        selected_predictions_are_calibrated=False,
    )

    guardrails: dict[str, dict[str, object]] = {}
    thresholds = {
        "CONFIRMATORY_EVALUATION": {
            "nll": max_confirmatory_nll_increase,
            "brier": max_confirmatory_brier_increase,
            "ece": max_confirmatory_ece_increase,
        },
        "EXTERNAL_VALIDATION": {
            "nll": max_external_nll_increase,
            "brier": max_external_brier_increase,
            "ece": max_external_ece_increase,
        },
    }
    for partition, tolerances in thresholds.items():
        for metric, tolerance in tolerances.items():
            raw_value = _metric_estimate(raw_report, partition, metric)
            calibrated_value = _metric_estimate(calibrated_report, partition, metric)
            delta = calibrated_value - raw_value
            guardrails[f"{partition}:{metric}"] = {
                "direction": "LOWER_IS_BETTER",
                "raw": raw_value,
                "calibrated": calibrated_value,
                "delta": delta,
                "maximum_allowed_increase": tolerance,
                "passed": delta <= tolerance + 1e-12,
            }

    labels = sorted(next(iter(predictions)).probabilities)
    classwise_guardrails = {}
    for partition in ("CONFIRMATORY_EVALUATION", "EXTERNAL_VALIDATION"):
        for label in labels:
            raw_brier, raw_ece = _classwise_estimates(raw_report, partition, label)
            cal_brier, cal_ece = _classwise_estimates(calibrated_report, partition, label)
            classwise_guardrails[f"{partition}:{label}"] = {
                "brier_delta": cal_brier - raw_brier,
                "maximum_brier_increase": max_classwise_brier_increase,
                "brier_passed": cal_brier - raw_brier <= max_classwise_brier_increase + 1e-12,
                "ece_delta": cal_ece - raw_ece,
                "maximum_ece_increase": max_classwise_ece_increase,
                "ece_passed": cal_ece - raw_ece <= max_classwise_ece_increase + 1e-12,
            }

    selective_guard = {
        "status": "NOT_EVALUABLE" if external_threshold["status"] != "EVALUATED_FIXED_DEVELOPMENT_THRESHOLD" else "EVALUATED",
        "passed": False,
        "maximum_allowed_selective_risk_increase": max_external_selective_risk_increase,
        "maximum_allowed_coverage_decrease": max_external_coverage_decrease,
        "comparison": "CALIBRATED_POLICY_VS_RAW_POLICY_EACH_SELECTED_ON_THE_SAME_DEVELOPMENT_GROUPS",
    }
    if external_threshold["status"] == "EVALUATED_FIXED_DEVELOPMENT_THRESHOLD" and raw_external_threshold["status"] == "EVALUATED_FIXED_DEVELOPMENT_THRESHOLD":
        calibrated_risk = external_threshold["overall"]["selective_risk"]
        raw_risk = raw_external_threshold["overall"]["selective_risk"]
        if calibrated_risk is None or raw_risk is None:
            selective_guard.update({"status": "NO_RETAINED_SAMPLES", "raw_selective_risk": raw_risk, "calibrated_selective_risk": calibrated_risk})
        else:
            delta = float(calibrated_risk) - float(raw_risk)
            calibrated_coverage = float(external_threshold["overall"]["coverage"])
            raw_coverage = float(raw_external_threshold["overall"]["coverage"])
            coverage_decrease = raw_coverage - calibrated_coverage
            selective_guard.update({
                "raw_selective_risk": raw_risk,
                "calibrated_selective_risk": calibrated_risk,
                "selective_risk_delta": delta,
                "raw_coverage": raw_coverage,
                "calibrated_coverage": calibrated_coverage,
                "coverage_decrease": coverage_decrease,
                "passed": (
                    delta <= max_external_selective_risk_increase + 1e-12
                    and coverage_decrease <= max_external_coverage_decrease + 1e-12
                ),
            })

    all_metric_pass = all(bool(item["passed"]) for item in guardrails.values())
    all_classwise_pass = all(bool(item["brier_passed"]) and bool(item["ece_passed"]) for item in classwise_guardrails.values())
    all_pass = all_metric_pass and all_classwise_pass and bool(selective_guard["passed"])

    external_rows = [(p, t, c) for p, t, c in rows if c.partition == "EXTERNAL_VALIDATION"]
    external_identity = _canonical_sha({"rows": [
        {
            "sample_id": p.sample_id,
            "group_id": p.group_id,
            "prediction_sha256": p.prediction_sha256,
            "true_id": t.true_id,
            "strata": dict(sorted((str(k), str(v)) for k, v in c.strata.items())),
        }
        for p, t, c in external_rows
    ]})
    report = {
        "schema_version": "astrograph-probability-calibration-qualification-report.v1",
        "calibration_id": verified_fit["calibration_id"],
        "model_id": model_id,
        "model_version": model_version,
        "fit_receipt_sha256": verified_fit["receipt_sha256"],
        "design_sha256": design_sha256,
        "admission_bundle_sha256": admission_bundle_sha256,
        "raw_calibration_transfer_report": raw_report,
        "calibrated_calibration_transfer_report": calibrated_report,
        "raw_calibration_transfer_report_sha256": raw_report["report_sha256"],
        "calibrated_calibration_transfer_report_sha256": calibrated_report["report_sha256"],
        "selective_threshold_receipt": threshold_receipt,
        "raw_baseline_selective_threshold_receipt": raw_threshold_receipt,
        "confirmatory_threshold_transfer": confirm_threshold,
        "external_threshold_transfer": external_threshold,
        "external_validation_dataset_id": external_validation_dataset_id,
        "external_validation_dataset_sha256": external_validation_dataset_sha256,
        "external_validation_identity_sha256": external_identity,
        "partition_group_counts": {name: len(partition_groups[name]) for name in sorted(partition_groups)},
        "positive_group_count_by_class": class_coverage_admission,
        "minimum_positive_groups_per_class": minimum_positive_groups_per_class,
        "guardrails": guardrails,
        "classwise_degradation_guardrails": classwise_guardrails,
        "external_selective_risk_degradation_guardrail": selective_guard,
        "qualification_status": "PASS_FOR_ENGINEERING_CALIBRATION_REVIEW" if all_pass else "BLOCKED_BY_CALIBRATION_GUARDRAIL",
        "claim_boundary": (
            "External-validation calibration qualification is an engineering review gate on frozen independent groups. "
            "It does not prove calibration under arbitrary distribution shift, causal robustness, HIL/flight qualification, or release safety."
        ),
    }
    report.update(bound_task)
    report["report_sha256"] = _canonical_sha(report)

    validation_ece = _metric_estimate(calibrated_report, "CONFIRMATORY_EVALUATION", "ece")
    external_ece = _metric_estimate(calibrated_report, "EXTERNAL_VALIDATION", "ece")
    qualification_receipt = ProbabilityCalibrationQualificationReceipt(
        **bound_task,
        calibration_id=str(verified_fit["calibration_id"]),
        model_id=model_id,
        model_version=model_version,
        calibration_dataset_id=str(verified_fit["calibration_dataset_id"]),
        calibration_dataset_sha256=str(verified_fit["calibration_dataset_sha256"]),
        model_artifact_sha256=model_artifact_sha256,
        calibration_method=str(verified_fit["method"]),
        fit_receipt_sha256=str(verified_fit["receipt_sha256"]),
        design_sha256=design_sha256,
        admission_bundle_sha256=admission_bundle_sha256,
        calibration_report_sha256=report["report_sha256"],
        validation_expected_calibration_error=validation_ece,
        external_validation_expected_calibration_error=external_ece,
        calibration_sample_count=int(verified_fit["fit_sample_count"]),
        calibration_group_count=int(verified_fit["fit_group_count"]),
        development_group_count=len(partition_groups["DEVELOPMENT_SELECTION"]),
        confirmatory_group_count=len(partition_groups["CONFIRMATORY_EVALUATION"]),
        external_validation_dataset_id=external_validation_dataset_id,
        external_validation_dataset_sha256=external_validation_dataset_sha256,
        external_validation_identity_sha256=external_identity,
        external_validation_group_count=len(partition_groups["EXTERNAL_VALIDATION"]),
        selective_threshold_receipt_sha256=str(threshold_receipt["receipt_sha256"]),
        qualification_status=report["qualification_status"],
    )
    return report, qualification_receipt


__all__ = [
    "temperature_scale_probability_map",
    "fit_temperature_scaler_receipt",
    "verify_temperature_scaler_fit_receipt",
    "apply_temperature_scaler_receipt",
    "qualify_temperature_calibration",
]
