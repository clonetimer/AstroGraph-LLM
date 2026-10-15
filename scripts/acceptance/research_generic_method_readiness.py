#!/usr/bin/env python3
"""Run the single Timer-authorized P-017 method-readiness study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/diagnosis/generic_method_readiness_protocol.v1.json"
GENERATOR_PATH = ROOT / "scripts/acceptance/research_thermal_observability_intervention.py"
CLASSES = ("NORMAL", "CAPACITY_LOSS", "DRIFT", "LEAKAGE")
ARCHETYPES = ("storage.energy_buffer", "thermal.lumped_capacity")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    authorization_path = ROOT / protocol["authorization"]["path"]
    if _sha(authorization_path) != protocol["authorization"]["sha256"]:
        raise ValueError("method-readiness authorization changed")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization["status"] != protocol["authorization"]["required_status"]:
        raise PermissionError("method-readiness study is not approved")
    if authorization["authorized_seeds"] != {
        "development": protocol["development_seed"],
        "calibration": protocol["calibration_seed"],
        "technical_validation": None,
    }:
        raise PermissionError("authorized seeds differ from the protocol")
    review_path = ROOT / protocol["literature_review"]["path"]
    if _sha(review_path) != protocol["literature_review"]["sha256"]:
        raise ValueError("literature review changed after protocol freeze")
    generator_path = ROOT / protocol["frozen_data_generator"]["implementation"]
    if _sha(generator_path) != protocol["frozen_data_generator"]["sha256"]:
        raise ValueError("frozen data generator changed")
    if protocol["single_model"]["model_count"] != 1:
        raise PermissionError("more than one model is configured")
    if protocol["technical_validation_seed"] is not None or protocol["stage_b_authorized"]:
        raise PermissionError("protocol exceeds Timer authority")
    exhausted = set(protocol["permanently_exhausted_validation_seeds"])
    if {protocol["development_seed"], protocol["calibration_seed"]} & exhausted:
        raise PermissionError("research reuses an exhausted validation seed")
    generator = _module("method_readiness_frozen_generator", GENERATOR_PATH)
    return protocol, generator


def _mapped(values: np.ndarray, indicator: float) -> np.ndarray:
    return np.concatenate([values, [indicator], indicator * values])


def _matrix(rows, indicators):
    return np.asarray(
        [_mapped(np.asarray(row.values, dtype=float), indicators[row.archetype]) for row in rows],
        dtype=float,
    )


def _quantile(values: list[float], alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def _conformal_components(x_dev, dev_rows, y_dev, protocol):
    cfg = protocol["conformal_rule"]
    shrinkage = float(cfg["covariance_shrinkage_to_diagonal"])
    floor = float(cfg["covariance_eigenvalue_floor"])
    components = {}
    thresholds = {}
    for archetype in ARCHETYPES:
        archetype_mask = np.asarray([row.archetype == archetype for row in dev_rows])
        for label in range(len(CLASSES)):
            matrix = x_dev[archetype_mask & (y_dev == label)]
            center = np.mean(matrix, axis=0)
            covariance = np.cov(matrix, rowvar=False, ddof=1)
            covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
                np.diag(covariance)
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T
            inverse = np.linalg.inv(covariance)
            distances = [
                float((row - center) @ inverse @ (row - center)) for row in matrix
            ]
            components[(archetype, label)] = (center, inverse)
            thresholds[(archetype, label)] = _quantile(distances, float(cfg["alpha"]))
    return components, thresholds


def _evaluate(rows, x, probabilities, components, thresholds):
    outcomes = []
    for index, row in enumerate(rows):
        scores = {}
        for label in range(len(CLASSES)):
            center, inverse = components[(row.archetype, label)]
            difference = x[index] - center
            scores[label] = float(difference @ inverse @ difference)
        prediction_set = [
            label for label in range(len(CLASSES))
            if scores[label] <= thresholds[(row.archetype, label)]
        ]
        predicted = (
            max(prediction_set, key=lambda label: probabilities[index, label])
            if prediction_set else None
        )
        known = row.label != 0
        anomalous = bool(predicted is None or predicted != 0)
        outcomes.append({
            "archetype": row.archetype,
            "label": row.label,
            "predicted": predicted,
            "ood": predicted is None,
            "classification_correct": bool(known and predicted == row.label),
            "known_retained": bool(known and predicted is not None and predicted != 0),
            "normal_correct": bool(not known and predicted == 0),
            "detection_correct": bool(anomalous if known else not anomalous),
        })
    return outcomes


def _metrics(outcomes):
    rate = lambda values: float(sum(values) / len(values))
    known = [item for item in outcomes if item["label"] != 0]
    normal = [item for item in outcomes if item["label"] == 0]
    result = {
        "known_classification_accuracy": rate([item["classification_correct"] for item in known]),
        "known_retention": rate([item["known_retained"] for item in known]),
        "normal_detection_accuracy": rate([item["normal_correct"] for item in normal]),
        "detection_accuracy": rate([item["detection_correct"] for item in outcomes]),
        "ood_rate_on_known_and_normal": rate([item["ood"] for item in outcomes]),
    }
    for archetype in ARCHETYPES:
        selected_known = [item for item in known if item["archetype"] == archetype]
        selected_normal = [item for item in normal if item["archetype"] == archetype]
        result[archetype] = {
            "known_classification_accuracy": rate(
                [item["classification_correct"] for item in selected_known]
            ),
            "known_retention": rate([item["known_retained"] for item in selected_known]),
            "normal_detection_accuracy": rate(
                [item["normal_correct"] for item in selected_normal]
            ),
        }
    labels = [-1, 0, 1, 2, 3]
    result["confusion_matrix_labels_OOD_NORMAL_CAPACITY_LOSS_DRIFT_LEAKAGE"] = labels
    result["confusion_matrix"] = confusion_matrix(
        [item["label"] for item in outcomes],
        [-1 if item["predicted"] is None else item["predicted"] for item in outcomes],
        labels=labels,
    ).tolist()
    return result


def _reachability(mean, scale, indicators, components, thresholds):
    result = {}
    base = 1.0e6 * np.linspace(1.0, 2.0, 22)
    for archetype in ARCHETYPES:
        mapped = (_mapped(base, indicators[archetype]) - mean) / scale
        result[archetype] = all(
            float((mapped - components[(archetype, label)][0])
                  @ components[(archetype, label)][1]
                  @ (mapped - components[(archetype, label)][0]))
            > thresholds[(archetype, label)]
            for label in range(len(CLASSES))
        )
    return result


def _generate_and_extract(protocol, generator, seed, count):
    old_protocol, stage_protocol, base_protocol, stage_a, _, candidate1, candidate2 = (
        generator._load()
    )
    cases, metadata, paired_identity = generator._generate_arm(
        old_protocol,
        stage_protocol,
        base_protocol,
        candidate1,
        seed,
        count,
        protocol["frozen_data_generator"]["thermal_arm"],
    )
    rows, failures = stage_a._extract(candidate2, cases)
    audit = {
        "case_count": len(cases),
        "thermal_case_count": len(metadata),
        "paired_identity_sha256": paired_identity,
        "thermal_horizon_seconds_minimum": float(
            min(item["horizon_seconds"] for item in metadata)
        ),
        "thermal_horizon_seconds_maximum": float(
            max(item["horizon_seconds"] for item in metadata)
        ),
    }
    return rows, failures, audit


def run(output: Path):
    started = time.monotonic()
    protocol, generator = _load()
    counts = protocol["case_counts"]
    development, development_failures, development_audit = _generate_and_extract(
        protocol,
        generator,
        protocol["development_seed"],
        counts["development_per_archetype_per_class"],
    )
    calibration, calibration_failures, calibration_audit = _generate_and_extract(
        protocol,
        generator,
        protocol["calibration_seed"],
        counts["calibration_per_archetype_per_class"],
    )
    if len(development) != counts["development_total"] or len(calibration) != counts["calibration_total"]:
        raise RuntimeError("generated case count differs from frozen denominator")
    indicators = protocol["feature_mapping"]["archetype_indicator"]
    x_dev_raw = _matrix(development, indicators)
    x_cal_raw = _matrix(calibration, indicators)
    mean = np.mean(x_dev_raw, axis=0)
    scale = np.maximum(
        np.std(x_dev_raw, axis=0),
        float(protocol["feature_mapping"]["standard_deviation_floor"]),
    )
    x_dev = (x_dev_raw - mean) / scale
    x_cal = (x_cal_raw - mean) / scale
    y_dev = np.asarray([row.label for row in development], dtype=int)
    y_cal = np.asarray([row.label for row in calibration], dtype=int)
    cfg = protocol["single_model"]
    model = LogisticRegression(
        penalty=cfg["penalty"],
        C=cfg["C"],
        fit_intercept=cfg["fit_intercept"],
        class_weight=cfg["class_weight"],
        solver=cfg["solver"],
        max_iter=cfg["maximum_iterations"],
        tol=cfg["tolerance"],
        random_state=cfg["random_state"],
    ).fit(x_dev, y_dev)
    components, thresholds = _conformal_components(x_dev, development, y_dev, protocol)
    dev_outcomes = _evaluate(
        development, x_dev, model.predict_proba(x_dev), components, thresholds
    )
    cal_outcomes = _evaluate(
        calibration, x_cal, model.predict_proba(x_cal), components, thresholds
    )
    development_metrics = _metrics(dev_outcomes)
    calibration_metrics = _metrics(cal_outcomes)
    reachability = _reachability(mean, scale, indicators, components, thresholds)
    gaps = {
        archetype: abs(
            development_metrics[archetype]["known_classification_accuracy"]
            - calibration_metrics[archetype]["known_classification_accuracy"]
        )
        for archetype in ARCHETYPES
    }
    failure_count = len(development_failures) + len(calibration_failures)
    elapsed = time.monotonic() - started
    exit_cfg = protocol["exit_checks"]
    checks = {
        "known_classification_accuracy": calibration_metrics["known_classification_accuracy"]
        >= exit_cfg["known_classification_accuracy"],
        "normal_detection_accuracy": calibration_metrics["normal_detection_accuracy"]
        >= exit_cfg["normal_detection_accuracy"],
        "storage_known_classification_accuracy": calibration_metrics[ARCHETYPES[0]][
            "known_classification_accuracy"
        ] >= exit_cfg["storage_known_classification_accuracy"],
        "thermal_known_classification_accuracy": calibration_metrics[ARCHETYPES[1]][
            "known_classification_accuracy"
        ] >= exit_cfg["thermal_known_classification_accuracy"],
        "known_retention": calibration_metrics["known_retention"]
        >= exit_cfg["known_retention"],
        "open_set_rule_reachable_for_each_archetype": all(reachability.values()),
        "development_calibration_gap_each_archetype": all(
            value <= exit_cfg["development_calibration_gap_each_archetype_maximum"]
            for value in gaps.values()
        ),
        "failed_case_count": failure_count == exit_cfg["failed_case_count"],
        "runtime_budget": elapsed <= exit_cfg["maximum_runtime_seconds"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    evidence = {
        "schema_version": "astrograph-p017-method-readiness-evidence.v1",
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "authorization_sha256": protocol["authorization"]["sha256"],
        "literature_review_sha256": protocol["literature_review"]["sha256"],
        "development_seed": protocol["development_seed"],
        "calibration_seed": protocol["calibration_seed"],
        "technical_validation_seed": None,
        "exhausted_validation_seeds_used": [],
        "formal_open_set_classes_generated": [],
        "production_candidate_fitted": False,
        "stage_b_started": False,
        "model_count": 1,
        "model_family": cfg["family"],
        "feature_dimension": protocol["feature_mapping"]["total_dimension"],
        "development_audit": development_audit,
        "calibration_audit": calibration_audit,
        "development_failure_count": len(development_failures),
        "calibration_failure_count": len(calibration_failures),
        "development_split_sha256": _canonical(
            [(row.case_key, row.label, row.values.tolist()) for row in development]
        ),
        "calibration_split_sha256": _canonical(
            [(row.case_key, row.label, row.values.tolist()) for row in calibration]
        ),
        "development_metrics": development_metrics,
        "calibration_metrics": calibration_metrics,
        "development_calibration_known_accuracy_gap": gaps,
        "open_set_rule_reachable": reachability,
        "same_score_semantics": True,
        "thresholds_sha256": _canonical(
            {f"{a}|{CLASSES[c]}": value for (a, c), value in thresholds.items()}
        ),
        "model_receipt": {
            "iterations_per_class": model.n_iter_.tolist(),
            "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
            "intercept_l2_norm": float(np.linalg.norm(model.intercept_)),
            "feature_mean_sha256": _canonical(mean.tolist()),
            "feature_scale_sha256": _canonical(scale.tolist()),
            "model_artifact_written": False,
        },
        "checks": checks,
        "method_readiness_status": status,
        "stage_b_proposal_eligible": status == "PASS",
        "stage_b_authorized": False,
        "elapsed_seconds": elapsed,
        "runtime_budget_seconds": exit_cfg["maximum_runtime_seconds"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "governance/evidence/0.7.1-extension/generic_method_readiness.v1.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise PermissionError("method-readiness evidence already exists")
    evidence = run(args.output)
    print(json.dumps({
        "status": evidence["method_readiness_status"],
        "stage_b_proposal_eligible": evidence["stage_b_proposal_eligible"],
        "calibration_metrics": evidence["calibration_metrics"],
        "gaps": evidence["development_calibration_known_accuracy_gap"],
        "checks": evidence["checks"],
        "elapsed_seconds": evidence["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0 if evidence["method_readiness_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
