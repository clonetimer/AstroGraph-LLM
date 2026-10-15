#!/usr/bin/env python3
"""Fit candidate 2 and consume its frozen validation seed at most once."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from astrograph.diagnosis.generic import (
    CalibratedSegmentedMechanismModel,
    FailureMechanism,
    OpenSetDecision,
    SegmentedMechanismCalibrationArtifact,
    SegmentedMechanismFeatureVector,
    extract_segmented_mechanism_features,
)
from astrograph.physics import default_dynamics_contracts


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/diagnosis/generic_head_calibration_protocol.v2.json"
AUTHORIZATION_GATE_PATH = (
    ROOT / "governance/evidence/0.7.1-extension/candidate2_authorization_gate.json"
)
KNOWN = (
    FailureMechanism.CAPACITY_LOSS,
    FailureMechanism.DRIFT,
    FailureMechanism.LEAKAGE,
)
OPEN_SET = (
    FailureMechanism.BIAS,
    FailureMechanism.INTERMITTENT,
    FailureMechanism.STUCK,
)
ARCHETYPES = ("storage.energy_buffer", "thermal.lumped_capacity")


@dataclass(frozen=True)
class ExtractedCase:
    case_key: str
    archetype: str
    mechanism: FailureMechanism | None
    features: SegmentedMechanismFeatureVector | None
    failed: bool
    failure_reason: str | None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _candidate_one_module():
    path = ROOT / "scripts/acceptance/calibrate_generic_diagnosis_heads.py"
    spec = importlib.util.spec_from_file_location("candidate_one_reference_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen candidate-one reference generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_version"] != "2.0.1":
        raise ValueError("candidate-2 protocol version differs from the frozen implementation")
    if tuple(protocol["known_mechanisms"]) != tuple(item.value for item in KNOWN):
        raise ValueError("known mechanism order differs from the frozen protocol")
    if tuple(protocol["open_set_mechanisms"]) != tuple(item.value for item in OPEN_SET):
        raise ValueError("open-set mechanism order differs from the frozen protocol")
    if tuple(protocol["component_archetypes"]) != ARCHETYPES:
        raise ValueError("archetype order differs from the frozen protocol")
    gate = json.loads(AUTHORIZATION_GATE_PATH.read_text(encoding="utf-8"))
    if gate["status"] != "PASS":
        raise PermissionError("candidate-2 authorization gate is not PASS")
    for item in protocol["frozen_g1_1_artifacts"]:
        path = ROOT / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise ValueError(f"frozen G1.1 artifact changed: {item['path']}")
    for stem in ("request", "timer_attestation", "authorization_gate"):
        path = ROOT / protocol["authorization"][f"{stem}_path"]
        if not path.is_file() or _sha(path) != protocol["authorization"][f"{stem}_sha256"]:
            raise ValueError(f"candidate-2 authorization artifact changed: {stem}")
    return protocol


def _generate(
    protocol: dict[str, Any],
    *,
    seed: int,
    count: int,
    mechanisms: tuple[FailureMechanism, ...],
    validation_domain: bool,
    include_normal: bool,
):
    module = _candidate_one_module()
    return module._generate_cases(
        protocol,
        seed=seed,
        count_per_archetype_per_mechanism=count,
        mechanisms=mechanisms,
        validation_domain=validation_domain,
        include_normal=include_normal,
    )


def _extract(cases: list[Any]) -> list[ExtractedCase]:
    contracts = default_dynamics_contracts()
    rows: list[ExtractedCase] = []
    for case in cases:
        try:
            features = extract_segmented_mechanism_features(
                case.frame, contracts[case.archetype]
            )
            rows.append(
                ExtractedCase(
                    case_key=case.case_key,
                    archetype=case.archetype,
                    mechanism=case.mechanism,
                    features=features,
                    failed=False,
                    failure_reason=None,
                )
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            rows.append(
                ExtractedCase(
                    case_key=case.case_key,
                    archetype=case.archetype,
                    mechanism=case.mechanism,
                    features=None,
                    failed=True,
                    failure_reason=str(exc),
                )
            )
    return rows


def _quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) == 0:
        raise ValueError("split-conformal quantile requires at least one value")
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def _temperature(
    logits: np.ndarray, labels: np.ndarray, protocol: dict[str, Any]
) -> float:
    configuration = protocol["production_candidate"]["temperature_scaling"]
    candidates = np.linspace(
        float(configuration["minimum_temperature"]),
        float(configuration["maximum_temperature"]),
        int(configuration["grid_point_count"]),
    )
    losses: list[float] = []
    for candidate in candidates:
        scaled = logits / candidate
        scaled -= np.max(scaled, axis=1, keepdims=True)
        log_probabilities = scaled - np.log(
            np.sum(np.exp(scaled), axis=1, keepdims=True)
        )
        losses.append(float(-np.mean(log_probabilities[np.arange(len(labels)), labels])))
    return float(candidates[int(np.argmin(losses))])


def _distance(
    vector: np.ndarray, center: np.ndarray, inverse_covariance: np.ndarray
) -> float:
    difference = vector - center
    return math.sqrt(
        max(0.0, float(difference @ inverse_covariance @ difference)) / len(vector)
    )


def _accuracy(
    model: CalibratedSegmentedMechanismModel, rows: list[ExtractedCase]
) -> float:
    correct = 0
    for row in rows:
        if row.features is None or row.mechanism is None:
            continue
        correct += int(model.predict(row.features).mechanism_id == row.mechanism)
    return correct / len(rows) if rows else 0.0


def fit(
    protocol: dict[str, Any],
) -> tuple[SegmentedMechanismCalibrationArtifact, dict[str, Any]]:
    started = time.monotonic()
    split = protocol["split"]
    development = _extract(
        _generate(
            protocol,
            seed=int(split["development_seed"]),
            count=int(split["development_count_per_archetype_per_known_mechanism"]),
            mechanisms=KNOWN,
            validation_domain=False,
            include_normal=False,
        )
    )
    calibration = _extract(
        _generate(
            protocol,
            seed=int(split["calibration_seed"]),
            count=int(split["calibration_count_per_archetype_per_known_mechanism"]),
            mechanisms=KNOWN,
            validation_domain=False,
            include_normal=False,
        )
    )
    normal_calibration = _extract(
        _generate(
            protocol,
            seed=int(split["calibration_seed"]),
            count=int(split["calibration_normal_count_per_archetype"]),
            mechanisms=(),
            validation_domain=False,
            include_normal=True,
        )
    )
    usable_development = [row for row in development if row.features is not None]
    usable_calibration = [row for row in calibration if row.features is not None]
    usable_normals = [row for row in normal_calibration if row.features is not None]
    if len(usable_development) != len(development):
        raise RuntimeError("candidate-2 development feature extraction failed")
    if len(usable_calibration) != len(calibration):
        raise RuntimeError("candidate-2 calibration feature extraction failed")
    if len(usable_normals) != len(normal_calibration):
        raise RuntimeError("candidate-2 normal calibration feature extraction failed")
    features = np.asarray([row.features.values for row in usable_development], dtype=float)
    labels = np.asarray(
        [KNOWN.index(row.mechanism) for row in usable_development], dtype=int
    )
    calibration_features = np.asarray(
        [row.features.values for row in usable_calibration], dtype=float
    )
    calibration_labels = np.asarray(
        [KNOWN.index(row.mechanism) for row in usable_calibration], dtype=int
    )
    scale_floor = float(protocol["production_candidate"]["feature_scale_floor"])
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), scale_floor)
    standardized = (features - mean) / scale
    standardized_calibration = (calibration_features - mean) / scale
    optimizer = protocol["production_candidate"]["optimizer"]
    classifier = LogisticRegression(
        C=float(optimizer["l2_inverse_regularization"]),
        solver=str(optimizer["solver"]),
        max_iter=int(optimizer["maximum_iterations"]),
        tol=float(optimizer["tolerance"]),
        random_state=int(optimizer["random_seed"]),
    ).fit(standardized, labels)
    if tuple(classifier.classes_.tolist()) != tuple(range(len(KNOWN))):
        raise RuntimeError("candidate-2 fitted class order differs from frozen order")
    temperature = _temperature(
        classifier.decision_function(standardized_calibration),
        calibration_labels,
        protocol,
    )
    shrinkage = float(
        protocol["production_candidate"]["class_covariance_shrinkage_to_diagonal"]
    )
    eigen_floor = float(protocol["production_candidate"]["matrix_eigenvalue_floor"])
    centers: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    for index in range(len(KNOWN)):
        class_rows = standardized[labels == index]
        center = np.mean(class_rows, axis=0)
        covariance = np.cov(class_rows, rowvar=False, ddof=1)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
            np.diag(covariance)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (
            eigenvectors
            @ np.diag(np.maximum(eigenvalues, eigen_floor))
            @ eigenvectors.T
        )
        centers.append(center)
        inverses.append(np.linalg.inv(covariance))
    alpha = float(protocol["thresholds"]["open_set_alpha"])
    distance_thresholds: list[float] = []
    for index in range(len(KNOWN)):
        class_rows = standardized_calibration[calibration_labels == index]
        distances = np.asarray(
            [_distance(row, centers[index], inverses[index]) for row in class_rows]
        )
        distance_thresholds.append(_quantile(distances, alpha))
    anomaly_alpha = float(protocol["thresholds"]["anomaly_false_alarm_alpha"])
    anomaly_threshold = _quantile(
        np.asarray([row.features.anomaly_score for row in usable_normals]),
        anomaly_alpha,
    )
    protocol_sha = _sha(PROTOCOL_PATH)
    artifact = SegmentedMechanismCalibrationArtifact(
        calibration_id="astrograph.generic-mechanism.segmented.synthetic-technical.v2",
        calibration_version="2.0.0",
        protocol_sha256=protocol_sha,
        model_family="L2_REGULARIZED_MULTINOMIAL_LOGISTIC_REGRESSION",
        model_id="astrograph.generic.mechanism.segmented-gray-box-logistic",
        model_version="2.0.0",
        feature_names=protocol["feature_schema"],
        class_order=list(KNOWN),
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        coefficients=classifier.coef_.tolist(),
        intercepts=classifier.intercept_.tolist(),
        temperature=temperature,
        anomaly_threshold=anomaly_threshold,
        class_centers=[item.tolist() for item in centers],
        class_inverse_covariances=[item.tolist() for item in inverses],
        class_distance_thresholds=distance_thresholds,
        open_set_alpha=alpha,
        anomaly_false_alarm_alpha=anomaly_alpha,
        covariance_shrinkage=shrinkage,
        covariance_eigenvalue_floor=eigen_floor,
        development_split_sha256=_canonical_sha(
            [
                (row.case_key, row.mechanism.value, row.features.values)
                for row in development
            ]
        ),
        calibration_split_sha256=_canonical_sha(
            [
                (row.case_key, row.mechanism.value, row.features.values)
                for row in calibration
            ]
        ),
        normal_calibration_split_sha256=_canonical_sha(
            [(row.case_key, "NORMAL", row.features.values) for row in normal_calibration]
        ),
        development_case_count=len(development),
        calibration_case_count=len(calibration),
        normal_calibration_case_count=len(normal_calibration),
    )
    model = CalibratedSegmentedMechanismModel(artifact)
    evidence = {
        "schema_version": "astrograph-segmented-generic-mechanism-fit-evidence.v1",
        "candidate_ordinal": 2,
        "protocol_sha256": protocol_sha,
        "formal_hidden_truth_used": False,
        "synthetic_controlled_reference_used": True,
        "development_seed": int(split["development_seed"]),
        "calibration_seed": int(split["calibration_seed"]),
        "exhausted_validation_seed_used": False,
        "new_validation_seed_used": False,
        "development_case_count": len(development),
        "calibration_case_count": len(calibration),
        "normal_calibration_case_count": len(normal_calibration),
        "development_failure_count": sum(row.failed for row in development),
        "calibration_failure_count": sum(row.failed for row in calibration),
        "normal_calibration_failure_count": sum(
            row.failed for row in normal_calibration
        ),
        "development_classification_accuracy": _accuracy(model, development),
        "calibration_classification_accuracy": _accuracy(model, calibration),
        "anomaly_threshold": anomaly_threshold,
        "temperature": temperature,
        "class_distance_thresholds": {
            mechanism.value: distance_thresholds[index]
            for index, mechanism in enumerate(KNOWN)
        },
        "elapsed_seconds": time.monotonic() - started,
        "runtime_budget_seconds": int(protocol["resource_budget"]["max_runtime_seconds"]),
        "technical_validation_read": False,
        "status": "FIT_COMPLETE",
        "candidate_bound_to_production": False,
        "claim_boundary": (
            "Candidate-2 development and calibration evidence only. The single-use technical "
            "validation seed was not generated or inspected; no promotion, P-014, P-015, T-310, "
            "T-311, G3 or G4 completion is established."
        ),
    }
    return artifact, evidence


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    groups = sorted({(row["archetype"], row["mechanism"]) for row in rows})
    for archetype, mechanism in groups:
        selected = [
            row
            for row in rows
            if row["archetype"] == archetype and row["mechanism"] == mechanism
        ]
        result[f"{archetype}|{mechanism}"] = {
            "count": len(selected),
            "detection_accuracy": _rate(
                [row["detection_correct"] for row in selected]
            ),
            "classification_accuracy": _rate(
                [row["classification_correct"] for row in selected]
            ),
            "known_retention": _rate([row["known_retained"] for row in selected]),
            "open_set_rejection": _rate(
                [row["open_set_rejected"] for row in selected]
            ),
        }
    return result


def validate(
    protocol: dict[str, Any],
    artifact: SegmentedMechanismCalibrationArtifact,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol_sha = _sha(PROTOCOL_PATH)
    if artifact.protocol_sha256 != protocol_sha:
        raise ValueError("candidate-2 artifact does not match the frozen protocol")
    split = protocol["split"]
    count = int(split["technical_validation_count_per_archetype_per_mechanism"])
    cases = _generate(
        protocol,
        seed=int(split["technical_validation_seed"]),
        count=count,
        mechanisms=KNOWN + OPEN_SET,
        validation_domain=True,
        include_normal=True,
    )
    extracted = _extract(cases)
    model = CalibratedSegmentedMechanismModel(artifact)
    rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    for case in extracted:
        prediction = model.predict(case.features) if case.features is not None else None
        if case.failed:
            reason = case.failure_reason or "UNSPECIFIED_FAILURE"
            failures[reason] = failures.get(reason, 0) + 1
        role = (
            "NORMAL"
            if case.mechanism is None
            else "KNOWN"
            if case.mechanism in KNOWN
            else "OPEN_SET"
        )
        anomalous = bool(prediction is not None and prediction.anomalous)
        detection_correct = (
            not case.failed and (not anomalous if role == "NORMAL" else anomalous)
        )
        retained = bool(
            role == "KNOWN"
            and anomalous
            and prediction is not None
            and prediction.open_set_decision == OpenSetDecision.IN_DISTRIBUTION
        )
        rejected = bool(
            role == "OPEN_SET"
            and anomalous
            and prediction is not None
            and prediction.open_set_decision == OpenSetDecision.OOD
        )
        classified = bool(
            retained
            and prediction is not None
            and prediction.mechanism_id == case.mechanism
        )
        rows.append(
            {
                "case_key": case.case_key,
                "archetype": case.archetype,
                "role": role,
                "mechanism": case.mechanism.value if case.mechanism else "NORMAL",
                "failed": case.failed,
                "failure_reason": case.failure_reason,
                "detection_correct": detection_correct,
                "known_retained": retained,
                "open_set_rejected": rejected,
                "classification_correct": classified,
                "localization_correct": classified,
                "reasoning_correct": classified,
                "anomaly_score": (
                    prediction.anomaly_score if prediction is not None else None
                ),
                "anomalous": anomalous,
                "predicted_mechanism": (
                    prediction.mechanism_id.value if prediction is not None else None
                ),
                "open_set_decision": (
                    prediction.open_set_decision.value if prediction is not None else None
                ),
                "class_distance": (
                    prediction.class_distance if prediction is not None else None
                ),
                "class_distance_threshold": (
                    prediction.class_distance_threshold
                    if prediction is not None
                    else None
                ),
            }
        )
    normal = [row for row in rows if row["role"] == "NORMAL"]
    known = [row for row in rows if row["role"] == "KNOWN"]
    opened = [row for row in rows if row["role"] == "OPEN_SET"]
    denominators = {
        "normal": len(normal),
        "known": len(known),
        "open_set": len(opened),
        "total": len(rows),
    }
    if denominators != protocol["denominators"]:
        raise RuntimeError(
            f"candidate-2 validation denominators changed: {denominators}"
        )
    metrics = {
        "detection_accuracy": _rate([row["detection_correct"] for row in rows]),
        "classification_accuracy": _rate(
            [row["classification_correct"] for row in known]
        ),
        "localization_accuracy": _rate(
            [row["localization_correct"] for row in known]
        ),
        "reasoning_correctness": _rate(
            [row["reasoning_correct"] for row in known]
        ),
        "known_class_retention": _rate([row["known_retained"] for row in known]),
        "open_set_rejection": _rate(
            [row["open_set_rejected"] for row in opened]
        ),
    }
    thresholds = protocol["thresholds"]
    checks = {
        "denominators_preserved": denominators == protocol["denominators"],
        "detection_accuracy": metrics["detection_accuracy"]
        >= float(thresholds["detection_accuracy"]),
        "classification_accuracy": metrics["classification_accuracy"]
        >= float(thresholds["classification_accuracy"]),
        "localization_accuracy": metrics["localization_accuracy"]
        >= float(thresholds["localization_accuracy"]),
        "reasoning_correctness": metrics["reasoning_correctness"]
        >= float(thresholds["reasoning_correctness"]),
        "known_class_retention": metrics["known_class_retention"]
        >= float(thresholds["known_class_retention"]),
        "open_set_rejection": metrics["open_set_rejection"]
        >= float(thresholds["open_set_rejection"]),
    }
    elapsed = time.monotonic() - started
    checks["runtime_budget"] = elapsed <= float(
        protocol["resource_budget"]["max_runtime_seconds"]
    )
    passed = all(checks.values())
    return {
        "schema_version": "astrograph-segmented-generic-head-technical-validation.v1",
        "candidate_ordinal": 2,
        "protocol_sha256": protocol_sha,
        "calibration_artifact_sha256": _canonical_sha(
            artifact.model_dump(mode="json")
        ),
        "formal_hidden_truth_used": False,
        "synthetic_controlled_reference_used": True,
        "validation_seed": int(split["technical_validation_seed"]),
        "exhausted_validation_seed_used": False,
        "validation_run_count": 1,
        "denominators": denominators,
        "failed_run_count": sum(row["failed"] for row in rows),
        "failure_reasons": failures,
        "metrics": metrics,
        "thresholds": {
            key: thresholds[key]
            for key in (
                "detection_accuracy",
                "classification_accuracy",
                "localization_accuracy",
                "reasoning_correctness",
                "known_class_retention",
                "open_set_rejection",
            )
        },
        "checks": checks,
        "per_declared_group": _group_rates(rows),
        "case_results": rows,
        "elapsed_seconds": elapsed,
        "technical_status": "PASS" if passed else "FAIL",
        "candidate_bound_to_production": False,
        "p014_status": "AWAITING_TIMER_REVIEW" if passed else "OPEN",
        "p015_status": "AWAITING_TIMER_REVIEW" if passed else "OPEN",
        "t310_status": "AWAITING_TIMER_REVIEW" if passed else "INCOMPLETE",
        "t311_status": "AWAITING_TIMER_REVIEW" if passed else "INCOMPLETE",
        "g3_status": "AWAITING_TIMER_REVIEW" if passed else "INCOMPLETE",
        "claim_boundary": (
            "Single-use synthetic controlled technical validation. PASS, if present, is still "
            "not model promotion or independent Timer acceptance; FAIL is retained without "
            "threshold, denominator, class or UNKNOWN changes."
        ),
    }


def _consume_validation_once(
    protocol: dict[str, Any],
    artifact_path: Path,
    evidence_path: Path,
    query_path: Path,
) -> dict[str, Any]:
    if query_path.exists() or evidence_path.exists():
        raise PermissionError(
            "candidate-2 validation seed is already consumed or an invocation already started"
        )
    artifact = SegmentedMechanismCalibrationArtifact.model_validate_json(
        artifact_path.read_text(encoding="utf-8")
    )
    query = {
        "schema_version": "astrograph-single-use-technical-validation-query.v1",
        "candidate_ordinal": 2,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "calibration_artifact_path": str(artifact_path.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "calibration_artifact_sha256": _canonical_sha(
            artifact.model_dump(mode="json")
        ),
        "validation_seed": int(protocol["split"]["technical_validation_seed"]),
        "exhausted_validation_seed_used": False,
        "invocation_started_at_utc": _now(),
        "status": "STARTED_AND_SEED_CONSUMED",
        "maximum_invocations": 1,
        "formal_hidden_truth_used": False,
        "claim_boundary": (
            "This receipt consumes candidate 2's sole technical-validation invocation even if "
            "execution fails. It cannot be deleted to authorize a rerun."
        ),
    }
    _write(query_path, query)
    try:
        result = validate(protocol, artifact)
    except Exception as exc:
        failed = {
            "schema_version": "astrograph-segmented-generic-head-technical-validation.v1",
            "candidate_ordinal": 2,
            "protocol_sha256": _sha(PROTOCOL_PATH),
            "calibration_artifact_sha256": query["calibration_artifact_sha256"],
            "validation_seed": int(protocol["split"]["technical_validation_seed"]),
            "validation_run_count": 1,
            "run_level_failure": True,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "denominators": protocol["denominators"],
            "metrics": {
                "detection_accuracy": 0.0,
                "classification_accuracy": 0.0,
                "localization_accuracy": 0.0,
                "reasoning_correctness": 0.0,
                "known_class_retention": 0.0,
                "open_set_rejection": 0.0,
            },
            "technical_status": "FAIL",
            "candidate_bound_to_production": False,
            "claim_boundary": "Run-level failure counts as failure for every frozen metric and denominator.",
        }
        _write(evidence_path, failed)
        raise
    _write(evidence_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "src/astrograph/resources/generic_mechanism_calibration.v2.json",
    )
    fit_parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT
        / "governance/evidence/0.7.1-extension/generic_mechanism_fit_calibration.v2.json",
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "src/astrograph/resources/generic_mechanism_calibration.v2.json",
    )
    validate_parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT
        / "governance/evidence/0.7.1-extension/generic_head_technical_validation.v2.json",
    )
    validate_parser.add_argument(
        "--query-receipt",
        type=Path,
        default=ROOT
        / "governance/evidence/0.7.1-extension/generic_head_technical_validation_query.v2.json",
    )
    args = parser.parse_args()
    protocol = _load_protocol()
    if args.command == "fit":
        artifact, evidence = fit(protocol)
        _write(args.artifact, artifact.model_dump(mode="json"))
        evidence["calibration_artifact_sha256"] = _canonical_sha(
            artifact.model_dump(mode="json")
        )
        _write(args.evidence, evidence)
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0
    result = _consume_validation_once(
        protocol, args.artifact.resolve(), args.evidence.resolve(), args.query_receipt.resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["technical_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
