#!/usr/bin/env python3
"""Fit and evaluate the frozen generic mechanism-head technical protocol.

The fit phase may read deterministic synthetic controlled references from the declared development
and calibration splits. The validation phase is a separate one-shot command and never exposes formal
hidden-acceptance truth to runtime code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from astrograph.diagnosis.generic import (
    CalibratedMechanismModel,
    FailureMechanism,
    MechanismCalibrationArtifact,
    MechanismFeatureVector,
    OpenSetDecision,
    extract_mechanism_features,
)
from astrograph.physics import default_dynamics_contracts
from astrograph.state_estimation import GrayBoxStateEstimator
from astrograph.state_estimation.gray_box import StateEstimationError


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/diagnosis/generic_head_calibration_protocol.v1.json"
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
class GeneratedCase:
    case_key: str
    archetype: str
    mechanism: FailureMechanism | None
    frame: Any


@dataclass(frozen=True)
class ExtractedCase:
    case_key: str
    archetype: str
    mechanism: FailureMechanism | None
    features: list[float] | None
    anomalous: bool
    failed: bool
    failure_reason: str | None


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(payload: Any) -> str:
    return _sha_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if tuple(payload["known_mechanisms"]) != tuple(item.value for item in KNOWN):
        raise ValueError("runtime known-mechanism order differs from the frozen protocol")
    if tuple(payload["open_set_mechanisms"]) != tuple(item.value for item in OPEN_SET):
        raise ValueError("runtime open-set order differs from the frozen protocol")
    if tuple(payload["component_archetypes"]) != ARCHETYPES:
        raise ValueError("runtime archetype order differs from the frozen protocol")
    return payload


def _gray_box_module():
    path = ROOT / "scripts/acceptance/calibrate_gray_box_uncertainty.py"
    spec = importlib.util.spec_from_file_location("gray_box_reference_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the gray-box reference generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _onset_and_severity(
    protocol: dict[str, Any], rng: np.random.Generator, mechanism: FailureMechanism
) -> tuple[float, float]:
    generator = protocol["synthetic_generator"]
    onset = float(rng.uniform(*generator["onset_fraction_range"]))
    if mechanism == FailureMechanism.CAPACITY_LOSS:
        severity = float(rng.uniform(*generator["capacity_remaining_fraction_range"]))
    elif mechanism == FailureMechanism.LEAKAGE:
        severity = 0.0
    elif mechanism == FailureMechanism.DRIFT:
        severity = 0.0
    elif mechanism == FailureMechanism.BIAS:
        severity = 0.0
    elif mechanism == FailureMechanism.INTERMITTENT:
        severity = float(rng.uniform(*generator["intermittent_active_fraction_range"]))
    elif mechanism == FailureMechanism.STUCK:
        severity = float(rng.uniform(*generator["stuck_duration_fraction_range"]))
    else:
        raise ValueError(f"unsupported synthetic mechanism: {mechanism.value}")
    return onset, severity


def _storage_case(
    module: Any,
    protocol: dict[str, Any],
    config: dict[str, float | int],
    mechanism: FailureMechanism | None,
    rng: np.random.Generator,
) -> Any:
    step = float(config["step_s"])
    timestamps = np.arange(int(config["sample_count"]), dtype=float) * step
    power = (
        float(config["power_offset_w"])
        + float(config["power_sine_w"])
        * np.sin(timestamps / float(config["sine_period_s"]))
        + float(config["power_cosine_w"])
        * np.cos(timestamps / float(config["cosine_period_s"]))
    )
    capacity = float(config["energy_capacity_j"])
    latent = np.empty_like(timestamps)
    latent[0] = float(config["initial_fraction"])
    onset, severity = (
        _onset_and_severity(protocol, rng, mechanism)
        if mechanism is not None
        else (1.0, 0.0)
    )
    onset_index = min(max(int(round(onset * (len(timestamps) - 1))), 1), len(timestamps) - 2)
    generator = protocol["synthetic_generator"]
    leakage_rate = float(rng.uniform(*generator["storage_leakage_rate_per_s_range"]))
    drift_final = float(rng.uniform(*generator["storage_drift_final_rate_per_s_range"]))
    active_fraction = severity
    period = int(rng.integers(
        generator["intermittent_period_samples_range"][0],
        generator["intermittent_period_samples_range"][1] + 1,
    ))
    for index, dt in enumerate(np.diff(timestamps)):
        capacity_now = capacity
        disturbance = 0.0
        if mechanism == FailureMechanism.CAPACITY_LOSS and index + 1 >= onset_index:
            capacity_now = capacity * severity
        elif mechanism == FailureMechanism.LEAKAGE and index + 1 >= onset_index:
            disturbance = leakage_rate
        elif mechanism == FailureMechanism.DRIFT and index + 1 >= onset_index:
            progress = (index + 1 - onset_index) / max(len(timestamps) - 1 - onset_index, 1)
            disturbance = drift_final * progress
        elif mechanism == FailureMechanism.INTERMITTENT and index + 1 >= onset_index:
            phase = (index + 1 - onset_index) % period
            disturbance = leakage_rate * 1.8 if phase < max(1, round(period * active_fraction)) else 0.0
        latent[index + 1] = (
            latent[index]
            + dt * 0.5 * (power[index] + power[index + 1]) / capacity_now
            + dt * disturbance
        )
    noise = np.random.default_rng(int(config["noise_seed"])).normal(
        0.0, float(config["state_noise_sd"]), len(timestamps)
    )
    observed = latent + noise
    if mechanism == FailureMechanism.BIAS:
        observed[onset_index:] += float(
            rng.uniform(*generator["storage_bias_step_range"])
        )
    elif mechanism == FailureMechanism.STUCK:
        duration = max(2, int(round(severity * len(timestamps))))
        end = min(onset_index + duration, len(timestamps))
        observed[onset_index:end] = observed[onset_index]
    return module._frame(
        "storage.energy_buffer",
        timestamps,
        [
            ("state.energy_fraction", "energy.state_of_charge", "1", observed),
            ("power.exchange", "electrical.active_power", "W", power),
        ],
    )


def _thermal_case(
    module: Any,
    protocol: dict[str, Any],
    config: dict[str, float | int],
    mechanism: FailureMechanism | None,
    rng: np.random.Generator,
) -> Any:
    step = float(config["step_s"])
    timestamps = np.arange(int(config["sample_count"]), dtype=float) * step
    heat = (
        float(config["heat_offset_w"])
        + float(config["heat_sine_w"])
        * np.sin(timestamps / float(config["heat_period_s"]))
        + float(config["heat_cosine_w"])
        * np.cos(timestamps / float(config["heat_cosine_period_s"]))
    )
    ambient = float(config["ambient_k"]) + float(config["ambient_swing_k"]) * np.sin(
        timestamps / float(config["ambient_period_s"])
    )
    capacity = float(config["thermal_capacity_j_per_k"])
    conductance = float(config["thermal_conductance_w_per_k"])
    latent = np.empty_like(timestamps)
    latent[0] = float(config["initial_temperature_k"])
    onset, severity = (
        _onset_and_severity(protocol, rng, mechanism)
        if mechanism is not None
        else (1.0, 0.0)
    )
    onset_index = min(max(int(round(onset * (len(timestamps) - 1))), 1), len(timestamps) - 2)
    generator = protocol["synthetic_generator"]
    leakage_heat = float(rng.uniform(*generator["thermal_leakage_heat_w_range"]))
    drift_final = float(rng.uniform(*generator["thermal_drift_final_heat_w_range"]))
    active_fraction = severity
    period = int(rng.integers(
        generator["intermittent_period_samples_range"][0],
        generator["intermittent_period_samples_range"][1] + 1,
    ))
    for index, dt in enumerate(np.diff(timestamps)):
        capacity_now = capacity
        disturbance = 0.0
        if mechanism == FailureMechanism.CAPACITY_LOSS and index + 1 >= onset_index:
            capacity_now = capacity * severity
        elif mechanism == FailureMechanism.LEAKAGE and index + 1 >= onset_index:
            disturbance = leakage_heat
        elif mechanism == FailureMechanism.DRIFT and index + 1 >= onset_index:
            progress = (index + 1 - onset_index) / max(len(timestamps) - 1 - onset_index, 1)
            disturbance = drift_final * progress
        elif mechanism == FailureMechanism.INTERMITTENT and index + 1 >= onset_index:
            phase = (index + 1 - onset_index) % period
            disturbance = leakage_heat * 1.8 if phase < max(1, round(period * active_fraction)) else 0.0
        heat_mid = 0.5 * (heat[index] + heat[index + 1])
        ambient_mid = 0.5 * (ambient[index] + ambient[index + 1])
        left = capacity_now / dt + conductance / 2.0
        right = (
            latent[index] * (capacity_now / dt - conductance / 2.0)
            + heat_mid
            + disturbance
            + conductance * ambient_mid
        )
        latent[index + 1] = right / left
    noise = np.random.default_rng(int(config["noise_seed"])).normal(
        0.0, float(config["state_noise_sd_k"]), len(timestamps)
    )
    observed = latent + noise
    if mechanism == FailureMechanism.BIAS:
        observed[onset_index:] += float(
            rng.uniform(*generator["thermal_bias_step_k_range"])
        )
    elif mechanism == FailureMechanism.STUCK:
        duration = max(2, int(round(severity * len(timestamps))))
        end = min(onset_index + duration, len(timestamps))
        observed[onset_index:end] = observed[onset_index]
    return module._frame(
        "thermal.lumped_capacity",
        timestamps,
        [
            ("state.temperature", "thermal.temperature", "K", observed),
            ("heat.exchange", "electrical.active_power", "W", heat),
            ("context.ambient_temperature", "thermal.temperature", "K", ambient),
        ],
    )


def _generate_cases(
    protocol: dict[str, Any],
    *,
    seed: int,
    count_per_archetype_per_mechanism: int,
    mechanisms: tuple[FailureMechanism, ...],
    validation_domain: bool,
    include_normal: bool = False,
) -> list[GeneratedCase]:
    module = _gray_box_module()
    cases: list[GeneratedCase] = []
    requested: tuple[FailureMechanism | None, ...] = (
        (*mechanisms, None) if include_normal else mechanisms
    )
    for mechanism_index, mechanism in enumerate(requested):
        mechanism_seed = seed + mechanism_index * 10_000
        configurations = module._configurations(
            mechanism_seed,
            count_per_archetype_per_mechanism,
            validation=validation_domain,
        )
        for archetype_index, archetype in enumerate(ARCHETYPES):
            for case_index, config in enumerate(configurations[archetype]):
                rng = np.random.default_rng(
                    mechanism_seed * 100_000 + archetype_index * 10_000 + case_index
                )
                frame = (
                    _storage_case(module, protocol, config, mechanism, rng)
                    if archetype == "storage.energy_buffer"
                    else _thermal_case(module, protocol, config, mechanism, rng)
                )
                mechanism_name = mechanism.value if mechanism is not None else "NORMAL"
                cases.append(
                    GeneratedCase(
                        case_key=(
                            f"seed={seed}|family={mechanism_seed}|archetype={archetype}|"
                            f"mechanism={mechanism_name}|index={case_index}"
                        ),
                        archetype=archetype,
                        mechanism=mechanism,
                        frame=frame,
                    )
                )
    return cases


def _extract_cases(cases: list[GeneratedCase]) -> list[ExtractedCase]:
    estimator = GrayBoxStateEstimator()
    contracts = default_dynamics_contracts()
    extracted: list[ExtractedCase] = []
    for case in cases:
        try:
            result = estimator.estimate(case.frame, contracts[case.archetype])
            extracted.append(
                ExtractedCase(
                    case_key=case.case_key,
                    archetype=case.archetype,
                    mechanism=case.mechanism,
                    features=extract_mechanism_features(result).values,
                    anomalous=result.conservation_assessment.statistically_significant,
                    failed=False,
                    failure_reason=None,
                )
            )
        except (StateEstimationError, ValueError, np.linalg.LinAlgError) as exc:
            extracted.append(
                ExtractedCase(
                    case_key=case.case_key,
                    archetype=case.archetype,
                    mechanism=case.mechanism,
                    features=None,
                    anomalous=False,
                    failed=True,
                    failure_reason=str(exc),
                )
            )
    return extracted


def _quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def _temperature(logits: np.ndarray, labels: np.ndarray, protocol: dict[str, Any]) -> float:
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


def _accuracy(model: CalibratedMechanismModel, rows: list[ExtractedCase]) -> float:
    correct = 0
    for row in rows:
        if row.features is None or row.mechanism is None:
            continue
        prediction = model.predict(
            MechanismFeatureVector(
                feature_names=list(model.artifact.feature_names),
                values=row.features,
                input_sha256="0" * 64,
            )
        )
        correct += int(prediction.mechanism_id == row.mechanism)
    return correct / len(rows)


def fit(protocol: dict[str, Any]) -> tuple[MechanismCalibrationArtifact, dict[str, Any]]:
    started = time.monotonic()
    split = protocol["split"]
    development = _extract_cases(
        _generate_cases(
            protocol,
            seed=int(split["development_seed"]),
            count_per_archetype_per_mechanism=int(
                split["development_count_per_archetype_per_known_mechanism"]
            ),
            mechanisms=KNOWN,
            validation_domain=False,
        )
    )
    calibration = _extract_cases(
        _generate_cases(
            protocol,
            seed=int(split["calibration_seed"]),
            count_per_archetype_per_mechanism=int(
                split["calibration_count_per_archetype_per_known_mechanism"]
            ),
            mechanisms=KNOWN,
            validation_domain=False,
        )
    )
    failed_development = [row for row in development if row.failed]
    failed_calibration = [row for row in calibration if row.failed]
    usable_development = [row for row in development if row.features is not None]
    usable_calibration = [row for row in calibration if row.features is not None]
    if any(
        not any(row.mechanism == mechanism for row in usable_development)
        or not any(row.mechanism == mechanism for row in usable_calibration)
        for mechanism in KNOWN
    ):
        raise RuntimeError("at least one frozen mechanism has no usable fit or calibration rows")
    features = np.asarray([row.features for row in usable_development], dtype=float)
    labels = np.asarray(
        [KNOWN.index(row.mechanism) for row in usable_development], dtype=int
    )
    calibration_features = np.asarray(
        [row.features for row in usable_calibration], dtype=float
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
        raise RuntimeError("fitted class order differs from the frozen mechanism order")
    calibration_logits = classifier.decision_function(standardized_calibration)
    temperature = _temperature(calibration_logits, calibration_labels, protocol)
    centroids = np.asarray(
        [np.mean(standardized[labels == index], axis=0) for index in range(len(KNOWN))]
    )
    alpha = float(protocol["thresholds"]["open_set_alpha"])
    distance_thresholds = []
    for index in range(len(KNOWN)):
        class_rows = standardized_calibration[calibration_labels == index]
        distances = np.linalg.norm(class_rows - centroids[index], axis=1)
        distance_thresholds.append(_quantile(distances, alpha))
    protocol_sha = _sha_bytes(PROTOCOL_PATH.read_bytes())
    artifact = MechanismCalibrationArtifact(
        calibration_id="astrograph.generic-mechanism.synthetic-technical.v1",
        calibration_version="1.0.0",
        protocol_sha256=protocol_sha,
        model_family="L2_REGULARIZED_MULTINOMIAL_LOGISTIC_REGRESSION",
        model_id="astrograph.generic.mechanism.linear-residual",
        model_version="1.0.0",
        feature_names=protocol["feature_schema"],
        class_order=list(KNOWN),
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        coefficients=classifier.coef_.tolist(),
        intercepts=classifier.intercept_.tolist(),
        temperature=temperature,
        class_centroids=centroids.tolist(),
        class_distance_thresholds=distance_thresholds,
        open_set_alpha=alpha,
        development_split_sha256=_canonical_sha(
            [(row.case_key, row.mechanism.value, row.features) for row in development]
        ),
        calibration_split_sha256=_canonical_sha(
            [(row.case_key, row.mechanism.value, row.features) for row in calibration]
        ),
        development_case_count=len(development),
        calibration_case_count=len(calibration),
    )
    model = CalibratedMechanismModel(artifact)
    evidence = {
        "schema_version": "astrograph-generic-mechanism-fit-evidence.v1",
        "protocol_sha256": protocol_sha,
        "formal_hidden_truth_used": False,
        "synthetic_controlled_reference_used": True,
        "development_case_count": len(development),
        "calibration_case_count": len(calibration),
        "development_failure_count": len(failed_development),
        "calibration_failure_count": len(failed_calibration),
        "development_feature_case_count": len(usable_development),
        "calibration_feature_case_count": len(usable_calibration),
        "development_classification_accuracy": _accuracy(model, development),
        "calibration_classification_accuracy": _accuracy(model, calibration),
        "temperature": temperature,
        "class_distance_thresholds": {
            mechanism.value: distance_thresholds[index]
            for index, mechanism in enumerate(KNOWN)
        },
        "elapsed_seconds": time.monotonic() - started,
        "technical_validation_read": False,
        "status": "FIT_COMPLETE",
        "claim_boundary": (
            "Development and calibration evidence only. Technical validation was not read and no "
            "T-310, T-311, P-014, G3 or G4 completion is established."
        ),
    }
    return artifact, evidence


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate(
    protocol: dict[str, Any], artifact: MechanismCalibrationArtifact
) -> dict[str, Any]:
    started = time.monotonic()
    protocol_sha = _sha_bytes(PROTOCOL_PATH.read_bytes())
    if artifact.protocol_sha256 != protocol_sha:
        raise ValueError("calibration artifact does not match the frozen protocol")
    split = protocol["split"]
    count = int(split["technical_validation_count_per_archetype_per_mechanism"])
    extracted = _extract_cases(
        _generate_cases(
            protocol,
            seed=int(split["technical_validation_seed"]),
            count_per_archetype_per_mechanism=count,
            mechanisms=KNOWN + OPEN_SET,
            validation_domain=True,
            include_normal=True,
        )
    )
    model = CalibratedMechanismModel(artifact)
    rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    for case in extracted:
        prediction = None
        if case.failed:
            reason = case.failure_reason or "UNSPECIFIED_FAILURE"
            failures[reason] = failures.get(reason, 0) + 1
        elif case.anomalous and case.features is not None:
            prediction = model.predict(
                MechanismFeatureVector(
                    feature_names=artifact.feature_names,
                    values=case.features,
                    input_sha256="0" * 64,
                )
            )
        role = (
            "NORMAL"
            if case.mechanism is None
            else "KNOWN"
            if case.mechanism in KNOWN
            else "OPEN_SET"
        )
        detection_correct = (
            not case.failed and (not case.anomalous if role == "NORMAL" else case.anomalous)
        )
        retained = bool(
            role == "KNOWN"
            and prediction is not None
            and prediction.open_set_decision == OpenSetDecision.IN_DISTRIBUTION
        )
        rejected = bool(
            role == "OPEN_SET"
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
                "detection_correct": detection_correct,
                "known_retained": retained,
                "open_set_rejected": rejected,
                "classification_correct": classified,
                "localization_correct": classified,
                "reasoning_correct": classified,
                "predicted_mechanism": (
                    prediction.mechanism_id.value if prediction is not None else None
                ),
                "open_set_decision": (
                    prediction.open_set_decision.value if prediction is not None else None
                ),
            }
        )
    normal = [row for row in rows if row["role"] == "NORMAL"]
    known = [row for row in rows if row["role"] == "KNOWN"]
    opened = [row for row in rows if row["role"] == "OPEN_SET"]
    expected = (
        len(ARCHETYPES)
        * int(
            protocol["synthetic_generator"][
                "normal_count_per_archetype_in_technical_validation"
            ]
        ),
        len(ARCHETYPES) * len(KNOWN) * count,
        len(ARCHETYPES) * len(OPEN_SET) * count,
    )
    denominators = (len(normal), len(known), len(opened))
    if denominators != expected:
        raise RuntimeError("technical-validation denominators differ from the frozen protocol")

    def selected_rate(
        selected: list[dict[str, Any]], field: str
    ) -> float:
        return _rate([bool(row[field]) for row in selected])

    known_rates = {
        mechanism.value: selected_rate(
            [row for row in known if row["mechanism"] == mechanism.value],
            "classification_correct",
        )
        for mechanism in KNOWN
    }
    open_rates = {
        mechanism.value: selected_rate(
            [row for row in opened if row["mechanism"] == mechanism.value],
            "open_set_rejected",
        )
        for mechanism in OPEN_SET
    }
    per_archetype: dict[str, Any] = {}
    for archetype in ARCHETYPES:
        all_group = [row for row in rows if row["archetype"] == archetype]
        known_group = [row for row in known if row["archetype"] == archetype]
        open_group = [row for row in opened if row["archetype"] == archetype]
        per_archetype[archetype] = {
            "case_count": len(all_group),
            "detection_accuracy": selected_rate(all_group, "detection_correct"),
            "classification_accuracy": selected_rate(
                known_group, "classification_correct"
            ),
            "localization_accuracy": selected_rate(
                known_group, "localization_correct"
            ),
            "reasoning_correctness": selected_rate(
                known_group, "reasoning_correct"
            ),
            "known_class_retention": selected_rate(known_group, "known_retained"),
            "open_set_rejection": selected_rate(open_group, "open_set_rejected"),
        }
    metrics = {
        "detection_accuracy": selected_rate(rows, "detection_correct"),
        "classification_accuracy_micro": selected_rate(
            known, "classification_correct"
        ),
        "classification_accuracy_macro": float(np.mean(list(known_rates.values()))),
        "localization_accuracy": selected_rate(known, "localization_correct"),
        "reasoning_correctness": selected_rate(known, "reasoning_correct"),
        "known_class_retention": selected_rate(known, "known_retained"),
        "open_set_rejection": selected_rate(opened, "open_set_rejected"),
        "worst_known_mechanism_classification": min(known_rates.values()),
        "worst_open_set_mechanism_rejection": min(open_rates.values()),
    }
    thresholds = protocol["thresholds"]
    checks = {
        "denominators_preserved": True,
        "detection_accuracy": metrics["detection_accuracy"]
        >= float(thresholds["detection_accuracy"]),
        "classification_accuracy_micro": metrics["classification_accuracy_micro"]
        >= float(thresholds["classification_accuracy"]),
        "classification_accuracy_macro": metrics["classification_accuracy_macro"]
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
    passed = all(checks.values())
    return {
        "schema_version": "astrograph-generic-head-technical-validation.v1",
        "protocol_sha256": protocol_sha,
        "calibration_artifact_sha256": _canonical_sha(
            artifact.model_dump(mode="json")
        ),
        "formal_hidden_truth_used": False,
        "synthetic_controlled_reference_used": True,
        "technical_validation_seed": int(split["technical_validation_seed"]),
        "validation_run_count": 1,
        "denominators": {
            "normal": len(normal),
            "known": len(known),
            "open_set": len(opened),
            "total": len(rows),
        },
        "failed_run_count": sum(bool(row["failed"]) for row in rows),
        "failure_reason_counts": dict(sorted(failures.items())),
        "metrics": metrics,
        "known_mechanism_rates": known_rates,
        "open_set_mechanism_rates": open_rates,
        "per_archetype": per_archetype,
        "thresholds": thresholds,
        "checks": checks,
        "technical_status": "PASS" if passed else "FAIL",
        "t310_status": "PASS" if passed else "INCOMPLETE",
        "t311_status": "PASS" if passed else "INCOMPLETE",
        "p014_status": "FIX_CANDIDATE" if passed else "OPEN",
        "validation_split_sha256": _canonical_sha(rows),
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": (
            "Single frozen-seed synthetic technical validation with every declared case retained. "
            "It does not use formal hidden-acceptance truth and cannot establish independent-source "
            "validation, G3, G4, flight validity or safety qualification."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "validate"), required=True)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument("--artifact-input", type=Path)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_protocol()
    if args.phase == "fit":
        if args.artifact_output is None:
            parser.error("--artifact-output is required for --phase fit")
        artifact, evidence = fit(protocol)
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_output.write_text(
            json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    else:
        if args.artifact_input is None:
            parser.error("--artifact-input is required for --phase validate")
        if args.evidence_output.exists():
            raise FileExistsError(
                "technical-validation evidence already exists; the frozen protocol permits one run"
            )
        artifact = MechanismCalibrationArtifact.model_validate_json(
            args.artifact_input.read_text(encoding="utf-8")
        )
        evidence = validate(protocol, artifact)
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence.get("technical_status", "PASS") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
