#!/usr/bin/env python3
"""Run the Timer-authorized P-017 thermal observability intervention study."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from astrograph.diagnosis.generic import FailureMechanism
from astrograph.diagnosis.generic.segmented_calibration import _physical_arrays
from astrograph.physics import default_dynamics_contracts


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/diagnosis/generic_observability_intervention_protocol.v1.json"
STAGE_A_V1 = ROOT / "scripts/acceptance/research_generic_diagnosis_stage_a.py"
STAGE_A_V3 = ROOT / "scripts/acceptance/research_generic_diagnosis_stage_a_v3.py"
CLASSES = ("NORMAL", "CAPACITY_LOSS", "DRIFT", "LEAKAGE")
KNOWN = (
    FailureMechanism.CAPACITY_LOSS,
    FailureMechanism.DRIFT,
    FailureMechanism.LEAKAGE,
)
ARCHETYPES = ("storage.energy_buffer", "thermal.lumped_capacity")
ARMS = ("PASSIVE_SHORT", "PASSIVE_LONG", "ACTIVE_MULTISCALE_LONG")


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
    authorization = ROOT / protocol["authorization"]["path"]
    attestation = json.loads(authorization.read_text(encoding="utf-8"))
    if attestation["status"] != protocol["authorization"]["status"]:
        raise PermissionError("observability intervention is not authorized")
    if attestation["authorized_seeds"] != {
        "development": protocol["development_seed"],
        "calibration": protocol["calibration_seed"],
        "technical_validation": None,
    }:
        raise PermissionError("authorized seeds differ from the frozen protocol")
    if protocol["technical_validation_seed"] is not None or protocol["stage_b_authorized"]:
        raise PermissionError("protocol exceeds Timer authority")
    exhausted = set(protocol["permanently_exhausted_validation_seeds"])
    if {protocol["development_seed"], protocol["calibration_seed"]} & exhausted:
        raise PermissionError("research reuses an exhausted validation seed")
    for item in protocol["retained_stage_a_failures"]:
        path = ROOT / item["path"]
        if _sha(path) != item["sha256"] or item["status"] != "FAIL":
            raise ValueError(f"retained Stage-A failure changed: {item['path']}")
    for method in ("primary_switching_physics_method", "nonlinear_separability_ceiling"):
        item = protocol[method]
        if _sha(ROOT / item["frozen_implementation_reference"]) != item["implementation_sha256"]:
            raise ValueError(f"frozen research method changed: {method}")
    stage_a = _module("observability_stage_a_base", STAGE_A_V1)
    physics = _module("observability_stage_a_physics", STAGE_A_V3)
    stage_protocol, generator_protocol = stage_a._load_protocols()
    candidate1, candidate2 = stage_a._dependencies()
    return protocol, stage_protocol, generator_protocol, stage_a, physics, candidate1, candidate2


def _square(time_s: np.ndarray, dwell_s: float, phase: float = 0.0) -> np.ndarray:
    return np.where(np.sin(np.pi * time_s / dwell_s + phase) >= 0.0, 1.0, -1.0)


def _thermal_case(gray_box, candidate1, generator, config, mechanism, rng, arm, horizon_s):
    step = float(config["step_s"])
    count = int(config["sample_count"]) if arm == "PASSIVE_SHORT" else int(math.floor(horizon_s / step)) + 1
    timestamps = np.arange(count, dtype=float) * step
    passive = (
        float(config["heat_offset_w"])
        + float(config["heat_sine_w"]) * np.sin(timestamps / float(config["heat_period_s"]))
        + float(config["heat_cosine_w"]) * np.cos(timestamps / float(config["heat_cosine_period_s"]))
    )
    target_rms = math.sqrt(
        (float(config["heat_sine_w"]) ** 2 + float(config["heat_cosine_w"]) ** 2) / 2.0
    )
    if arm == "ACTIVE_MULTISCALE_LONG":
        excitation = (
            _square(timestamps, 90.0)
            + _square(timestamps, 360.0, math.pi / 4.0)
        ) / math.sqrt(2.0)
        heat = float(config["heat_offset_w"]) + target_rms * excitation
    else:
        heat = passive
    ambient = float(config["ambient_k"]) + float(config["ambient_swing_k"]) * np.sin(
        timestamps / float(config["ambient_period_s"])
    )
    capacity = float(config["thermal_capacity_j_per_k"])
    conductance = float(config["thermal_conductance_w_per_k"])
    latent = np.empty_like(timestamps)
    latent[0] = float(config["initial_temperature_k"])
    onset, severity = (
        candidate1._onset_and_severity(generator, rng, mechanism)
        if mechanism is not None
        else (1.0, 0.0)
    )
    onset_index = min(max(int(round(onset * (len(timestamps) - 1))), 1), len(timestamps) - 2)
    synthetic = generator["synthetic_generator"]
    leakage_heat = float(rng.uniform(*synthetic["thermal_leakage_heat_w_range"]))
    drift_final = float(rng.uniform(*synthetic["thermal_drift_final_heat_w_range"]))
    period = int(rng.integers(
        synthetic["intermittent_period_samples_range"][0],
        synthetic["intermittent_period_samples_range"][1] + 1,
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
        heat_mid = 0.5 * (heat[index] + heat[index + 1])
        ambient_mid = 0.5 * (ambient[index] + ambient[index + 1])
        left = capacity_now / dt + conductance / 2.0
        right = (
            latent[index] * (capacity_now / dt - conductance / 2.0)
            + heat_mid + disturbance + conductance * ambient_mid
        )
        latent[index + 1] = right / left
    noise = np.random.default_rng(int(config["noise_seed"])).normal(
        0.0, float(config["state_noise_sd_k"]), len(timestamps)
    )
    observed = latent + noise
    frame = gray_box._frame(
        "thermal.lumped_capacity",
        timestamps,
        [
            ("state.temperature", "thermal.temperature", "K", observed),
            ("heat.exchange", "electrical.active_power", "W", heat),
            ("context.ambient_temperature", "thermal.temperature", "K", ambient),
        ],
    )
    metadata = {
        "onset_fraction": onset,
        "severity": severity,
        "leakage_heat_w": leakage_heat,
        "drift_final_heat_w": drift_final,
        "unused_intermittent_period": period,
        "time_constant_seconds": capacity / conductance,
        "horizon_seconds": float(timestamps[-1]),
        "post_fault_time_constant_coverage": (
            float(timestamps[-1]) * (1.0 - onset) / (capacity / conductance)
            if mechanism is not None else None
        ),
        "heat_centered_rms_ratio": float(np.std(heat)) / target_rms,
        "minimum_heat_w": float(np.min(heat)),
    }
    return frame, metadata


def _generate_arm(protocol, stage_protocol, generator, candidate1, seed, count, arm):
    cases = []
    thermal_metadata = []
    pair_identity = []
    gray_box = candidate1._gray_box_module()
    horizon = float(protocol["thermal_arms"][1]["observation_horizon_seconds"])
    requested = (None, *KNOWN)
    for class_index, mechanism in enumerate(requested):
        family_seed = seed + class_index * 10_000
        configurations = _module_configurations(stage_protocol, family_seed, count)
        for case_index, config in enumerate(configurations[ARCHETYPES[0]]):
            rng = np.random.default_rng(family_seed * 100_000 + case_index)
            frame = candidate1._storage_case(gray_box, generator, config, mechanism, rng)
            label = "NORMAL" if mechanism is None else mechanism.value
            cases.append(candidate1.GeneratedCase(
                case_key=f"p017-observability|seed={seed}|arm=STORAGE_CONTROL|class={label}|index={case_index}",
                archetype=ARCHETYPES[0], mechanism=mechanism, frame=frame,
            ))
        for case_index, config in enumerate(configurations[ARCHETYPES[1]]):
            rng = np.random.default_rng(family_seed * 100_000 + 10_000 + case_index)
            frame, metadata = _thermal_case(
                gray_box, candidate1, generator, config, mechanism, rng, arm, horizon
            )
            label = "NORMAL" if mechanism is None else mechanism.value
            cases.append(candidate1.GeneratedCase(
                case_key=f"p017-observability|seed={seed}|arm={arm}|class={label}|index={case_index}",
                archetype=ARCHETYPES[1], mechanism=mechanism, frame=frame,
            ))
            thermal_metadata.append(metadata)
            pair_identity.append({
                "family_seed": family_seed,
                "case_index": case_index,
                "class": label,
                "configuration": config,
                "onset_fraction": metadata["onset_fraction"],
                "severity": metadata["severity"],
                "leakage_heat_w": metadata["leakage_heat_w"],
                "drift_final_heat_w": metadata["drift_final_heat_w"],
            })
    return cases, thermal_metadata, _canonical(pair_identity)


def _module_configurations(stage_protocol, seed, count):
    # Kept separate so the frozen Stage-A configuration generator is the only randomizer.
    stage_a = sys.modules["observability_stage_a_base"]
    return stage_a._configurations(stage_protocol, seed, count)


def _compact_physics(cases, physics, fractions, floor):
    contracts = default_dynamics_contracts()
    rows, failures, representatives = [], [], {}
    design_metrics = []
    for case in cases:
        try:
            _, derivative, design = _physical_arrays(case.frame, contracts[case.archetype])
            label = "NORMAL" if case.mechanism is None else case.mechanism.value
            rows.append({
                "archetype": case.archetype,
                "label": label,
                "hypotheses": physics._hypotheses(derivative, design, fractions, floor),
            })
            representatives.setdefault(case.archetype, (derivative.copy(), design.copy()))
            if case.archetype == ARCHETYPES[1]:
                singular = np.linalg.svd(design, compute_uv=False)
                design_metrics.append({
                    "condition_number": float(singular[0] / max(singular[-1], 1.0e-15)),
                    "minimum_normalized_singular_value": float(singular[-1] / math.sqrt(len(design))),
                    "column_correlation": float(np.corrcoef(design[:, 0], design[:, 1])[0, 1]),
                })
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures.append({"case_key": case.case_key, "error": str(exc)})
    return rows, failures, representatives, design_metrics


def _summarize(values):
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _process_split(protocol, stage_protocol, generator, stage_a, physics, candidate1, candidate2, seed, count, arm):
    cases, metadata, identity = _generate_arm(
        protocol, stage_protocol, generator, candidate1, seed, count, arm
    )
    features, feature_failures = stage_a._extract(candidate2, cases)
    primary = protocol["primary_switching_physics_method"]
    rows, physics_failures, representatives, design = _compact_physics(
        cases, physics, primary["change_point_grid"], 1.0e-18
    )
    meta_summary = {
        "time_constant_seconds": _summarize([item["time_constant_seconds"] for item in metadata]),
        "horizon_seconds": _summarize([item["horizon_seconds"] for item in metadata]),
        "post_fault_time_constant_coverage": _summarize([
            item["post_fault_time_constant_coverage"] for item in metadata
            if item["post_fault_time_constant_coverage"] is not None
        ]),
        "heat_centered_rms_ratio": _summarize([item["heat_centered_rms_ratio"] for item in metadata]),
        "minimum_heat_w": _summarize([item["minimum_heat_w"] for item in metadata]),
        "design_condition_number": _summarize([item["condition_number"] for item in design]),
        "design_minimum_normalized_singular_value": _summarize([
            item["minimum_normalized_singular_value"] for item in design
        ]),
        "design_absolute_column_correlation": _summarize([
            abs(item["column_correlation"]) for item in design
        ]),
    }
    count_generated = len(cases)
    del cases, metadata, design
    gc.collect()
    return {
        "features": features,
        "feature_failures": feature_failures,
        "physics": rows,
        "physics_failures": physics_failures,
        "representatives": representatives,
        "paired_identity_sha256": identity,
        "case_count": count_generated,
        "audit": meta_summary,
    }


def _oracle_metrics(rows, predicted):
    labels = np.asarray([row.label for row in rows], dtype=int)
    known = labels != 0
    normal = labels == 0
    metrics = {
        "known_classification_accuracy": float(np.mean(predicted[known] == labels[known])),
        "normal_detection_accuracy": float(np.mean(predicted[normal] == 0)),
    }
    for archetype in ARCHETYPES:
        mask = np.asarray([row.archetype == archetype for row in rows])
        metrics[archetype] = {
            "known_classification_accuracy": float(np.mean(predicted[mask & known] == labels[mask & known])),
            "normal_detection_accuracy": float(np.mean(predicted[mask & normal] == 0)),
        }
    return metrics


def _oracle(dev, cal, protocol):
    indicators = {ARCHETYPES[0]: 0.0, ARCHETYPES[1]: 1.0}
    matrix = lambda rows: np.asarray([
        np.concatenate([row.values, [indicators[row.archetype]]]) for row in rows
    ], dtype=float)
    x_dev, x_cal = matrix(dev), matrix(cal)
    y_dev = np.asarray([row.label for row in dev], dtype=int)
    cfg = protocol["nonlinear_separability_ceiling"]
    model = ExtraTreesClassifier(
        n_estimators=cfg["n_estimators"], criterion=cfg["criterion"],
        max_features=cfg["max_features"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"], bootstrap=cfg["bootstrap"],
        class_weight=cfg["class_weight"], random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    ).fit(x_dev, y_dev)
    dev_predicted = model.predict(x_dev)
    cal_predicted = model.predict(x_cal)
    cal_labels = np.asarray([row.label for row in cal], dtype=int)
    return {
        "development": _oracle_metrics(dev, dev_predicted),
        "calibration": _oracle_metrics(cal, cal_predicted),
        "calibration_confusion_matrix": confusion_matrix(
            cal_labels, cal_predicted, labels=[0, 1, 2, 3]
        ).tolist(),
    }


def _physics_metrics(rows, thresholds):
    outcomes = []
    for row in rows:
        prediction_set = [
            name for name in CLASSES
            if row["hypotheses"][name]["adequacy"] <= thresholds[(row["archetype"], name)]
        ]
        predicted = min(prediction_set, key=lambda name: row["hypotheses"][name]["bic"]) if prediction_set else None
        known = row["label"] != "NORMAL"
        outcomes.append({
            "archetype": row["archetype"], "label": row["label"], "predicted": predicted,
            "ood": not prediction_set,
            "classification_correct": bool(known and predicted == row["label"]),
            "known_retained": bool(known and prediction_set and predicted != "NORMAL"),
            "normal_correct": bool(not known and predicted == "NORMAL"),
        })
    rate = lambda values: float(sum(values) / len(values))
    known = [item for item in outcomes if item["label"] != "NORMAL"]
    normal = [item for item in outcomes if item["label"] == "NORMAL"]
    metrics = {
        "known_classification_accuracy": rate([item["classification_correct"] for item in known]),
        "known_retention": rate([item["known_retained"] for item in known]),
        "normal_detection_accuracy": rate([item["normal_correct"] for item in normal]),
    }
    for archetype in ARCHETYPES:
        ak = [item for item in known if item["archetype"] == archetype]
        an = [item for item in normal if item["archetype"] == archetype]
        metrics[archetype] = {
            "known_classification_accuracy": rate([item["classification_correct"] for item in ak]),
            "known_retention": rate([item["known_retained"] for item in ak]),
            "normal_detection_accuracy": rate([item["normal_correct"] for item in an]),
        }
    return metrics


def _physics(dev, cal, representatives, physics, protocol):
    alpha = float(protocol["primary_switching_physics_method"]["conformal_alpha"])
    thresholds = {}
    for archetype in ARCHETYPES:
        for name in CLASSES:
            values = [
                row["hypotheses"][name]["adequacy"] for row in dev
                if row["archetype"] == archetype and row["label"] == name
            ]
            thresholds[(archetype, name)] = physics._quantile(values, alpha)
    reachable = {}
    fractions = protocol["primary_switching_physics_method"]["change_point_grid"]
    for archetype, (derivative, design) in representatives.items():
        alternating = np.where(np.arange(len(derivative)) % 2 == 0, 1.0, -1.0)
        stressed = derivative + 1000.0 * alternating
        scores = physics._hypotheses(stressed, design, fractions, 1.0e-18)
        reachable[archetype] = all(
            scores[name]["adequacy"] > thresholds[(archetype, name)] for name in CLASSES
        )
    return {
        "development": _physics_metrics(dev, thresholds),
        "calibration": _physics_metrics(cal, thresholds),
        "open_set_rule_reachable": reachable,
        "same_score_semantics": True,
        "thresholds_sha256": _canonical({f"{a}|{c}": v for (a, c), v in thresholds.items()}),
    }


def _numerical_verification(protocol):
    cfg = protocol["simulation_verification"]
    p = cfg["reference_parameters"]
    capacity, conductance = p["thermal_capacity_j_per_k"], p["thermal_conductance_w_per_k"]
    equilibrium = p["ambient_temperature_k"] + p["constant_heat_w"] / conductance
    horizon = cfg["deterministic_convergence_horizon_seconds"]
    exact = equilibrium + (p["initial_temperature_k"] - equilibrium) * math.exp(
        -conductance * horizon / capacity
    )
    values = []
    for dt in cfg["deterministic_convergence_steps_seconds"]:
        state = p["initial_temperature_k"]
        for _ in range(int(round(horizon / dt))):
            left = capacity / dt + conductance / 2.0
            right = state * (capacity / dt - conductance / 2.0) + p["constant_heat_w"] + conductance * p["ambient_temperature_k"]
            state = right / left
        values.append(state)
    coarse, medium, fine = values
    order = math.log(abs((coarse - medium) / (medium - fine))) / math.log(2.0)
    fine_relative_error = abs((fine - exact) / exact)
    thresholds = protocol["root_cause_decision_thresholds"]
    return {
        "steps_seconds_coarse_to_fine": cfg["deterministic_convergence_steps_seconds"],
        "final_temperature_k_coarse_to_fine": values,
        "analytic_final_temperature_k": exact,
        "observed_order": order,
        "fine_grid_relative_error": fine_relative_error,
        "gci_fine": cfg["gci_safety_factor"] * abs((fine - medium) / fine) / (2.0 ** order - 1.0),
        "pass": order >= thresholds["numerical_observed_order_minimum"] and fine_relative_error <= thresholds["numerical_fine_grid_relative_error_maximum"],
    }


def _checks(primary, failures, elapsed, protocol):
    cfg = protocol["exit_checks"]
    cal = primary["calibration"]
    return {
        "known_classification_accuracy": cal["known_classification_accuracy"] >= cfg["known_classification_accuracy"],
        "normal_detection_accuracy": cal["normal_detection_accuracy"] >= cfg["normal_detection_accuracy"],
        "each_archetype_known_classification_accuracy": all(
            cal[a]["known_classification_accuracy"] >= cfg["each_archetype_known_classification_accuracy"] for a in ARCHETYPES
        ),
        "known_retention": cal["known_retention"] >= cfg["known_retention"],
        "open_set_rule_reachable_for_each_archetype": all(primary["open_set_rule_reachable"].values()),
        "failed_case_count": failures == 0,
        "runtime_budget": elapsed <= cfg["maximum_runtime_seconds"],
    }


def _attribution(arms, numerical, protocol):
    threshold = protocol["root_cause_decision_thresholds"]["material_accuracy_improvement_absolute"]
    gap_threshold = protocol["root_cause_decision_thresholds"]["material_generalization_gap_absolute"]
    def thermal(method, arm, split="calibration"):
        return arms[arm][method][split][ARCHETYPES[1]]["known_classification_accuracy"]
    deltas = {}
    for method in ("switching_physics", "nonlinear_ceiling"):
        deltas[method] = {
            "horizon_delta": thermal(method, "PASSIVE_LONG") - thermal(method, "PASSIVE_SHORT"),
            "active_excitation_delta": thermal(method, "ACTIVE_MULTISCALE_LONG") - thermal(method, "PASSIVE_LONG"),
            "active_development_calibration_gap": thermal(method, "ACTIVE_MULTISCALE_LONG", "development") - thermal(method, "ACTIVE_MULTISCALE_LONG"),
        }
    findings = []
    findings.append({
        "factor": "simulation_numerical_implementation",
        "classification": "NOT_SUPPORTED_AS_PRIMARY_CAUSE" if numerical["pass"] else "BLOCKING_CAUSE",
        "evidence": "deterministic Crank-Nicolson convergence",
    })
    findings.append({
        "factor": "observation_horizon",
        "classification": "CAUSAL" if max(item["horizon_delta"] for item in deltas.values()) >= threshold else "NOT_MATERIAL_AT_FROZEN_DELTA",
        "evidence": deltas,
    })
    findings.append({
        "factor": "active_excitation",
        "classification": "CAUSAL" if max(item["active_excitation_delta"] for item in deltas.values()) >= threshold else "NOT_MATERIAL_AT_FROZEN_DELTA",
        "evidence": deltas,
    })
    findings.append({
        "factor": "cross_configuration_generalization",
        "classification": "MATERIAL" if max(item["active_development_calibration_gap"] for item in deltas.values()) > gap_threshold else "NOT_PRIMARY",
        "evidence": deltas,
    })
    active_oracle = arms["ACTIVE_MULTISCALE_LONG"]["nonlinear_ceiling"]["calibration"][ARCHETYPES[1]]["known_classification_accuracy"]
    active_physics = arms["ACTIVE_MULTISCALE_LONG"]["switching_physics"]["calibration"][ARCHETYPES[1]]["known_classification_accuracy"]
    findings.append({
        "factor": "current_governed_diagnosis_method",
        "classification": "INSUFFICIENT_WHILE_REPRESENTATION_IS_SEPARABLE" if active_oracle >= 0.9 and active_physics < 0.9 else "NOT_ISOLATED",
        "evidence": {"active_nonlinear_ceiling": active_oracle, "active_switching_physics": active_physics},
    })
    findings.append({
        "factor": "simulator_external_validity",
        "classification": "INSUFFICIENT_EVIDENCE",
        "evidence": "Synthetic equations and ranges are internally auditable, but no independent real or higher-fidelity thermal reference is introduced by this authorization.",
    })
    return {"frozen_deltas": deltas, "findings": findings}


def run(output: Path):
    started = time.monotonic()
    protocol, stage_protocol, generator, stage_a, physics, candidate1, candidate2 = _load()
    arm_results = {}
    identity_hashes = {"development": {}, "calibration": {}}
    failure_count = 0
    for arm in ARMS:
        dev = _process_split(
            protocol, stage_protocol, generator, stage_a, physics, candidate1, candidate2,
            protocol["development_seed"], protocol["case_counts_per_arm"]["development_per_archetype_per_class"], arm,
        )
        cal = _process_split(
            protocol, stage_protocol, generator, stage_a, physics, candidate1, candidate2,
            protocol["calibration_seed"], protocol["case_counts_per_arm"]["calibration_per_archetype_per_class"], arm,
        )
        identity_hashes["development"][arm] = dev["paired_identity_sha256"]
        identity_hashes["calibration"][arm] = cal["paired_identity_sha256"]
        failure_count += len(dev["feature_failures"]) + len(cal["feature_failures"])
        failure_count += len(dev["physics_failures"]) + len(cal["physics_failures"])
        oracle = _oracle(dev["features"], cal["features"], protocol)
        primary = _physics(dev["physics"], cal["physics"], dev["representatives"], physics, protocol)
        arm_results[arm] = {
            "development_case_count": dev["case_count"],
            "calibration_case_count": cal["case_count"],
            "development_audit": dev["audit"],
            "calibration_audit": cal["audit"],
            "nonlinear_ceiling": oracle,
            "switching_physics": primary,
        }
        del dev, cal
        gc.collect()
    paired_controls_pass = all(
        len(set(identity_hashes[split].values())) == 1 for split in ("development", "calibration")
    )
    numerical = _numerical_verification(protocol)
    elapsed = time.monotonic() - started
    for arm in ARMS:
        arm_results[arm]["primary_exit_checks"] = _checks(
            arm_results[arm]["switching_physics"], failure_count, elapsed, protocol
        )
        arm_results[arm]["primary_exit_pass"] = all(arm_results[arm]["primary_exit_checks"].values())
    rms_tolerance = protocol["root_cause_decision_thresholds"]["rms_match_relative_tolerance"]
    rms_match_pass = abs(
        arm_results["ACTIVE_MULTISCALE_LONG"]["calibration_audit"]["heat_centered_rms_ratio"]["median"]
        - arm_results["PASSIVE_LONG"]["calibration_audit"]["heat_centered_rms_ratio"]["median"]
    ) <= rms_tolerance
    attribution = _attribution(arm_results, numerical, protocol)
    active_exit = arm_results["ACTIVE_MULTISCALE_LONG"]["primary_exit_pass"]
    evidence = {
        "schema_version": "astrograph-p017-observability-intervention-evidence.v1",
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "authorization_sha256": _sha(ROOT / protocol["authorization"]["path"]),
        "development_seed": protocol["development_seed"],
        "calibration_seed": protocol["calibration_seed"],
        "technical_validation_seed": None,
        "exhausted_validation_seeds_used": [],
        "formal_open_set_classes_generated": [],
        "production_candidate_fitted": False,
        "stage_b_started": False,
        "paired_control_identity_sha256": identity_hashes,
        "paired_controls_pass": paired_controls_pass,
        "rms_match_pass": rms_match_pass,
        "failed_case_count": failure_count,
        "simulation_numerical_verification": numerical,
        "arms": arm_results,
        "root_cause_attribution": attribution,
        "observability_intervention_status": "PASS" if active_exit and paired_controls_pass and rms_match_pass and numerical["pass"] else "FAIL",
        "stage_b_proposal_eligible": bool(active_exit and paired_controls_pass and rms_match_pass and numerical["pass"]),
        "stage_b_authorized": False,
        "elapsed_seconds": elapsed,
        "runtime_budget_seconds": protocol["exit_checks"]["maximum_runtime_seconds"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "governance/evidence/0.7.1-extension/generic_thermal_observability_intervention.v1.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise PermissionError("observability intervention evidence already exists")
    evidence = run(args.output)
    print(json.dumps({
        "status": evidence["observability_intervention_status"],
        "stage_b_proposal_eligible": evidence["stage_b_proposal_eligible"],
        "elapsed_seconds": evidence["elapsed_seconds"],
        "failed_case_count": evidence["failed_case_count"],
        "arms": {
            arm: {
                "oracle": result["nonlinear_ceiling"]["calibration"],
                "physics": result["switching_physics"]["calibration"],
                "exit_pass": result["primary_exit_pass"],
            } for arm, result in evidence["arms"].items()
        },
        "root_causes": evidence["root_cause_attribution"],
    }, ensure_ascii=False, indent=2))
    return 0 if evidence["observability_intervention_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
