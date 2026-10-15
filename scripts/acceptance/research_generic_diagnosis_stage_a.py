#!/usr/bin/env python3
"""Run the Timer-authorized P-017 Stage-A identifiability experiment.

This script is development/calibration research only. It deliberately has no validation
subcommand, no validation seed, and no production-artifact output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from astrograph.diagnosis.generic import FailureMechanism


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/diagnosis/generic_head_research_protocol.stage_a.v1.json"
BASE_PROTOCOL_PATH = ROOT / "configs/diagnosis/generic_head_calibration_protocol.v2.json"
CLASSES = ("NORMAL", "CAPACITY_LOSS", "DRIFT", "LEAKAGE")
KNOWN = (
    FailureMechanism.CAPACITY_LOSS,
    FailureMechanism.DRIFT,
    FailureMechanism.LEAKAGE,
)
ARCHETYPES = ("storage.energy_buffer", "thermal.lumped_capacity")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen research dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dependencies():
    candidate2 = _load_module(
        "stage_a_candidate2_frozen_generator",
        ROOT / "scripts/acceptance/calibrate_segmented_generic_diagnosis.py",
    )
    candidate1 = candidate2._candidate_one_module()
    return candidate1, candidate2


def _load_protocols() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    authorization = ROOT / protocol["authorization"]["path"]
    if _sha(authorization) != protocol["authorization"]["sha256"]:
        raise ValueError("Stage-A Timer authorization changed")
    attestation = json.loads(authorization.read_text(encoding="utf-8"))
    if attestation["status"] != "APPROVED_STAGE_A_ONLY":
        raise PermissionError("Stage A is not approved")
    if protocol["technical_validation_seed"] is not None:
        raise PermissionError("Stage-A protocol unexpectedly contains a validation seed")
    if protocol["production_candidate_fit"] or protocol["technical_validation_authorized"]:
        raise PermissionError("Stage-A protocol exceeds the Timer authorization")
    exhausted = set(protocol["permanently_exhausted_validation_seeds"])
    if protocol["development_seed"] in exhausted or protocol["calibration_seed"] in exhausted:
        raise PermissionError("Stage-A split reuses an exhausted validation seed")
    return protocol, base


def _configurations(protocol: dict[str, Any], seed: int, count: int):
    rng = np.random.default_rng(seed)
    domain = protocol["domain_randomization"]
    storage: list[dict[str, float | int]] = []
    thermal: list[dict[str, float | int]] = []
    for index in range(count):
        storage_period = float(rng.uniform(*domain["storage_sine_period_s_range"]))
        storage.append(
            {
                "step_s": float(rng.choice(domain["storage_step_s_choices"])),
                "sample_count": int(domain["sample_count"]["storage"]),
                "energy_capacity_j": float(
                    rng.uniform(*domain["storage_energy_capacity_j_range"])
                ),
                "initial_fraction": float(rng.uniform(0.35, 0.55)),
                "power_offset_w": float(rng.uniform(-1.0, 3.0)),
                "power_sine_w": float(rng.uniform(10.0, 18.0)),
                "power_cosine_w": float(rng.uniform(3.0, 8.0)),
                "sine_period_s": storage_period,
                "cosine_period_s": float(storage_period * rng.uniform(0.45, 0.7)),
                "state_noise_sd": float(rng.uniform(1.0e-4, 4.0e-4)),
                "noise_seed": seed * 10_000 + index,
            }
        )
        heat_period = float(rng.uniform(*domain["thermal_heat_period_s_range"]))
        thermal.append(
            {
                "step_s": float(rng.choice(domain["thermal_step_s_choices"])),
                "sample_count": int(domain["sample_count"]["thermal"]),
                "thermal_capacity_j_per_k": float(
                    rng.uniform(*domain["thermal_capacity_j_per_k_range"])
                ),
                "thermal_conductance_w_per_k": float(
                    rng.uniform(*domain["thermal_conductance_w_per_k_range"])
                ),
                "initial_temperature_k": float(rng.uniform(293.0, 298.0)),
                "heat_offset_w": float(rng.uniform(12.0, 22.0)),
                "heat_sine_w": float(rng.uniform(5.0, 10.0)),
                "heat_cosine_w": float(rng.uniform(1.0, 4.0)),
                "heat_period_s": heat_period,
                "heat_cosine_period_s": float(heat_period * rng.uniform(0.4, 0.75)),
                "ambient_k": float(rng.uniform(289.0, 294.0)),
                "ambient_swing_k": float(rng.uniform(0.3, 1.2)),
                "ambient_period_s": float(rng.uniform(31.0, 53.0)),
                "state_noise_sd_k": float(rng.uniform(0.01, 0.05)),
                "noise_seed": seed * 10_000 + 5_000 + index,
            }
        )
    return {ARCHETYPES[0]: storage, ARCHETYPES[1]: thermal}


def _generate(
    candidate1,
    base: dict[str, Any],
    protocol: dict[str, Any],
    *,
    seed: int,
    count: int,
):
    gray = candidate1._gray_box_module()
    rows = []
    requested: tuple[FailureMechanism | None, ...] = (None, *KNOWN)
    for class_index, mechanism in enumerate(requested):
        family_seed = seed + class_index * 10_000
        configurations = _configurations(protocol, family_seed, count)
        for archetype_index, archetype in enumerate(ARCHETYPES):
            for case_index, configuration in enumerate(configurations[archetype]):
                rng = np.random.default_rng(
                    family_seed * 100_000 + archetype_index * 10_000 + case_index
                )
                frame = (
                    candidate1._storage_case(gray, base, configuration, mechanism, rng)
                    if archetype == ARCHETYPES[0]
                    else candidate1._thermal_case(gray, base, configuration, mechanism, rng)
                )
                mechanism_name = mechanism.value if mechanism is not None else "NORMAL"
                rows.append(
                    candidate1.GeneratedCase(
                        case_key=(
                            f"stage-a|seed={seed}|family={family_seed}|archetype={archetype}|"
                            f"class={mechanism_name}|index={case_index}"
                        ),
                        archetype=archetype,
                        mechanism=mechanism,
                        frame=frame,
                    )
                )
    return rows


@dataclass
class ResearchRow:
    case_key: str
    archetype: str
    label: int
    values: np.ndarray


def _extract(candidate2, generated) -> tuple[list[ResearchRow], list[str]]:
    extracted = candidate2._extract(generated)
    rows: list[ResearchRow] = []
    failures: list[str] = []
    for row in extracted:
        if row.failed or row.features is None:
            failures.append(row.case_key)
            continue
        label = 0 if row.mechanism is None else CLASSES.index(row.mechanism.value)
        rows.append(
            ResearchRow(
                case_key=row.case_key,
                archetype=row.archetype,
                label=label,
                values=np.asarray(row.features.values, dtype=float),
            )
        )
    return rows, failures


def _quantile(values: list[float], alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def _normalizers(rows: list[ResearchRow], alternative: dict[str, Any]):
    groups = ARCHETYPES if alternative["archetype_conditioned"] else ("SHARED",)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group in groups:
        matrix = np.asarray(
            [
                row.values
                for row in rows
                if group == "SHARED" or row.archetype == group
            ]
        )
        if alternative.get("normalization") == "MEDIAN_AND_1.4826_MAD":
            center = np.median(matrix, axis=0)
            scale = np.maximum(
                1.4826 * np.median(np.abs(matrix - center), axis=0), 1.0e-9
            )
        else:
            center = np.mean(matrix, axis=0)
            scale = np.maximum(np.std(matrix, axis=0), 1.0e-9)
        result[group] = (center, scale)
    return result


def _normalized(row: ResearchRow, normalizers, conditioned: bool) -> np.ndarray:
    key = row.archetype if conditioned else "SHARED"
    center, scale = normalizers[key]
    return (row.values - center) / scale


def _components(rows, alternative, normalizers):
    conditioned = alternative["archetype_conditioned"]
    groups = ARCHETYPES if conditioned else ("SHARED",)
    result = {}
    for group in groups:
        for label in range(len(CLASSES)):
            matrix = np.asarray(
                [
                    _normalized(row, normalizers, conditioned)
                    for row in rows
                    if row.label == label
                    and (group == "SHARED" or row.archetype == group)
                ]
            )
            mean = np.mean(matrix, axis=0)
            covariance = np.cov(matrix, rowvar=False, ddof=1)
            if alternative["covariance"] == "DIAGONAL":
                floor = float(alternative["variance_floor"])
                covariance = np.diag(np.maximum(np.diag(covariance), floor))
            else:
                shrinkage = float(alternative["shrinkage_to_diagonal"])
                covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
                    np.diag(covariance)
                )
                floor = float(alternative["eigenvalue_floor"])
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise RuntimeError("research covariance is not positive definite")
            result[(group, label)] = {
                "mean": mean,
                "inverse": np.linalg.inv(covariance),
                "logdet": float(logdet),
            }
    return result


def _scores(row, alternative, normalizers, components):
    conditioned = alternative["archetype_conditioned"]
    group = row.archetype if conditioned else "SHARED"
    vector = _normalized(row, normalizers, conditioned)
    distances = []
    negative_logs = []
    for label in range(len(CLASSES)):
        component = components[(group, label)]
        difference = vector - component["mean"]
        distance = float(difference @ component["inverse"] @ difference)
        distances.append(max(0.0, distance))
        negative_logs.append(max(0.0, distance) + component["logdet"])
    return distances, negative_logs


def _fit_and_evaluate(development, calibration, alternative, alpha):
    normalizers = _normalizers(development, alternative)
    components = _components(development, alternative, normalizers)
    conditioned = alternative["archetype_conditioned"]
    groups = ARCHETYPES if conditioned else ("SHARED",)
    thresholds = {}
    for group in groups:
        for label in range(len(CLASSES)):
            selected = [
                row
                for row in calibration
                if row.label == label and (group == "SHARED" or row.archetype == group)
            ]
            thresholds[(group, label)] = _quantile(
                [_scores(row, alternative, normalizers, components)[0][label] for row in selected],
                alpha,
            )

    outcomes = []
    for row in calibration:
        group = row.archetype if conditioned else "SHARED"
        distances, negative_logs = _scores(row, alternative, normalizers, components)
        prediction_set = [
            label
            for label in range(len(CLASSES))
            if distances[label] <= thresholds[(group, label)]
        ]
        ood = not prediction_set
        predicted = (
            min(prediction_set, key=lambda label: negative_logs[label])
            if prediction_set
            else None
        )
        anomalous = bool(ood or predicted != 0)
        known = row.label != 0
        outcomes.append(
            {
                "archetype": row.archetype,
                "label": row.label,
                "ood": ood,
                "predicted": predicted,
                "detection_correct": anomalous if known else not anomalous,
                "known_retained": bool(known and not ood and predicted != 0),
                "classification_correct": bool(
                    known and not ood and predicted == row.label
                ),
                "normal_correct": bool(not known and not ood and predicted == 0),
            }
        )

    known_rows = [item for item in outcomes if item["label"] != 0]
    normal_rows = [item for item in outcomes if item["label"] == 0]
    rate = lambda values: sum(values) / len(values)
    per_archetype = {}
    for archetype in ARCHETYPES:
        selected_known = [item for item in known_rows if item["archetype"] == archetype]
        selected_normal = [item for item in normal_rows if item["archetype"] == archetype]
        per_archetype[archetype] = {
            "known_classification_accuracy": rate(
                [item["classification_correct"] for item in selected_known]
            ),
            "known_retention": rate([item["known_retained"] for item in selected_known]),
            "normal_detection_accuracy": rate(
                [item["normal_correct"] for item in selected_normal]
            ),
        }

    reachable = {}
    for archetype in ARCHETYPES:
        group = archetype if conditioned else "SHARED"
        component_means = np.asarray(
            [components[(group, label)]["mean"] for label in range(len(CLASSES))]
        )
        direction = np.linspace(1.0, 2.0, component_means.shape[1])
        stress = np.max(component_means, axis=0) + 100.0 * direction
        scores = []
        for label in range(len(CLASSES)):
            component = components[(group, label)]
            difference = stress - component["mean"]
            scores.append(float(difference @ component["inverse"] @ difference))
        reachable[archetype] = all(
            scores[label] > thresholds[(group, label)]
            for label in range(len(CLASSES))
        )

    return {
        "research_id": alternative["research_id"],
        "calibration_known_classification_accuracy": rate(
            [item["classification_correct"] for item in known_rows]
        ),
        "calibration_normal_detection_accuracy": rate(
            [item["normal_correct"] for item in normal_rows]
        ),
        "calibration_known_retention": rate(
            [item["known_retained"] for item in known_rows]
        ),
        "calibration_detection_accuracy": rate(
            [item["detection_correct"] for item in outcomes]
        ),
        "calibration_ood_rate_on_known_and_normal": rate(
            [item["ood"] for item in outcomes]
        ),
        "per_archetype": per_archetype,
        "open_set_rule_reachable": reachable,
        "score_semantics_identical": True,
        "thresholds_sha256": _canonical(
            {f"{group}|{CLASSES[label]}": value for (group, label), value in thresholds.items()}
        ),
    }


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    protocol, base = _load_protocols()
    candidate1, candidate2 = _dependencies()
    development_generated = _generate(
        candidate1,
        base,
        protocol,
        seed=int(protocol["development_seed"]),
        count=int(protocol["case_counts"]["development_per_archetype_per_class"]),
    )
    calibration_generated = _generate(
        candidate1,
        base,
        protocol,
        seed=int(protocol["calibration_seed"]),
        count=int(protocol["case_counts"]["calibration_per_archetype_per_class"]),
    )
    development, development_failures = _extract(candidate2, development_generated)
    calibration, calibration_failures = _extract(candidate2, calibration_generated)
    results = [
        _fit_and_evaluate(
            development,
            calibration,
            alternative,
            float(protocol["conformal_rule"]["alpha"]),
        )
        for alternative in protocol["research_alternatives"]
    ]
    checks = protocol["stage_a_exit_checks"]
    for result in results:
        result["checks"] = {
            "calibration_classification_accuracy": result[
                "calibration_known_classification_accuracy"
            ]
            >= checks["calibration_classification_accuracy"],
            "calibration_normal_detection_accuracy": result[
                "calibration_normal_detection_accuracy"
            ]
            >= checks["calibration_normal_detection_accuracy"],
            "calibration_known_retention": result["calibration_known_retention"]
            >= checks["calibration_known_retention"],
            "open_set_rule_reachable_for_each_archetype": all(
                result["open_set_rule_reachable"].values()
            ),
            "failed_feature_extractions": not development_failures
            and not calibration_failures,
        }
        result["stage_a_pass"] = all(result["checks"].values())
    passing = [item for item in results if item["stage_a_pass"]]
    selected = (
        max(
            passing,
            key=lambda item: (
                item["calibration_known_classification_accuracy"],
                item["calibration_normal_detection_accuracy"],
            ),
        )["research_id"]
        if passing
        else None
    )
    elapsed = time.monotonic() - started
    evidence = {
        "schema_version": "astrograph-generic-diagnosis-stage-a-evidence.v1",
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "authorization_sha256": protocol["authorization"]["sha256"],
        "development_seed": protocol["development_seed"],
        "calibration_seed": protocol["calibration_seed"],
        "technical_validation_seed": None,
        "exhausted_validation_seeds_used": [],
        "formal_open_set_classes_generated": [],
        "formal_hidden_truth_used": False,
        "production_candidate_fitted": False,
        "development_case_count": len(development_generated),
        "calibration_case_count": len(calibration_generated),
        "development_failure_count": len(development_failures),
        "calibration_failure_count": len(calibration_failures),
        "development_split_sha256": _canonical(
            [(row.case_key, row.label, row.values.tolist()) for row in development]
        ),
        "calibration_split_sha256": _canonical(
            [(row.case_key, row.label, row.values.tolist()) for row in calibration]
        ),
        "research_results": results,
        "stage_a_selected_research_id": selected,
        "stage_a_status": "PASS" if selected is not None else "FAIL",
        "stage_b_authorized": False,
        "elapsed_seconds": elapsed,
        "runtime_budget_seconds": checks["maximum_runtime_seconds"],
        "runtime_budget_pass": elapsed <= checks["maximum_runtime_seconds"],
        "claim_boundary": "Development/calibration research only; no open-set performance, production fit, validation, Stage B or G3 claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v1.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise PermissionError("Stage-A evidence already exists")
    evidence = run(args.output)
    print(json.dumps({key: value for key, value in evidence.items() if key != "research_results"}, indent=2))
    for item in evidence["research_results"]:
        print(json.dumps(item, indent=2))
    return 0 if evidence["stage_a_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
