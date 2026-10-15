#!/usr/bin/env python3
"""Evaluate direct switching-physics mechanism evidence under P-017 Stage A."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from astrograph.diagnosis.generic.segmented_calibration import _physical_arrays
from astrograph.physics import default_dynamics_contracts


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/diagnosis/generic_head_research_protocol.stage_a.v3.json"
STAGE_A_V1 = ROOT / "scripts/acceptance/research_generic_diagnosis_stage_a.py"
CLASSES = ("NORMAL", "CAPACITY_LOSS", "DRIFT", "LEAKAGE")
ARCHETYPES = ("storage.energy_buffer", "thermal.lumped_capacity")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location("stage_a_v1_physics_generator", STAGE_A_V1)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage-A generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fit(y: np.ndarray, matrix: np.ndarray, floor: float):
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ coefficients
    sse = max(float(residual @ residual), floor)
    n = len(y)
    k = matrix.shape[1]
    bic = n * math.log(sse / n) + k * math.log(n)
    return coefficients, sse, bic, math.log(sse / n)


def _hypotheses(
    y: np.ndarray,
    design: np.ndarray,
    fractions: list[float],
    floor: float,
):
    n = len(y)
    parameter_count = design.shape[1]
    _, normal_sse, normal_bic, normal_adequacy = _fit(y, design, floor)
    result = {
        "NORMAL": {
            "bic": normal_bic,
            "adequacy": normal_adequacy,
            "sse": normal_sse,
            "split_fraction": None,
        }
    }
    for class_name in ("CAPACITY_LOSS", "DRIFT", "LEAKAGE"):
        best = None
        for fraction in fractions:
            split = min(max(int(round(fraction * n)), parameter_count + 3), n - 4)
            post = np.zeros(n)
            post[split:] = 1.0
            if class_name == "CAPACITY_LOSS":
                matrix = np.column_stack([design, design * post[:, None]])
            elif class_name == "LEAKAGE":
                matrix = np.column_stack([design, post])
            else:
                ramp = np.zeros(n)
                ramp[split:] = np.linspace(0.0, 1.0, n - split)
                matrix = np.column_stack([design, ramp])
            coefficients, sse, bic, adequacy = _fit(y, matrix, floor)
            added = coefficients[parameter_count:]
            valid = (
                bool(np.all(added > 0.0))
                if class_name == "CAPACITY_LOSS"
                else bool(added[-1] < 0.0)
            )
            if not valid:
                continue
            item = {
                "bic": bic,
                "adequacy": adequacy,
                "sse": sse,
                "split_fraction": fraction,
            }
            if best is None or item["bic"] < best["bic"]:
                best = item
        result[class_name] = best or {
            "bic": float("inf"),
            "adequacy": float("inf"),
            "sse": float("inf"),
            "split_fraction": None,
        }
    return result


def _rows(generated, fractions, floor):
    contracts = default_dynamics_contracts()
    rows = []
    failures = []
    for case in generated:
        try:
            _, derivative, design = _physical_arrays(case.frame, contracts[case.archetype])
            label = "NORMAL" if case.mechanism is None else case.mechanism.value
            rows.append(
                {
                    "case_key": case.case_key,
                    "archetype": case.archetype,
                    "label": label,
                    "hypotheses": _hypotheses(derivative, design, fractions, floor),
                    "derivative": derivative,
                    "design": design,
                }
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures.append({"case_key": case.case_key, "error": str(exc)})
    return rows, failures


def _quantile(values: list[float], alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def run(output: Path):
    started = time.monotonic()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    authorization = ROOT / protocol["authorization"]["path"]
    if _sha(authorization) != protocol["authorization"]["sha256"]:
        raise ValueError("Stage-A authorization changed")
    if protocol["technical_validation_seed"] is not None:
        raise PermissionError("Stage-A v3 unexpectedly has a validation seed")
    exhausted = set(protocol["permanently_exhausted_validation_seeds"])
    if protocol["development_seed"] in exhausted or protocol["calibration_seed"] in exhausted:
        raise PermissionError("Stage-A v3 reuses an exhausted validation seed")
    retained = protocol["retained_stage_a_failures"]
    retained_paths = [
        ROOT / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v1.json",
        ROOT / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v2.json",
    ]
    for item, path in zip(retained, retained_paths):
        if _sha(path) != item["evidence_sha256"] or item["status"] != "FAIL":
            raise ValueError("retained Stage-A failure changed")

    base = _module()
    stage_protocol, generator_protocol = base._load_protocols()
    candidate1, _ = base._dependencies()
    development_generated = base._generate(
        candidate1,
        generator_protocol,
        stage_protocol,
        seed=protocol["development_seed"],
        count=stage_protocol["case_counts"]["development_per_archetype_per_class"],
    )
    calibration_generated = base._generate(
        candidate1,
        generator_protocol,
        stage_protocol,
        seed=protocol["calibration_seed"],
        count=stage_protocol["case_counts"]["calibration_per_archetype_per_class"],
    )
    fractions = protocol["change_point_grid"]
    floor = float(protocol["fit"]["sse_floor"])
    development, development_failures = _rows(development_generated, fractions, floor)
    calibration, calibration_failures = _rows(calibration_generated, fractions, floor)
    alpha = float(protocol["adequacy_and_open_set_design"]["alpha"])
    thresholds = {}
    for archetype in ARCHETYPES:
        for class_name in CLASSES:
            values = [
                row["hypotheses"][class_name]["adequacy"]
                for row in development
                if row["archetype"] == archetype and row["label"] == class_name
            ]
            thresholds[(archetype, class_name)] = _quantile(values, alpha)

    outcomes = []
    for row in calibration:
        prediction_set = [
            class_name
            for class_name in CLASSES
            if row["hypotheses"][class_name]["adequacy"]
            <= thresholds[(row["archetype"], class_name)]
        ]
        ood = not prediction_set
        predicted = (
            min(
                prediction_set,
                key=lambda name: row["hypotheses"][name]["bic"],
            )
            if prediction_set
            else None
        )
        known = row["label"] != "NORMAL"
        anomalous = bool(ood or predicted != "NORMAL")
        outcomes.append(
            {
                "archetype": row["archetype"],
                "label": row["label"],
                "predicted": predicted,
                "ood": ood,
                "detection_correct": anomalous if known else not anomalous,
                "known_retained": bool(known and not ood and predicted != "NORMAL"),
                "classification_correct": bool(known and predicted == row["label"]),
                "normal_correct": bool(not known and predicted == "NORMAL"),
            }
        )
    rate = lambda values: sum(values) / len(values)
    known_rows = [item for item in outcomes if item["label"] != "NORMAL"]
    normal_rows = [item for item in outcomes if item["label"] == "NORMAL"]
    per_archetype = {}
    for archetype in ARCHETYPES:
        known = [item for item in known_rows if item["archetype"] == archetype]
        normal = [item for item in normal_rows if item["archetype"] == archetype]
        per_archetype[archetype] = {
            "known_classification_accuracy": rate(
                [item["classification_correct"] for item in known]
            ),
            "known_retention": rate([item["known_retained"] for item in known]),
            "normal_detection_accuracy": rate([item["normal_correct"] for item in normal]),
        }

    reachability = {}
    for archetype in ARCHETYPES:
        representative = next(row for row in development if row["archetype"] == archetype)
        alternating = np.where(np.arange(len(representative["derivative"])) % 2 == 0, 1.0, -1.0)
        stressed = representative["derivative"] + 1000.0 * alternating
        scores = _hypotheses(stressed, representative["design"], fractions, floor)
        reachability[archetype] = all(
            scores[class_name]["adequacy"] > thresholds[(archetype, class_name)]
            for class_name in CLASSES
        )

    known_accuracy = rate([item["classification_correct"] for item in known_rows])
    normal_accuracy = rate([item["normal_correct"] for item in normal_rows])
    known_retention = rate([item["known_retained"] for item in known_rows])
    checks_cfg = protocol["exit_checks"]
    checks = {
        "known_classification_accuracy": known_accuracy
        >= checks_cfg["known_classification_accuracy"],
        "normal_detection_accuracy": normal_accuracy
        >= checks_cfg["normal_detection_accuracy"],
        "each_archetype_known_classification_accuracy": all(
            item["known_classification_accuracy"]
            >= checks_cfg["each_archetype_known_classification_accuracy"]
            for item in per_archetype.values()
        ),
        "known_retention": known_retention >= checks_cfg["known_retention"],
        "open_set_rule_reachable_for_each_archetype": all(reachability.values()),
        "failed_case_count": not development_failures and not calibration_failures,
    }
    elapsed = time.monotonic() - started
    checks["runtime_budget"] = elapsed <= checks_cfg["maximum_runtime_seconds"]
    evidence = {
        "schema_version": "astrograph-generic-diagnosis-stage-a-switching-physics.v1",
        "protocol_sha256": _sha(PROTOCOL),
        "authorization_sha256": protocol["authorization"]["sha256"],
        "development_seed": protocol["development_seed"],
        "calibration_seed": protocol["calibration_seed"],
        "technical_validation_seed": None,
        "exhausted_validation_seeds_used": [],
        "formal_open_set_classes_generated": [],
        "production_candidate_fitted": False,
        "development_case_count": len(development_generated),
        "calibration_case_count": len(calibration_generated),
        "development_failure_count": len(development_failures),
        "calibration_failure_count": len(calibration_failures),
        "known_classification_accuracy": known_accuracy,
        "normal_detection_accuracy": normal_accuracy,
        "known_retention": known_retention,
        "detection_accuracy": rate([item["detection_correct"] for item in outcomes]),
        "per_archetype": per_archetype,
        "open_set_rule_reachable": reachability,
        "same_score_semantics": True,
        "thresholds_sha256": hashlib.sha256(
            json.dumps(
                {f"{key[0]}|{key[1]}": value for key, value in thresholds.items()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "checks": checks,
        "stage_a_v3_status": "PASS" if all(checks.values()) else "FAIL",
        "stage_b_authorized": False,
        "elapsed_seconds": elapsed,
        "claim_boundary": protocol["claim_boundary"],
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
        / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v3.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise PermissionError("Stage-A v3 evidence already exists")
    evidence = run(args.output)
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["stage_a_v3_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
