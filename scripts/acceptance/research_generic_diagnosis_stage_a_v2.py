#!/usr/bin/env python3
"""Run the frozen P-017 Stage-A nonlinear separability ceiling."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from astrograph.diagnosis.generic import SEGMENTED_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/diagnosis/generic_head_research_protocol.stage_a.v2.json"
STAGE_A_V1 = ROOT / "scripts/acceptance/research_generic_diagnosis_stage_a.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location("stage_a_v1_frozen", STAGE_A_V1)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage-A v1 generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _matrix(rows, indicators):
    return np.asarray(
        [np.concatenate([row.values, [indicators[row.archetype]]]) for row in rows],
        dtype=float,
    )


def run(output: Path):
    started = time.monotonic()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    authorization = ROOT / protocol["authorization"]["path"]
    if _sha(authorization) != protocol["authorization"]["sha256"]:
        raise ValueError("Stage-A authorization changed")
    retained = protocol["retained_stage_a_v1_failure"]
    if _sha(STAGE_A_V1) != retained["implementation_sha256"]:
        raise ValueError("Stage-A v1 implementation changed")
    old_evidence = ROOT / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v1.json"
    if _sha(old_evidence) != retained["evidence_sha256"]:
        raise ValueError("Stage-A v1 failure evidence changed")
    if protocol["technical_validation_seed"] is not None:
        raise PermissionError("Stage-A v2 unexpectedly has a validation seed")
    exhausted = set(protocol["permanently_exhausted_validation_seeds"])
    if protocol["development_seed"] in exhausted or protocol["calibration_seed"] in exhausted:
        raise PermissionError("Stage-A v2 reuses an exhausted seed")

    base = _module()
    stage_protocol, generator_protocol = base._load_protocols()
    candidate1, candidate2 = base._dependencies()
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
    development, development_failures = base._extract(candidate2, development_generated)
    calibration, calibration_failures = base._extract(candidate2, calibration_generated)
    indicators = protocol["input_mapping"]["archetype_indicator"]
    x_dev = _matrix(development, indicators)
    y_dev = np.asarray([row.label for row in development], dtype=int)
    x_cal = _matrix(calibration, indicators)
    y_cal = np.asarray([row.label for row in calibration], dtype=int)
    cfg = protocol["oracle"]
    model = ExtraTreesClassifier(
        n_estimators=cfg["n_estimators"],
        criterion=cfg["criterion"],
        max_features=cfg["max_features"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        bootstrap=cfg["bootstrap"],
        class_weight=cfg["class_weight"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    ).fit(x_dev, y_dev)
    predicted = model.predict(x_cal)
    known_mask = y_cal != 0
    normal_mask = y_cal == 0
    known_accuracy = float(np.mean(predicted[known_mask] == y_cal[known_mask]))
    normal_accuracy = float(np.mean(predicted[normal_mask] == 0))
    per_archetype = {}
    for archetype in base.ARCHETYPES:
        mask = np.asarray([row.archetype == archetype for row in calibration])
        per_archetype[archetype] = {
            "known_classification_accuracy": float(
                np.mean(predicted[mask & known_mask] == y_cal[mask & known_mask])
            ),
            "normal_detection_accuracy": float(
                np.mean(predicted[mask & normal_mask] == 0)
            ),
        }
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
        "failed_feature_extractions": not development_failures and not calibration_failures,
    }
    elapsed = time.monotonic() - started
    checks["runtime_budget"] = elapsed <= checks_cfg["maximum_runtime_seconds"]
    importances = sorted(
        enumerate(model.feature_importances_.tolist()), key=lambda item: item[1], reverse=True
    )[:10]
    feature_names = [*stage_protocol["frozen_feature_source"].get("feature_names", []), "archetype_indicator"]
    # The v1 protocol binds the implementation and dimension; names come from the frozen module.
    if len(feature_names) != 23:
        feature_names = [*SEGMENTED_FEATURE_NAMES, "archetype_indicator"]
    evidence = {
        "schema_version": "astrograph-generic-diagnosis-stage-a-separability.v1",
        "protocol_sha256": _sha(PROTOCOL),
        "authorization_sha256": protocol["authorization"]["sha256"],
        "retained_stage_a_v1_evidence_sha256": retained["evidence_sha256"],
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
        "per_archetype": per_archetype,
        "confusion_matrix_classes_NORMAL_CAPACITY_LOSS_DRIFT_LEAKAGE": confusion_matrix(
            y_cal, predicted, labels=[0, 1, 2, 3]
        ).tolist(),
        "top_feature_importances": [
            {"feature": feature_names[index], "importance": value}
            for index, value in importances
        ],
        "checks": checks,
        "stage_a_v2_status": "PASS" if all(checks.values()) else "FAIL",
        "interpretation": (
            protocol["interpretation_rule"]["PASS"]
            if all(checks.values())
            else protocol["interpretation_rule"]["FAIL"]
        ),
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
        / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v2.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise PermissionError("Stage-A v2 evidence already exists")
    evidence = run(args.output)
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["stage_a_v2_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
