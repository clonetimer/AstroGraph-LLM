#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.evaluation.calibration_campaign import CalibrationContext
from astrograph.evaluation.calibration_fitting import (
    apply_temperature_scaler_receipt,
    fit_temperature_scaler_receipt,
    qualify_temperature_calibration,
    verify_temperature_scaler_fit_receipt,
)
from astrograph.evaluation.selective_risk import FrozenPrediction, TruthRecord


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit/admit one group-disjoint probability temperature calibrator and evaluate it on locked "
            "confirmatory + external-validation groups. No production/flight qualification is implied."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--design", type=Path, default=Path("configs/evaluation/calibration_fitting_r19.v1.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fit-receipt", type=Path, help="Use an already-frozen fit receipt instead of fitting again.")
    parser.add_argument('--exposure-dataset-id', required=True, choices=['SYNTHETIC_PROTOCOL'], help='R21 permits this historical full qualification runner only for declared synthetic protocol fixtures.')
    args = parser.parse_args()

    from astrograph.evaluation.data_exposure import dataset_use
    with dataset_use(args.exposure_dataset_id, 'CALIBRATION_FIT', 'CALIBRATE'):
        return _run_synthetic_protocol(args)

def _run_synthetic_protocol(args):
    bundle_bytes = args.input.read_bytes()
    design_bytes = args.design.read_bytes()
    bundle = json.loads(bundle_bytes.decode("utf-8"))
    design = json.loads(design_bytes.decode("utf-8"))
    admission_bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    design_sha256 = hashlib.sha256(design_bytes).hexdigest()
    predictions = [FrozenPrediction(**row) for row in bundle["predictions"]]
    truths = [TruthRecord(**row) for row in bundle["truths"]]
    contexts = [CalibrationContext(**row) for row in bundle["contexts"]]
    calibrator = design["calibrator"]

    if args.fit_receipt is None:
        fit_receipt = fit_temperature_scaler_receipt(
            predictions,
            truths,
            contexts,
            calibration_id=bundle["calibration_id"],
            model_id=bundle["model_id"],
            model_version=bundle["model_version"],
            calibration_dataset_id=bundle["calibration_dataset_id"],
            calibration_dataset_sha256=bundle["calibration_dataset_sha256"],
            model_artifact_sha256=bundle["model_artifact_sha256"],
            minimum_fit_groups=int(calibrator["minimum_fit_groups"]),
            minimum_positive_groups_per_class=int(calibrator["minimum_positive_groups_per_class"]),
            temperature_lower=float(calibrator["temperature_lower"]),
            temperature_upper=float(calibrator["temperature_upper"]),
            coarse_points=int(calibrator["coarse_points"]),
            refine_iterations=int(calibrator["refine_iterations"]),
        )
    else:
        fit_receipt = verify_temperature_scaler_fit_receipt(_load(args.fit_receipt))

    guards = design["guardrails"]
    report, qualification_receipt = qualify_temperature_calibration(
        predictions,
        truths,
        contexts,
        fit_receipt=fit_receipt,
        required_dimensions=design["required_dimensions"],
        baseline_values=bundle["baseline_values"],
        model_id=bundle["model_id"],
        model_version=bundle["model_version"],
        external_validation_dataset_id=bundle["external_validation_dataset_id"],
        external_validation_dataset_sha256=bundle["external_validation_dataset_sha256"],
        model_artifact_sha256=bundle["model_artifact_sha256"],
        design_sha256=design_sha256,
        admission_bundle_sha256=admission_bundle_sha256,
        minimum_development_groups=int(design["minimum_development_groups"]),
        minimum_confirmatory_groups=int(design["minimum_confirmatory_groups"]),
        minimum_external_groups=int(design["minimum_external_validation_groups"]),
        minimum_positive_groups_per_class=int(design["minimum_positive_groups_per_class"]),
        ece_bins=int(design["ece_bins"]),
        bootstrap_resamples=int(design["bootstrap_resamples"]),
        bootstrap_seed=int(design["bootstrap_seed"]),
        max_development_selective_risk=float(design["max_development_selective_risk"]),
        max_confirmatory_nll_increase=float(guards["max_confirmatory_nll_increase"]),
        max_confirmatory_brier_increase=float(guards["max_confirmatory_brier_increase"]),
        max_confirmatory_ece_increase=float(guards["max_confirmatory_ece_increase"]),
        max_external_nll_increase=float(guards["max_external_nll_increase"]),
        max_external_brier_increase=float(guards["max_external_brier_increase"]),
        max_external_ece_increase=float(guards["max_external_ece_increase"]),
        max_classwise_brier_increase=float(guards["max_classwise_brier_increase"]),
        max_classwise_ece_increase=float(guards["max_classwise_ece_increase"]),
        max_external_selective_risk_increase=float(guards["max_external_selective_risk_increase"]),
        max_external_coverage_decrease=float(guards["max_external_coverage_decrease"]),
    )
    calibrated = apply_temperature_scaler_receipt(predictions, fit_receipt)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "calibrator_fit_receipt.json", fit_receipt)
    _write(args.output_dir / "calibration_qualification_report.json", report)
    _write(args.output_dir / "probability_calibration_qualification_receipt.json", qualification_receipt.model_dump(mode="json") | {"receipt_sha256": qualification_receipt.receipt_sha256})
    _write(args.output_dir / "calibrated_predictions.json", {
        "schema_version": "astrograph-calibrated-frozen-predictions.v1",
        "calibration_id": fit_receipt["calibration_id"],
        "fit_receipt_sha256": fit_receipt["receipt_sha256"],
        "predictions": [
            {
                "sample_id": p.sample_id,
                "group_id": p.group_id,
                "probabilities": dict(sorted((str(k), float(v)) for k, v in p.probabilities.items())),
                "predicted_id": p.predicted_id,
                "prediction_sha256": p.prediction_sha256,
            }
            for p in calibrated
        ],
        "claim_boundary": "Offline calibrated predictions bound to one fit receipt; not release evidence by themselves.",
    })
    return 0 if report["qualification_status"] == "PASS_FOR_ENGINEERING_CALIBRATION_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
