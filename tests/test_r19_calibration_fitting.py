from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from astrograph.evaluation.calibration_campaign import CalibrationContext
from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.calibration_fitting import (
    apply_temperature_scaler_receipt,
    fit_temperature_scaler_receipt,
    qualify_temperature_calibration,
    verify_temperature_scaler_fit_receipt,
)
from astrograph.evaluation.selective_risk import FrozenPrediction, TruthRecord

DIMS = ["simulator_version", "mission_mode", "fault_severity", "noise_condition"]
BASELINE = {"simulator_version": "sim1", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"}


def _fixture():
    predictions = []
    truths = []
    contexts = []
    index = 0
    for partition, count in [
        ("CALIBRATION_FIT", 8),
        ("DEVELOPMENT_SELECTION", 4),
        ("CONFIRMATORY_EVALUATION", 8),
        ("EXTERNAL_VALIDATION", 8),
    ]:
        prefix = {
            "CALIBRATION_FIT": "fit",
            "DEVELOPMENT_SELECTION": "dev",
            "CONFIRMATORY_EVALUATION": "conf",
            "EXTERNAL_VALIDATION": "ext",
        }[partition]
        for item in range(count):
            group_id = f"{prefix}-g{item}"
            true_id = "A" if item % 2 == 0 else "B"
            wrong = item in ({6, 7} if count >= 8 else {3})
            if wrong:
                p_a = 0.8 if true_id == "B" else 0.2
            else:
                p_a = 0.9 if true_id == "A" else 0.1
            probs = {"A": p_a, "B": 1.0 - p_a}
            sample_id = f"sample-{index}"
            index += 1
            predictions.append(FrozenPrediction(sample_id, group_id, probs, max(probs, key=probs.get), f"{index:064x}"[-64:]))
            truths.append(TruthRecord(sample_id, group_id, true_id))
            contexts.append(CalibrationContext(sample_id, group_id, partition, BASELINE))
    return predictions, truths, contexts


def _fit(predictions, truths, contexts):
    with dataset_use("SYNTHETIC_PROTOCOL", "CALIBRATION_FIT", "CALIBRATE"):
        return fit_temperature_scaler_receipt(
            predictions,
            truths,
            contexts,
        calibration_id="temp-r19",
        model_id="model-r19",
        model_version="1",
        calibration_dataset_id="calibration-dataset-r19",
        calibration_dataset_sha256="1" * 64,
        model_artifact_sha256="2" * 64,
        minimum_fit_groups=8,
        minimum_positive_groups_per_class=2,
    )


def _qualify(predictions, truths, contexts, fit):
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT", "SELECT"):
        return qualify_temperature_calibration(
            predictions,
            truths,
            contexts,
        fit_receipt=fit,
        required_dimensions=DIMS,
        baseline_values=BASELINE,
        model_id="model-r19",
        model_version="1",
        external_validation_dataset_id="external-r19",
        external_validation_dataset_sha256="3" * 64,
        model_artifact_sha256="2" * 64,
        design_sha256="4" * 64,
        admission_bundle_sha256="5" * 64,
        minimum_development_groups=4,
        minimum_confirmatory_groups=8,
        minimum_external_groups=8,
        minimum_positive_groups_per_class=2,
        ece_bins=5,
        bootstrap_resamples=500,
        bootstrap_seed=17,
        max_development_selective_risk=0.30,
        max_confirmatory_nll_increase=0.02,
        max_confirmatory_brier_increase=0.02,
        max_confirmatory_ece_increase=0.05,
        max_external_nll_increase=0.02,
        max_external_brier_increase=0.02,
        max_external_ece_increase=0.05,
        max_classwise_brier_increase=0.05,
        max_classwise_ece_increase=0.10,
        max_external_selective_risk_increase=0.05,
        max_external_coverage_decrease=0.25,
    )


def test_temperature_fit_is_group_macro_and_content_bound():
    predictions, truths, contexts = _fixture()
    receipt = _fit(predictions, truths, contexts)
    assert receipt["fit_group_count"] == 8
    assert receipt["fit_objective"] == "EQUAL_INDEPENDENT_GROUP_MACRO_NLL"
    assert receipt["fit_partition"] == "CALIBRATION_FIT"
    assert receipt["fit_equal_group_macro_nll_after"] <= receipt["fit_equal_group_macro_nll_before"]
    assert 0.05 <= receipt["temperature"] <= 20.0
    assert verify_temperature_scaler_fit_receipt(receipt)["receipt_sha256"] == receipt["receipt_sha256"]


def test_fit_receipt_rejects_self_hash_tamper():
    predictions, truths, contexts = _fixture()
    receipt = _fit(predictions, truths, contexts)
    tampered = dict(receipt)
    tampered["temperature"] = float(receipt["temperature"]) + 0.1
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_temperature_scaler_fit_receipt(tampered)


def test_external_qualification_binds_fit_content_and_external_dataset_identity():
    predictions, truths, contexts = _fixture()
    fit = _fit(predictions, truths, contexts)
    report, receipt = _qualify(predictions, truths, contexts, fit)
    assert report["qualification_status"] == "PASS_FOR_ENGINEERING_CALIBRATION_REVIEW"
    assert report["partition_group_counts"]["EXTERNAL_VALIDATION"] == 8
    assert report["external_validation_dataset_sha256"] == "3" * 64
    assert report["external_selective_risk_degradation_guardrail"]["comparison"].startswith("CALIBRATED_POLICY_VS_RAW_POLICY")
    assert receipt.external_validation_dataset_id == "external-r19"
    assert receipt.external_validation_dataset_sha256 == "3" * 64
    assert receipt.model_artifact_sha256 == "2" * 64
    assert receipt.qualification_status == "PASS_FOR_ENGINEERING_CALIBRATION_REVIEW"
    assert len(receipt.receipt_sha256) == 64


def test_external_qualification_rejects_fit_prediction_drift_even_with_same_group_ids():
    predictions, truths, contexts = _fixture()
    fit = _fit(predictions, truths, contexts)
    mutated = list(predictions)
    first = mutated[0]
    mutated[0] = FrozenPrediction(first.sample_id, first.group_id, {"A": 0.8, "B": 0.2}, "A", "f" * 64)
    with pytest.raises(ValueError, match="fit receipt content identity mismatch"):
        _qualify(mutated, truths, contexts, fit)


def test_external_groups_are_required_and_cannot_overlap_fit():
    predictions, truths, contexts = _fixture()
    fit = _fit(predictions, truths, contexts)
    without_external = [c for c in contexts if c.partition != "EXTERNAL_VALIDATION"]
    keep = {c.sample_id for c in without_external}
    p2 = [p for p in predictions if p.sample_id in keep]
    t2 = [t for t in truths if t.sample_id in keep]
    with pytest.raises(ValueError, match="missing required partitions"):
        _qualify(p2, t2, without_external, fit)

    contexts2 = list(contexts)
    ext_index = next(i for i, c in enumerate(contexts2) if c.partition == "EXTERNAL_VALIDATION")
    external = contexts2[ext_index]
    contexts2[ext_index] = CalibrationContext(external.sample_id, "fit-g0", external.partition, external.strata)
    p3 = list(predictions)
    old = p3[ext_index]
    p3[ext_index] = FrozenPrediction(old.sample_id, "fit-g0", old.probabilities, old.predicted_id, old.prediction_sha256)
    t3 = list(truths)
    old_t = t3[ext_index]
    t3[ext_index] = TruthRecord(old_t.sample_id, "fit-g0", old_t.true_id)
    with pytest.raises(ValueError, match="share independent groups"):
        _qualify(p3, t3, contexts2, fit)


def test_calibrated_predictions_are_bound_to_fit_receipt():
    predictions, truths, contexts = _fixture()
    fit = _fit(predictions, truths, contexts)
    calibrated = apply_temperature_scaler_receipt(predictions, fit)
    assert len(calibrated) == len(predictions)
    assert calibrated[0].prediction_sha256 != predictions[0].prediction_sha256
    assert calibrated[0].predicted_id == predictions[0].predicted_id
    assert sum(calibrated[0].probabilities.values()) == pytest.approx(1.0)


def test_r19_design_freezes_fit_and_external_validation_boundaries():
    design = json.loads(Path("configs/evaluation/calibration_fitting_r19.v1.json").read_text(encoding="utf-8"))
    assert design["calibrator"]["fit_objective"] == "EQUAL_INDEPENDENT_GROUP_MACRO_NLL"
    assert "EXTERNAL_VALIDATION" in design["required_partitions"]
    assert design["guardrail_status"] == "ENGINEERING_REVIEW_FLOORS_NOT_FLIGHT_CALIBRATED"

def test_external_validation_requires_positive_independent_groups_for_every_class():
    predictions, truths, contexts = _fixture()
    fit = _fit(predictions, truths, contexts)
    mutated_truths = list(truths)
    for index, (truth, context) in enumerate(zip(mutated_truths, contexts)):
        if context.partition == "EXTERNAL_VALIDATION" and truth.true_id == "B":
            mutated_truths[index] = TruthRecord(truth.sample_id, truth.group_id, "A")
    with pytest.raises(ValueError, match="positive independent groups per class"):
        _qualify(predictions, mutated_truths, contexts, fit)
