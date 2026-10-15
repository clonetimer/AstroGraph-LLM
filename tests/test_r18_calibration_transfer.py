import json
import math
from pathlib import Path

import pytest

from astrograph.evaluation.calibration_campaign import (
    CalibrationContext,
    evaluate_calibration_transfer_campaign,
    evaluate_fixed_threshold_transfer,
    group_bootstrap_mean_ci,
)
from astrograph.evaluation.selective_risk import FrozenPrediction, TruthRecord, select_confidence_threshold
from astrograph.evaluation.data_exposure import dataset_use

DIMS = ["simulator_version", "mission_mode", "fault_severity", "noise_condition"]


def _rows():
    predictions = []
    truths = []
    contexts = []
    specs = [
        ("fit1", "gfit1", "A", {"A": 0.8, "B": 0.2}, "CALIBRATION_FIT", "sim1", "MODE_A", "LOW", "CLEAN"),
        ("dev1", "gdev1", "A", {"A": 0.7, "B": 0.3}, "DEVELOPMENT_SELECTION", "sim1", "MODE_A", "LOW", "CLEAN"),
        ("c1", "g1", "A", {"A": 0.9, "B": 0.1}, "CONFIRMATORY_EVALUATION", "sim1", "MODE_A", "LOW", "CLEAN"),
        ("c2", "g2", "B", {"A": 0.2, "B": 0.8}, "CONFIRMATORY_EVALUATION", "sim1", "MODE_A", "LOW", "CLEAN"),
        ("c3", "g3", "A", {"A": 0.55, "B": 0.45}, "CONFIRMATORY_EVALUATION", "sim2", "MODE_B", "HIGH", "NOISY"),
        ("c4", "g4", "B", {"A": 0.75, "B": 0.25}, "CONFIRMATORY_EVALUATION", "sim2", "MODE_B", "HIGH", "NOISY"),
    ]
    for sample_id, group_id, truth, probs, partition, sim, mode, severity, noise in specs:
        predictions.append(FrozenPrediction(sample_id, group_id, probs, max(probs, key=probs.get), (sample_id[0] * 64)[:64]))
        truths.append(TruthRecord(sample_id, group_id, truth))
        contexts.append(CalibrationContext(sample_id, group_id, partition, {
            "simulator_version": sim,
            "mission_mode": mode,
            "fault_severity": severity,
            "noise_condition": noise,
        }))
    return predictions, truths, contexts


def test_group_bootstrap_is_deterministic_and_group_based():
    first = group_bootstrap_mean_ci({"g1": 0.0, "g2": 1.0, "g3": 1.0}, resamples=500, seed=7)
    second = group_bootstrap_mean_ci({"g1": 0.0, "g2": 1.0, "g3": 1.0}, resamples=500, seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(2 / 3)
    assert first["method"] == "INDEPENDENT_GROUP_PERCENTILE_BOOTSTRAP_DESCRIPTIVE"


def test_calibration_transfer_separates_partitions_and_reports_shift_strata():
    p, t, c = _rows()
    report = evaluate_calibration_transfer_campaign(
        p, t, c,
        required_dimensions=DIMS,
        baseline_values={"simulator_version": "sim1", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"},
        ece_bins=5,
        bootstrap_resamples=500,
        bootstrap_seed=11,
    )
    assert report["independent_group_count"] == 6
    assert report["partitions"]["CONFIRMATORY_EVALUATION"]["group_count"] == 4
    assert report["confirmatory_strata"]["simulator_version"]["sim2"]["group_count"] == 2
    assert "delta_vs_declared_baseline_equal_group_macro" in report["confirmatory_strata"]["simulator_version"]["sim2"]
    assert report["calibration_qualification_status"] == "EVALUATION_ONLY_NOT_CALIBRATION_RECEIPT"
    assert report["partitions"]["CONFIRMATORY_EVALUATION"]["classwise_one_vs_rest"]["A"]["positive_group_count"] == 2
    assert len(report["report_sha256"]) == 64


def test_group_cannot_cross_shift_stratum():
    p, t, c = _rows()
    p.append(FrozenPrediction("c5", "g1", {"A": 0.8, "B": 0.2}, "A", "e" * 64))
    t.append(TruthRecord("c5", "g1", "A"))
    c.append(CalibrationContext("c5", "g1", "CONFIRMATORY_EVALUATION", {
        "simulator_version": "sim2", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"
    }))
    with pytest.raises(ValueError, match="crosses stratum simulator_version"):
        evaluate_calibration_transfer_campaign(
            p, t, c,
            required_dimensions=DIMS,
            baseline_values={"simulator_version": "sim1", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"},
            bootstrap_resamples=500,
        )


def test_fixed_threshold_cannot_be_selected_on_confirmatory_groups():
    p, t, c = _rows()
    dev_p = [x for x in p if x.group_id == "gdev1"]
    dev_t = [x for x in t if x.group_id == "gdev1"]
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT", "SELECT", group_ids=["r18-selective-synth"]):
        valid = select_confidence_threshold(
            dev_p, dev_t, allowed_groups={"gdev1"}, forbidden_groups={"g1", "g2", "g3", "g4"}, max_selective_risk=0.0
        )
    tampered = dict(valid)
    tampered["selected_on_groups"] = ["g1"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluate_fixed_threshold_transfer(p, t, c, threshold_receipt=tampered, required_dimensions=DIMS)
    result = evaluate_fixed_threshold_transfer(p, t, c, threshold_receipt=valid, required_dimensions=DIMS)
    assert result["status"] == "EVALUATED_FIXED_DEVELOPMENT_THRESHOLD"
    assert result["confirmatory_group_count"] == 4
    assert result["overall"]["coverage"] == pytest.approx(0.75)
    assert result["overall"]["equal_group_macro_coverage"] == pytest.approx(0.75)



def test_probability_universe_drift_and_missing_true_label_fail_closed():
    p, t, c = _rows()
    p[2] = FrozenPrediction("c1", "g1", {"A": 1.0}, "A", "c" * 64)
    with pytest.raises(ValueError, match="class universe drift"):
        evaluate_calibration_transfer_campaign(
            p, t, c, required_dimensions=DIMS,
            baseline_values={"simulator_version": "sim1", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"},
            bootstrap_resamples=500,
        )
    p, t, c = _rows()
    t[2] = TruthRecord("c1", "g1", "C")
    with pytest.raises(ValueError, match="truth label absent"):
        evaluate_calibration_transfer_campaign(
            p, t, c, required_dimensions=DIMS,
            baseline_values={"simulator_version": "sim1", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"},
            bootstrap_resamples=500,
        )


def test_shift_delta_is_marked_confounding_when_axes_move_together():
    p, t, c = _rows()
    report = evaluate_calibration_transfer_campaign(
        p, t, c, required_dimensions=DIMS,
        baseline_values={"simulator_version": "sim1", "mission_mode": "MODE_A", "fault_severity": "LOW", "noise_condition": "CLEAN"},
        bootstrap_resamples=500,
    )
    interp = report["confirmatory_strata"]["simulator_version"]["_interpretation"]
    assert interp["single_dimension_isolation_status"] == "CONFOUNDED_WITH_OTHER_DECLARED_DIMENSIONS"
    assert set(interp["confounded_with_dimensions"]) == {"mission_mode", "fault_severity", "noise_condition"}


def test_r18_design_freezes_group_shift_method():
    design = json.loads(Path("configs/evaluation/calibration_transfer_r18.v1.json").read_text(encoding="utf-8"))
    assert design["uncertainty_method"] == "INDEPENDENT_GROUP_PERCENTILE_BOOTSTRAP_DESCRIPTIVE"
    assert design["required_dimensions"] == DIMS
    assert "CONFIRMATORY_EVALUATION" in design["required_partitions"]
