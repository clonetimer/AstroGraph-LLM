from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astrograph.data_driven.adcs_metric_ledger import load_ledger, verify_ledger_sources
from astrograph.data_driven.adcs_posthoc import ADCSPosthocQualificationArtifactV2

ROOT = Path(__file__).resolve().parents[3]
DEV23 = ROOT / "governance/implementation/post48_dev23"
CURRENT_MODEL_SHA = "39062457c1ed6d68c1478afa1d125ba54b80a6e07936de0a4078dca865852732"


def _json(name: str) -> dict:
    return json.loads((DEV23 / name).read_text(encoding="utf-8"))


def test_stage91_metric_ledger_separates_protocols_and_verifies_sources():
    ledger = load_ledger(DEV23 / "ADCS_METRIC_PROTOCOL_LEDGER.json")
    verify_ledger_sources(ROOT, ledger)
    roles = {record.evidence_role: record for record in ledger.records}
    assert set(roles) == {
        "RESEARCH_GROUP_CV_BASELINE",
        "HISTORICAL_PRODUCTION_QUALIFICATION",
        "USED_GROUP_RETROSPECTIVE_REPLAY",
        "CURRENT_MISSION_BINDING",
    }
    assert ledger.current_mission_model_sha256 == CURRENT_MODEL_SHA
    assert roles["HISTORICAL_PRODUCTION_QUALIFICATION"].model_sha256 != CURRENT_MODEL_SHA
    assert roles["USED_GROUP_RETROSPECTIVE_REPLAY"].model_sha256 == CURRENT_MODEL_SHA
    assert roles["USED_GROUP_RETROSPECTIVE_REPLAY"].independent_confirmation is False
    assert roles["USED_GROUP_RETROSPECTIVE_REPLAY"].metrics["detection_macro_f1_executable"] > 0.98
    assert roles["HISTORICAL_PRODUCTION_QUALIFICATION"].metrics["detection_macro_f1"] < 0.92


def test_stage92_current_v2_status_keeps_metrics_denominators_and_reuse_boundary():
    status = _json("ADCS_CURRENT_V2_STATUS.json")
    assert status["current_model_sha256"] == CURRENT_MODEL_SHA
    assert status["three_task_gate_passed"] is True
    assert status["three_task_metrics"]["detection_macro_f1_executable"] >= 0.95
    assert status["three_task_metrics"]["localization_system_derived_macro_f1"] >= 0.95
    assert status["three_task_metrics"]["classification_system_macro_f1"] >= 0.95
    assert status["execution"]["windows_all"] == 546
    assert status["execution"]["windows_executable"] == 501
    assert status["execution"]["windows_unavailable"] == 45
    assert status["full_window_coverage_complete"] is False
    assert status["all_window_detection_macro_f1"] < status["three_task_metrics"]["detection_macro_f1_executable"]
    assert status["independent_confirmation"] is False


def test_stage93_uploaded_dataset_is_not_relabelled_as_new_independent_confirmation():
    status = _json("ADCS_INDEPENDENT_CONFIRMATION_STATUS.json")
    assert status["dataset_sample_ids"] == [f"{i:02d}" for i in range(10)]
    assert status["historically_consumed_sample_ids"] == [f"{i:02d}" for i in range(10)]
    assert status["unused_sample_ids"] == []
    assert status["new_independent_group_available_in_uploaded_dataset"] is False
    assert status["status"] == "NEW_GROUP_REQUIRED"
    assert status["independent_confirmation_performed"] is False
    contract = status["required_confirmation_contract"]
    assert contract["labels_withheld_until_prediction_freeze"] is True
    assert contract["threshold_tuning_on_confirmation_data"] is False
    assert contract["all_windows_retained"] is True


def test_stage94_calibration_ood_is_bounded_and_does_not_change_three_task_baseline():
    raw = _json("ADCS_CALIBRATION_OOD_QUALIFICATION.json")
    provenance = raw.pop("provenance")
    artifact = ADCSPosthocQualificationArtifactV2.model_validate(raw)
    assert artifact.model_artifact_sha256 == CURRENT_MODEL_SHA
    assert artifact.test_partition_used_for_fitting is False
    assert artifact.weights_modified is False
    assert provenance["test_trajectory_count_loaded"] == 0
    assert artifact.probability_calibration_qualified is True
    assert artifact.ood_score_qualified is True
    assert artifact.formal_probability_and_ood_eligible is True
    calibration = artifact.joint_probability_calibration
    assert calibration.mean_nll_after < calibration.mean_nll_before
    assert calibration.mean_brier_after <= calibration.mean_brier_before + 0.001
    # ECE may trade slightly against NLL/Brier; the qualification rule is explicit rather than hidden.
    assert calibration.mean_ece_after <= calibration.mean_ece_before + 0.005
    ood = artifact.ood_profile
    assert ood.validation_id_false_reject_rate <= 0.01
    assert all(value >= 0.95 for value in ood.primary_stress_recall.values())
    assert ood.secondary_stress_recall["temporal_reverse"] < 0.95
    assert "arbitrary" in ood.claim_boundary.lower()
    assert "cross-satellite" in ood.claim_boundary.lower()


def test_dev23_model_file_remains_the_current_mission_binding():
    path = ROOT / "models/adcs_v5_2_2/p2_innovation_v2.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CURRENT_MODEL_SHA
