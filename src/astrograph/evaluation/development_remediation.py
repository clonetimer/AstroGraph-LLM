"""Finite, preregistered R21 development remediation orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

from .data_exposure import capture_development_accesses, dataset_use
from .external_dataset_contract import verify_seal

CONFIG = Path(__file__).resolve().parents[3] / "configs/evaluation/r21_remediation_preregistration.v1.json"


def load_preregistration(path: str | Path = CONFIG) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_seal(plan)
    if not plan.get("frozen_before_execution"):
        raise ValueError("R21_PREREGISTRATION_NOT_FROZEN")
    return plan


def _select(trials: list[dict], *, metric: Callable[[dict], float], tie_break: Callable[[dict], tuple]) -> dict:
    if not trials:
        raise ValueError("R21_NO_DEVELOPMENT_TRIALS")
    return max(trials, key=lambda row: (metric(row),) + tuple(-x if isinstance(x, (int, float)) else x for x in tie_break(row)))


def run_ops_development_study(*, group_ids_by_partition: Mapping[str, list[str]],
                              evaluate_candidate: Callable[[str, float], dict],
                              plan: Mapping | None = None) -> dict:
    """Evaluate exactly the nine registered OPS trials on development groups."""
    frozen = load_preregistration()
    if plan is not None and dict(plan) != frozen:
        raise ValueError("R21_PREREGISTRATION_CHANGED")
    spec = frozen["OPS"]
    trials = []
    with capture_development_accesses() as accesses:
        with dataset_use("OPS_SAT", "MODEL_TRAIN", "FIT", group_ids=group_ids_by_partition["MODEL_TRAIN"]):
            # The callback owns fitting and must enter guarded shared APIs.
            pass
        with dataset_use("OPS_SAT", "DEVELOPMENT_SELECTION", "SELECT", group_ids=group_ids_by_partition["DEVELOPMENT_SELECTION"]):
            for family_index, family in enumerate(spec["candidate_family"]):
                for threshold_index, threshold in enumerate(spec["thresholds"][family]):
                    result = dict(evaluate_candidate(family, threshold))
                    result.update(candidate_family=family, threshold=threshold,
                                  family_index=family_index, threshold_index=threshold_index)
                    trials.append(result)
    if len(trials) != spec["maximum_trials"]:
        raise ValueError("R21_OPS_TRIAL_BUDGET_MISMATCH")
    selected = max(trials, key=lambda r: (r["selection_objective"], -r["covered_metrics"]["false_positive_segment_count"],
                                          -r["family_index"], -r["threshold_index"]))
    return {"schema_version": "astrograph-r21-development-remediation-report.v1", "family": "OPS",
            "status": "DEVELOPMENT_REMEDIATION_READY_FOR_NEW_REPLICATION",
            "preregistration_sha256": frozen["receipt_sha256"], "development_accesses": accesses,
            "trials": trials, "selected_candidate": selected,
            "score_semantics": "UNCALIBRATED_ANOMALY_SCORE", "probability_metrics_computed": False,
            "external_predictions_generated": False, "qualification": "NOT_FLIGHT_QUALIFIED"}


def run_esa_development_study(*, group_ids_by_partition: Mapping[str, list[str]],
                              evaluate_candidate: Callable[[str], dict],
                              plan: Mapping | None = None) -> dict:
    """Evaluate exactly the four registered ESA candidates on development groups."""
    frozen = load_preregistration()
    if plan is not None and dict(plan) != frozen:
        raise ValueError("R21_PREREGISTRATION_CHANGED")
    spec = frozen["ESA"]
    trials = []
    with capture_development_accesses() as accesses:
        with dataset_use("ESA_MISSION1", "MODEL_TRAIN", "FIT", group_ids=group_ids_by_partition["MODEL_TRAIN"]):
            pass
        with dataset_use("ESA_MISSION1", "DEVELOPMENT_SELECTION", "SELECT", group_ids=group_ids_by_partition["DEVELOPMENT_SELECTION"]):
            for index, family in enumerate(spec["candidate_family"]):
                result = dict(evaluate_candidate(family)); result.update(candidate_family=family, family_index=index)
                trials.append(result)
    if len(trials) != spec["maximum_trials"]:
        raise ValueError("R21_ESA_TRIAL_BUDGET_MISMATCH")
    selected = max(trials, key=lambda r: (r["selection_metric"], -r["metrics"]["false_alarm_duration_s"], -r["family_index"]))
    return {"schema_version": "astrograph-r21-development-remediation-report.v1", "family": "ESA",
            "status": "DEVELOPMENT_REMEDIATION_READY_FOR_NEW_REPLICATION",
            "preregistration_sha256": frozen["receipt_sha256"], "development_accesses": accesses,
            "trials": trials, "selected_candidate": selected,
            "score_semantics": "UNCALIBRATED_ANOMALY_SCORE", "external_predictions_generated": False,
            "qualification": "NOT_FLIGHT_QUALIFIED"}


def validate_existing_development_reports(ops: Mapping, esa: Mapping) -> dict:
    """Validate already-executed reports after guard hardening without rerunning trials."""
    frozen = load_preregistration()
    if ops.get("preregistration_sha256") != frozen["receipt_sha256"] or esa.get("preregistration_sha256") != frozen["receipt_sha256"]:
        raise ValueError("R21_DEVELOPMENT_REPORT_PLAN_MISMATCH")
    if len(ops.get("trials", [])) != frozen["OPS"]["maximum_trials"]:
        raise ValueError("R21_OPS_TRIAL_BUDGET_MISMATCH")
    if len(esa.get("trials", [])) != frozen["ESA"]["maximum_trials"]:
        raise ValueError("R21_ESA_TRIAL_BUDGET_MISMATCH")
    if ops.get("external_predictions_generated") or esa.get("external_predictions_generated"):
        raise ValueError("R21_EXTERNAL_PREDICTIONS_FORBIDDEN")
    return {"status": "PASS", "OPS_trials": len(ops["trials"]), "ESA_trials": len(esa["trials"]),
            "numerical_candidate_study_rerun": False}
