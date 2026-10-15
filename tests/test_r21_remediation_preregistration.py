import json
import pytest

from astrograph.evaluation.development_remediation import load_preregistration, validate_existing_development_reports


def test_candidate_budgets_are_finite_and_frozen():
    plan = load_preregistration()
    assert plan["OPS"]["maximum_trials"] == 9
    assert plan["ESA"]["maximum_trials"] == 4
    assert plan["frozen_before_execution"] is True


def test_existing_reports_must_bind_exact_plan_and_trial_budgets():
    plan = load_preregistration()
    ops = {"preregistration_sha256": plan["receipt_sha256"], "trials": [{}] * 9, "external_predictions_generated": False}
    esa = {"preregistration_sha256": plan["receipt_sha256"], "trials": [{}] * 4, "external_predictions_generated": False}
    assert validate_existing_development_reports(ops, esa)["status"] == "PASS"
    ops["trials"].append({})
    with pytest.raises(ValueError, match="TRIAL_BUDGET"):
        validate_existing_development_reports(ops, esa)
