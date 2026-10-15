from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.data_driven.fair_comparison import (
    FairThreeTaskComparisonLedger,
    build_fair_three_task_comparison_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "governance/evidence/research/data_driven_runs/fair_three_task_comparison_20260831"


def test_governed_comparison_reports_all_tasks_without_suppressing_methods(tmp_path: Path) -> None:
    ledger = FairThreeTaskComparisonLedger.model_validate_json(
        (RUN / "fairness_audit.json").read_text(encoding="utf-8")
    )
    assert ledger.status == "PASS_FAIR_COMPARISON"
    assert all(ledger.anti_suppression_checks.values())
    assert {item.dataset_id for item in ledger.dataset_audits} == {"adcs.v5_2_2", "phm.v6_0"}
    assert all(item.common_test_split for item in ledger.dataset_audits)
    assert all(item.common_training_budget for item in ledger.dataset_audits)
    assert all(item.common_seed_budget for item in ledger.dataset_audits)
    assert all(item.parameter_capacity_ratio <= 4.0 for item in ledger.dataset_audits)
    assert len(ledger.task_leaders) == 12
    assert not any(key.startswith("phm.v6_0:detection") for key in ledger.task_leaders)
    assert {
        item.information_regime for item in ledger.runs if item.dataset_id == "phm.v6_0"
    } == {"RAW_SUMMARY_ONLY", "MATCHED_NOMINAL_DIGITAL_TWIN_REQUIRED"}


def test_unequal_method_budget_is_rejected(tmp_path: Path) -> None:
    source = json.loads((RUN / "adcs/report.json").read_text(encoding="utf-8"))
    source["models"][0]["validation_history"] = source["models"][0]["validation_history"][:-1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(source), encoding="utf-8")
    adcs_index = tmp_path / "adcs-index.csv"
    phm_index = tmp_path / "phm-index.csv"
    adcs_index.write_text("sample_id\nA\n", encoding="utf-8")
    phm_index.write_text("sample_id\nP\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unequal epoch budgets"):
        build_fair_three_task_comparison_ledger(
            adcs_report_path=bad,
            phm_report_path=RUN / "phm/report.json",
            adcs_script_path=ROOT / "scripts/data_driven/compare_adcs_deep_encoders.py",
            phm_script_path=ROOT / "scripts/data_driven/compare_phm_deep_models.py",
            adcs_dataset_index_path=adcs_index,
            phm_dataset_index_path=phm_index,
            output_path=tmp_path / "ledger.json",
        )
