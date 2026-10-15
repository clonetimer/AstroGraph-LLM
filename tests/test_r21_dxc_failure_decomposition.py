import pytest

from astrograph.evaluation.real_data_failure_analysis import decompose_fault_sets, summarize_fault_cases


def test_fault_set_mechanisms_separate_partial_overlap_and_over_isolation():
    row = decompose_fault_sets(["A:stuck", "B:offset"], ["A:stuck", "C:offset"])
    assert row["flags"]["PARTIAL_OVERLAP"] and row["flags"]["OVER_ISOLATION"] and row["flags"]["UNDER_ISOLATION"]
    assert not row["flags"]["EXACT_SET_MATCH"]


def test_micro_metrics_use_fault_elements_not_case_average():
    rows = [decompose_fault_sets(["A:x", "B:x"], ["A:x"]), decompose_fault_sets(["C:x"], ["Z:x", "Q:x"])]
    summary = summarize_fault_cases(rows)
    assert summary["fault_element_recall_micro"] == pytest.approx(1 / 3)
    assert summary["fault_element_precision_micro"] == pytest.approx(1 / 3)
