from __future__ import annotations

from astrograph.diagnosis.model_based import enumerate_minimal_diagnoses


def test_subset_minimal_and_minimum_cardinality_are_not_conflated():
    result = enumerate_minimal_diagnoses([("A", "B"), ("B", "C")])
    assert result.search_complete is True
    assert [item.fault_ids for item in result.subset_minimal_diagnoses] == [("B",), ("A", "C")]
    assert [item.fault_ids for item in result.minimum_cardinality_diagnoses] == [("B",)]


def test_cardinality_bound_is_explicitly_incomplete_not_silently_normal():
    result = enumerate_minimal_diagnoses([("A",), ("B",)], max_cardinality=1)
    assert result.subset_minimal_diagnoses == ()
    assert result.search_complete is False
    assert "MAX_CARDINALITY_EXCLUDED_CANDIDATE" in result.reason_codes
