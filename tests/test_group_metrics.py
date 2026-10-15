from __future__ import annotations

import pytest

from astrograph.experiments import (
    GroupedPrediction,
    paired_group_accuracy_delta,
    summarize_grouped_classification,
)


def _row(oid: str, group: str, truth: str, pred: str | None) -> GroupedPrediction:
    return GroupedPrediction(observation_id=oid, group_id=group, truth=truth, prediction=pred)


def test_group_summary_resamples_groups_and_counts_abstention() -> None:
    rows = [
        _row("a1", "g1", "F1", "F1"),
        _row("a2", "g1", "F2", None),
        _row("b1", "g2", "F1", "F2"),
        _row("b2", "g2", "F2", "F2"),
    ]
    result = summarize_grouped_classification(
        rows, label_universe=("F1", "F2"), bootstrap_iterations=100, bootstrap_seed=7
    )
    assert result.accuracy == pytest.approx(0.5)
    assert result.abstention_rate == pytest.approx(0.25)
    assert result.independent_group_count == 2
    assert result.group_macro_accuracy == pytest.approx(0.5)
    assert result.cluster_bootstrap_accuracy_ci95[0] <= result.accuracy
    assert result.cluster_bootstrap_accuracy_ci95[1] >= result.accuracy


def test_paired_group_delta_is_group_level_not_row_independent() -> None:
    reference = [
        _row("a1", "g1", "F1", "F2"),
        _row("a2", "g1", "F1", "F2"),
        _row("b1", "g2", "F1", "F1"),
    ]
    candidate = [
        _row("a1", "g1", "F1", "F1"),
        _row("a2", "g1", "F1", "F1"),
        _row("b1", "g2", "F1", "F1"),
    ]
    result = paired_group_accuracy_delta(
        reference,
        candidate,
        reference_id="A0",
        candidate_id="A7",
        bootstrap_iterations=100,
        bootstrap_seed=11,
    )
    assert result.group_accuracy_delta == {"g1": 1.0, "g2": 0.0}
    assert result.mean_group_accuracy_delta == pytest.approx(0.5)


def test_summary_rejects_duplicate_observations() -> None:
    rows = [_row("x", "g1", "F1", "F1"), _row("x", "g2", "F1", "F1")]
    with pytest.raises(ValueError, match="observation_id"):
        summarize_grouped_classification(rows, label_universe=("F1",))
