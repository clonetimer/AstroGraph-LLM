from __future__ import annotations

from astrograph.experiments.soft_kg_comparison import (
    METHOD_IDS,
    ComparisonObservation,
    evaluate_comparison,
    predict_comparison_methods,
)


def _row(index: int, truth: str, data: dict[str, float], hard: dict[str, float], soft: dict[str, float]):
    return ComparisonObservation(
        observation_id=f"obs-{index}",
        domain_id="PHM_V6_0",
        group_id=str(3 if index % 2 else 8),
        truth_fault_id=truth,
        fault_universe=("F1", "F2", "F3"),
        data_probabilities=data,
        hard_predicate_scores=hard,
        soft_predicate_scores=soft,
        executable_coverage=2 / 3,
        contextual_reliability=0.7,
        source_bundle_sha256=f"{index:064x}",
    )


def test_frozen_method_set_is_complete_and_normalized():
    row = _row(1, "F1", {"F1": 0.6, "F2": 0.3, "F3": 0.1}, {"F1": 1}, {"F1": 0.8})
    predictions = predict_comparison_methods(row)
    assert tuple(predictions) == METHOD_IDS
    for prediction in predictions.values():
        assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-12


def test_comparison_reports_required_selective_and_group_metrics():
    report = evaluate_comparison([
        _row(1, "F1", {"F1": 0.8, "F2": 0.1, "F3": 0.1}, {"F1": 1}, {"F1": 0.9}),
        _row(2, "F2", {"F1": 0.2, "F2": 0.7, "F3": 0.1}, {"F2": 1}, {"F2": 0.8}),
        _row(3, "F3", {"F1": 0.2, "F2": 0.2, "F3": 0.6}, {}, {}),
    ])
    assert len(report["report_sha256"]) == 64
    required = {
        "macro_f1", "balanced_accuracy", "nll", "brier", "ece", "coverage",
        "selective_risk", "aurc", "worst_group_accuracy", "failure_to_abstain",
        "risk_coverage_curve",
    }
    assert required <= report["metrics"]["PSL_STYLE_SOFT_KG_CHALLENGER"].keys()
    assert "data_vs_hard_phi" in report["branch_error_correlation"]


def test_missing_branches_abstain_instead_of_fabricating_evidence():
    row = ComparisonObservation(
        observation_id="missing", domain_id="ADCS_V5_2_2", group_id="3",
        truth_fault_id="F1", fault_universe=("F1", "F2"), data_probabilities={},
        hard_predicate_scores={}, soft_predicate_scores={}, data_available=False,
        executable_coverage=0.0, contextual_reliability=0.0,
        source_bundle_sha256="0" * 64,
    )
    predictions = predict_comparison_methods(row)
    assert predictions["A0_DATA_ONLY"].abstained
    assert predictions["HARD_RULE_ONLY"].abstained
    assert predictions["PSL_STYLE_SOFT_KG_CHALLENGER"].abstained
