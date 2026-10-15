from __future__ import annotations

import pytest

from astrograph.data_driven.three_stage_protocol import (
    UNKNOWN,
    CascadeEpisodeResult,
    CascadeTruth,
    CausalStreamContract,
    CausalTelemetrySample,
    ClassificationTruth,
    HierarchicalClassificationPrediction,
    HierarchicalFaultOntology,
    LocalizationDecision,
    LocalizationPrediction,
    LocalizationTruth,
    evaluate_cascade,
    hierarchical_classification_metrics,
    localization_metrics,
    stratified_hierarchical_classification_metrics,
    stratified_localization_metrics,
    validate_open_set_partition,
)


@pytest.fixture
def ontology() -> HierarchicalFaultOntology:
    return HierarchicalFaultOntology(
        {
            "rw": {
                "MECHANICAL": ("JAM", "FRICTION_INCREASE"),
                "ELECTRICAL": ("MOTOR_OPEN",),
            },
            "star_tracker": {
                "SIGNAL": ("BIAS", "DROPOUT"),
                "DATA_QUALITY": ("NOISE_BURST",),
            },
        }
    )


def test_causal_stream_rejects_truth_and_noncausal_order() -> None:
    guard = CausalStreamContract()
    accepted = guard.accept(CausalTelemetrySample(1.0, {"rw.speed_rpm": 10.0}))
    assert accepted.timestamp_s == 1.0

    with pytest.raises(ValueError, match="evaluator-only"):
        CausalTelemetrySample(2.0, {"fault_active": 1.0})
    with pytest.raises(ValueError, match="strictly increasing"):
        guard.accept(CausalTelemetrySample(1.0, {"rw.speed_rpm": 11.0}))


def test_localization_keeps_source_and_affected_metrics_separate() -> None:
    result = localization_metrics(
        [
            LocalizationTruth("rw", frozenset({"cmg"})),
            LocalizationTruth("star_tracker", frozenset({"cmg", "mtb"})),
        ],
        [
            LocalizationPrediction(("rw",), frozenset({"cmg"})),
            LocalizationPrediction(("cmg",), frozenset({"star_tracker", "mtb"})),
        ],
        top_k=2,
    )

    assert result["source_top1_accuracy"] == pytest.approx(0.5)
    assert result["source_top2_recall"] == pytest.approx(0.5)
    assert result["affected_as_top_source_count"] == 1
    assert result["source_as_affected_count"] == 1


def test_localization_prediction_forbids_double_role() -> None:
    with pytest.raises(ValueError, match="simultaneously"):
        LocalizationPrediction(("rw",), frozenset({"rw"}))


def test_ambiguous_localization_is_not_a_top1_decision() -> None:
    result = localization_metrics(
        [LocalizationTruth("rw"), LocalizationTruth("star_tracker")],
        [
            LocalizationPrediction(("rw",), decision=LocalizationDecision.AMBIGUOUS),
            LocalizationPrediction(("star_tracker",)),
        ],
    )
    assert result["source_top1_accuracy"] == pytest.approx(0.5)
    assert result["identified_coverage"] == pytest.approx(0.5)
    assert result["identified_selective_accuracy"] == pytest.approx(1.0)
    assert result["source_macro_f1"] == pytest.approx(0.5)


def test_hierarchical_classifier_cannot_emit_another_components_mode(
    ontology: HierarchicalFaultOntology,
) -> None:
    invalid = HierarchicalClassificationPrediction("rw", "SIGNAL", "BIAS")
    with pytest.raises(ValueError, match="outside"):
        invalid.validate(ontology)

    invalid_abstention = HierarchicalClassificationPrediction("rw", UNKNOWN, "JAM", abstained=True)
    with pytest.raises(ValueError, match="UNKNOWN"):
        invalid_abstention.validate(ontology)


def test_hierarchical_metrics_report_path_and_abstention(
    ontology: HierarchicalFaultOntology,
) -> None:
    truth = [
        ClassificationTruth("rw", "MECHANICAL", "JAM", "rw/JAM"),
        ClassificationTruth("rw", "MECHANICAL", "FRICTION_INCREASE", "rw/FRICTION_INCREASE"),
    ]
    predicted = [
        HierarchicalClassificationPrediction("rw", "MECHANICAL", "JAM"),
        HierarchicalClassificationPrediction("rw", UNKNOWN, UNKNOWN, abstained=True),
    ]
    result = hierarchical_classification_metrics(truth, predicted, ontology=ontology)

    assert result["hierarchical_exact_match"] == pytest.approx(0.5)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["selective_exact_match"] == pytest.approx(1.0)
    assert result["family"]["evaluated_class_count"] == 1


def test_hierarchical_metrics_support_open_set_unknown_truth(
    ontology: HierarchicalFaultOntology,
) -> None:
    truth = [
        ClassificationTruth("rw", "MECHANICAL", "JAM", "rw/JAM"),
        ClassificationTruth("rw", UNKNOWN, UNKNOWN, "rw/HELDOUT_MODE", is_open_set=True),
    ]
    predicted = [
        HierarchicalClassificationPrediction("rw", "MECHANICAL", "JAM"),
        HierarchicalClassificationPrediction("rw", UNKNOWN, UNKNOWN, abstained=True),
    ]
    result = hierarchical_classification_metrics(truth, predicted, ontology=ontology)
    assert result["hierarchical_exact_match"] == 1.0
    assert result["open_set"]["unknown_rejection_recall"] == 1.0
    assert result["open_set"]["false_acceptance_rate"] == 0.0
    assert result["known_class"]["coverage"] == 1.0


def test_open_set_truth_requires_unknown_labels(ontology: HierarchicalFaultOntology) -> None:
    with pytest.raises(ValueError, match="UNKNOWN"):
        ClassificationTruth("rw", "MECHANICAL", "JAM", "rw/JAM", is_open_set=True).validate(ontology)


def test_open_set_preflight_rejects_canonical_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_open_set_partition(
            known_mode_ids={"rw/JAM", "rw/FRICTION"},
            open_set_mode_ids={"rw/JAM"},
        )


def test_classification_topk_unknown_aupr_ece_and_strata(
    ontology: HierarchicalFaultOntology,
) -> None:
    truth = [
        ClassificationTruth("rw", "MECHANICAL", "JAM", "rw/JAM"),
        ClassificationTruth("rw", "ELECTRICAL", "MOTOR_OPEN", "rw/MOTOR_OPEN"),
        ClassificationTruth("rw", UNKNOWN, UNKNOWN, "rw/HELDOUT", is_open_set=True),
    ]
    predicted = [
        HierarchicalClassificationPrediction(
            "rw", "MECHANICAL", "FRICTION_INCREASE",
            family_scores={"MECHANICAL": 0.9, "ELECTRICAL": 0.1},
            local_mode_scores={"JAM": 0.4, "FRICTION_INCREASE": 0.5, "MOTOR_OPEN": 0.1},
            unknown_score=0.1,
        ),
        HierarchicalClassificationPrediction(
            "rw", "ELECTRICAL", "MOTOR_OPEN",
            family_scores={"MECHANICAL": 0.2, "ELECTRICAL": 0.8},
            local_mode_scores={"JAM": 0.1, "FRICTION_INCREASE": 0.1, "MOTOR_OPEN": 0.8},
            unknown_score=0.2,
        ),
        HierarchicalClassificationPrediction(
            "rw", UNKNOWN, UNKNOWN, abstained=True, unknown_score=0.9,
        ),
    ]
    result = hierarchical_classification_metrics(
        truth, predicted, ontology=ontology, local_mode_top_k=2
    )
    assert result["component_local_mode_top2_accuracy"] == 1.0
    assert result["open_set"]["unknown_rejection_aupr"] == 1.0
    assert result["open_set"]["unknown_score_coverage"] == 1.0
    assert result["calibration"]["coverage"] == 1.0
    assert result["calibration"]["frozen_bin_ece"]["bin_edges"] == [
        index / 10 for index in range(11)
    ]
    by_stratum = stratified_hierarchical_classification_metrics(
        truth,
        predicted,
        ontology=ontology,
        strata={"profile": ["p1", "p1", "p2"], "severity": ["low", "high", "low"]},
        local_mode_top_k=2,
    )
    assert by_stratum["profile"]["p1"]["n_fault_episodes"] == 2
    assert by_stratum["severity"]["low"]["n_fault_episodes"] == 2


def test_localization_strata_keep_full_metric_contract() -> None:
    truth = [LocalizationTruth("rw"), LocalizationTruth("star_tracker")]
    predicted = [LocalizationPrediction(("rw",)), LocalizationPrediction(("rw", "star_tracker"))]
    result = stratified_localization_metrics(
        truth,
        predicted,
        strata={"profile": ["nadir", "sun"], "severity": ["low", "low"]},
        top_k=2,
    )
    assert result["profile"]["nadir"]["source_mrr"] == 1.0
    assert result["severity"]["low"]["source_top2_recall"] == 1.0


def _fault_truth(
    episode_id: str,
    component: str,
    family: str,
    mode: str,
    affected: frozenset[str] = frozenset(),
) -> CascadeTruth:
    return CascadeTruth(
        episode_id,
        True,
        LocalizationTruth(component, affected),
        ClassificationTruth(component, family, mode, f"{component}/{mode}"),
    )


def _oracle_fields(
    component: str, family: str, mode: str
) -> dict[str, object]:
    return {
        "localization_with_oracle_window": LocalizationPrediction((component,)),
        "classification_with_oracle_upstream": HierarchicalClassificationPrediction(
            component, family, mode
        ),
    }


def test_cascade_reports_oracle_predicted_and_first_wrong_stage(
    ontology: HierarchicalFaultOntology,
) -> None:
    correct_truth = _fault_truth("correct", "rw", "MECHANICAL", "JAM")
    missed_truth = _fault_truth("missed", "rw", "MECHANICAL", "FRICTION_INCREASE")
    mislocated_truth = _fault_truth(
        "mislocated", "rw", "ELECTRICAL", "MOTOR_OPEN", frozenset({"star_tracker"})
    )
    misclassified_truth = _fault_truth("misclassified", "star_tracker", "SIGNAL", "BIAS")

    episodes = [
        CascadeEpisodeResult(
            correct_truth,
            True,
            **_oracle_fields("rw", "MECHANICAL", "JAM"),
            localization_with_predicted_window=LocalizationPrediction(("rw",)),
            classification_with_predicted_upstream=HierarchicalClassificationPrediction(
                "rw", "MECHANICAL", "JAM"
            ),
        ),
        CascadeEpisodeResult(
            missed_truth,
            False,
            **_oracle_fields("rw", "MECHANICAL", "FRICTION_INCREASE"),
        ),
        CascadeEpisodeResult(
            mislocated_truth,
            True,
            **_oracle_fields("rw", "ELECTRICAL", "MOTOR_OPEN"),
            localization_with_predicted_window=LocalizationPrediction(("star_tracker", "rw")),
        ),
        CascadeEpisodeResult(
            misclassified_truth,
            True,
            **_oracle_fields("star_tracker", "SIGNAL", "BIAS"),
            localization_with_predicted_window=LocalizationPrediction(("star_tracker",)),
            classification_with_predicted_upstream=HierarchicalClassificationPrediction(
                "star_tracker", "DATA_QUALITY", "NOISE_BURST"
            ),
        ),
        CascadeEpisodeResult(CascadeTruth("nominal", False), True),
    ]

    result = evaluate_cascade(episodes, ontology=ontology, include_confidence_intervals=True)

    assert result["localization"]["oracle_detection_window"]["source_top1_accuracy"] == 1.0
    assert result["classification"]["oracle_upstream"]["hierarchical_exact_match"] == 1.0
    assert result["localization"]["predicted_upstream_source_top1_accuracy"] == pytest.approx(0.5)
    assert result["localization"]["predicted_upstream"]["source_macro_f1"] >= 0.0
    assert "component_local_mode_top3_accuracy" in result["classification"]["predicted_upstream"]
    assert result["first_wrong_stage"] == {
        "detection": 2,
        "localization": 1,
        "classification_abstention": 0,
        "classification_family": 1,
        "classification_mode": 0,
        "correct": 1,
    }
    assert result["end_to_end_exact_match"] == pytest.approx(0.2)
    assert result["fault_only_exact_match"] == pytest.approx(0.25)
    assert result["confidence_intervals"]["classification_predicted_exact"]["successes"] == 1


def test_cascade_blocks_downstream_execution_after_missed_detection(
    ontology: HierarchicalFaultOntology,
) -> None:
    truth = _fault_truth("miss", "rw", "MECHANICAL", "JAM")
    episode = CascadeEpisodeResult(
        truth,
        False,
        **_oracle_fields("rw", "MECHANICAL", "JAM"),
        localization_with_predicted_window=LocalizationPrediction(("rw",)),
    )
    with pytest.raises(ValueError, match="must not run"):
        episode.validate(ontology)
