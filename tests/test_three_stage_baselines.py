from __future__ import annotations

import pytest

from astrograph.data_driven.three_stage_baselines import (
    CausalResidualCusumConfig,
    CausalResidualCusumDetector,
    ComponentConditionedLinearClassifier,
    ComponentLinearHead,
    ComponentResidualEvidence,
    ResidualSourceLocalizer,
    ResidualSourceLocalizerConfig,
)
from astrograph.data_driven.three_stage_protocol import (
    UNKNOWN,
    CausalTelemetrySample,
    HierarchicalFaultOntology,
    LocalizationDecision,
)


def test_causal_cusum_uses_only_past_samples_and_frozen_parameters() -> None:
    detector = CausalResidualCusumDetector(
        CausalResidualCusumConfig(
            channel_center={"rw.speed": 0.0},
            channel_scale={"rw.speed": 1.0},
            drift=1.0,
            alarm_threshold=5.0,
            suspected_component="rw",
        )
    )
    assert detector.update(CausalTelemetrySample(0.0, {"rw.speed": 0.2})) is None
    assert detector.update(CausalTelemetrySample(1.0, {"rw.speed": 3.0})) is None
    assert detector.update(CausalTelemetrySample(2.0, {"rw.speed": 3.0})) is None
    alarm = detector.update(CausalTelemetrySample(3.0, {"rw.speed": 3.0}))
    assert alarm is not None
    assert alarm.timestamp_s == 3.0
    assert alarm.suspected_component == "rw"
    with pytest.raises(ValueError, match="strictly increasing"):
        detector.update(CausalTelemetrySample(2.0, {"rw.speed": 9.0}))


def test_source_localizer_penalizes_public_propagation_response() -> None:
    localizer = ResidualSourceLocalizer(
        ResidualSourceLocalizerConfig(
            source_local_weight=1.0,
            command_response_weight=0.5,
            propagation_penalty=0.5,
            affected_threshold=2.0,
            minimum_source_margin=0.1,
        )
    )
    result = localizer.localize(
        [
            ComponentResidualEvidence("rw", 3.0, 2.0, 0.2),
            ComponentResidualEvidence("cmg", 0.2, 0.1, 6.0),
        ]
    )
    assert result.top_source == "rw"
    assert result.affected_components == frozenset({"cmg"})
    assert result.decision == LocalizationDecision.IDENTIFIED


def test_source_localizer_abstains_on_empty_and_marks_small_margin_ambiguous() -> None:
    localizer = ResidualSourceLocalizer(ResidualSourceLocalizerConfig(minimum_source_margin=1.0))
    assert localizer.localize([]).decision == LocalizationDecision.ABSTAIN
    result = localizer.localize(
        [
            ComponentResidualEvidence("rw", 1.0, 0.0, 0.0),
            ComponentResidualEvidence("star", 0.9, 0.0, 0.0),
        ]
    )
    assert result.decision == LocalizationDecision.AMBIGUOUS


def _classifier(threshold: float) -> ComponentConditionedLinearClassifier:
    ontology = HierarchicalFaultOntology(
        {"rw": {"MECHANICAL": ("JAM", "FRICTION"), "ELECTRICAL": ("MOTOR_OPEN",)}}
    )
    head = ComponentLinearHead(
        mode_weights={"JAM": (2.0, 0.0), "FRICTION": (0.0, 2.0), "MOTOR_OPEN": (-1.0, -1.0)},
        mode_bias={"JAM": 0.0, "FRICTION": 0.0, "MOTOR_OPEN": 0.0},
        unknown_probability_threshold=threshold,
    )
    return ComponentConditionedLinearClassifier(ontology, {"rw": head})


def test_component_conditioned_linear_classifier_emits_ranked_probabilities() -> None:
    result = _classifier(0.5).classify("rw", [2.0, 0.0])
    assert result.conditioned_component == "rw"
    assert result.mechanism_family == "MECHANICAL"
    assert result.local_fault_mode == "JAM"
    assert sum(result.local_mode_scores.values()) == pytest.approx(1.0)
    assert result.unknown_score is not None


def test_component_conditioned_linear_classifier_can_reject_unknown() -> None:
    result = _classifier(0.8).classify("rw", [0.0, 0.0])
    assert result.abstained is True
    assert result.mechanism_family == UNKNOWN
    assert result.local_fault_mode == UNKNOWN
    with pytest.raises(ValueError, match="no frozen"):
        _classifier(0.5).classify("star_tracker", [1.0, 0.0])
