"""AI-Native upgrade roadmap A1: Harness four-quadrant governance tests."""
from __future__ import annotations

import pytest

from astrograph.governance.harness import (
    DiagnosticPrinciple, HarnessQuadrant, PeerReviewRecord, RiskLevel, resolve_autonomy,
    assess_harness_completeness,
)


def _principles() -> list[DiagnosticPrinciple]:
    return [
        DiagnosticPrinciple(
            principle_id='P-001', name='conservatism',
            statement='When evidence is ambiguous, prefer fail-closed over speculative diagnosis.',
        ),
        DiagnosticPrinciple(
            principle_id='P-002', name='priority',
            statement='Life-critical subsystems (ADCS attitude control) take precedence over minor telemetry drift.',
        ),
    ]


def _reviews() -> list[PeerReviewRecord]:
    return [
        PeerReviewRecord(
            review_id='RVW-001', reviewed_artifact='hypothesis_set:EP-123',
            reviewer='independent-deterministic-checker', verdict='APPROVED',
            findings=['No unsupported fault claim.'],
        ),
    ]


def test_complete_four_quadrants():
    assessment = assess_harness_completeness(
        principles=_principles(), peer_reviews=_reviews(),
    )
    assert assessment.overall == 'COMPLETE'
    assert assessment.missing_quadrants == []
    assert assessment.quadrants[HarnessQuadrant.FEEDFORWARD_REASONING.value]['count'] == 2
    assert assessment.quadrants[HarnessQuadrant.FEEDBACK_REASONING.value]['count'] == 1


def test_missing_reasoning_quadrants_reported():
    assessment = assess_harness_completeness(principles=[], peer_reviews=[])
    assert assessment.overall == 'GAPS'
    assert HarnessQuadrant.FEEDFORWARD_REASONING.value in assessment.missing_quadrants
    assert HarnessQuadrant.FEEDBACK_REASONING.value in assessment.missing_quadrants


def test_deterministic_quadrants_present_by_default():
    assessment = assess_harness_completeness(principles=[], peer_reviews=[])
    assert assessment.quadrants[HarnessQuadrant.FEEDFORWARD_DETERMINISTIC.value]['present'] is True
    assert assessment.quadrants[HarnessQuadrant.FEEDBACK_DETERMINISTIC.value]['present'] is True


def test_principle_validates_fields():
    with pytest.raises(Exception):
        DiagnosticPrinciple(principle_id='', name='x', statement='y')


def test_review_verdict_enum():
    with pytest.raises(Exception):
        PeerReviewRecord(review_id='r', reviewed_artifact='a', reviewer='b', verdict='MAYBE')


def test_missing_principle_statement_rejected():
    with pytest.raises(Exception):
        DiagnosticPrinciple(principle_id='P-x', name='n', statement='')


def test_autonomy_low_risk_no_gate():
    rule = resolve_autonomy(RiskLevel.R1)
    assert rule.requires_human_gate is False
    assert rule.autonomy == 'SEMI_AUTONOMOUS'


def test_autonomy_high_risk_requires_gate():
    rule = resolve_autonomy(RiskLevel.R3)
    assert rule.requires_human_gate is True
    assert rule.autonomy == 'HUMAN_GATE'


def test_autonomy_safety_critical_dual_review():
    rule = resolve_autonomy(RiskLevel.R4)
    assert rule.requires_human_gate is True
    assert rule.requires_dual_review is True


def test_autonomy_unknown_risk_rejected():
    import pytest
    with pytest.raises(ValueError):
        resolve_autonomy('R9')
