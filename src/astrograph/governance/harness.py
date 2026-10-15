from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class HarnessQuadrant(str, Enum):
    FEEDFORWARD_DETERMINISTIC = 'FEEDFORWARD_DETERMINISTIC'
    FEEDFORWARD_REASONING = 'FEEDFORWARD_REASONING'
    FEEDBACK_DETERMINISTIC = 'FEEDBACK_DETERMINISTIC'
    FEEDBACK_REASONING = 'FEEDBACK_REASONING'


class DiagnosticPrinciple(BaseModel):
    """One feedforward reasoning-layer principle (conservatism / priority)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-diagnostic-principle.v1'
    principle_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    domain: str = Field(default='diagnosis')
    rationale: str = ''
    source_ref: str = ''


class PeerReviewRecord(BaseModel):
    """One feedback reasoning-layer AI/peer review record."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-peer-review-record.v1'
    review_id: str = Field(min_length=1)
    reviewed_artifact: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    verdict: str = Field(pattern=r'^(APPROVED|REVISIONS|REJECTED)$')
    findings: list[str] = Field(default_factory=list)
    reviewed_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class HarnessAssessment(BaseModel):
    """Four-quadrant Harness completeness assessment."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-harness-assessment.v1'
    assessed_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))
    quadrants: dict[str, dict] = Field(default_factory=dict)
    missing_quadrants: list[str] = Field(default_factory=list)
    overall: str = Field(pattern=r'^(COMPLETE|GAPS)$')


class RiskLevel(str, Enum):
    R0 = 'R0'
    R1 = 'R1'
    R2 = 'R2'
    R3 = 'R3'
    R4 = 'R4'


class AutonomyRule(BaseModel):
    """One risk-level -> autonomy mapping (A2 risk-based autonomy)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-autonomy-rule.v1'
    rule_id: str = Field(min_length=1)
    risk_level: RiskLevel
    action: str = Field(min_length=1)
    autonomy: str = Field(pattern=r'^(FULL_AUTONOMOUS|SEMI_AUTONOMOUS|HUMAN_GATE|HUMAN_GATE_DUAL_REVIEW)$')
    requires_human_gate: bool = False
    requires_dual_review: bool = False


DEFAULT_AUTONOMY_RULES = [
    AutonomyRule(rule_id='AUT-R0', risk_level=RiskLevel.R0, action='metrics/ledger/triage', autonomy='FULL_AUTONOMOUS'),
    AutonomyRule(rule_id='AUT-R1', risk_level=RiskLevel.R1, action='feature/evidence/kg-readonly', autonomy='SEMI_AUTONOMOUS'),
    AutonomyRule(rule_id='AUT-R2', risk_level=RiskLevel.R2, action='hypothesis/threshold-suggestion', autonomy='HUMAN_GATE', requires_human_gate=True),
    AutonomyRule(rule_id='AUT-R3', risk_level=RiskLevel.R3, action='promotion/config-change/knowledge-adoption', autonomy='HUMAN_GATE', requires_human_gate=True),
    AutonomyRule(rule_id='AUT-R4', risk_level=RiskLevel.R4, action='attitude-critical-diagnosis-publish', autonomy='HUMAN_GATE_DUAL_REVIEW', requires_human_gate=True, requires_dual_review=True),
]


def resolve_autonomy(risk_level: RiskLevel, rules: Sequence[AutonomyRule] | None = None) -> AutonomyRule:
    """Return the autonomy rule for a given risk level (A2)."""
    if isinstance(risk_level, str):
        try:
            risk_level = RiskLevel(risk_level)
        except ValueError as exc:
            raise ValueError(f'unknown risk level: {risk_level}') from exc
    table = {rule.risk_level: rule for rule in (rules or DEFAULT_AUTONOMY_RULES)}
    try:
        return table[risk_level]
    except KeyError as exc:
        raise ValueError(f'no autonomy rule for risk level: {risk_level.value}') from exc


def assess_harness_completeness(
    *,
    principles: Sequence[DiagnosticPrinciple] | None = None,
    peer_reviews: Sequence[PeerReviewRecord] | None = None,
    has_feedforward_deterministic: bool = True,
    has_feedback_deterministic: bool = True,
) -> HarnessAssessment:
    quadrants = {
        HarnessQuadrant.FEEDFORWARD_DETERMINISTIC.value: {'present': has_feedforward_deterministic, 'count': 1 if has_feedforward_deterministic else 0},
        HarnessQuadrant.FEEDFORWARD_REASONING.value: {'present': bool(principles), 'count': len(principles or [])},
        HarnessQuadrant.FEEDBACK_DETERMINISTIC.value: {'present': has_feedback_deterministic, 'count': 1 if has_feedback_deterministic else 0},
        HarnessQuadrant.FEEDBACK_REASONING.value: {'present': bool(peer_reviews), 'count': len(peer_reviews or [])},
    }
    missing = [name for name, item in quadrants.items() if not item['present']]
    return HarnessAssessment(
        quadrants=quadrants,
        missing_quadrants=missing,
        overall='COMPLETE' if not missing else 'GAPS',
    )


__all__ = [
    'HarnessQuadrant', 'DiagnosticPrinciple', 'PeerReviewRecord', 'HarnessAssessment',
    'RiskLevel', 'AutonomyRule', 'DEFAULT_AUTONOMY_RULES', 'resolve_autonomy',
    'assess_harness_completeness',
]