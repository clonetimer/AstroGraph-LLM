"""A4: human-gate blind re-review (anti-de-skilling) tests."""
from __future__ import annotations

import pytest

from astrograph.governance.blind_review import (
    BlindReviewLedger, BlindReviewRecord, BlindReviewStatus, GateDecision, schedule_blind_review,
)


def _decisions(n: int = 10) -> list[GateDecision]:
    return [
        GateDecision(decision_id=f'G-{i}', artifact=f'candidate-{i}', decision='APPROVED', rationale=f'r{i}')
        for i in range(n)
    ]


def test_schedule_blind_review_samples_ratio():
    decisions = _decisions(10)
    sampled = schedule_blind_review(decisions, sample_ratio=0.2)
    assert len(sampled) == 2
    assert sampled == schedule_blind_review(list(reversed(decisions)), sample_ratio=0.2)
    assert sampled != decisions[:2]


def test_schedule_min_one_sample():
    decisions = _decisions(5)
    sampled = schedule_blind_review(decisions, sample_ratio=0.05)
    assert len(sampled) == 1


def test_schedule_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        schedule_blind_review(_decisions(), sample_ratio=0.0)
    with pytest.raises(ValueError):
        schedule_blind_review(_decisions(), sample_ratio=1.5)


def test_blind_review_record_agreement():
    record = BlindReviewRecord(
        review_id='BR-1', decision_id='G-0', artifact='candidate-0',
        original_decision='APPROVED', blind_verdict='APPROVED', agreement=True,
        reviewer='human-2',
    )
    assert record.status == BlindReviewStatus.REVIEWED
    assert record.agreement is True


def test_blind_review_ledger_append_and_audit(tmp_path):
    ledger = BlindReviewLedger(tmp_path)
    record = BlindReviewRecord(
        review_id='BR-2', decision_id='G-1', artifact='candidate-1',
        original_decision='APPROVED', blind_verdict='REJECTED', agreement=False,
        reviewer='human-2', notes='discrepancy found',
    )
    ledger.append(record)
    audit = ledger.audit()
    assert audit['review_count'] == 1
    assert audit['agreement_rate'] == 0.0
    assert audit['discrepant'] == ['BR-2']


def test_blind_review_ledger_empty_audit(tmp_path):
    ledger = BlindReviewLedger(tmp_path)
    audit = ledger.audit()
    assert audit['status'] == 'PASS'
    assert audit['review_count'] == 0
    assert audit['agreement_rate'] == 1.0
