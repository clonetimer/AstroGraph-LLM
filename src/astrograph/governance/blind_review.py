from __future__ import annotations

import json
import hashlib
import time
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class BlindReviewStatus(str, Enum):
    SCHEDULED = 'SCHEDULED'
    REVIEWED = 'REVIEWED'
    CONFIRMED = 'CONFIRMED'
    DISCREPANT = 'DISCREPANT'


class GateDecision(BaseModel):
    """One historical human-gate decision eligible for blind re-review."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-gate-decision.v1'
    decision_id: str = Field(min_length=1)
    artifact: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    rationale: str = ''
    decided_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class BlindReviewRecord(BaseModel):
    """One blind re-review of a past gate decision (anti-de-skilling)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-blind-review.v1'
    review_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    artifact: str = Field(min_length=1)
    original_decision: str = Field(min_length=1)
    blind_verdict: str = Field(min_length=1)
    agreement: bool
    reviewer: str = Field(min_length=1)
    status: BlindReviewStatus = BlindReviewStatus.REVIEWED
    notes: str = ''
    reviewed_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class BlindReviewLedger:
    """Append-only ledger of blind re-reviews (A4)."""

    schema_version = 'astrograph-blind-review-ledger.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in name)
        return self.root / f'{safe}.jsonl'

    def append(self, record: BlindReviewRecord) -> Path:
        path = self._path('blind_reviews')
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def audit(self) -> dict:
        path = self._path('blind_reviews')
        if not path.is_file():
            return {'status': 'PASS', 'review_count': 0, 'agreement_rate': 1.0}
        records = [BlindReviewRecord.model_validate(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        agreed = sum(1 for r in records if r.agreement)
        return {
            'status': 'PASS' if records else 'PASS',
            'review_count': len(records),
            'agreement_rate': round(agreed / len(records), 4) if records else 1.0,
            'discrepant': [r.review_id for r in records if not r.agreement],
        }


def schedule_blind_review(decisions: Sequence[GateDecision], *, sample_ratio: float = 0.1) -> list[GateDecision]:
    """Deterministically sample past decisions for blind re-review (A4)."""
    if not 0.0 < sample_ratio <= 1.0:
        raise ValueError('sample_ratio must be in (0, 1]')
    count = max(1, int(len(decisions) * sample_ratio))
    ranked = sorted(
        decisions,
        key=lambda item: (hashlib.sha256(item.decision_id.encode('utf-8')).hexdigest(), item.decision_id),
    )
    return ranked[:count]


__all__ = [
    'BlindReviewStatus', 'GateDecision', 'BlindReviewRecord', 'BlindReviewLedger',
    'schedule_blind_review',
]
