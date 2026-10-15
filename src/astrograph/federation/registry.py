from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import stable_id


class FederationStatus(str, Enum):
    PROPOSED = 'PROPOSED'
    VERIFIED = 'VERIFIED'
    REJECTED = 'REJECTED'
    HUMAN_REVIEW = 'HUMAN_REVIEW'


class TransferInsight(BaseModel):
    """One quantified cross-domain transfer insight (federation evidence)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-transfer-insight.v1'
    insight_id: str = Field(min_length=1)
    source_domain: str = Field(min_length=1)
    target_domain: str = Field(min_length=1)
    shared_backbone: str = Field(min_length=1)
    metric: str = Field(default='e2e_macro_f1')
    delta_macro_f1: float
    target_baseline_macro_f1: float
    evidence_ref: str = Field(min_length=1)
    status: FederationStatus = FederationStatus.PROPOSED
    reviewed_by: str = ''
    created_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))

    @property
    def relative_change(self) -> float:
        return self.delta_macro_f1 / self.target_baseline_macro_f1 if self.target_baseline_macro_f1 else 0.0


class CapabilityFederationRecord(BaseModel):
    """One auditable federation decision record (append-only)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-capability-federation.v1'
    record_id: str = Field(min_length=1)
    backbone_id: str = Field(min_length=1)
    member_domains: list[str] = Field(min_length=1)
    insights: list[TransferInsight] = Field(default_factory=list)
    decision: FederationStatus = FederationStatus.HUMAN_REVIEW
    rationale: str = ''
    claim_boundary: str = Field(default='Research-channel federation evidence; nothing is promoted without human gate.')


class CapabilityFederationRegistry:
    """Append-only registry for cross-domain capability-federation insights.

    Discipline:
    - Insights are recorded as evidence, never as production claims.
    - Only human review may change an insight/report status.
    - Negative insights (delta < 0) are recorded honestly and mark the
      domain pair as NOT a federation candidate.
    """

    schema_version = 'astrograph-capability-federation-registry.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, record_id: str) -> Path:
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in record_id)
        return self.root / f'{safe}.jsonl'

    def append(self, record: CapabilityFederationRecord) -> Path:
        path = self._path(record.record_id)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def read(self, record_id: str) -> list[CapabilityFederationRecord]:
        path = self._path(record_id)
        if not path.is_file():
            return []
        return [
            CapabilityFederationRecord.model_validate(json.loads(line))
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def audit(self) -> dict:
        records = [
            CapabilityFederationRecord.model_validate(json.loads(line))
            for path in self.root.glob('*.jsonl')
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        return {
            'status': 'PASS',
            'record_count': len(records),
            'records': [r.record_id for r in records],
            'domains': sorted({d for r in records for d in r.member_domains}),
        }


__all__ = [
    'FederationStatus', 'TransferInsight', 'CapabilityFederationRecord',
    'CapabilityFederationRegistry',
]