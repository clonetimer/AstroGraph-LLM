from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class ProvenanceMarkedEvidence(BaseModel):
    """E2: evidence carrying a source marker for idempotent aggregation."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-provenance-marked-evidence.v1'
    evidence_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    feature_definition_id: str = Field(min_length=1)
    value: float
    source_marker: str = Field(min_length=1)  # episode_id:feature:index -> idempotency key


class IdempotentEvidenceAggregator:
    """Deduplicates evidence by source marker so replayed episodes are not double-counted (E2)."""

    schema_version = 'astrograph-idempotent-evidence-aggregator.v1'

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._aggregates: dict[str, list[float]] = {}

    def ingest(self, evidence: Sequence[ProvenanceMarkedEvidence]) -> dict[str, list[float]]:
        """Ingest evidence, skipping already-seen source markers; return per-feature observations."""
        added = 0
        for item in evidence:
            if item.source_marker in self._seen:
                continue
            self._seen.add(item.source_marker)
            self._aggregates.setdefault(item.feature_definition_id, []).append(item.value)
            added += 1
        return {k: list(v) for k, v in self._aggregates.items()}

    def replay_count(self) -> int:
        return len(self._seen)


__all__ = ['ProvenanceMarkedEvidence', 'IdempotentEvidenceAggregator']