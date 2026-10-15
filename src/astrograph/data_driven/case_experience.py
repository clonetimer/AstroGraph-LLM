from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ConfirmedCase(BaseModel):
    """D4: a human-gate-confirmed diagnosis case with provenance (experience library)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-confirmed-case.v1'
    case_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    truth_fault_id: str = Field(min_length=1)
    predicted_fault_id: str = Field(min_length=1)
    source_hashes: list[str] = Field(default_factory=list)
    human_confirmed: bool = False
    confirmed_by: str = ''
    is_negative: bool = False
    notes: str = ''
    confirmed_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class RetrievedCase(BaseModel):
    """A provenance-preserving exact-match retrieval result."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-retrieved-case.v1'
    case: ConfirmedCase
    match_score: float = Field(ge=0.0, le=1.0)
    matched_fields: list[str] = Field(default_factory=list)


class CaseExperienceLibrary:
    """Append-only library of human-confirmed cases (D4)."""

    schema_version = 'astrograph-case-experience-library.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / 'confirmed_cases.jsonl'

    def add(self, case: ConfirmedCase) -> Path:
        """Only human-confirmed cases may enter the library (D4 gate)."""
        if not case.human_confirmed:
            raise ValueError('case requires explicit human confirmation')
        if not case.confirmed_by.strip():
            raise ValueError('confirmed_by is required for a human-confirmed case')
        if any(re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None for digest in case.source_hashes):
            raise ValueError('source_hashes must contain SHA-256 hex digests')
        if any(existing.case_id == case.case_id for existing in self.cases()):
            raise ValueError(f'duplicate case: {case.case_id}')
        path = self._path()
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(case.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def cases(self) -> list[ConfirmedCase]:
        path = self._path()
        if not path.is_file():
            return []
        return [ConfirmedCase.model_validate(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def audit(self) -> dict:
        cases = self.cases()
        return {
            'status': 'PASS',
            'case_count': len(cases),
            'confirmed': sum(1 for c in cases if c.human_confirmed),
            'negative_samples': sum(1 for c in cases if c.is_negative),
            'episodes': sorted({c.episode_id for c in cases}),
        }

    def retrieve(
        self,
        *,
        truth_fault_id: str | None = None,
        predicted_fault_id: str | None = None,
        include_negative: bool = False,
        limit: int = 10,
    ) -> list[RetrievedCase]:
        """Retrieve confirmed cases while excluding negative cases by default."""
        if limit < 1:
            raise ValueError('limit must be positive')
        requested = {
            key: value
            for key, value in {
                'truth_fault_id': truth_fault_id,
                'predicted_fault_id': predicted_fault_id,
            }.items()
            if value is not None
        }
        results: list[RetrievedCase] = []
        for case in self.cases():
            if case.is_negative and not include_negative:
                continue
            matched = [key for key, value in requested.items() if getattr(case, key) == value]
            if requested and len(matched) != len(requested):
                continue
            score = len(matched) / len(requested) if requested else 1.0
            results.append(RetrievedCase(case=case, match_score=score, matched_fields=matched))
        return sorted(results, key=lambda item: item.case.case_id)[:limit]


__all__ = ['ConfirmedCase', 'RetrievedCase', 'CaseExperienceLibrary']
