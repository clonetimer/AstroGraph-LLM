from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field


class FailureCase(BaseModel):
    """One recorded misdiagnosis / missed diagnosis (D2 regression case)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-failure-case.v1'
    case_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    failure_type: str = Field(pattern=r'^(MISDIAGNOSIS|MISSED|FALSE_POSITIVE)$')
    truth_fault_id: str = Field(min_length=1)
    predicted_fault_id: str = Field(min_length=1)
    evidence_summary: str = ''
    feedback_id: str | None = None
    task_id: str | None = None
    result_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    input_data_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    model_refs: list[str] = Field(default_factory=list)
    knowledge_version: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class FailureRegressionSet:
    """Append-only failure-case regression set with replay (D2)."""

    schema_version = 'astrograph-failure-regression-set.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / 'failure_regression_set.jsonl'

    def add(self, case: FailureCase) -> Path:
        path = self._path()
        existing = {item.case_id: item for item in self.cases()}
        if case.case_id in existing:
            if existing[case.case_id] != case:
                raise ValueError('failure case ID reused with different content')
            return path
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(case.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def cases(self) -> list[FailureCase]:
        path = self._path()
        if not path.is_file():
            return []
        return [FailureCase.model_validate(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def replay(self, predictor) -> dict[str, Any]:
        """Re-run every recorded failure case through a predictor and report regressions."""
        cases = self.cases()
        results = []
        for case in cases:
            predicted = predictor(case)
            fixed = predicted == case.truth_fault_id
            results.append({
                'case_id': case.case_id, 'truth': case.truth_fault_id,
                'predicted': predicted, 'fixed': fixed, 'failure_type': case.failure_type,
            })
        fixed_count = sum(1 for r in results if r['fixed'])
        return {
            'case_count': len(cases),
            'fixed_count': fixed_count,
            'fix_rate': round(fixed_count / len(cases), 4) if cases else 1.0,
            'results': results,
        }


__all__ = ['FailureCase', 'FailureRegressionSet']