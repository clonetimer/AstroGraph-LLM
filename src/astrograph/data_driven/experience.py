from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExperienceStatus(str, Enum):
    RECORDED = 'RECORDED'
    OUTCOME_BACKFILLED = 'OUTCOME_BACKFILLED'
    LESSON_EXTRACTED = 'LESSON_EXTRACTED'
    REUSE_APPROVED = 'REUSE_APPROVED'


class ExperienceRecord(BaseModel):
    """One experience-asset record with outcome backfill and lesson (D3)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-experience-record.v1'
    experience_id: str = Field(min_length=1)
    source_cycle_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    proposal_summary: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    outcome: str = ''
    lesson: str = ''
    fault_ids: list[str] = Field(default_factory=list)
    evidence_definition_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    reuse_reviewed_by: str = ''
    status: ExperienceStatus = ExperienceStatus.RECORDED
    recorded_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))

    @field_validator('fault_ids', 'evidence_definition_ids', 'source_refs')
    @classmethod
    def unique_nonempty_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError('experience lineage lists must contain unique nonempty values')
        return value


class ExperienceLedger:
    """Append-only experience ledger (D3: from audit asset to experience asset)."""

    schema_version = 'astrograph-experience-ledger.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / 'experience_ledger.jsonl'

    def record(self, entry: ExperienceRecord) -> Path:
        path = self._path()
        if any(record.experience_id == entry.experience_id for record in self.records()):
            raise ValueError(f'duplicate experience: {entry.experience_id}')
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def backfill_outcome(self, experience_id: str, *, outcome: str, lesson: str) -> None:
        """Backfill outcome + lesson after the decision's real-world effect is known."""
        path = self._path()
        if not path.is_file():
            raise KeyError(f'experience ledger empty: {experience_id}')
        current = next(
            (record for record in reversed(self.records()) if record.experience_id == experience_id),
            None,
        )
        if current is None:
            raise KeyError(f'unknown experience: {experience_id}')
        updated = current.model_copy(
            update={
                'outcome': outcome,
                'lesson': lesson,
                'status': ExperienceStatus.LESSON_EXTRACTED,
            }
        )
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(updated.model_dump(mode='json'), ensure_ascii=False) + chr(10))

    def approve_reuse(
        self,
        experience_id: str,
        *,
        reviewer: str,
        fault_ids: list[str],
        evidence_definition_ids: list[str] | None = None,
        source_refs: list[str] | None = None,
    ) -> ExperienceRecord:
        """Approve an extracted lesson for read-only retrieval reuse.

        Approval creates another append-only event.  It does not publish to the
        active knowledge graph or change diagnosis/fusion policy.
        """
        if not reviewer.strip():
            raise ValueError('experience reuse approval requires reviewer identity')
        if not fault_ids or len(fault_ids) != len(set(fault_ids)):
            raise ValueError('experience reuse approval requires unique fault_ids')
        current = next(
            (record for record in reversed(self.records()) if record.experience_id == experience_id),
            None,
        )
        if current is None:
            raise KeyError(f'unknown experience: {experience_id}')
        if current.status not in {ExperienceStatus.LESSON_EXTRACTED, ExperienceStatus.REUSE_APPROVED}:
            raise ValueError('experience lesson must be extracted before reuse approval')
        if not current.lesson.strip() or not current.outcome.strip():
            raise ValueError('experience reuse requires nonempty outcome and lesson')
        updated = current.model_copy(update={
            'fault_ids': sorted(fault_ids),
            'evidence_definition_ids': sorted(set(evidence_definition_ids or [])),
            'source_refs': sorted(set(source_refs or [])),
            'reuse_reviewed_by': reviewer.strip(),
            'status': ExperienceStatus.REUSE_APPROVED,
        })
        with self._path().open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(updated.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return updated

    def records(self) -> list[ExperienceRecord]:
        path = self._path()
        if not path.is_file():
            return []
        return [
            ExperienceRecord.model_validate(json.loads(line))
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def audit(self) -> dict:
        path = self._path()
        if not path.is_file():
            return {'status': 'PASS', 'record_count': 0, 'lessons_extracted': 0}
        history = self.records()
        records = {record.experience_id: record for record in history}
        return {
            'status': 'PASS',
            'record_count': len(records),
            'event_count': len(history),
            'lessons_extracted': sum(
                1 for record in records.values()
                if record.status in {ExperienceStatus.LESSON_EXTRACTED, ExperienceStatus.REUSE_APPROVED}
            ),
            'reuse_approved': sum(
                1 for record in records.values()
                if record.status == ExperienceStatus.REUSE_APPROVED
            ),
        }


__all__ = ['ExperienceStatus', 'ExperienceRecord', 'ExperienceLedger']
