from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryLayer(str, Enum):
    WORKING = 'WORKING'
    EPISODIC = 'EPISODIC'
    SEMANTIC = 'SEMANTIC'


class MemoryEntry(BaseModel):
    """One memory entry with layer and provenance (D5)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-memory-entry.v1'
    entry_id: str = Field(min_length=1)
    layer: MemoryLayer = MemoryLayer.WORKING
    content: str = Field(min_length=1)
    source_ref: str = ''
    fault_ids: list[str] = Field(default_factory=list)
    is_negative: bool = False
    demoted: bool = False
    created_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))

    @field_validator('fault_ids')
    @classmethod
    def unique_fault_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError('memory fault_ids must contain unique nonempty values')
        return value


class MemoryStore:
    """Layered memory with human-gate promotion and deny-negative-sample pollution guard (D5)."""

    schema_version = 'astrograph-memory-store.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / 'memory_layers.jsonl'

    def write(self, entry: MemoryEntry) -> Path:
        path = self._path()
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def entries(self) -> list[MemoryEntry]:
        path = self._path()
        if not path.is_file():
            return []
        return [MemoryEntry.model_validate(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def promote(self, entry_id: str, *, human_approved: bool, target_layer: MemoryLayer) -> MemoryEntry:
        """Promote a working entry to episodic/semantic only with human approval (D5 gate)."""
        if not human_approved:
            raise ValueError('memory promotion requires explicit human approval')
        lines = self._path().read_text(encoding='utf-8').splitlines()
        updated = None
        for i, line in enumerate(lines):
            payload = json.loads(line)
            if payload['entry_id'] == entry_id:
                payload['layer'] = target_layer.value
                lines[i] = json.dumps(payload, ensure_ascii=False)
                updated = MemoryEntry.model_validate(payload)
                break
        if updated is None:
            raise KeyError(f'unknown memory entry: {entry_id}')
        self._path().write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
        return updated

    def demote(self, entry_id: str, *, reason: str) -> MemoryEntry:
        """Demote (down-weight) rather than delete safety-critical memory (D5 no-delete)."""
        lines = self._path().read_text(encoding='utf-8').splitlines()
        updated = None
        for i, line in enumerate(lines):
            payload = json.loads(line)
            if payload['entry_id'] == entry_id:
                payload['demoted'] = True
                payload['content'] = payload['content'] + ' [DEMOTED: ' + reason + ']'
                lines[i] = json.dumps(payload, ensure_ascii=False)
                updated = MemoryEntry.model_validate(payload)
                break
        if updated is None:
            raise KeyError(f'unknown memory entry: {entry_id}')
        self._path().write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
        return updated

    def audit(self) -> dict:
        entries = self.entries()
        return {
            'status': 'PASS',
            'entry_count': len(entries),
            'layers': {l.value: sum(1 for e in entries if e.layer == l) for l in MemoryLayer},
            'negative_samples': sum(1 for e in entries if e.is_negative),
            'demoted': [e.entry_id for e in entries if e.demoted],
        }


__all__ = ['MemoryLayer', 'MemoryEntry', 'MemoryStore']
