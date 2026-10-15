from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field


class AggregationCheckpoint(BaseModel):
    """E1: persistent state of a multi-episode aggregation run (resumable)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-aggregation-checkpoint.v1'
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    completed_episodes: list[str] = Field(default_factory=list)
    pending_episodes: list[str] = Field(default_factory=list)
    partial_observations: dict[str, list[float]] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class CheckpointStore:
    """Append-only JSONL checkpoint store with resume (E1)."""

    schema_version = 'astrograph-checkpoint-store.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in run_id)
        return self.root / f'{safe}.jsonl'

    def save(self, checkpoint: AggregationCheckpoint) -> Path:
        path = self._path(checkpoint.run_id)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(checkpoint.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def latest(self, run_id: str) -> AggregationCheckpoint | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not lines:
            return None
        return AggregationCheckpoint.model_validate(json.loads(lines[-1]))

    def resume(self, run_id: str, *, all_episodes: Sequence[str]) -> tuple[AggregationCheckpoint, list[str]]:
        """Resume from the latest checkpoint; return (checkpoint, still_pending)."""
        latest = self.latest(run_id)
        if latest is None:
            checkpoint = AggregationCheckpoint(run_id=run_id, dataset_id='UNKNOWN', pending_episodes=list(all_episodes))
            return checkpoint, list(all_episodes)
        done = set(latest.completed_episodes)
        pending = [e for e in all_episodes if e not in done]
        return latest, pending


__all__ = ['AggregationCheckpoint', 'CheckpointStore']