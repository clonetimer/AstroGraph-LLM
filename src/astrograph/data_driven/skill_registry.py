from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field


class SkillLoadTier(str, Enum):
    CORE = 'CORE'
    EXTENDED = 'EXTENDED'
    ON_DEMAND = 'ON_DEMAND'


class DiagnosticSkill(BaseModel):
    """A loadable, governable diagnostic skill (D1 assetization)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-diagnostic-skill.v1'
    skill_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    load_tier: SkillLoadTier = SkillLoadTier.ON_DEMAND
    description: str = ''
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    version: str = '1.0.0'
    failure_notes: list[str] = Field(default_factory=list)


class DiagnosticSkillRegistry:
    """Registry of loadable diagnostic skills (D1)."""

    schema_version = 'astrograph-diagnostic-skill-registry.v1'

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / 'diagnostic_skills.jsonl'

    def register(self, skill: DiagnosticSkill) -> Path:
        path = self._path()
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(skill.model_dump(mode='json'), ensure_ascii=False) + chr(10))
        return path

    def load_by_tier(self, tier: SkillLoadTier | None = None) -> list[DiagnosticSkill]:
        path = self._path()
        if not path.is_file():
            return []
        skills = [DiagnosticSkill.model_validate(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if tier is None:
            return skills
        return [s for s in skills if s.load_tier == tier]

    def record_failure(self, skill_id: str, note: str) -> None:
        """Append a failure note to a skill (failure-archive discipline)."""
        path = self._path()
        if not path.is_file():
            raise KeyError(f'skill registry empty: {skill_id}')
        lines = path.read_text(encoding='utf-8').splitlines()
        updated = False
        for i, line in enumerate(lines):
            payload = json.loads(line)
            if payload['skill_id'] == skill_id:
                payload.setdefault('failure_notes', []).append(note)
                lines[i] = json.dumps(payload, ensure_ascii=False)
                updated = True
                break
        if not updated:
            raise KeyError(f'unknown skill: {skill_id}')
        path.write_text(chr(10).join(lines) + chr(10), encoding='utf-8')

    def audit(self) -> dict:
        skills = self.load_by_tier()
        return {
            'status': 'PASS',
            'skill_count': len(skills),
            'tiers': {t.value: sum(1 for s in skills if s.load_tier == t) for t in SkillLoadTier},
            'skills_with_failures': [s.skill_id for s in skills if s.failure_notes],
        }


__all__ = ['SkillLoadTier', 'DiagnosticSkill', 'DiagnosticSkillRegistry']