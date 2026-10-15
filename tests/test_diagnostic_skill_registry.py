"""D1: diagnostic skill assetization tests."""
from __future__ import annotations

import pytest

from astrograph.data_driven.skill_registry import DiagnosticSkill, DiagnosticSkillRegistry, SkillLoadTier


def _skill(skill_id: str = 'S-1', tier: SkillLoadTier = SkillLoadTier.CORE) -> DiagnosticSkill:
    return DiagnosticSkill(
        skill_id=skill_id, name='residual-evidence-operator', load_tier=tier,
        description='Computes command-residual evidence fact.',
        inputs=['command_residual_norm'], outputs=['evidence_fact'],
    )


def test_register_and_load(tmp_path):
    registry = DiagnosticSkillRegistry(tmp_path)
    registry.register(_skill())
    skills = registry.load_by_tier()
    assert len(skills) == 1
    assert skills[0].skill_id == 'S-1'


def test_load_by_tier(tmp_path):
    registry = DiagnosticSkillRegistry(tmp_path)
    registry.register(_skill('S-1', SkillLoadTier.CORE))
    registry.register(_skill('S-2', SkillLoadTier.ON_DEMAND))
    assert len(registry.load_by_tier(SkillLoadTier.CORE)) == 1
    assert len(registry.load_by_tier(SkillLoadTier.ON_DEMAND)) == 1


def test_record_failure_archives(tmp_path):
    registry = DiagnosticSkillRegistry(tmp_path)
    registry.register(_skill())
    registry.record_failure('S-1', 'observed activity below threshold on one episode')
    skills = registry.load_by_tier()
    assert skills[0].failure_notes == ['observed activity below threshold on one episode']


def test_record_failure_unknown_skill(tmp_path):
    registry = DiagnosticSkillRegistry(tmp_path)
    registry.register(_skill())
    with pytest.raises(KeyError):
        registry.record_failure('NOPE', 'x')


def test_audit(tmp_path):
    registry = DiagnosticSkillRegistry(tmp_path)
    registry.register(_skill())
    registry.register(_skill('S-2', SkillLoadTier.EXTENDED))
    audit = registry.audit()
    assert audit['status'] == 'PASS'
    assert audit['skill_count'] == 2
    assert audit['tiers']['CORE'] == 1
    assert audit['tiers']['EXTENDED'] == 1


def test_skill_requires_id():
    with pytest.raises(Exception):
        DiagnosticSkill(skill_id='', name='x')