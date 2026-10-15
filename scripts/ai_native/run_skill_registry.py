#!/usr/bin/env python3
"""D1: register core diagnostic skills as governable assets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.data_driven.skill_registry import DiagnosticSkill, DiagnosticSkillRegistry, SkillLoadTier

REGISTRY_ROOT = ROOT / 'governance/evidence/ai_native/skills'


def main() -> int:
    registry = DiagnosticSkillRegistry(REGISTRY_ROOT)
    skills = [
        DiagnosticSkill(skill_id='SKL-RW-COMMAND-RESIDUAL', name='rw-command-residual-operator', load_tier=SkillLoadTier.CORE,
                        description='Computes RW command-execution residual max fact (ADCS_RW_JAM evidence).',
                        inputs=['command_execution_residual_norm'], outputs=['evidence_fact'], version='1.0.0'),
        DiagnosticSkill(skill_id='SKL-GYRO-RW-DISAMBIG', name='gyro-rw-disambiguation-guard', load_tier=SkillLoadTier.CORE,
                        description='AND-conjunctive guard: star-gyro innovation band + RW command residual <= 0.33 (disambiguates GYRO_BIAS_DRIFT from RW_JAM).',
                        inputs=['star_gyro_innovation', 'command_execution_residual_norm'], outputs=['evidence_fact'], version='1.0.0',
                        failure_notes=[]),
        DiagnosticSkill(skill_id='SKL-STAR-DROPOUT', name='star-dropout-validity-operator', load_tier=SkillLoadTier.EXTENDED,
                        description='Computes star-tracker validity false-fraction fact (ADCS_STAR_TRACKER_DROPOUT evidence).',
                        inputs=['solution_valid'], outputs=['evidence_fact'], version='1.0.0'),
        DiagnosticSkill(skill_id='SKL-MULTI-EPISODE-AGG', name='multi-episode-aggregation', load_tier=SkillLoadTier.ON_DEMAND,
                        description='Multi-episode threshold-aggregation with class separation (Phase 4).',
                        inputs=['episode_evidence_bundles'], outputs=['aggregated_suggestions'], version='1.0.0'),
    ]
    for skill in skills:
        registry.register(skill)
        print(f'[d1] registered {skill.skill_id} ({skill.load_tier.value})', flush=True)
    # Record a real failure note (honest: single-episode threshold suggestions were rejected).
    registry.record_failure('SKL-MULTI-EPISODE-AGG', '2026-08-24: single-episode threshold suggestions rejected by frozen regression gate; multi-episode agreement required')
    print(f'[d1] audit: {registry.audit()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())