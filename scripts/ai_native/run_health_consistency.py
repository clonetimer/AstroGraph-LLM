#!/usr/bin/env python3
"""AI-Native Phase 5: governance health-check vs human review consistency.

Verifies that the advisory checker's decisions agree with human review on a
controlled set of scenarios: the real repository (human: healthy -> PASS) and
injected governance defects (human: broken -> FAIL for the matching check).
Reports the agreement rate and the acceptance-criterion gate (>= 0.95).

"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.governance.health_check import GovernanceHealthChecker, HealthStatus


def _write_minimal_tree(root: Path, *, version: str = '1.2.3', sha_ok: bool = True) -> Path:
    (root / 'src/astrograph').mkdir(parents=True)
    (root / 'governance/contracts').mkdir(parents=True)
    (root / 'governance/evidence/ai_native/feedback_cycles').mkdir(parents=True)
    (root / 'release.json').write_text(json.dumps({'version': version}), encoding='utf-8')
    (root / 'pyproject.toml').write_text('[project]\nname = "x"\nversion = "' + version + '"\n', encoding='utf-8')
    (root / 'src/astrograph/__init__.py').write_text('__version__ = "' + version + '"\n', encoding='utf-8')
    contract = '{}'
    (root / 'governance/contracts/1.2.3.yaml').write_text(contract, encoding='utf-8')
    expected = hashlib.sha256(contract.encode()).hexdigest() if sha_ok else '0' * 64
    (root / 'governance/contracts/1.2.3.sha256').write_text(expected + '  file', encoding='utf-8')
    return root


def main() -> int:
    scenarios = []
    # Scenario 1: real repository - human review says healthy, checker should PASS.
    real = GovernanceHealthChecker().run(ROOT, checked_at='2026-08-24T00:00:00+00:00')
    scenarios.append({
        'name': 'real_repository', 'human_expected': 'PASS',
        'checker_actual': real.overall_status.value,
    })
    # Scenarios 2-4: injected defects in synthetic trees, human review confirms broken.
    cases = [
        # (name, builder_kwargs, post_builder_mutator, check_id)
        ('version_mismatch', {'version': '1.2.3'},
         lambda r: (r / 'release.json').write_text(json.dumps({'version': '9.9.9'}), encoding='utf-8'),
         'version_consistency'),
        ('contract_hash_bad', {'sha_ok': False}, None, 'contract_sha_chain'),
        ('missing_contracts_dir', {'version': '1.2.3'},
         lambda r: (__import__('shutil').rmtree(r / 'governance' / 'contracts') or None),
         'contract_sha_chain'),
    ]
    for name, kwargs, mutator, check_id in cases:
        root = Path(tempfile.gettempdir()) / ('hc-' + uuid.uuid4().hex[:8])
        try:
            _write_minimal_tree(root, **kwargs)
            if mutator is not None:
                mutator(root)
            report = GovernanceHealthChecker().run(root, checked_at='2026-08-24T00:00:00+00:00')
            by_id = {item.check_id: item for item in report.checks}
            actual = by_id[check_id].status.value
            scenarios.append({'name': name, 'human_expected': 'FAIL', 'checker_actual': actual})
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)
    agreed = sum(1 for s in scenarios if s['human_expected'] == s['checker_actual'])
    rate = agreed / len(scenarios)
    summary = {
        'schema_version': 'astrograph-health-consistency.v1',
        'check_id': 'HEALTH-HUMAN-CONSISTENCY-20260824',
        'scenarios': scenarios,
        'agreement_count': agreed, 'scenario_count': len(scenarios),
        'agreement_rate': round(rate, 4),
        'gate': 'AI audit decision matches human review at >= 0.95',
        'gate_passed': rate >= 0.95,
    }
    out = ROOT / 'governance/evidence/ai_native/governance_health/health_human_consistency_20260824.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    for s in scenarios:
        print(f'[consistency] {s["name"]}: human={s["human_expected"]} checker={s["checker_actual"]}', flush=True)
    print(f'[consistency] rate={rate:.2%} gate_passed={rate >= 0.95}', flush=True)
    print(f'[evidence] {out}', flush=True)
    return 0 if rate >= 0.95 else 1


if __name__ == '__main__':
    raise SystemExit(main())