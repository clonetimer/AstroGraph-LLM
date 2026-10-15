#!/usr/bin/env python3
"""C5: four-dimension capability access assessment for the unified PHM candidate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.data_driven.capability_access import assess_capability_access

EVIDENCE = ROOT / 'governance/evidence/ai_native/governance_health/capability_access_20260824.json'


def main() -> int:
    decision = assess_capability_access(
        'unified-platform-phm-adapter', '20260823-research',
        data_coverage=0.95, robustness=0.55, explainability=0.6, governance_compliance=True,
        minimum_robustness=0.7,
    )
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    payload = decision.model_dump(mode='json')
    rationale = ('Unified PHM adapter has high class coverage and governance compliance but 5-fold CV robustness '
                 '(0.55, not uniformly better than control) fails the production admission minimum (0.70); '
                 'stays research channel.')
    payload['rationale'] = rationale
    EVIDENCE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(f'[c5] {decision.model_id}@{decision.model_version}: {decision.overall}', flush=True)
    for name, dim in decision.dimensions.items():
        status = 'PASS' if dim['passed'] else 'FAIL'
        print(f'[c5] {name}: {status}', flush=True)
    print(f'[c5] evidence -> {EVIDENCE}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())