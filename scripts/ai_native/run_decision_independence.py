#!/usr/bin/env python3
"""B4: audit and record decision-path independence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.governance.decision_independence import audit_decision_independence

EVIDENCE = ROOT / 'governance/evidence/ai_native/governance_health/decision_independence_20260824.json'


def main() -> int:
    report = audit_decision_independence(ROOT)
    payload = report.model_dump(mode='json')
    note = ('Arbitration path (DiagnosticDecisionGovernor + deterministic core comparison) is explicitly '
            'independent of the LLM reasoning backend; generated hypotheses are shadow conclusions until the '
            'governor and deterministic core confirm them.')
    payload['architecture_note'] = note
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(f'[b4] overall={report.overall}', flush=True)
    for c in report.checks:
        status = 'PASS' if c['passed'] else 'FAIL'
        print(f'[b4] {c["check"]}: {status} - {c["detail"]}', flush=True)
    print(f'[b4] evidence -> {EVIDENCE}', flush=True)
    return 0 if report.overall == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())