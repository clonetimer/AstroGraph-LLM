#!/usr/bin/env python3
"""E4: failure-threshold alarm demo (deep-LLM path degrades to deterministic on repeated failure)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.data_driven.failure_alarm import FailureAlarmMonitor

EVIDENCE = ROOT / 'governance/evidence/ai_native/governance_health/failure_alarm_20260824.json'


def main() -> int:
    monitor = FailureAlarmMonitor(threshold=3)
    route = 'deep-kg-llm'
    events = []
    for i in range(4):
        state = monitor.record_failure(route)
        events.append({'attempt': i + 1, 'consecutive_failures': state.consecutive_failures, 'degraded': state.degraded, 'alarm': state.alarm_raised})
        if state.degraded:
            print(f'[e4] attempt {i+1}: DEGRADED -> route to fast-deterministic + alarm raised', flush=True)
        else:
            print(f'[e4] attempt {i+1}: failure {state.consecutive_failures}/{monitor.threshold}', flush=True)
    import json
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        'route': route, 'threshold': monitor.threshold, 'events': events,
        'action': 'continuous failures past threshold degrade deep-LLM path to fast-deterministic and raise alarm (E4)',
    }, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(f'[e4] evidence -> {EVIDENCE}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())