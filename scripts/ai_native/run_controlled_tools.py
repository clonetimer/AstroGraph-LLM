#!/usr/bin/env python3
"""B3: constrained Agentic - LLM agent tools with S1 permission + budget + audit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.agents.controlled_tools import ControlledToolRegistry, RegisteredTool, ToolPermission

EVIDENCE = ROOT / 'governance/evidence/ai_native/governance_health/controlled_tools_20260824.json'


def main() -> int:
    import json
    registry = ControlledToolRegistry(max_calls_per_agent=10)
    registry.register(RegisteredTool(tool_id='kg.query', description='read-only KG fault query (S1 ALLOW)'),
                      lambda a: {'fault': 'ADCS_RW_JAM', 'evidence_nodes': 3})
    registry.register(RegisteredTool(tool_id='evidence.fetch', description='fetch episode evidence facts'),
                      lambda a: {'facts': ['command_residual 0.4', 'speed_residual 0.3']})
    registry.register(RegisteredTool(tool_id='config.write', description='write active config', permission=ToolPermission.DENY),
                      lambda a: {})
    results = []
    results.append(registry.invoke('llm-reasoner', 'kg.query', {'fault_id': 'ADCS_RW_JAM'}))
    results.append(registry.invoke('llm-reasoner', 'evidence.fetch', {'episode': 'EP-1'}))
    results.append(registry.invoke('llm-reasoner', 'config.write', {'threshold': 0.01}))
    for r in results:
        print(f'[b3] {r["record"]["tool_id"]}: permitted={r["permitted"]}', flush=True)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        'agent': 'llm-reasoner', 'results': results,
        'audit': registry.audit_summary(),
        'note': 'LLM agent may query KG/evidence read-only; active-config writes DENY; every call audited within S1 budget (B3)',
    }, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(f'[b3] audit: {registry.audit_summary()}', flush=True)
    print(f'[b3] evidence -> {EVIDENCE}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())