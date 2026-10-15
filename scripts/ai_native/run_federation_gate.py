#!/usr/bin/env python3
"""AI-Native Phase 5: capability-federation human gate drill.

Applies a human review to the federation insights recorded on 2026-08-24: the PHM
transfer insight (frozen-backbone finetune +0.0044 E2E) is VERIFIED as a research
federation candidate, and the ADCS insight (-0.0305 E2E) is REJECTED. Both are
appended to the append-only federation ledger with the human decision recorded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.federation import (
    CapabilityFederationRecord, CapabilityFederationRegistry, FederationStatus, TransferInsight,
)

REGISTRY_ROOT = ROOT / 'governance/evidence/ai_native/federation'


def main() -> int:
    registry = CapabilityFederationRegistry(REGISTRY_ROOT)
    records = registry.read('fed-unified-backbone-20260824')
    if not records:
        print('[error] federation record not found', file=sys.stderr)
        return 1
    base = records[-1]
    reviewed_insights = []
    for insight in base.insights:
        if 'PHM' in insight.insight_id and insight.delta_macro_f1 > 0:
            updated = insight.model_copy(update={'status': FederationStatus.VERIFIED, 'reviewed_by': 'user-approved 2026-08-24 (research-channel federation gate)'})
            verdict = 'VERIFIED (research candidate)'
        else:
            updated = insight.model_copy(update={'status': FederationStatus.REJECTED, 'reviewed_by': 'user-approved 2026-08-24 (negative transfer)'})
            verdict = 'REJECTED (negative transfer)'
        reviewed_insights.append(updated)
        print(f'[gate] {insight.insight_id}: {verdict} (delta={insight.delta_macro_f1:+.4f})', flush=True)
    reviewed = CapabilityFederationRecord(
        record_id='fed-unified-backbone-20260824-human-gate',
        backbone_id=base.backbone_id,
        member_domains=base.member_domains,
        insights=reviewed_insights,
        decision=FederationStatus.VERIFIED,
        rationale='Human gate applied: PHM transfer insight verified as research federation candidate; ADCS negative transfer rejected. No production promotion.',
    )
    path = registry.append(reviewed)
    print(f'[gate] appended -> {path}', flush=True)
    audit = registry.audit()
    print(f'[audit] {json.dumps(audit, ensure_ascii=False)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())