#!/usr/bin/env python3
"""AI-Native Phase 5: capability-federation insight registration from Phase 3 results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.federation import (
    CapabilityFederationRecord, CapabilityFederationRegistry, TransferInsight,
)

REPORT = ROOT / 'governance/evidence/ai_native/unified_baseline_20260823/unified_baseline_report.json'
REGISTRY_ROOT = ROOT / 'governance/evidence/ai_native/federation'


def main() -> int:
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    probes = report['unified']['transfer_probes']
    phm_probe = probes['phm_adapter_finetune']
    adcs_probe = probes['adcs_adapter_finetune']
    phm_insight = TransferInsight(
        insight_id='TI-20260824-PHM-TRANSFER',
        source_domain='adcs', target_domain='phm',
        shared_backbone='unified-dilated-residual-v1', metric='e2e_macro_f1',
        delta_macro_f1=phm_probe['delta_e2e_macro_f1'],
        target_baseline_macro_f1=0.8439,
        evidence_ref='unified_baseline_20260823/unified_baseline_report.json#transfer_probes.phm_adapter_finetune',
    )
    adcs_insight = TransferInsight(
        insight_id='TI-20260824-ADCS-TRANSFER',
        source_domain='phm', target_domain='adcs',
        shared_backbone='unified-dilated-residual-v1', metric='e2e_macro_f1',
        delta_macro_f1=adcs_probe['delta_e2e_macro_f1'],
        target_baseline_macro_f1=0.8293,
        evidence_ref='unified_baseline_20260823/unified_baseline_report.json#transfer_probes.adcs_adapter_finetune',
    )
    rationale = (
        'Frozen-backbone transfer probes from Phase 3: PHM adapter finetune gains +0.0044 E2E from the shared '
        'backbone (federation candidate), ADCS adapter finetune loses -0.0305 E2E (not a federation candidate '
        'for ADCS; independent training retained).'
    )
    record = CapabilityFederationRecord(
        record_id='fed-unified-backbone-20260824',
        backbone_id='unified-dilated-residual-v1',
        member_domains=['adcs', 'phm'],
        insights=[phm_insight, adcs_insight],
        rationale=rationale,
    )
    registry = CapabilityFederationRegistry(REGISTRY_ROOT)
    path = registry.append(record)
    print(f'[federation] {path}', flush=True)
    print(f'[insights] {len(record.insights)}', flush=True)
    for item in record.insights:
        print(f'  - {item.insight_id}: delta={item.delta_macro_f1:+.4f} relative={item.relative_change:+.2%}', flush=True)
    print(f'[audit] {json.dumps(registry.audit(), ensure_ascii=False)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())