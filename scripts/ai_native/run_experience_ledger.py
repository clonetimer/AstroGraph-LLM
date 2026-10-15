#!/usr/bin/env python3
"""D3: record and backfill experience assets from this session's feedback cycles."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.data_driven.experience import ExperienceLedger, ExperienceRecord, ExperienceStatus
from astrograph.data_driven.memory_layers import MemoryEntry, MemoryLayer, MemoryStore

LEDGER_ROOT = ROOT / 'governance/evidence/ai_native/experience'
MEMORY_ROOT = ROOT / 'governance/evidence/ai_native/memory'


def main() -> int:
    ledger = ExperienceLedger(LEDGER_ROOT)
    entries = [
        ExperienceRecord(
            experience_id='EXP-CYCLE1-PHM', source_cycle_id='cycle-1-unified-phm', dataset_id='phm',
            proposal_summary='Promote unified PHM adapter to production Registry', decision='WITHHELD (PENDING_HUMAN_GATE)',
        ),
        ExperienceRecord(
            experience_id='EXP-GYRO-DISAMBIG', source_cycle_id='p4-suggestion-gate', dataset_id='adcs',
            proposal_summary='GYRO upper threshold recalibration with AND guard', decision='APPROVED (human gate)',
        ),
    ]
    latest = {item.experience_id: item for item in ledger.records()}
    for e in entries:
        if e.experience_id not in latest:
            ledger.record(e)
            print(f'[d3] recorded {e.experience_id} ({e.decision})', flush=True)
    # Backfill outcomes + lessons (real results from this session).
    latest = {item.experience_id: item for item in ledger.records()}
    if latest['EXP-CYCLE1-PHM'].status is ExperienceStatus.RECORDED:
        ledger.backfill_outcome('EXP-CYCLE1-PHM',
            outcome='5-fold CV: unified 0.8578 vs control 0.8410, not uniformly better; C1/C2 stabilization both negative',
            lesson='PHM unified adapter stays research channel; single-split gains must be multi-fold validated; full-task equal-weight training is optimal')
    if latest['EXP-GYRO-DISAMBIG'].status is ExperienceStatus.RECORDED:
        ledger.backfill_outcome('EXP-GYRO-DISAMBIG',
            outcome='gyro_bias_step now supported {ADCS_GYRO_BIAS_DRIFT}, rw_jamming keeps {ADCS_RW_JAM}; frozen regression 37+ passed',
            lesson='Conjunctive (AND) evidence conditions enable safe threshold broadening for overlapping-fault disambiguation')
    latest = {item.experience_id: item for item in ledger.records()}
    if latest['EXP-GYRO-DISAMBIG'].status is not ExperienceStatus.REUSE_APPROVED:
        ledger.approve_reuse(
            'EXP-GYRO-DISAMBIG',
            reviewer='user-approved-new-scheme-20260831',
            fault_ids=['ADCS_GYRO_BIAS_DRIFT', 'ADCS_RW_JAM'],
            evidence_definition_ids=[
                'ED-ADCS-V522-GYRO-BIAS-INNOVATION-LOWER-V1',
                'ED-ADCS-V522-GYRO-BIAS-INNOVATION-UPPER-V1',
                'ED-ADCS-V522-RW-JAM-COMMAND-RESIDUAL-V1',
                'ED-ADCS-V522-RW-JAM-SPEED-RESIDUAL-V1',
            ],
            source_refs=['docs/CURRENT_ENGINEERING_POLICY.md', 'approval:new-scheme-20260831'],
        )
    memory = MemoryStore(MEMORY_ROOT)
    if not any(item.entry_id == 'MEM-GYRO-AND-V2' for item in memory.entries()):
        memory.write(MemoryEntry(
            entry_id='MEM-GYRO-AND-V2',
            layer=MemoryLayer.WORKING,
            content='GYRO_BIAS_DRIFT supported after AND-disambiguation fix; RW_JAM preserved',
            source_ref='approval:new-scheme-20260831',
            fault_ids=['ADCS_GYRO_BIAS_DRIFT', 'ADCS_RW_JAM'],
        ))
        memory.promote(
            'MEM-GYRO-AND-V2', human_approved=True, target_layer=MemoryLayer.EPISODIC
        )
    print(f'[d3] audit: {ledger.audit()}', flush=True)
    print(f'[d5] memory audit: {memory.audit()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
