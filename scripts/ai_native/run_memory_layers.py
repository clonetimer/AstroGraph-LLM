#!/usr/bin/env python3
"""D5: layered memory with human-gate promotion and pollution guard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.data_driven.memory_layers import MemoryEntry, MemoryLayer, MemoryStore

STORE_ROOT = ROOT / 'governance/evidence/ai_native/memory'


def main() -> int:
    store = MemoryStore(STORE_ROOT)
    entries = [
        MemoryEntry(entry_id='MEM-PHM-CV', content='PHM unified adapter: single-split gain must be multi-fold validated; promotion withheld'),
        MemoryEntry(entry_id='MEM-GYRO-AND', layer=MemoryLayer.EPISODIC, content='GYRO_BIAS_DRIFT supported after AND-disambiguation fix; RW_JAM preserved', source_ref='approval_gyro_disambig_20260824.json'),
        MemoryEntry(entry_id='MEM-NEG-THRESH', content='single-episode threshold 0.015945 caused RW_JAM false positive; rejected by regression gate', is_negative=True),
        MemoryEntry(entry_id='MEM-NEG-FILTER', content='C1 task filtering (exclude comm_data) degraded PHM -0.20; negative-transfer hypothesis rejected', is_negative=True),
    ]
    for e in entries:
        store.write(e)
        print(f'[d5] wrote {e.entry_id} ({e.layer.value}{" NEG" if e.is_negative else ""})', flush=True)
    # Human-gate promotion: PHM-CV lesson promoted to semantic (durable experience).
    promoted = store.promote('MEM-PHM-CV', human_approved=True, target_layer=MemoryLayer.SEMANTIC)
    print(f'[d5] promoted {promoted.entry_id} -> {promoted.layer.value}', flush=True)
    # Demote a superseded working note (never delete).
    demoted = store.demote('MEM-NEG-THRESH', reason='superseded by multi-episode aggregation requirement')
    print(f'[d5] demoted {demoted.entry_id} (kept, flagged)', flush=True)
    print(f'[d5] audit: {store.audit()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())