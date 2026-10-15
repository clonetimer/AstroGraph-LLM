#!/usr/bin/env python3
"""A5: independent deterministic verification of generated hypothesis sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.governance.output_verifier import verify_hypothesis_set

UNIVERSE = json.loads((ROOT / 'configs/datasets/adcs_v5_2_2/candidate_universe.json').read_text(encoding='utf-8'))['runtime_fault_ids']
EVIDENCE_OUT = ROOT / 'governance/evidence/ai_native/governance_health/output_verification_20260824.json'


def main() -> int:
    # Representative hypothesis sets from this session's frozen test cases.
    samples = [
        {'episode_id': 'EP-rw-jam-mc04', 'hypotheses': [
            {'hypothesis_id': 'H-1', 'target_fault_id': 'ADCS_RW_JAM', 'status': 'SUPPORTED', 'supporting_evidence_fact_ids': ['EF-a', 'EF-b']},
        ]},
        {'episode_id': 'EP-gyro-bias', 'hypotheses': [
            {'hypothesis_id': 'H-1', 'target_fault_id': 'ADCS_GYRO_BIAS_DRIFT', 'status': 'SUPPORTED', 'supporting_evidence_fact_ids': ['EF-c']},
        ]},
    ]
    records = [verify_hypothesis_set(s, candidate_universe=UNIVERSE).model_dump(mode='json') for s in samples]
    EVIDENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_OUT.write_text(json.dumps({'records': records}, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    for r in records:
        print(f'[a5] {r["artifact_id"]}: verdict={r["verdict"]}', flush=True)
    print(f'[a5] evidence -> {EVIDENCE_OUT}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())