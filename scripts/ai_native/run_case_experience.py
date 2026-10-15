#!/usr/bin/env python3
"""D4: register human-confirmed diagnosis cases as experience-library assets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.data_driven.case_experience import CaseExperienceLibrary, ConfirmedCase

LIBRARY_ROOT = ROOT / 'governance/evidence/ai_native/case_experience'


def main() -> int:
    lib = CaseExperienceLibrary(LIBRARY_ROOT)
    cases = [
        ConfirmedCase(
            case_id='CC-GYRO-001', episode_id='EP-gyro-bias',
            truth_fault_id='ADCS_GYRO_BIAS_DRIFT', predicted_fault_id='ADCS_GYRO_BIAS_DRIFT',
            source_hashes=['b6adf2dab53e2675b171fc0d1a4a36e14d90bcedc0409dc9a54016729006ba6c'],
            human_confirmed=True, confirmed_by='user-approved-20260824',
            notes='GYRO_BIAS_DRIFT supported after AND-disambiguation fix (threshold 0.01 + RW residual <= 0.33)',
        ),
        ConfirmedCase(
            case_id='CC-RWJAM-001', episode_id='EP-rw-jam-mc04',
            truth_fault_id='ADCS_RW_JAM', predicted_fault_id='ADCS_RW_JAM',
            human_confirmed=True, confirmed_by='user-approved-20260824',
            notes='RW_JAM preserved under AND guard; frozen regression confirmed supported={ADCS_RW_JAM}',
        ),
        ConfirmedCase(
            case_id='CC-NEG-001', episode_id='EP-threshold-015945',
            truth_fault_id='ADCS_RW_JAM', predicted_fault_id='ADCS_GYRO_BIAS_DRIFT',
            human_confirmed=True, confirmed_by='user-approved-20260824',
            is_negative=True,
            notes='single-episode threshold 0.015945 caused false positive; rejected by regression gate (negative sample for pollution guard)',
        ),
    ]
    for c in cases:
        lib.add(c)
        tag = ' NEG' if c.is_negative else ''
        print(f'[d4] confirmed {c.case_id}: {c.truth_fault_id} -> {c.predicted_fault_id}{tag}', flush=True)
    print(f'[d4] audit: {lib.audit()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())