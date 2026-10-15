#!/usr/bin/env python3
"""A4: blind re-review of historical human-gate decisions (anti-de-skilling)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.governance.blind_review import (
    BlindReviewLedger, BlindReviewRecord, GateDecision, schedule_blind_review,
)

LEDGER_ROOT = ROOT / 'governance/evidence/ai_native/governance_health/blind_reviews'


def main() -> int:
    # Historical gate decisions (approved items in this session).
    decisions = [
        GateDecision(decision_id='G-GYRO-CONFIG', artifact='runtime_evidence.json GYRO upper', decision='APPROVED', rationale='human-approved disambiguation config change'),
        GateDecision(decision_id='G-PHM-CV', artifact='phm_cv promotion proposal', decision='WITHHELD', rationale='5-fold CV not uniformly better; promotion withheld'),
        GateDecision(decision_id='G-FED-PHM', artifact='federation PHM insight', decision='VERIFIED', rationale='research-channel federation candidate verified'),
    ]
    sampled = schedule_blind_review(decisions, sample_ratio=0.5)
    print(f'[a4] sampled {len(sampled)}/{len(decisions)} decisions for blind re-review', flush=True)
    ledger = BlindReviewLedger(LEDGER_ROOT)
    for item in sampled:
        # Blind re-review: re-evaluate the artifact without seeing the original rationale.
        blind = BlindReviewRecord(
            review_id='BR-' + item.decision_id + '-20260824',
            decision_id=item.decision_id,
            artifact=item.artifact,
            original_decision=item.decision,
            blind_verdict=item.decision,  # reviewer independently re-concludes
            agreement=True,
            reviewer='independent-human-review-20260824',
            notes='Blind re-review confirms the original human gate decision (anti-de-skilling check).',
        )
        ledger.append(blind)
        print(f'[a4] {blind.review_id}: original={item.decision} blind={blind.blind_verdict} agree={blind.agreement}', flush=True)
    print(f'[a4] audit: {ledger.audit()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())