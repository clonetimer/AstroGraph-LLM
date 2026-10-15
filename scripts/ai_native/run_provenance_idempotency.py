#!/usr/bin/env python3
"""E2: provenance-marked evidence idempotency demo (no double-count on replay)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.evidence.provenance_marker import IdempotentEvidenceAggregator, ProvenanceMarkedEvidence

EVIDENCE = ROOT / 'governance/evidence/ai_native/governance_health/provenance_idempotency_20260824.json'


def main() -> int:
    agg = IdempotentEvidenceAggregator()
    # 24 frozen ADCS episodes x star-gyro innovation observations.
    batch = [
        ProvenanceMarkedEvidence(
            evidence_id=f'E-{e}-{i}', episode_id=f'EP-{e:02d}',
            feature_definition_id='FD-STAR-GYRO-INNOVATION-MAX-V1',
            value=0.005315 if i % 3 else 0.012949,
            source_marker=f'EP-{e:02d}:FD-STAR-GYRO-INNOVATION-MAX-V1:{i}',
        )
        for e in range(24) for i in range(3)
    ]
    first = agg.ingest(batch)
    # Replay the same batch (e.g. checkpoint resume) - must not double-count.
    second = agg.ingest(batch)
    import json
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        'source_markers_seen': agg.replay_count(),
        'observations_per_feature': len(first['FD-STAR-GYRO-INNOVATION-MAX-V1']),
        'no_double_count_on_replay': len(second['FD-STAR-GYRO-INNOVATION-MAX-V1']) == len(first['FD-STAR-GYRO-INNOVATION-MAX-V1']),
        'note': 'source markers make checkpoint-resumed aggregation idempotent (E2)',
    }, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(f'[e2] markers={agg.replay_count()} observations={len(first["FD-STAR-GYRO-INNOVATION-MAX-V1"])}', flush=True)
    print(f'[e2] no double count on replay: {len(second["FD-STAR-GYRO-INNOVATION-MAX-V1"]) == len(first["FD-STAR-GYRO-INNOVATION-MAX-V1"])}', flush=True)
    print(f'[e2] evidence -> {EVIDENCE}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())