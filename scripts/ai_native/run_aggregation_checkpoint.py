#!/usr/bin/env python3
"""E1: aggregation checkpoint persistence + resume demo."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.evidence.aggregation_checkpoint import AggregationCheckpoint, CheckpointStore

STORE_ROOT = ROOT / 'governance/evidence/ai_native/aggregation'


def main() -> int:
    store = CheckpointStore(STORE_ROOT)
    run_id = 'agg-20260824-adcs-8class'
    all_episodes = ['EP-A'] * 8 + ['EP-B'] * 8 + ['EP-C'] * 8 + ['EP-D'] * 8
    # Simulate an interrupted run: 16 episodes completed, 16 pending.
    store.save(AggregationCheckpoint(
        run_id=run_id, dataset_id='whole_spacecraft_adcs_paired_dataset_v5_2_2',
        completed_episodes=all_episodes[:16], pending_episodes=all_episodes[16:],
        partial_observations={'FD-STAR-GYRO': [0.005315, 0.004993, 0.012949]},

    ))
    cp, pending = store.resume(run_id, all_episodes=all_episodes)
    print(f'[e1] resumed {run_id}: {len(cp.completed_episodes)} completed, {len(pending)} pending', flush=True)
    print(f'[e1] partial observations preserved: {len(cp.partial_observations)} feature(s)', flush=True)
    print(f'[e1] store -> {STORE_ROOT / (run_id + ".jsonl")}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())