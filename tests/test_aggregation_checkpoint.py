"""E1: aggregation checkpoint persistence and resume tests."""
from __future__ import annotations

from astrograph.evidence.aggregation_checkpoint import AggregationCheckpoint, CheckpointStore


def _all_episodes() -> list[str]:
    return ['EP-1', 'EP-2', 'EP-3', 'EP-4', 'EP-5']


def test_save_and_latest(tmp_path):
    store = CheckpointStore(tmp_path)
    cp = AggregationCheckpoint(run_id='run-1', dataset_id='adcs', completed_episodes=['EP-1', 'EP-2'])
    store.save(cp)
    latest = store.latest('run-1')
    assert latest is not None
    assert latest.completed_episodes == ['EP-1', 'EP-2']


def test_resume_no_checkpoint(tmp_path):
    store = CheckpointStore(tmp_path)
    cp, pending = store.resume('run-new', all_episodes=_all_episodes())
    assert len(pending) == 5
    assert cp.completed_episodes == []


def test_resume_partial(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(AggregationCheckpoint(run_id='run-2', dataset_id='adcs', completed_episodes=['EP-1', 'EP-2', 'EP-3']))
    cp, pending = store.resume('run-2', all_episodes=_all_episodes())
    assert pending == ['EP-4', 'EP-5']
    assert cp.completed_episodes == ['EP-1', 'EP-2', 'EP-3']


def test_append_only_snapshots(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(AggregationCheckpoint(run_id='run-3', dataset_id='adcs', completed_episodes=['EP-1']))
    store.save(AggregationCheckpoint(run_id='run-3', dataset_id='adcs', completed_episodes=['EP-1', 'EP-2']))
    latest = store.latest('run-3')
    assert latest.completed_episodes == ['EP-1', 'EP-2']
    # Append-only: both lines retained.
    lines = (tmp_path / 'run-3.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2


def test_observations_persisted(tmp_path):
    store = CheckpointStore(tmp_path)
    cp = AggregationCheckpoint(run_id='run-4', dataset_id='adcs', completed_episodes=['EP-1'], partial_observations={'FD-1': [0.005, 0.006]})
    store.save(cp)
    latest = store.latest('run-4')
    assert latest.partial_observations['FD-1'] == [0.005, 0.006]