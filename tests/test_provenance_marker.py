"""E2: provenance-marked evidence idempotency tests."""
from __future__ import annotations

from astrograph.evidence.provenance_marker import IdempotentEvidenceAggregator, ProvenanceMarkedEvidence


def _evidence(episode: str, feature: str, value: float, index: int = 0) -> ProvenanceMarkedEvidence:
    return ProvenanceMarkedEvidence(
        evidence_id=f'{episode}-{feature}-{index}', episode_id=episode,
        feature_definition_id=feature, value=value,
        source_marker=f'{episode}:{feature}:{index}',
    )


def test_ingest_collects_observations():
    agg = IdempotentEvidenceAggregator()
    result = agg.ingest([
        _evidence('EP-1', 'FD-1', 0.005, index=0), _evidence('EP-1', 'FD-1', 0.006, index=1),
    ])
    assert result['FD-1'] == [0.005, 0.006]
    assert agg.replay_count() == 2


def test_replayed_episode_deduplicated():
    agg = IdempotentEvidenceAggregator()
    batch = [_evidence('EP-1', 'FD-1', 0.005), _evidence('EP-2', 'FD-1', 0.007)]
    agg.ingest(batch)
    agg.ingest(batch)  # replay the same batch
    assert agg.replay_count() == 2  # no double count
    assert len(agg.ingest([])['FD-1']) == 2


def test_distinct_markers_across_episodes():
    agg = IdempotentEvidenceAggregator()
    agg.ingest([_evidence('EP-1', 'FD-1', 0.005), _evidence('EP-2', 'FD-1', 0.007)])
    assert len(agg.ingest([])['FD-1']) == 2


def test_multiple_features():
    agg = IdempotentEvidenceAggregator()
    agg.ingest([
        _evidence('EP-1', 'FD-1', 0.005), _evidence('EP-1', 'FD-2', 0.3),
    ])
    result = agg.ingest([])
    assert 'FD-1' in result and 'FD-2' in result