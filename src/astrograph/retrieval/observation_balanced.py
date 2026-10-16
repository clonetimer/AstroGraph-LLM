"""Recall-oriented, non-voting multi-view retrieval for user-reported observations.

Query construction depends *only* on submitted observations and already frozen
upstream candidates. Adjudicated diagnoses are intentionally not accepted here.
The resulting hits are unreviewed source pointers, not fault votes or proof.
"""
from __future__ import annotations

import hashlib
from typing import Any

SCHEMA_VERSION = 'astrograph-observation-balanced-retrieval.v1'


def retrieve_observation_balanced(index: Any, chunks: list[Any], *,
                                  controlled_query: str, task: str,
                                  observation_ledger: dict[str, Any],
                                  max_observation_queries: int = 8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(max_observation_queries) is not int or not 1 <= max_observation_queries <= 12:
        raise ValueError('invalid query bound')
    observations = (observation_ledger.get('observations') or [])[:max_observation_queries]
    original_query = ' '.join((controlled_query, task.strip()[:1000])).strip()
    # Preserve the legacy behavior when no numbered observations are available.
    if not observations:
        hits = index.retrieve(original_query, chunks)
        return hits, {
            'schema_version': SCHEMA_VERSION, 'strategy': 'LEGACY_SINGLE_QUERY',
            'query_count': 1 if original_query else 0,
            'observation_queries': 0, 'observation_coverage_count': 0,
            'uncovered_observation_ids': [], 'diagnostic_vote_allowed': False,
        }
    # Each numbered line is an independent query; the combined request remains
    # an additional query to recover cross-statement context. Never use gold labels.
    channels: list[tuple[str, str]] = []
    if original_query:
        channels.append(('context', original_query))
    for row in observations:
        statement = str(row.get('reported_text') or '').strip()[:600]
        eid = str(row.get('evidence_id') or '')
        if statement and eid:
            channels.append((eid, statement))
    scores: dict[str, float] = {}
    matching: dict[str, set[str]] = {}
    hit_by_id: dict[str, dict[str, Any]] = {}
    per_observation: list[tuple[str, list[str]]] = []
    for source, query in channels:
        found = index.retrieve(query, chunks)
        good_ids: list[str] = []
        for rank, hit in enumerate(found, 1):
            if not isinstance(hit.get('knowledge_id'), str):
                continue
            cid = hit['knowledge_id']
            hit_by_id.setdefault(cid, dict(hit))
            # Reciprocal-rank aggregation: bounded and order-stable; no
            # relevance probability or diagnostic confidence is implied.
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
            if source != 'context':
                matching.setdefault(cid, set()).add(source)
                good_ids.append(cid)
        if source != 'context':
            per_observation.append((source, good_ids))
    # Reserve one distinct top hit per covered observation where possible, then
    # finish by aggregate rank. This prevents one high-frequency symptom from
    # occupying every displayed snippet in a multi-system incident report.
    selected: list[str] = []
    bound = index.top_k
    for _, ids in per_observation:
        if len(selected) >= bound:
            break
        chosen = next((cid for cid in ids if cid not in selected), None)
        if chosen is not None:
            selected.append(chosen)
    for cid in sorted(scores, key=lambda name: (-scores[name], name)):
        if len(selected) >= bound:
            break
        if cid not in selected:
            selected.append(cid)
    hits: list[dict[str, Any]] = []
    for rank, cid in enumerate(selected, 1):
        row = dict(hit_by_id[cid])
        row['rank'] = rank
        row['retrieval_fusion_score'] = round(scores[cid], 9)
        row['matched_user_observation_ids'] = sorted(matching.get(cid, set()))
        row['diagnostic_vote_allowed'] = False
        hits.append(row)
    covered = {x for h in hits for x in h['matched_user_observation_ids']}
    present_ids = [src for src, _ in per_observation]
    receipt = {
        'schema_version': SCHEMA_VERSION,
        'strategy': 'OBSERVATION_BALANCED_RRF_NON_VOTING',
        'query_count': len(channels),
        'observation_queries': len(per_observation),
        'observation_coverage_count': len(covered),
        'uncovered_observation_ids': [eid for eid in present_ids if eid not in covered],
        'query_sha256': [hashlib.sha256(query.encode('utf-8')).hexdigest() for _, query in channels],
        'retrieval_coverage_is_not_mechanism_validation': True,
        'diagnostic_vote_allowed': False,
    }
    return hits, receipt
