"""R10: independently reviewed, phase-aware causal *reference* constraints.

No automatic event detection, flight-state attestation, or diagnosis vote. Temporal
metadata must be attached through a privileged review, never by the LLM graph
proposal. R9 static intersection remains the default for historical records.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from astrograph.integrations.material_applicability import (
    assess_contexts, normalize_reviewed_applicability,
)

SCHEMA = 'astrograph-reviewed-causal-temporal.v1'
_PROCESS = re.compile(r'^[A-Z][A-Z0-9_:-]{3,79}$')
_MAX_SECONDS = 1_000_000_000.0


def normalize_reviewed_temporal(value: Mapping[str, Any], *, equipment_scope: str) -> dict[str, Any]:
    """Validate provenance-backed stages; a state-change claim is not telemetry."""
    required = {'schema_version', 'equipment_model', 'process_id', 'stage_index',
                'pre_context', 'post_context', 'lag_seconds',
                'minimum_source_dwell_seconds', 'review_basis_ref'}
    if not isinstance(value, Mapping) or set(value) != required or value['schema_version'] != SCHEMA:
        raise ValueError('invalid reviewed temporal schema')
    if value['equipment_model'] != equipment_scope:
        raise ValueError('reviewed temporal equipment scope mismatch')
    process = value['process_id']
    if not isinstance(process, str) or not _PROCESS.fullmatch(process):
        raise ValueError('invalid reviewed temporal process id')
    stage = value['stage_index']
    if type(stage) is not int or not 0 <= stage <= 1000:
        raise ValueError('invalid reviewed temporal stage')
    basis = value['review_basis_ref']
    if not isinstance(basis, str) or not 8 <= len(basis.strip()) <= 300:
        raise ValueError('independent temporal basis required')
    lag = value['lag_seconds']
    if not isinstance(lag, Mapping) or set(lag) != {'min', 'max'}:
        raise ValueError('invalid causal lag bounds')
    low, high, dwell = lag['min'], lag['max'], value['minimum_source_dwell_seconds']
    if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= _MAX_SECONDS
            for v in (low, high, dwell)) or low > high or dwell > high):
        raise ValueError('invalid/nonfinite temporal delay or dwell')
    pre = normalize_reviewed_applicability(value['pre_context'], equipment_scope=equipment_scope)
    post = normalize_reviewed_applicability(value['post_context'], equipment_scope=equipment_scope)
    return {'schema_version': SCHEMA, 'equipment_model': equipment_scope,
            'process_id': process, 'stage_index': stage,
            'pre_context': pre, 'post_context': post,
            'lag_seconds': {'min': float(low), 'max': float(high)},
            'minimum_source_dwell_seconds': float(dwell),
            'review_basis_ref': basis.strip()}


def assess_temporal_causal_path(edges: Sequence[Mapping[str, Any]], *,
                                equipment_scope: str) -> dict[str, Any]:
    """Check sequential stage compatibility, never pretend states were observed.

    A complete independently reviewed stage chain may change modes, ranges and
    environment between stages. An incomplete one cannot waive R9 restrictions.
    """
    static = assess_contexts(edges, equipment_scope=equipment_scope)
    base = {'flight_applicability_verified': False, 'causal_proof': False,
            'reviewed_stage_count': 0, 'static_context_review': static,
            'transition_receipts': [], 'cumulative_lag_seconds': None,
            'conflict_reasons': []}
    if not any(edge.get('reviewed_temporal') is not None for edge in edges):
        return {**base, 'mode': 'R9_STATIC_LEGACY',
                'status': ('REJECTED_STATIC_CONTEXT' if static['conflict_reasons']
                           else 'STATIC_CONTEXT_ONLY_TEMPORAL_UNKNOWN'),
                'conflict_reasons': list(static['conflict_reasons'])}
    if any(edge.get('reviewed_temporal') is None for edge in edges):
        return {**base, 'mode': 'INCOMPLETE_TEMPORAL_REVIEW',
                'status': 'REJECTED_PARTIALLY_ANNOTATED_TIMELINE',
                'conflict_reasons': ['INCOMPLETE_REVIEWED_TEMPORAL_CHAIN']}
    normalized = []
    try:
        for edge in edges:
            entry = normalize_reviewed_temporal(edge['reviewed_temporal'], equipment_scope=equipment_scope)
            if edge.get('reviewed_applicability') is None:
                raise ValueError('reviewed edge applicability required for temporal override')
            parent = normalize_reviewed_applicability(edge['reviewed_applicability'],
                                                      equipment_scope=equipment_scope)
            pre = entry['pre_context']
            # Stage-preconditions may narrow an approved edge's scope but may
            # not silently widen any established categorical or numeric bounds.
            if not set(pre['operating_modes']).issubset(parent['operating_modes']):
                raise ValueError('temporal pre-mode widens approved edge scope')
            if any(pre['environment'].get(k) != v for k, v in parent['environment'].items()):
                raise ValueError('temporal pre-environment widens approved edge scope')
            pre_ranges = {item['name']: item for item in pre['parameter_ranges']}
            for bound in parent['parameter_ranges']:
                p = pre_ranges.get(bound['name'])
                if (not p or p['unit'] != bound['unit'] or
                    p['min'] < bound['min'] or p['max'] > bound['max']):
                    raise ValueError('temporal pre-range widens approved edge scope')
            normalized.append(entry)
    except (ValueError, TypeError, OverflowError, KeyError):
        return {**base, 'mode': 'REVIEWED_STAGES', 'status': 'REJECTED_INVALID_TEMPORAL_REVIEW',
                'conflict_reasons': ['INVALID_OR_SCOPE_WIDENING_REVIEWED_TIMELINE']}
    reasons: set[str] = set()
    if len({item['process_id'] for item in normalized}) != 1:
        reasons.add('TEMPORAL_PROCESS_MISMATCH')
    for prev, current in zip(normalized, normalized[1:]):
        if current['stage_index'] != prev['stage_index'] + 1:
            reasons.add('TEMPORAL_STAGE_ORDER_OR_GAP')
        junction = assess_contexts(({'reviewed_applicability': prev['post_context']},
                                    {'reviewed_applicability': current['pre_context']}),
                                   equipment_scope=equipment_scope)
        if junction['conflict_reasons']:
            reasons.add('TEMPORAL_STATE_TRANSITION_INCOMPATIBLE')
    cumulative = {'min': sum(entry['lag_seconds']['min'] for entry in normalized),
                  'max': sum(entry['lag_seconds']['max'] for entry in normalized)}
    receipts = [{'edge_id': edge.get('edge_id') or edge.get('proposal_id'),
                 'process_id': entry['process_id'], 'stage_index': entry['stage_index'],
                 'pre_modes': entry['pre_context']['operating_modes'],
                 'post_modes': entry['post_context']['operating_modes'],
                 'lag_seconds': entry['lag_seconds'],
                 'minimum_source_dwell_seconds': entry['minimum_source_dwell_seconds'],
                 'review_basis_ref': entry['review_basis_ref']}
                for edge, entry in zip(edges, normalized)]
    return {**base, 'mode': 'REVIEWED_STAGES',
            'status': ('REJECTED_TEMPORAL_TRANSITION' if reasons else
                       'REVIEWED_SEQUENCE_COMPATIBLE_NOT_FLIGHT_VERIFIED'),
            'reviewed_stage_count': len(normalized),
            'transition_receipts': receipts, 'cumulative_lag_seconds': cumulative,
            'conflict_reasons': sorted(reasons)}
