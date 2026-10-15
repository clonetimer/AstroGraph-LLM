"""Bounded diagnostic hypotheses from data and executable mechanism evidence.

This layer retains unresolved hypotheses, never promotes a set to a confirmed
class and never treats a model prediction as an observed propagation event.
"""
from __future__ import annotations

from collections import defaultdict
from .diagnostic_scope import diagnostic_scope
from pathlib import Path
import hashlib
import json


def mechanism_confirmation_candidate(decision: dict, *, evidence_package_sha256: str) -> str | None:
    """Narrow, version-pinned disagreement policy; not a generic KG override."""
    if evidence_package_sha256 not in {'6ee2ab138a1001e00db509796eef68df6540c93fb457f543aa6bd9a24ca74148', '27c25dcdcf2deb05c475ae32132a6e4c137658c714afd4c22e90262bc7deecef'}:
        return None
    if (decision['status'] != 'CANDIDATE_SET' or not decision['detector_mechanism_disagreement'] or
            decision['blocking_conflicts'] or len(decision['candidate_fault_ids']) != 1 or
            decision['data_candidate_count'] > 2 or
            decision['data_candidate_source'] != 'PRE_HIERARCHY_CLASS_DISTRIBUTION'):
        return None
    fid = decision['candidate_fault_ids'][0]
    if fid not in {'ADCS_RW_JAM', 'ADCS_RW_TORQUE_AUTHORITY_LOSS', 'ADCS_RW_SPEED_SENSOR_BIAS'}:
        return None
    if evidence_package_sha256 == '27c25dcdcf2deb05c475ae32132a6e4c137658c714afd4c22e90262bc7deecef':
        # This pinned package keeps original channels/features/definitions byte-
        # equivalent in JSON values. New zero rules cannot grant an override.
        if not any(row['fault_id'] == fid and row['state'] == 'SATISFIED'
                   and row['definition_id'].startswith('ED-RW-WINDOW-')
                   and row['definition_id'].endswith('-V1')
                   and '-ZERO-' not in row['definition_id']
                   for row in decision['evidence_state_matrix']):
            return None
    support = decision['support_by_fault'][fid]
    if not support['complete_mechanism_support'] or not support['data_candidate']:
        return None
    return fid


def complementary_diagnosis(bundle, summary: dict, *, maximum_candidates: int = 3,
                            data_candidate_limit: int = 2, paired_reference_identical: bool = False) -> dict:
    if not 1 <= data_candidate_limit <= maximum_candidates <= 10:
        raise ValueError('candidate bounds must satisfy 1 <= data <= maximum <= 10')
    features = {f.feature_definition_id: f for f in bundle.feature_results}
    if any(f.window_end_s > bundle.episode.end_s + 1e-8 or f.window_start_s > f.window_end_s
           for f in bundle.feature_results):
        raise ValueError('mechanism evidence cannot use future or reversed feature windows')
    definitions = {d.evidence_definition_id: d for d in bundle.evidence_definitions}
    facts = defaultdict(list)
    for fact in bundle.evidence_facts:
        facts[fact.evidence_definition_id].append(fact)
    matrix = []
    for d in bundle.evidence_definitions:
        fs = facts[d.evidence_definition_id]
        states = sorted({f.status.value for f in fs})
        # Missing, invalid and unevaluated remain distinct from observed failure.
        state = states[0] if len(states) == 1 else 'UNEVALUATED' if not states else 'MIXED'
        conditions = [(d.feature_definition_id, d.comparison.value, d.threshold, d.threshold_unit)]
        conditions += [(c.feature_definition_id, c.comparison.value, c.threshold, c.threshold_unit)
                       for c in d.and_feature_conditions]
        matrix.append({
            'definition_id': d.evidence_definition_id, 'fault_id': d.target_fault_id,
            'role': d.role.value, 'state': state,
            'fact_ids': sorted(f.evidence_fact_id for f in fs),
            'conditions': [{
                'feature_definition_id': fid, 'field': features[fid].resolved_field if fid in features else None,
                'value': features[fid].value if fid in features else None,
                'quality_flags': features[fid].quality_flags if fid in features else ['NO_FEATURE_RESULT'],
                'comparison': op, 'threshold': threshold, 'unit': unit,
            } for fid, op, threshold, unit in conditions],
        })
    selected = summary['suspected_fault_id']
    heads = summary.get('three_task_results', {})
    detected = heads.get('detection', {}).get('detected', False)
    candidates = {c['fault_id']: c for c in summary['candidates']}
    supported = {fid for fid, c in candidates.items() if c['eligible_for_explanation']}
    contradicted = {fid for fid, c in candidates.items() if c['contradicting_evidence_ids']}
    ranking = heads.get('classification', [])
    trace = heads.get('head_trace')
    if trace:
        mass = defaultdict(float)
        for label, probability in trace['raw_class_probabilities'].items():
            if label in trace['class_to_runtime_fault']:
                mass[trace['class_to_runtime_fault'][label]] += probability
        scores = sorted(set(mass.values()), reverse=True)
        ranking = [{'fault_id': fid, 'score': score, 'rank': scores.index(score)+1}
                   for fid, score in mass.items()]
    data = [c['fault_id'] for c in sorted(ranking, key=lambda c: (c['rank'], c['fault_id']))
            if c['score'] is not None and c['score'] > 0 and c['fault_id'] not in contradicted]
    # A tie at the boundary is retained; overflow is explicit, never silent truncation.
    if data:
        rank_by_id = {c['fault_id']: c['rank'] for c in ranking}
        cutoff = rank_by_id[data[min(data_candidate_limit, len(data))-1]]
        data = [fid for fid in data if rank_by_id[fid] <= cutoff]
    proposed = supported | (set(data) if detected else set())
    hard_blocks = [b for b in summary['blocking_conflicts']
                   if b != 'THREE_TASK_DETECTOR_NOT_TRIGGERED']
    if summary['open_set_actions']:
        hard_blocks.append('OPEN_SET_BLOCK')
    if paired_reference_identical:
        result_ids, status, selected = [], 'NORMAL_REFERENCE_MATCH', None
    elif hard_blocks:
        result_ids, status, selected = [], 'BLOCKED', None
    elif selected:
        result_ids, status = [selected], 'SUPPORTED_SINGLETON'
    elif len(proposed) > maximum_candidates:
        result_ids, status = [], 'CANDIDATE_OVERFLOW'
    elif proposed:
        result_ids, status = sorted(proposed), 'CANDIDATE_SET'
    else:
        result_ids, status = [], 'INSUFFICIENT_EVIDENCE'
    # Explain current-window mechanism links. These are expected responses, not
    # proof of temporal direction or newly observed causal events.
    paths = []
    for fid in result_ids:
        for row in matrix:
            if row['fault_id'] != fid:
                continue
            d = definitions[row['definition_id']]
            paths.append({
                'fault_id': fid,
                'nodes': [fid, *[c['feature_definition_id'] for c in row['conditions']], row['definition_id']],
                'relation': 'PREDICTS_RESPONSE_CHECKED_BY_EXECUTABLE_CONDITION',
                'condition_logic': 'AND', 'state': row['state'], 'fact_ids': row['fact_ids'],
                'source_refs': list(d.source_refs),
                'window': dict(summary['window']),
                'temporal_propagation_observed': False,
            })
    # Group identical requirements so a shared symptom is not advertised as a
    # discriminator. A differing requirement is a proposed check, not proof that
    # the pair is identifiable under every operating condition.
    groups = {}
    for row in matrix:
        if row['fault_id'] not in result_ids or row['role'] not in {'REQUIRED', 'SUPPORTING'}:
            continue
        key = tuple((c['feature_definition_id'], c['comparison'], c['threshold'], c['unit'])
                    for c in row['conditions'])
        group = groups.setdefault(key, {'fault_ids': set(), 'rows': []})
        group['fault_ids'].add(row['fault_id'])
        group['rows'].append(row)
    checks = []
    for group in groups.values():
        rows = group['rows']
        if all(r['state'] == 'SATISFIED' for r in rows):
            continue
        states = sorted({r['state'] for r in rows})
        unavailable = any(s in {'MISSING_DATA', 'INVALID_DATA', 'UNKNOWN', 'UNEVALUATED', 'MIXED'} for s in states)
        checks.append({
            'definition_ids': sorted(r['definition_id'] for r in rows),
            'required_by_fault_ids': sorted(group['fault_ids']),
            'other_candidate_fault_ids': sorted(set(result_ids) - group['fault_ids']),
            'candidate_requirement_differs': bool(set(result_ids) - group['fault_ids']),
            'states': states,
            'action': ('CHECK_APPLICABILITY' if 'NOT_APPLICABLE' in states else
                       'ACQUIRE_OR_REPAIR_OBSERVATION' if unavailable else 'REASSESS_OBSERVED_MISMATCH'),
            'conditions': rows[0]['conditions'],
            'distinguishability': 'REQUIRES_OPERATING_CONDITION_AND_SENSOR_VALIDATION',
        })
    checks.sort(key=lambda c: (not c['candidate_requirement_differs'], c['definition_ids']))
    if result_ids and not checks:
        checks.append({'action': 'CHECK_INDEPENDENT_SENSOR_OR_LONGER_CAUSAL_WINDOW',
                       'reason': 'Existing satisfied conditions do not settle detector disagreement or exclude alternative mechanisms.',
                       'required_by_fault_ids': result_ids,
                       'distinguishability': 'NOT_ESTABLISHED'})
    catalog_path = Path(__file__).resolve().parents[1] / 'resources/adcs_mechanism_hypotheses.v1.json'
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    mechanism_hypotheses = []
    for fid in result_ids:
        if fid not in catalog['mechanisms']:
            continue
        model = catalog['mechanisms'][fid]
        mechanism_hypotheses.append({'fault_id': fid, **model,
                                    'source_refs': catalog['source_refs'],
                                    'propagation_is': 'MODEL_PREDICTION_NOT_OBSERVED_EVENT_ORDER'})
        if not selected:
            checks.append({'action': 'VERIFY_MECHANISM_WITH_INDEPENDENT_OBSERVATION',
                           'required_by_fault_ids': [fid],
                           'observation_roles': model['observation_roles'],
                           'applicability_conditions': model['conditions'],
                           'check': model['discriminating_check'],
                           'distinguishability': 'CONDITIONAL_NOT_GUARANTEED'})
    return {
        'schema_version': 'astrograph-complementary-diagnosis.v1',
        'status': status, 'candidate_fault_ids': result_ids,
        'selected_fault_id': selected, 'maximum_candidates': maximum_candidates,
        'diagnostic_scopes': [diagnostic_scope(fid) for fid in result_ids],
        'proposed_count': len(proposed), 'candidate_set_is_confirmed_class': False,
        'data_candidate_source': 'PRE_HIERARCHY_CLASS_DISTRIBUTION' if trace else 'EXISTING_CLASS_RANKING',
        'data_candidate_count': len(data),
        'support_by_fault': {fid: {'data_candidate': fid in data,
                                 'complete_mechanism_support': fid in supported,
                                 'executable_knowledge_coverage': any(d.target_fault_id == fid for d in definitions.values())
                                 } for fid in result_ids},
        'detector_mechanism_disagreement': bool(supported and not detected),
        'blocking_conflicts': hard_blocks,
        'evidence_state_matrix': matrix, 'mechanism_paths': paths,
        'mechanism_hypotheses': mechanism_hypotheses,
        'mechanism_catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
        'additional_observations': checks,
        'calibrated_probability': None, 'coverage_guarantee': None,
        'causal_identification': 'NOT_ESTABLISHED',
        'observation_contract': {
            'reference': 'PAIRED_TIME_ALIGNED_NOMINAL_REQUIRED',
            'paired_reference_identical': paired_reference_identical,
            'window': dict(summary['window']),
            'observed_fields': sorted({f.resolved_field for f in bundle.feature_results}),
            'domain_id': bundle.domain_id,
            'transfer_requirement': 'Revalidate sensor availability, units, excitation, topology and mechanism parameters for each new domain.',
        },
        'claim_boundary': 'Bounded hypotheses with current-window response checks; no temporal propagation identification, new independent votes, or control authority.',
    }
