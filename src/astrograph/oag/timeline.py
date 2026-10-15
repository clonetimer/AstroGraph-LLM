"""Preserve every scheduled window without turning old evidence into new votes."""
from __future__ import annotations
import math
from .contracts import content_hash


def build_evidence_timeline(windows: list[dict]) -> dict:
    if not windows:
        raise ValueError('timeline requires scheduled windows')
    endpoints = [w['endpoint_s'] for w in windows]
    if any(not math.isfinite(x) for x in endpoints) or endpoints != sorted(set(endpoints)):
        raise ValueError('window endpoints must be finite, unique and increasing')
    history, points, component, graph = {}, [], None, None
    for w in windows:
        endpoint = w['endpoint_s']
        s = w.get('summary')
        if w['status'] != 'SUCCEEDED':
            points.append({'endpoint_s': endpoint, 'state': 'DATA_UNAVAILABLE', 'candidate': None,
                           'error': w.get('error'), 'historical_candidates': sorted(history)})
            continue
        if not s or s.get('summary_sha256') != content_hash({k:v for k,v in s.items() if k != 'summary_sha256'}):
            raise ValueError('timeline summary hash mismatch')
        if s['window']['end_s'] > endpoint + 1e-8 or s['window']['start_s'] > s['window']['end_s']:
            raise ValueError('future or reversed evidence window')
        if component is None: component, graph = s['component'], s['graph_identity']
        if component != s['component'] or graph != s['graph_identity']:
            raise ValueError('timeline cannot mix component or graph identities')
        fault = s['suspected_fault_id']
        if fault:
            record = history.setdefault(fault, {'fault_id': fault, 'first_supported_s': endpoint,
                'last_supported_s': endpoint, 'supporting_summary_hashes': []})
            record['last_supported_s'] = endpoint
            record['supporting_summary_hashes'].append(s['summary_sha256'])
        detected = s.get('three_task_results', {}).get('detection', {}).get('detected')
        # Different model versions expose the explicit top-level detected field.
        if detected is None: detected = s.get('three_task_results', {}).get('detected', False)
        state = 'SUPPORTED_ANOMALY' if fault else 'UNCONFIRMED_MODEL_ALERT' if detected else 'NO_SUPPORTED_ANOMALY'
        points.append({'endpoint_s': endpoint, 'window': s['window'], 'state': state, 'candidate': fault,
                       'historical_candidates': sorted(history), 'summary_sha256': s['summary_sha256'],
                       **({'diagnostic_decision': {
                           k: s['diagnostic_decision'][k] for k in (
                               'status', 'candidate_fault_ids', 'selected_fault_id',
                               'detector_mechanism_disagreement', 'additional_observations')
                       }} if 'diagnostic_decision' in s else {}),
                       'detector_triggered': bool(detected)})
    result = {'schema_version': 'astrograph-evidence-timeline.v1', 'scheduled_windows': len(windows),
        'successful_windows': sum(w['status']=='SUCCEEDED' for w in windows),
        'unavailable_windows': sum(w['status']!='SUCCEEDED' for w in windows), 'points': points,
        'historical_candidates': list(history.values()), 'current': points[-1],
        'additional_fusion_votes': 0, 'historical_evidence_is_current': False,
        'claim_boundary': 'Historical candidates retain their own windows; no truth-driven selection, interpolation, model recalibration or new fusion votes.'}
    result['timeline_sha256'] = content_hash(result)
    return result
