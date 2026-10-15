"""Evaluator-only set metrics. Truth is joined to completed inference outputs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def predictions(row):
    methods = ['DATA_CLASS', 'DATA_RAW_TOP2', 'FUSION_CLASS', 'COMPLEMENTARY']
    if row['status'] != 'SUCCEEDED' or not row.get('summary'):
        return {m: {'kind': 'UNAVAILABLE', 'ids': []} for m in methods}
    s = row['summary']
    h = s['three_task_results']
    identical = row['paired_reference_identical']
    detected = h['detection']['detected']
    ranking = h['classification']
    result = {}
    result['DATA_CLASS'] = ({'kind': 'NORMAL', 'ids': []} if identical or not detected else
                            {'kind': 'CLASS', 'ids': [ranking[0]['fault_id']]})
    mass = defaultdict(float)
    trace = h.get('head_trace')
    if trace:
        for label, p in trace['raw_class_probabilities'].items():
            if label in trace['class_to_runtime_fault']:
                mass[trace['class_to_runtime_fault'][label]] += p
    else:
        mass.update({c['fault_id']: c['score'] for c in ranking if c['score'] is not None})
    ordered = sorted(((p, fid) for fid, p in mass.items() if p > 0), reverse=True)
    cutoff = ordered[min(2, len(ordered))-1][0] if ordered else 1.
    ids = sorted(fid for p, fid in ordered if p >= cutoff)
    result['DATA_RAW_TOP2'] = ({'kind': 'NORMAL', 'ids': []} if identical or not detected else
                              {'kind': 'SET', 'ids': ids})
    result['FUSION_CLASS'] = ({'kind': 'NORMAL', 'ids': []} if identical else
                              {'kind': 'CLASS', 'ids': [s['suspected_fault_id']]} if s['suspected_fault_id'] else
                              {'kind': 'ABSTAIN', 'ids': []})
    d = s.get('diagnostic_decision')
    if d is None:
        raise ValueError('complementary diagnosis missing from completed inference')
    kind = ('NORMAL' if d['status'] == 'NORMAL_REFERENCE_MATCH' else
            'CLASS' if d['status'] == 'SUPPORTED_SINGLETON' else
            'SET' if d['status'] == 'CANDIDATE_SET' else 'ABSTAIN')
    result['COMPLEMENTARY'] = {'kind': kind, 'ids': d['candidate_fault_ids']}
    return result


def measure(rows):
    result = {}
    for method in ['DATA_CLASS', 'DATA_RAW_TOP2', 'FUSION_CLASS', 'COMPLEMENTARY']:
        pairs = [(r['target'], r['predictions'][method]) for r in rows]
        active = [(t, p) for t, p in pairs if t is not None]
        nsets = sum(bool(p['ids']) for _, p in active)
        memberships = sum(len(p['ids']) for _, p in active)
        hits = sum(t in p['ids'] for t, p in active)
        result[method] = {
            'n': len(pairs), 'fault_active_n': len(active),
            'strict_class_correct': sum((t is None and p['kind'] == 'NORMAL') or
                                        (t is not None and p['kind'] == 'CLASS' and p['ids'] == [t]) for t, p in pairs),
            'wrong_specific_fault': sum(p['kind'] == 'CLASS' and p['ids'] != [t] for t, p in pairs),
            'fault_candidate_hits': hits, 'fault_candidate_hit_rate': hits/len(active) if active else None,
            'fault_candidate_outputs': nsets,
            'fault_candidate_mean_size': memberships/nsets if nsets else None,
            'fault_candidate_precision': hits/memberships if memberships else None,
            'set_size_histogram': dict(Counter(len(p['ids']) for _, p in active)),
            'normal_or_pre_event_n': len(pairs)-len(active),
            'normal_or_pre_event_candidate_alerts': sum(t is None and bool(p['ids']) for t, p in pairs),
            'unavailable': sum(p['kind'] == 'UNAVAILABLE' for _, p in pairs),
            'abstained': sum(p['kind'] == 'ABSTAIN' for _, p in pairs),
            'singleton_hypothesis_not_confirmed': sum(p['kind'] == 'SET' and len(p['ids']) == 1 for _, p in pairs),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads((args.evaluation/'evaluation.json').read_text())
    rows = []
    head_changes = Counter()
    for record in evaluation['records']:
        metadata = json.loads((args.dataset/record['source_path']).with_name('sample_metadata.json').read_text())
        windows = json.loads((args.evaluation/record['case_id']/'windows.json').read_text())
        for w in windows:
            active = metadata['event_start_s'] is not None and w['endpoint_s'] >= metadata['event_start_s']
            row = {'case_id': record['case_id'], 'endpoint_s': w['endpoint_s'],
                   'mission': metadata['mission_id'], 'initial_condition_family': metadata['pair_seed'],
                   'target': record['evaluation_target'] if active else None,
                   'predictions': predictions(w)}
            rows.append(row)
            h = (w.get('summary') or {}).get('three_task_results', {})
            trace = h.get('head_trace')
            if trace and h.get('classification'):
                raw = trace['class_to_runtime_fault'].get(trace['raw_top_class'])
                post = h['classification'][0]['fault_id']
                head_changes['windows'] += 1
                if raw != post:
                    head_changes['raw_top_changed'] += 1
                    if active:
                        head_changes['raw_correct_lost'] += raw == record['evaluation_target'] and post != record['evaluation_target']
                        head_changes['post_correct_gained'] += post == record['evaluation_target'] and raw != record['evaluation_target']
    report = {
        'schema_version': 'astrograph-complementary-evaluation.v1',
        'source_evaluation': str(args.evaluation),
        'all_windows': measure(rows),
        'terminal': measure([r for r in rows if r['endpoint_s'] == max(x['endpoint_s'] for x in rows)]),
        'by_mission': {m: measure([r for r in rows if r['mission'] == m]) for m in sorted({r['mission'] for r in rows})},
        'by_fault': {str(t): measure([r for r in rows if r['target'] == t]) for t in sorted({r['target'] for r in rows}, key=str)},
        'head_trace_audit': dict(head_changes),
        'initial_condition_families': len({r['initial_condition_family'] for r in rows}),
        'rows': rows,
        'claim_boundary': 'Sets are hypotheses, not correct specific diagnoses. All scheduled failures remain in denominators. Paired reference available to every method; no LLM invoked. Windows and paired trajectories are dependent; no independent-window significance claim.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'rows', 'by_fault'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
