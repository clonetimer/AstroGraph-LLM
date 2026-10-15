"""Paired adoption gate for frozen baseline/challenger inference campaigns."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def compare(baseline: dict, challenger: dict) -> dict:
    key = lambda r: (r['case_id'], r['endpoint_s'])
    left = {key(r): r for r in baseline['rows']}
    right = {key(r): r for r in challenger['rows']}
    if set(left) != set(right) or len(left) != len(baseline['rows']) or len(right) != len(challenger['rows']):
        raise ValueError('paired campaigns must have identical unique scheduled windows')
    totals, by_mission = Counter(), defaultdict(Counter)
    changes = []
    for k, a in left.items():
        b = right[k]
        if any(a[f] != b[f] for f in ('target', 'mission', 'initial_condition_family')):
            raise ValueError('paired target/domain/group mismatch')
        target = a['target']
        p, q = a['predictions']['COMPLEMENTARY'], b['predictions']['COMPLEMENTARY']
        correct = lambda x: (x['kind'] == 'NORMAL' if target is None else x['kind'] == 'CLASS' and x['ids'] == [target])
        wrong = lambda x: x['kind'] == 'CLASS' and x['ids'] != [target]
        values = {
            'scheduled': 1,
            'correct_preserved': int(correct(p) and correct(q)),
            'added_correct_specific': int(target is not None and not correct(p) and correct(q)),
            'lost_correct': int(correct(p) and not correct(q)),
            'added_wrong_specific': int(wrong(q) and (not wrong(p) or p['ids'] != q['ids'])),
            'added_normal_candidate_alert': int(target is None and bool(q['ids']) and not p['ids']),
            'availability_mismatch': int((p['kind'] == 'UNAVAILABLE') != (q['kind'] == 'UNAVAILABLE')),
        }
        totals.update(values)
        by_mission[a['mission']].update(values)
        if p != q:
            changes.append({'case_id': k[0], 'endpoint_s': k[1], 'target': target,
                            'mission': a['mission'], 'before': p, 'after': q})
    failed = any(totals[k] for k in ('lost_correct', 'added_wrong_specific', 'added_normal_candidate_alert', 'availability_mismatch'))
    return {'adoption_gate': 'PASS' if not failed and totals['added_correct_specific'] > 0 else 'NOT_PASSED',
            'totals': dict(totals), 'by_mission': {k: dict(v) for k, v in by_mission.items()}, 'changes': changes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--challenger', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    a = json.loads(args.baseline.read_text())
    b = json.loads(args.challenger.read_text())
    report = compare(a, b)
    raw_matches = 0
    for case in Path(a['source_evaluation']).glob('case-*/windows.json'):
        other = Path(b['source_evaluation'])/case.parent.name/'windows.json'
        x, y = json.loads(case.read_text()), json.loads(other.read_text())
        if len(x) != len(y):
            raise ValueError('raw window schedule mismatch')
        for l, r in zip(x, y, strict=True):
            if l['endpoint_s'] != r['endpoint_s'] or l['status'] != r['status']:
                raise ValueError('raw window status mismatch')
            if l['status'] == 'SUCCEEDED':
                if (l['summary']['three_task_results'] != r['summary']['three_task_results'] or
                        l['paired_reference_identical'] != r['paired_reference_identical']):
                    raise ValueError('model output or reference control differs across policies')
                raw_matches += 1
    report.update({'schema_version': 'astrograph-complementary-adoption.v1',
                   'same_model_output_and_reference_windows': raw_matches,
                   'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (args.baseline, args.challenger)},
                   'claim_boundary': 'Descriptive paired acceptance on the frozen pilot only; dependent windows, no universal generalization or flight qualification.'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'changes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
