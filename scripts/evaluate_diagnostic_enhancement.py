"""Compare frozen diagnosis runs at injection and functional-effect granularity.

Labels are joined here after inference. Neither this mapper nor metadata enters
native model/evidence processing. Candidate hits are never counted as decisions.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
from astrograph.integrations.brb_diagnosis_trial import base_prediction


def effect_target(target):
    return 'ADCS_RW_TORQUE_AUTHORITY_LOSS' if target=='ADCS_RW_MOTOR_FAILURE' else target


def load_run(path,dataset):
    rows={}
    for case in json.loads((path/'evaluation.json').read_text())['records']:
        meta=json.loads((dataset/case['source_path']).with_name('sample_metadata.json').read_text())
        for w in json.loads((path/case['case_id']/'windows.json').read_text()):
            key=(case['source_path'],w['endpoint_s'])
            if key in rows:raise ValueError('duplicate evaluation window')
            target=case['evaluation_target'] if meta['event_start_s'] is not None and w['endpoint_s']>=meta['event_start_s'] else None
            prediction=base_prediction(w['summary']) if w['status']=='SUCCEEDED' else {'kind':'UNAVAILABLE','ids':[]}
            rows[key]={'target':target,'family':meta['pair_seed'],'prediction':prediction,
                'degraded':w.get('degraded_evidence'), 'event_start_s':meta['event_start_s']}
    return rows


def measure(rows,*,effect=False):
    c=Counter();families=defaultdict(Counter)
    for row in rows.values():
        target=effect_target(row['target']) if effect else row['target']
        p=row['prediction'];kind=p['kind'];ids=p['ids'];f=families[str(row['family'])]
        counts={'windows':1,'fault_active':int(target is not None),kind:1}
        decided=kind in {'CLASS','NORMAL'}
        correct=(kind=='CLASS' and ids==[target]) or (kind=='NORMAL' and target is None)
        counts.update(decided=int(decided),correct=int(correct),wrong=int(decided and not correct),
            candidate_hit=int(target is not None and target in ids),
            candidate_output=int(target is not None and bool(ids)),
            normal_alert=int(target is None and bool(ids)))
        d=row['degraded']
        if d:
            counts['degraded_observation_receipts']=1
            counts['degraded_candidate_output']=int(bool(d['candidate_fault_ids']))
            counts['degraded_candidate_hit']=int(target is not None and target in d['candidate_fault_ids'])
            counts['degraded_normal_alert']=int(target is None and bool(d['candidate_fault_ids']))
        c.update(counts);f.update(counts)
    return {**dict(c),'decision_coverage':c['decided']/c['windows'],
        'conditional_correctness':c['correct']/c['decided'] if c['decided'] else None,
        'family_count':len(families),'by_family':dict(families)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',action='append',required=True,help='NAME=completed evaluation directory')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'claim_boundary':'Effect-level alignment is an additional metric, not fine root-cause identification. Degraded candidates are separate from available model decisions.','runs':{}}
    reference=None
    for item in a.run:
        name,location=item.split('=',1);path=Path(location);rows=load_run(path,a.dataset)
        plan=json.loads((path/'plan.json').read_text())
        identity=(set(rows),plan['telemetry_sha256'],plan['model_sha256'])
        if reference is None:reference=identity
        elif identity!=reference:raise ValueError('cohort, windows or model differ between comparisons')
        result['runs'][name]={'injection_label_metrics':measure(rows),'effect_level_metrics':measure(rows,effect=True),
            'plan':plan,'evaluation_sha256':hashlib.sha256((path/'evaluation.json').read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{level:{x:y for x,y in v[level].items() if x!='by_family'} for level in ['injection_label_metrics','effect_level_metrics']} for k,v in result['runs'].items()},indent=2))

if __name__=='__main__':main()
