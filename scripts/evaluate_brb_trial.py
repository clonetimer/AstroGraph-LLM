"""Fit/freeze or evaluate an offline fusion challenger; labels never enter predict()."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from astrograph.fusion.belief_rules import activation
from astrograph.integrations.brb_diagnosis_trial import packet,predict,base_prediction,specificity_only,soft_features


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def load(evaluation,dataset):
    cases=json.loads((evaluation/'evaluation.json').read_text())['records'];out=[]
    for case in cases:
        # Evaluation joins, outside the inference adapter. Inference receives
        # only the completed summary below, never this row or its metadata.
        meta=json.loads((dataset/case['source_path']).with_name('sample_metadata.json').read_text())
        for w in json.loads((evaluation/case['case_id']/'windows.json').read_text()):
            active=meta['event_start_s'] is not None and w['endpoint_s']>=meta['event_start_s']
            out.append({'case':case['case_id'],'endpoint':w['endpoint_s'],'family':meta['pair_seed'],
                        'mission':meta['mission_id'],'target':case['evaluation_target'] if active else None,
                        'summary':w.get('summary') if w['status']=='SUCCEEDED' else None})
    return out


def outcome(row,p):
    target=row['target'];correct=(target is None and p['kind']=='NORMAL') or (target is not None and p['kind']=='CLASS' and p['ids']==[target])
    return {'correct':int(correct),'wrong':int(p['kind']=='CLASS' and p['ids']!=[target]),
            'candidate_hit':int(target is not None and target in p['ids']),
            'specific':int(p['kind']=='CLASS'),'members':len(p['ids']) if target is not None else 0,
            'candidate_output':int(target is not None and bool(p['ids'])),
            'normal_alert':int(target is None and bool(p['ids'])),'unavailable':int(p['kind']=='UNAVAILABLE')}


def measure(rows,outputs):
    counts=defaultdict(int)
    for r,p in zip(rows,outputs,strict=True):
        for k,v in outcome(r,p).items():counts[k]+=v
    return {'n':len(rows),'fault_active':sum(r['target'] is not None for r in rows),**counts,
            'specific_risk':counts['wrong']/counts['specific'] if counts['specific'] else None,
            'mean_candidate_size':counts['members']/counts['candidate_output'] if counts['candidate_output'] else None}


def outputs(rows,model=None,repair_only=False):
    results=[]
    for r in rows:
        s=r['summary']
        results.append({'kind':'UNAVAILABLE','ids':[]} if s is None else
                       predict(s,model)['prediction'] if model else specificity_only(s) if repair_only else base_prediction(s))
    return results


def fit(rows,method,repair):
    if method=='BRB':
        successes=np.zeros(12);counts=np.zeros(12)
        for r in rows:
            if not r['summary']:continue
            for fid,x in packet(r['summary'],repair_specificity=repair)['candidates'].items():
                a=np.array(activation(x['data'],x['mechanism']))/4
                counts+=a;successes+=a*(r['target']==fid)
        completeness=counts/(counts+2.)
        positive=(successes+1.)/(counts+2.)
        rules=np.column_stack([positive*completeness,(1-positive)*completeness]).tolist()
        model={'method':method,'rules':rules,'training_rule_effective_counts':counts.tolist(),
               'rule_weights':[1.]*12,'attribute_weights':[1.,1.],
               'parameter_note':'12 support proportions and 12 completeness estimates; fixed grid/weights; Laplace smoothing strength 2, not claimed optimal.'}
    else:
        xrows=[];ys=[]
        for r in rows:
            if not r['summary']:continue
            for fid,x in packet(r['summary'],repair_specificity=repair)['candidates'].items():
                xrows.append(soft_features(x));ys.append(int(r['target']==fid))
        model_fit=LogisticRegression(C=1.,max_iter=1000,random_state=0).fit(xrows,ys,sample_weight=np.full(len(ys),.25))
        model={'method':method,'coefficients':model_fit.coef_[0].tolist(),'intercept':float(model_fit.intercept_[0]),'parameter_note':'Five coefficients plus intercept; L2 C=1; score calibration not established.'}
    return {**model,'repair_specificity':repair,'threshold':.5,'margin':.1}


def tune(rows,model):
    baseline=measure(rows,outputs(rows));trials=[]
    for threshold in [.2,.3,.4,.5,.6,.7,.8,.9,.99]:
        for margin in [0.,.05,.1,.2]:
            candidate={**model,'threshold':threshold,'margin':margin}
            m=measure(rows,outputs(rows,candidate));trials.append({'threshold':threshold,'margin':margin,'metrics':m})
    feasible=[t for t in trials if t['metrics']['wrong']<=baseline['wrong'] and t['metrics']['normal_alert']<=baseline['normal_alert']]
    chosen=max(feasible or trials,key=lambda t:(t['metrics']['correct'],-t['metrics']['wrong'],t['metrics']['candidate_hit'],-t['threshold'],-t['margin']))
    return {**model,'threshold':chosen['threshold'],'margin':chosen['margin']},trials


def report(rows,models):
    start=time.perf_counter();preds={'BASELINE':outputs(rows),'SPECIFICITY_ONLY':outputs(rows,repair_only=True)}
    for name,model in models.items():preds[name]=outputs(rows,model)
    metrics={n:measure(rows,p) for n,p in preds.items()}
    domains={m:{n:measure([r for r in rows if r['mission']==m],[p[i] for i,r in enumerate(rows) if r['mission']==m]) for n,p in preds.items()} for m in sorted({r['mission'] for r in rows})}
    # Paired cluster bootstrap: all missions/trajectories/windows sharing a seed
    # family move together. These intervals do not establish new-domain validity.
    families=sorted({r['family'] for r in rows});rng=np.random.default_rng(1927)
    draws=rng.integers(len(families),size=(1000,len(families)))
    intervals={}
    for name in models:
        intervals[name]={}
        for comparator in ['BASELINE','SPECIFICITY_ONLY','SOFT_REPAIRED']:
            if comparator==name:continue
            diffs=np.array([[sum(outcome(r,preds[name][i])[metric]-outcome(r,preds[comparator][i])[metric] for i,r in enumerate(rows) if r['family']==f) for metric in ['correct','wrong']] for f in families])
            sample=diffs[draws].sum(axis=1)
            intervals[name][comparator]={'correct_count_difference_95_percentile':np.quantile(sample[:,0],[.025,.975]).tolist(),'wrong_count_difference_95_percentile':np.quantile(sample[:,1],[.025,.975]).tolist()}
    gates={}
    for name in ['BRB_RAW','BRB_REPAIRED']:
        # BRB must beat both mechanism-only correction and the simpler fitted
        # soft fusion, not just undo known baseline mistakes.
        candidate=metrics[name];comparators=['BASELINE','SPECIFICITY_ONLY','SOFT_REPAIRED']
        ok=all(candidate['correct']>=metrics[n]['correct'] and candidate['wrong']<=metrics[n]['wrong'] and candidate['normal_alert']<=metrics[n]['normal_alert'] and candidate['candidate_hit']>=metrics[n]['candidate_hit'] for n in comparators)
        better=any(candidate['correct']>metrics[n]['correct'] or candidate['wrong']<metrics[n]['wrong'] for n in comparators)
        domain_ok=all(domains[m][name]['correct']>=domains[m]['BASELINE']['correct'] and domains[m][name]['wrong']<=domains[m]['BASELINE']['wrong'] for m in domains)
        gates[name]='PASS_PILOT_ONLY' if ok and better and domain_ok else 'NOT_PASSED'
    return {'metrics':metrics,'by_mission':domains,'cluster_bootstrap':intervals,'family_count':len(families),'adoption_gate':gates,'duration_s':time.perf_counter()-start,
            'rows':[{k:v for k,v in r.items() if k!='summary'}|{'predictions':{n:p[i] for n,p in preds.items()}} for i,r in enumerate(rows)]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','evaluate']);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frozen-model',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=load(a.evaluation,a.dataset)
    if a.mode=='fit':
        families=sorted({r['family'] for r in rows});assert len(families)>=10
        train_ids=families[:6];validation_ids=families[6:];train=[r for r in rows if r['family'] in train_ids];validation=[r for r in rows if r['family'] in validation_ids]
        models={};search={}
        for name,method,repair in [('SOFT_REPAIRED','SOFT',True),('BRB_RAW','BRB',False),('BRB_REPAIRED','BRB',True)]:
            models[name],search[name]=tune(validation,fit(train,method,repair))
        freeze={'schema_version':'astrograph-small-brb-trial.v1','models':models,'training_families':train_ids,'validation_families':validation_ids,'development_source':str(a.evaluation),'fresh_seed':20261219,'endpoints':[40,120,200,300],
                'source_sha256':{str(f):sha(f) for f in [Path('scripts/evaluate_brb_trial.py'),Path('src/astrograph/fusion/belief_rules.py'),Path('src/astrograph/integrations/brb_diagnosis_trial.py')]},
                'adoption_criterion':'At selected frozen operating points, no fewer correct/hits or more wrong/normal alerts than baseline, specificity repair, and simple soft fusion; strict improvement; no per-mission baseline regression. Descriptive pilot gate, not universal or flight qualification.',
                'claim_boundary':'Offline only, no runtime/default changes. Same original model, same raw observations, no LLM. Mechanism correction means demoting a non-identifying residual envelope, not adding a sensor.'}
        write(a.output/'frozen-model.json',freeze);write(a.output/'selection.json',search);result=report(validation,models)
    else:
        if a.frozen_model is None:p.error('--frozen-model required')
        freeze=json.loads(a.frozen_model.read_text())
        for f,digest in freeze['source_sha256'].items():
            if sha(f)!=digest:raise ValueError('scientific source changed after freeze: '+f)
        result=report(rows,freeze['models']);result['frozen_model_sha256']=sha(a.frozen_model)
    result['source_evaluation_sha256']=sha(a.evaluation/'evaluation.json');write(a.output/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['metrics','adoption_gate','family_count']},indent=2))

if __name__=='__main__':main()
