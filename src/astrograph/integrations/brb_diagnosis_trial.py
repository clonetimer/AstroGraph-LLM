"""Offline ADCS fusion challenger. Never called by the production runtime.

Consumes a whitelisted completed summary, never labels or case identity.
Only the profile identifies low-specificity mechanisms. No automatic KG update.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import math
from astrograph.fusion.belief_rules import MECHANISM_STATES, infer


def packet(summary: dict, *, repair_specificity: bool) -> dict:
    decision = summary['diagnostic_decision']
    head = summary['three_task_results']
    trace = head.get('head_trace')
    if not trace:
        raise ValueError('raw pre-hierarchy trace is required')
    mass = defaultdict(float)
    for label,p in trace['raw_class_probabilities'].items():
        if label in trace['class_to_runtime_fault']:
            mass[trace['class_to_runtime_fault'][label]] += p
    records = {c['fault_id']:c for c in summary['candidates']}
    states = {}
    for fid in sorted(set(mass)|set(records)):
        c = records.get(fid, {})
        rows = [x for x in decision['evidence_state_matrix'] if x['fault_id']==fid]
        if not rows or any(x['state'] in {'MISSING_DATA','INVALID_DATA','UNKNOWN','UNEVALUATED','MIXED'} for x in rows):
            state = 'UNKNOWN'
        else:
            state = 'SUPPORTED' if c.get('eligible_for_explanation') else 'UNSUPPORTED'
        # This frozen source-specific profile marks the legacy residual envelope
        # as non-identifying. It does NOT fabricate a new gyro observation.
        if repair_specificity and fid=='ADCS_GYRO_BIAS_DRIFT' and state=='SUPPORTED':
            state = 'NONSPECIFIC'
        states[fid] = {'data':mass.get(fid,0.),'mechanism':state,
                       'contradicted':bool(c.get('contradicting_evidence_ids')),
                       'evidence_ids':c.get('supporting_evidence_ids',[])}
    ordered = sorted(mass, key=lambda f:(-mass[f],f))
    cut = mass[ordered[min(2,len(ordered))-1]] if ordered else 1.
    alternatives = {f for f in ordered if mass[f]>=cut and mass[f]>0}
    alternatives |= {f for f,x in states.items() if x['mechanism'] in {'SUPPORTED','NONSPECIFIC'}}
    alternatives -= {f for f,x in states.items() if x['contradicted']}
    return {'candidates':states,'alternatives':sorted(alternatives),
            'status':decision['status'],'old_ids':list(decision['candidate_fault_ids']),
            'detected':head['detection']['detected'],
            'blocking_conflicts':list(decision['blocking_conflicts']),
            'paired_identical':decision['observation_contract']['paired_reference_identical']}


def base_prediction(summary):
    d=summary['diagnostic_decision']
    kind={'NORMAL_REFERENCE_MATCH':'NORMAL','SUPPORTED_SINGLETON':'CLASS','CANDIDATE_SET':'SET'}.get(d['status'],'ABSTAIN')
    return {'kind':kind,'ids':list(d['candidate_fault_ids'])}


def bounded(ids):
    return {'kind':'SET','ids':sorted(ids)} if 0<len(ids)<=3 else {'kind':'ABSTAIN','ids':[]}


def specificity_only(summary):
    old=base_prediction(summary)
    if old=={'kind':'CLASS','ids':['ADCS_GYRO_BIAS_DRIFT']}:
        p=packet(summary,repair_specificity=True)
        return bounded(p['alternatives'])
    return old


def soft_features(x):
    return [x['data'], *[float(x['mechanism']==s) for s in MECHANISM_STATES]]


def predict(summary: dict, model: dict) -> dict:
    p=packet(summary,repair_specificity=model['repair_specificity'])
    if p['paired_identical']:
        return {'prediction':{'kind':'NORMAL','ids':[]},'scores':{}}
    if p['blocking_conflicts'] or not p['detected']:
        # Preserve the qualified native detector-disagreement policy, never
        # invent a new detection trigger in an offline fusion experiment.
        return {'prediction':specificity_only(summary) if model['repair_specificity'] else base_prediction(summary),'scores':{},'policy':'PRESERVED_GATE'}
    scores={}
    for fid,x in p['candidates'].items():
        if x['contradicted']:continue
        if model['method']=='BRB':
            result=infer(x['data'],x['mechanism'],model['rules'])
            scores[fid]={'support':result['beliefs'][0],'ignorance':result['ignorance']}
        elif model['method']=='SOFT':
            z=model['intercept']+sum(a*b for a,b in zip(model['coefficients'],soft_features(x),strict=True))
            scores[fid]={'support':1/(1+math.exp(-max(-700.,min(700.,z)))),'ignorance':None}
        else:raise ValueError('unknown fusion challenger')
    ranked=sorted(scores,key=lambda f:(-scores[f]['support'],f))
    if not ranked:return {'prediction':{'kind':'ABSTAIN','ids':[]},'scores':scores}
    top=ranked[0];score=scores[top]['support'];second=scores[ranked[1]]['support'] if len(ranked)>1 else 0.
    unique=len(ranked)==1 or score>second+1e-12
    if unique and score>=model['threshold'] and score-second>=model['margin']:
        result={'kind':'CLASS','ids':[top]}
    else:result=bounded(p['alternatives'])
    return {'prediction':result,'scores':scores,'needed_observations':
            ['Independent sensor validity, rate consistency and operating-condition checks; residual envelope alone is not identifying.'] if result['kind']!='CLASS' else []}
