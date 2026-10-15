"""Offline truth join AFTER frozen prediction generation; retain all planned rows.

Reports independent heads and detector-gated system outputs separately. Knowledge
and paired-reference normal mode is a different-information track, not a claim
that fusion alone outperforms an officially reproduced SOTA system.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from astrograph.agents.mission.storage import file_sha
from astrograph.evaluation.three_task_replay import (
    classification_metrics, _fault_prediction, _detection_prediction,
    NORMAL, UNAVAILABLE, ABSTAIN,
)


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def score_predictions(truth, predictions, *, faults, components, fault_component):
    active = [i for i, r in enumerate(truth) if r['anomaly'] == 1]
    yd = ['ANOMALY' if r['anomaly'] else NORMAL for r in truth]
    yf = [faults[truth[i]['fault']] for i in active]
    yc = [components[truth[i]['component']] for i in active]
    pd = [_detection_prediction(r['prediction']) for r in predictions]
    pf = [_fault_prediction(predictions[i]['prediction']) for i in active]
    pc = [fault_component[p] if p in fault_component else p for p in pf]
    # Component agreement is not the same as agreement on a particular fault.
    for n, i in enumerate(active):
        pred = predictions[i]['prediction']
        if pred['kind'] == 'SET' and pred['ids']:
            cs = {fault_component[f] for f in pred['ids']}
            if len(cs) == 1:
                pc[n] = next(iter(cs))
    valid = [i for i, p in enumerate(pd) if p != UNAVAILABLE]
    dc = classification_metrics(yd, pd, labels=[NORMAL, 'ANOMALY'])
    dv = classification_metrics([yd[i] for i in valid], [pd[i] for i in valid], labels=[NORMAL, 'ANOMALY'])
    fc = classification_metrics(yf, pf, labels=faults)
    cc = classification_metrics(yc, pc, labels=components)
    direct_c = [predictions[i].get('component', UNAVAILABLE) for i in active]
    direct_f = [predictions[i].get('fault', UNAVAILABLE) for i in active]
    return {
        'detection_all': dc, 'detection_executable': dv,
        'localization_independent_head': classification_metrics(yc, direct_c, labels=components),
        'classification_raw_head': classification_metrics(yf, direct_f, labels=faults),
        'localization_system_derived': cc, 'classification_system': fc,
        'fault_counts': {
            'n': len(active), 'correct_class': sum(t == p for t, p in zip(yf, pf)),
            'wrong_fault_class': sum(p in faults and t != p for t, p in zip(yf, pf)),
            'missed_as_normal': sum(p == NORMAL for p in pf),
            'abstained': sum(p == ABSTAIN for p in pf),
            'unavailable': sum(p == UNAVAILABLE for p in pf),
        },
        'detection_counts': {
            'false_positive': sum(t == NORMAL and p == 'ANOMALY' for t, p in zip(yd, pd)),
            'false_negative_normal': sum(t == 'ANOMALY' and p == NORMAL for t, p in zip(yd, pd)),
            'abstained': sum(p == ABSTAIN for p in pd),
            'unavailable': sum(p == UNAVAILABLE for p in pd),
        },
        'denominators': {'planned':len(truth), 'executable':len(valid), 'fault_windows':len(active)},
    }


def checked_probs(values, rows, cols):
    values = np.asarray(values)
    if values.shape != (rows, cols) or not np.isfinite(values).all():
        raise ValueError('probability shape/nonfinite error')
    if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(1), 1., atol=1e-5):
        raise ValueError('invalid probability distribution')
    return values


def mean_sd(values):
    return {'mean':statistics.mean(values), 'sample_sd':statistics.stdev(values) if len(values)>1 else None,
            'n':len(values), 'values':values,
            'meaning':'training-seed variation only; not an independent-test confidence interval'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--normal-runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((a.data/'protocol.json').read_text());protocol_hash=file_sha(a.data/'protocol.json')
    truth=json.loads((a.data/'test_labels.json').read_text())
    rows=json.loads((a.data/'test_rows.json').read_text())
    if [r['id'] for r in truth] != [r['id'] for r in rows] or len({r['id'] for r in truth})!=len(truth):
        raise ValueError('truth/metadata ID mismatch')
    ids=[r['id'] for r in rows];valid_ids=[r['id'] for r in rows if r['valid']]
    classes=protocol['classes'];faults=[protocol['class_map'][x] for x in classes];components=protocol['components']
    fault_component={protocol['class_map'][f]:c for f,c in protocol['fault_to_component'].items()}
    reports={};predictions={};loaded=[]
    for family in protocol['families']:
        for seed in protocol['seeds']:
            key=family+'__'+str(seed);root=a.runs/key
            receipt=json.loads((root/'receipt.json').read_text())
            if receipt['protocol_sha256']!=protocol_hash or receipt['test_labels_read']:
                raise ValueError('protocol or training-label isolation violation')
            if receipt['test_predictions_sha256']!=file_sha(root/'test_probabilities.npz'):
                raise ValueError('prediction hash mismatch')
            raw=np.load(root/'test_probabilities.npz',allow_pickle=False)
            if list(raw['ids']) != valid_ids:raise ValueError('missing/unexpected prediction rows')
            d=checked_probs(raw['detector'],len(valid_ids),2);c=checked_probs(raw['localizer'],len(valid_ids),len(components))
            f=checked_probs(raw['classifier'],len(valid_ids),len(faults));threshold=float(raw['threshold'])
            if threshold != receipt['threshold']:raise ValueError('threshold receipt mismatch')
            by_id={}
            for j,identity in enumerate(valid_ids):
                target=faults[int(f[j].argmax())]
                by_id[identity]={'id':identity,'prediction':{'kind':'CLASS','ids':[target]} if d[j,1]>=threshold else {'kind':'NORMAL','ids':[]},
                    'component':components[int(c[j].argmax())], 'fault':target}
            pred=[by_id.get(i,{'id':i,'prediction':{'kind':'UNAVAILABLE','ids':[]}}) for i in ids]
            metrics=score_predictions(truth,pred,faults=faults,components=components,fault_component=fault_component)
            reports[key]={'name':family,'seed':seed,'track':'controlled_same33channels','metrics':metrics,'receipt':receipt}
            predictions[key]=pred;loaded.append({'path':str(root/'test_probabilities.npz'),'sha256':file_sha(root/'test_probabilities.npz')})
    parity={}
    for name in ['production']+[str(s) for s in protocol['seeds']]:
        root=a.normal_runs/name;receipt=json.loads((root/'receipt.json').read_text())
        rr=[json.loads(l) for l in (root/'inference.jsonl').read_text().splitlines()]
        if [r['id'] for r in rr]!=ids or receipt['labels_read'] or receipt['protocol_sha256']!=protocol_hash:
            raise ValueError('normal replay isolation/identity mismatch')
        if receipt['inference_sha256']!=file_sha(root/'inference.jsonl'):raise ValueError('normal prediction hash mismatch')
        pred=[];dp=[]
        for r in rr:
            head={}
            if r.get('direct_component_probabilities'):
                cc=r['direct_component_probabilities'];fc=r['raw_class_probabilities']
                if list(cc)!=components or list(fc)!=classes:raise ValueError('head class order differs')
                checked_probs([list(cc.values())],1,len(components));checked_probs([list(fc.values())],1,len(classes))
                head={'component':max(cc,key=cc.get), 'fault':protocol['class_map'][max(fc,key=fc.get)]}
            pred.append({'id':r['id'],'prediction':r['normal'],**head});dp.append({'id':r['id'],'prediction':r['data_only'],**head})
        key='normal__'+name;predictions[key]=pred
        reports[key]={'name':'production_normal' if name=='production' else 'controlled_tcn_plus_normal',
            'seed':None if name=='production' else int(name),'track':'extra_knowledge_reference_observables',
            'metrics':score_predictions(truth,pred,faults=faults,components=components,fault_component=fault_component),'receipt':receipt,
            'data_only_same_replay':score_predictions(truth,dp,faults=faults,components=components,fault_component=fault_component)}
        if name!='production':
            base=predictions['residual_tcn__'+name]
            raw=np.load(a.runs/('residual_tcn__'+name)/'test_probabilities.npz',allow_pickle=False)
            vrows=[r for r in rr if r['valid'] and r['status'] in {'SUCCEEDED','PARTIAL'}]
            dpdiff=sum(x['prediction']!=y['prediction'] for x,y in zip(dp,base))
            diffd=max(abs(float(r['detector_probability'])-float(raw['detector'][j,1])) for j,r in enumerate(vrows)) if len(vrows)==len(valid_ids) else None
            parity[name]={'data_only_prediction_differences':dpdiff,'max_detector_probability_difference':diffd,
                'valid_normal_execution':len(vrows),'compared_to_same_trained_tcn':True}
        loaded.append({'path':str(root/'inference.jsonl'),'sha256':file_sha(root/'inference.jsonl')})
    metric_keys=['detection_all','detection_executable','localization_independent_head','classification_raw_head','localization_system_derived','classification_system']
    aggregate={}
    for family in protocol['families']+['controlled_tcn_plus_normal']:
        group=[r for r in reports.values() if r['name']==family]
        aggregate[family]={m:mean_sd([r['metrics'][m]['macro_f1'] for r in group]) for m in metric_keys}
    # Pairwise changes are evaluated on the exact same windows and same model seed.
    changes={}
    for seed in protocol['seeds']:
        key=str(seed);old=predictions['residual_tcn__'+key];new=predictions['normal__'+key]
        benefit=harm=both_wrong=both_correct=0;allchanges=[]
        for t,b,n in zip(truth,old,new):
            expected=faults[t['fault']] if t['anomaly'] else NORMAL
            bp=_fault_prediction(b['prediction']);npred=_fault_prediction(n['prediction'])
            bc=bp==expected;nc=npred==expected
            benefit+=not bc and nc;harm+=bc and not nc;both_wrong+=not bc and not nc;both_correct+=bc and nc
            if bp!=npred:allchanges.append({'id':t['id'],'truth':expected,'data':b['prediction'],'normal':n['prediction'],'corrected':not bc and nc,'harmed':bc and not nc})
        changes[key]={'corrected':benefit,'harmed':harm,'net_correct_change':benefit-harm,'both_wrong':both_wrong,'both_correct':both_correct,'changed_windows':allchanges}
    out={'schema_version':'astrograph-work-modes-three-task-comparison.v1','evaluated_unix':time.time(),
         'protocol':protocol,'protocol_sha256':protocol_hash,'truth_sha256':file_sha(a.data/'test_labels.json'),
         'reports':reports,'seed_summary':aggregate,'same_model_normal_changes':changes,'same_model_input_parity':parity,
         'prediction_artifacts':loaded,'independent_confirmation':False,
         'limitations':['Previously used two global test seed families; no fresh external validation.',
             'Five neural architectures are compact three-task adaptations, not official full SOTA reproductions.',
             'Same epoch ceiling does not give equal FLOPs; no architecture-specific tuning. Sparse training windows.',
             'Normal mode has paired healthy reference, approved graph and extra sensor observations; unequal information.',
             'Mean ± SD is across three training seeds only; no test-set model or threshold selection.',
             'Existing production model trained under a different larger budget; context result, not fair retraining winner.',
             'All 45 unavailable windows retained in all-window metrics; class F1 includes abstention FN.',
             'No actual LLM endpoint used; these results concern normal-mode core and deterministic checks.']}
    dump(a.output/'COMPARISON_RESULTS.json',out)
    small=[]
    for family,met in aggregate.items():
        small.append({'model':family,**{k:met[k] for k in metric_keys}})
    dump(a.output/'METRICS_TABLE.json',{'rows':small,'production_context':reports['normal__production']['metrics'],'paired_changes':changes,'input_parity':parity})
    print(json.dumps({'output':str(a.output),'controlled_runs':18,'normal_runs':4,'paired_changes':{k:{x:v[x] for x in ('corrected','harmed','net_correct_change')} for k,v in changes.items()}},ensure_ascii=False))

if __name__=='__main__':main()
