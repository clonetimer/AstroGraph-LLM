"""Frozen-cohort ADCS replay: labels are joined only after all inference finishes.

The supplied model split is preserved: development=3/8, confirmation=4/9.
Confirmation freezes this iteration only; these are existing same-simulator
project data, not a new external dataset or cross-spacecraft evidence. All windows
are endpoint-selected a priori, including normal prefixes. No model is trained.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction(decision):
    status=decision['status'];selected=decision.get('selected_fault_id')
    if selected:
        return {'kind':'MODEL_ONLY' if status=='MODEL_ONLY' else 'CLASS','ids':[selected]}
    if status in {'NORMAL_REFERENCE_MATCH','NO_DETECTED_ANOMALY'}:
        return {'kind':'NORMAL','ids':[]}
    ids=decision.get('candidate_fault_ids',[])
    return {'kind':'SET' if ids else 'ABSTAIN','ids':ids}


def working_prediction(decision):
    strict = prediction(decision)
    preferred = decision.get('working_hypothesis_fault_id')
    if strict['kind'] in {'SET','ABSTAIN'} and preferred:
        return {'kind':'PREFERRED_UNCONFIRMED','ids':[preferred]}
    return strict


def metrics(rows,branch):
    c=Counter()
    for row in rows:
        p=row[branch];target=row['target']
        selected=p['kind'] in {'CLASS','MODEL_ONLY','PREFERRED_UNCONFIRMED','NORMAL'}
        correct=selected and ((p['kind']=='NORMAL' and target is None) or (p['kind']!='NORMAL' and p['ids']==[target]))
        c.update(windows=1,active_fault_windows=int(target is not None),decided=int(selected),correct=int(correct),
                 wrong=int(selected and not correct),unresolved=int(not selected),model_only=int(p['kind']=='MODEL_ONLY'),
                 class_outputs=int(p['kind']=='CLASS'),preferred_unconfirmed=int(p['kind']=='PREFERRED_UNCONFIRMED'),normal_outputs=int(p['kind']=='NORMAL'),
                 normal_alert=int(target is None and bool(p['ids'])),
                 candidate_hit=int(target is not None and target in p['ids']))
    return {**dict(c),'selected_label_accuracy_all_windows':c['correct']/max(1,c['windows']),
            'conditional_correctness':c['correct']/c['decided'] if c['decided'] else None,
            'decision_coverage':c['decided']/max(1,c['windows'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-zip',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cohort',choices=['development','confirmation'],default='development')
    p.add_argument('--samples',type=int,nargs='+',default=None)
    p.add_argument('--endpoints',type=float,nargs='+',default=[30,150,300])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    expected_samples = [3,8] if a.cohort=='development' else [4,9]
    if a.samples is None:a.samples=expected_samples
    if sorted(a.samples)!=expected_samples:raise ValueError('Use the complete predefined split, not a selected subset')
    import torch
    torch.set_num_threads(1)
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    from astrograph.simulation import SimulationRunCompleted
    from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,build_registry_from_adcs_three_task_provider
    from astrograph.oag.native_bridge import from_native_frames,write_package_graph
    from astrograph.oag.runtime import build_csv_oag_runtime,OAG_PIPELINE_ID
    from astrograph.oag.kernel import EvidenceDiagnosisKernel
    from astrograph.oag.physical_diagnosis import diagnose_physical_complement
    from astrograph.integrations.brb_diagnosis_trial import base_prediction
    from astrograph.runtime import RuntimeRequest
    from astrograph.oag.contracts import content_hash
    root=a.project_root.resolve()
    model=root/'models/adcs_v5_2_2/p2_innovation_v2.pt'
    package_path=root/'configs/datasets/adcs_v5_2_2/runtime_window_integrated_evidence.json'
    with zipfile.ZipFile(a.dataset_zip) as z:
        dataset_root=next(n for n in z.namelist() if n.endswith('/dataset_manifest.json')).rsplit('/',1)[0]
        manifest=json.loads(z.read(dataset_root+'/dataset_manifest.json'))
        mission_by_scenario={s['scenario_id']:s['mission_id'] for s in manifest['operating_scenarios']}
        files=sorted(n for n in z.namelist() if n.endswith('/telemetry.csv') and any('/sample_%02d/'%i in n for i in a.samples))
        plan={'cohort_kind':a.cohort,'samples':a.samples,'endpoints':a.endpoints,'window_length':20,'trajectory_count':len(files),
              'dataset_zip_sha256':digest(a.dataset_zip),'model_sha256':digest(model),'package_sha256':digest(package_path),
              'time_equality_tolerance_s':1e-8,'cohort':files,'code_sha256':{str(f.relative_to(root)):digest(f) for folder in ['src/astrograph/oag','src/astrograph/fusion','src/astrograph/memory'] for f in sorted((root/folder).glob('*.py'))},
              'model_training_receipt':json.loads((model.parent/'p2_innovation.receipt.json').read_text())['split_audit']['assignment'],
              'scope':a.cohort+' split of existing dataset, not independent new simulator data. Fixed existing weights and label taxonomy. No empirical estimate of Agent tool selection or memory benefit: evidence is fully computed before this replay. Working hypotheses are explicitly distinct from verified singletons.',
              'script_sha256':digest(Path(__file__)), 'paired_reference_ablation':'data_plus_reference uses the same exact-input identity rule, to prevent attributing reference-only gains to KG fusion.'}
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        provider=load_adcs_three_task_torch_package(model,expected_sha256=digest(model),device='cpu')
        registry=build_registry_from_adcs_three_task_provider(provider)
        builder=ConfiguredRuntimeEvidenceBuilder(package_path)
        graph=write_package_graph(package_path,a.output/'graph')
        legacy_runtime=build_csv_oag_runtime(graph)
        kernel=EvidenceDiagnosisKernel(graph)
        nominal_cache={};inferences=[]
        started=time.monotonic()
        with (a.output/'inference.jsonl').open('w') as log:
            for number,name in enumerate(files):
                rel=name[len(dataset_root)+1:];parts=Path(rel).parts
                mission=parts[1] if parts[0]=='nominal_baseline' else mission_by_scenario[parts[1]]
                nominal_name=dataset_root+'/nominal_baseline/'+mission+'/'+parts[-2]+'/telemetry.csv'
                if nominal_name not in nominal_cache:
                    nominal_cache[nominal_name]=pd.read_csv(io.BytesIO(z.read(nominal_name)),low_memory=False)
                nominal=nominal_cache[nominal_name]
                source=pd.read_csv(io.BytesIO(z.read(name)),low_memory=False)
                opaque=hashlib.sha256(name.encode()).hexdigest()[:20]
                # Files, truth-bearing metadata and path-derived fault names never
                # become an input to the model or graph reasoning functions.
                for endpoint in a.endpoints:
                    record={'source':name,'endpoint_s':endpoint,'case_id':opaque}
                    try:
                        event=SimulationRunCompleted(event_id='EVENT-'+opaque,run_id='RUN-'+opaque,
                              run_dir='/admitted/'+opaque,nominal_run_dir='/admitted/reference',runtime_id='fixed-development-replay')
                        prefix=source[source.time_s<=endpoint+1e-8].copy()
                        baseline=nominal[nominal.time_s<=endpoint+1e-8].copy()
                        data=from_native_frames(builder=builder,event=event,source=prefix,nominal=baseline,
                                                provider=provider,registry=registry,review_evidence=True)
                        old=legacy_runtime.execute(RuntimeRequest(request_id='old-'+opaque+str(endpoint),pipeline_id=OAG_PIPELINE_ID,
                                                                 payload=data.model_dump(mode='json')))
                        if old.status.value!='SUCCEEDED':raise RuntimeError(str(old.error))
                        legacy=base_prediction(old.output['summary'])
                        new=kernel.evaluate(data)
                        heads=data.three_task_result
                        active=[c for c in data.diagnostic_ranking.candidates if c.contributes_to_fusion]
                        top=[c.fault_id for c in active if c.rank==min((r.rank for r in active),default=1)]
                        model_prediction={'kind':'NORMAL','ids':[]} if not heads.detected else {'kind':'CLASS' if len(top)==1 else 'SET','ids':top}
                        try:
                            physical=diagnose_physical_complement(prefix,baseline,old.output['summary'])['prediction']
                        except Exception as exc:
                            physical={'kind':'UNAVAILABLE','ids':[]}
                            record['optional_physical_error']=type(exc).__name__+': '+str(exc)
                        record.update(status='SUCCEEDED',data_only=model_prediction,
                                      data_plus_reference=({'kind':'NORMAL','ids':[]} if new['diagnosis']['status']=='NORMAL_REFERENCE_MATCH' else model_prediction),
                                      legacy=legacy,evidence_aware=prediction(new['diagnosis']),working_hypothesis=working_prediction(new['diagnosis']),
                                      post47_physical=physical,new_status=new['diagnosis']['status'],new_reason_codes=new['diagnosis']['reason_codes'],
                                      initial_bundle_sha256=content_hash(data.bundle.model_dump(mode='json')),
                                      diagnostic_summary_sha256=new['summary']['summary_sha256'])
                        if number<2 and endpoint==a.endpoints[-1]:
                            (a.output/(opaque+'_explanation.json')).write_text(json.dumps(new,indent=2)+'\n')
                    except Exception as exc:
                        record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
                        for branch in ('data_only','data_plus_reference','legacy','evidence_aware','working_hypothesis','post47_physical'):
                            record[branch]={'kind':'UNAVAILABLE','ids':[]}
                    inferences.append(record);log.write(json.dumps(record)+'\n');log.flush()
                if (number+1)%20==0:print(f'{number+1}/{len(files)} trajectories, {time.monotonic()-started:.1f}s',flush=True)
        # Only now open supervision. This is the existing 14-class diagnostic
        # taxonomy, which groups several injection root causes together.
        mapping=json.loads((root/'configs/datasets/adcs_v5_2_2/fault_class_map.json').read_text())['mappings']
        joined=[];labels_cache={}
        for record in inferences:
            lp=record['source'].replace('/telemetry.csv','/labels.csv')
            if lp not in labels_cache:labels_cache[lp]=pd.read_csv(io.BytesIO(z.read(lp)))
            label_rows=labels_cache[lp].loc[lambda frame:np.isclose(frame.time_s,record['endpoint_s'],rtol=0,atol=1e-8)]
            if len(label_rows)!=1:raise ValueError('endpoint does not have exactly one time-aligned evaluation label')
            label=label_rows.iloc[0]
            active=bool(int(label['event_active']))
            target=mapping[str(label['diagnostic_class'])] if active else None
            joined.append({**record,'target':target,'diagnostic_class':str(label['diagnostic_class']),
                           'condition_for_evaluator_only':str(label['condition_id'])})
    branches=('data_only','data_plus_reference','legacy','evidence_aware','working_hypothesis','post47_physical')
    report={'claim_boundary':a.cohort+' replay, existing grouped diagnostic labels (not fine injected root-cause identification), not deployment qualification. Data/knowledge use identical admitted windows. Post47 physical has the existing longer causal prefix and is contextual, not an equal-information fusion-only comparison.',
            'trajectory_count':len(files),'window_count':len(joined),'execution_failures':sum(r['status']!='SUCCEEDED' for r in joined),
            'runtime_s':time.monotonic()-started,'branches':{b:metrics(joined,b) for b in branches},
            'policy_status_counts':dict(Counter(r.get('new_status','UNAVAILABLE') for r in joined))}
    for reference in ('data_only','data_plus_reference','legacy'):
        def correct(r,b):
            q=r[b]
            return (q['kind']=='NORMAL' and r['target'] is None) or (q['kind'] in {'CLASS','MODEL_ONLY','PREFERRED_UNCONFIRMED'} and q['ids']==[r['target']])
        for branch in ('evidence_aware','working_hypothesis'):
            c=sum(not correct(r,reference) and correct(r,branch) for r in joined)
            h=sum(correct(r,reference) and not correct(r,branch) for r in joined)
            report[branch+'_vs_'+reference]={'corrected':c,'harmed':h,'net_correct':c-h}
    (a.output/'joined_evaluation.json').write_text(json.dumps(joined,indent=2)+'\n')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
