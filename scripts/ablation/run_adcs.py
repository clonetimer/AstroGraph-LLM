"""Frozen retrospective ADCS ablation: labels read only after inference closes."""
from __future__ import annotations
import argparse,sys,io,json,time,zipfile,hashlib,platform
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from common import *
from evaluate_three_task_agent_zip import head_inference
from evaluate_core_fusion_zip import prediction

ARMS=('data_only','data_reference','no_checks','full_adaptive','full_eager',*SCORE_ARMS,'no_executable_knowledge','explicit_counter_rules_off')


def run(args):
    import numpy as np,pandas as pd,torch
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder,EpisodeEvidenceBundle
    from astrograph.simulation import SimulationRunCompleted
    from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,build_registry_from_adcs_three_task_provider
    from astrograph.oag.native_bridge import from_native_frames,write_package_graph,project_registered_telemetry
    from astrograph.oag.kernel import EvidenceDiagnosisKernel
    from astrograph.oag.sensor_checks import sensor_check_tools
    from astrograph.oag.agent import AutonomousDiagnosisService,_apply_features
    from astrograph.orchestration.models import WorkflowContext,PermissionScope
    from astrograph.evaluation.three_task_replay import evaluate_three_task_replay
    torch.set_num_threads(1)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    model=ROOT/'models/adcs_v5_2_2/p2_innovation_v2.pt'
    package=ROOT/'configs/datasets/adcs_v5_2_2/runtime_sensor_checks.research.json'
    provider=load_adcs_three_task_torch_package(model,expected_sha256=sha_file(model),device='cpu')
    registry=build_registry_from_adcs_three_task_provider(provider)
    builder=ConfiguredRuntimeEvidenceBuilder(package)
    kernel=EvidenceDiagnosisKernel(write_package_graph(package,out/'graph'))
    schema=read(ROOT/'configs/datasets/adcs_v5_2_2/telemetry_projection.json')['canonical_sha256']
    ctx=WorkflowContext(workflow_id='ablation-offline',run_id='frozen',actor='diagnostic-agent',permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE])
    samples=[3,8] if args.cohort=='development' else [4,9]
    endpoints=[30.,150.,300.];start=time.monotonic();records=[];nominal_cache={}
    with zipfile.ZipFile(args.dataset_zip) as z:
        root=next(n for n in z.namelist() if n.endswith('/dataset_manifest.json')).rsplit('/',1)[0]
        missions={s['scenario_id']:s['mission_id'] for s in json.loads(z.read(root+'/dataset_manifest.json'))['operating_scenarios']}
        names=sorted(n for n in z.namelist() if n.endswith('/telemetry.csv') and any('/sample_%02d/'%s in n for s in samples))
        if args.smoke: names=names[:1]
        plan={'schema_version':'astrograph-ablation-adcs.v1','scope':'RETROSPECTIVE_PREVIOUSLY_USED_SIMULATION_SPLIT_NOT_NEW_HOLDOUT',
          'cohort':args.cohort,'samples':samples,'endpoints_s':endpoints,'window_length':20,'trajectories':len(names),'planned_windows':len(names)*3,
          'arms':ARMS,'source_files':names,'dataset_sha256':sha_file(args.dataset_zip),'model_sha256':sha_file(model),'package_sha256':sha_file(package),
          'scripts':{p.name:sha_file(p) for p in Path(__file__).parent.glob('*.py')},
          'same_weight_inference_reused_across_arms':True,'model_retrained':False,'llm_requests':0,'memory':'DISABLED',
          'fixed_evidence_arms':list(SCORE_ARMS)+['no_executable_knowledge','explicit_counter_rules_off'],
          'timing':'shared inference/facts; do not compare evaluator wall time as production latency',
          'feature_scope':'same 20-point time window; sensor checks add channels beyond model 33; reference already used in evidence builder',
          'label_read_policy':'OPEN_LABELS_ONLY_AFTER_ALL_INFERENCE_CLOSED','smoke':args.smoke,'python':platform.python_version(),'torch':torch.__version__}
        write(out/'plan.json',plan);event(out,'PLAN_FROZEN',plan_sha256=sha_file(out/'plan.json'))
        with (out/'inference.jsonl').open('x',encoding='utf-8') as log:
            for number,name in enumerate(names):
                parts=Path(name[len(root)+1:]).parts
                mission=parts[1] if parts[0]=='nominal_baseline' else missions[parts[1]]
                nname=root+'/nominal_baseline/'+mission+'/'+parts[-2]+'/telemetry.csv'
                if nname not in nominal_cache: nominal_cache[nname]=project_registered_telemetry(pd.read_csv(io.BytesIO(z.read(nname)),low_memory=False),schema_sha256=schema)
                nominal=nominal_cache[nname]
                source=project_registered_telemetry(pd.read_csv(io.BytesIO(z.read(name)),low_memory=False),schema_sha256=schema)
                opaque=hashlib.sha256(name.encode()).hexdigest()[:20]
                for endpoint in endpoints:
                    r={'source':name,'case_id':opaque,'endpoint_s':endpoint,'heads':None,'errors':{},'details':{},'checks_executed':0,'eager_calls':0}
                    for arm in ARMS:r[arm]={'kind':'UNAVAILABLE','ids':[]}
                    prefix=source[source.time_s<=endpoint+1e-8].copy();baseline=nominal[nominal.time_s<=endpoint+1e-8].copy()
                    try:
                        heads=head_inference(prefix,provider=provider,registry=registry,episode_id='HEAD-'+opaque+str(endpoint));r['heads']=heads
                        target=provider.class_map[heads['predicted_dataset_class']]
                        r['data_only']={'kind':'CLASS','ids':[target]} if heads['detected'] else {'kind':'NORMAL','ids':[]}
                        window=prefix.iloc[-20:];paired=baseline[baseline.time_s.between(window.time_s.iloc[0],window.time_s.iloc[-1])]
                        same=window.to_csv(index=False)==paired.to_csv(index=False)
                        r['data_reference']={'kind':'NORMAL','ids':[]} if same else dict(r['data_only']);r['reference_exact_match']=same
                    except Exception as exc:r['errors']['model']={'type':type(exc).__name__,'message':str(exc)}
                    try:
                        ev=SimulationRunCompleted(event_id='EVENT-'+opaque,run_id='RUN-'+opaque,run_dir='/admitted/'+opaque,nominal_run_dir='/admitted/reference',runtime_id='ablation-window')
                        data=from_native_frames(builder=builder,event=ev,source=prefix,nominal=baseline,provider=provider,registry=registry,review_evidence=True,decision_policy='evidence_aware')
                        assert r['heads'] is not None
                        assert data.three_task_result.input_data_sha256==r['heads']['input_data_sha256']
                        root_hash=sha_obj(data.model_dump(mode='json'));initial=kernel.evaluate(data)
                        r['no_checks']=prediction(initial['diagnosis']);r['details']['no_checks']=pack_result(initial)
                        agent=AutonomousDiagnosisService(kernel,tools=sensor_check_tools(data,source=prefix,nominal=baseline),max_checks=1)
                        full=agent.run(data,context=ctx);r['checks_executed']=full['checks_executed'];r['trace']=full['planning_trace']
                        r['full_adaptive']=prediction(full['diagnosis']);r['details']['full_adaptive']=pack_result(full)
                        effective=EpisodeEvidenceBundle.model_validate(full['effective_evidence_bundle'])
                        eager_bundle=data.bundle
                        for tool in sensor_check_tools(data,source=prefix,nominal=baseline):
                            req=tuple(sorted(tool.feature_definition_ids));r['eager_calls']+=1
                            eager_bundle,_=_apply_features(eager_bundle,tool,req,tool.handler(req))
                        eager=kernel.evaluate(data,bundle=eager_bundle);r['full_eager']=prediction(eager['diagnosis']);r['details']['full_eager']=pack_result(eager)
                        for arm in (*SCORE_ARMS,'no_executable_knowledge','explicit_counter_rules_off'):
                            try:
                                res=intervene(data,effective,kernel,arm=arm)
                                r[arm]=prediction(res['diagnosis']);r['details'][arm]=pack_result(res)
                            except Exception as exc:r['errors'][arm]={'type':type(exc).__name__,'message':str(exc)}
                        assert root_hash==sha_obj(data.model_dump(mode='json')),'INPUT_MUTATED'
                        r['original_input_sha256']=root_hash;r['effective_evidence_sha256']=sha_obj(effective.model_dump(mode='json'))
                        if number==0 and endpoint==150:
                            write(out/'exemplar-full.json',full)
                    except Exception as exc:r['errors']['preparation_or_core']={'type':type(exc).__name__,'message':str(exc)}
                    records.append(r);log.write(canonical(r)+'\n');log.flush()
                if (number+1)%10==0:print(json.dumps({'cohort':args.cohort,'trajectories':number+1,'total':len(names),'seconds':round(time.monotonic()-start,1)}),flush=True)
        ih=sha_file(out/'inference.jsonl');event(out,'INFERENCE_CLOSED_BEFORE_LABEL_JOIN',sha256=ih,records=len(records))
        mapping=read(ROOT/'configs/datasets/adcs_v5_2_2/fault_class_map.json')['mappings']
        hierarchy=read(ROOT/'configs/datasets/adcs_v5_2_2/diagnosis_hierarchy.json')['fault_to_component'];comp={mapping[k]:v for k,v in hierarchy.items()}
        labels={};joined=[]
        for r in records:
            lp=r['source'].replace('/telemetry.csv','/labels.csv')
            if lp not in labels:labels[lp]=pd.read_csv(io.BytesIO(z.read(lp)))
            lr=labels[lp].loc[np.isclose(labels[lp].time_s,r['endpoint_s'],rtol=0,atol=1e-8)]
            if len(lr)!=1:raise ValueError('AMBIGUOUS_LABEL')
            l=lr.iloc[0];target=mapping[str(l['diagnostic_class'])] if bool(int(l['event_active'])) else None
            joined.append({**r,'target':target,'target_component':comp[target] if target else None,'label_join_inference_sha256':ih})
    report=evaluate_three_task_replay(joined,fault_to_component=comp,dataset_class_to_fault=mapping,branches=ARMS)
    report.update(cohort=args.cohort,scope=plan['scope'],inference_sha256=ih,checks_executed=sum(r['checks_executed'] for r in records),eager_calls=sum(r['eager_calls'] for r in records),elapsed_s=time.monotonic()-start,
        errors=dict(Counter((k+':'+v['type']+':'+v['message']) for r in records for k,v in r['errors'].items())))
    def correct(r,b):
        v=r[b];return v['kind']=='NORMAL' and r['target'] is None or v['kind'] in {'CLASS','MODEL_ONLY'} and v['ids']==[r['target']]
    report['paired_vs_full']={}
    for arm in ARMS:
        both=[r for r in records if arm in r['details'] and 'full_adaptive' in r['details']]
        report['paired_vs_full'][arm]={'full_correct_removed_by_ablation':sum(correct(r,'full_adaptive') and not correct(r,arm) for r in joined),
            'ablation_correct_full_wrong':sum(correct(r,arm) and not correct(r,'full_adaptive') for r in joined),
            'prediction_changes':sum(r[arm]!=r['full_adaptive'] for r in joined),
            'status_changes':sum(r['details'][arm]['status']!=r['details']['full_adaptive']['status'] for r in both),
            'ranking_changes':sum(r['details'][arm]['ranking']!=r['details']['full_adaptive']['ranking'] for r in both)}
    report['fixed_evidence_branch_note']='Score removal is NOT complete module removal. Data nominations/evidence adjudication remain; no-KG removes executable evidence only on an evaluator copy.'
    write(out/'joined.json',joined);write(out/'report.json',report)
    event(out,'SCORED',report_sha256=sha_file(out/'report.json'),labels_sent_to_agent=False)
    print(json.dumps({'cohort':args.cohort,'windows':len(records),'errors':report['errors'],'complete':True}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',choices=['development','used-test-regression'],required=True);p.add_argument('--smoke',action='store_true');run(p.parse_args())
