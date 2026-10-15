"""Full-head retrospective replay, with equal-window evidence/Agent ablations.

Development samples 03/08 and previously used test samples 04/09 are NOT new
independent confirmation. Labels are opened only after inference.jsonl closes.
All planned windows remain in the denominator, including invalid packets.
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

from evaluate_core_fusion_zip import digest, prediction, working_prediction
from astrograph.evaluation.replay_integrity import semantic_inference_digest

BRANCHES = ('data_only', 'data_plus_reference', 'dev2_single', 'sensor_initial',
            'sensor_eager', 'sensor_agent', 'sensor_working')


def head_inference(source, *, provider, registry, episode_id):
    from astrograph.data_driven import ModelInferenceRequest, ObservationScope
    from astrograph.assurance import canonical_sha256
    window = source.iloc[-int(provider.manifest.input_contract['window_length']):]
    times = window.time_s.to_numpy(float)
    channels = list(provider.selected_channels)
    values = window[channels].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    if len(values) != int(provider.manifest.input_contract['window_length']) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('invalid model window length/time')
    if not np.isfinite(values).all():
        bad = [channels[j] for j in np.flatnonzero(~np.isfinite(values).all(axis=0))]
        raise ValueError('non-finite required model channels: '+','.join(bad))
    payload = {'channel_order':channels, 'telemetry_window':values.tolist()}
    data_hash = canonical_sha256(payload)
    source_hash = hashlib.sha256(window.to_csv(index=False).encode()).hexdigest()
    request = ModelInferenceRequest(episode_id=episode_id,model_id=provider.manifest.model_id,
        model_version=provider.manifest.model_version,feature_schema_id=provider.manifest.feature_schema_ids[0],
        input_data_sha256=data_hash,registered_fault_ids=sorted(provider.runtime_fault_ids),payload=payload,
        observation_scope=ObservationScope(scope_id='S-'+source_hash[:20],source_id='admitted-telemetry',
            window_start_s=float(times[0]),window_end_s=float(times[-1]),source_data_sha256=source_hash))
    result = registry.predict(request)
    raw = result.metadata.get('raw_output_metadata',result.metadata)
    trace = raw['head_trace']
    return {
        'prediction_id':result.prediction_id, 'model_id':result.model_id, 'model_version':result.model_version,
        'input_data_sha256':data_hash, 'window_start_s':float(times[0]), 'window_end_s':float(times[-1]),
        'channel_order':channels, 'class_order':list(provider.class_names), 'component_order':list(provider.component_names),
        'detection_probability':raw['detection_probability'], 'detection_threshold':raw['detection_threshold'],
        'detected':raw['detected'], 'predicted_component':raw['predicted_component'],
        'predicted_dataset_class':raw['predicted_dataset_class'],
        'component_probabilities':raw['component_probabilities'],
        'dataset_class_probabilities':raw['dataset_class_probabilities'],
        **{k:trace[k] for k in ('raw_class_probabilities','direct_component_probabilities',
                              'derived_component_probabilities','classification_policy','discarded_class_mass')},
        'score_semantics':'UNCALIBRATED_MODEL_OUTPUT',
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-zip',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cohort',choices=['development','used-test-regression'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1)
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    from astrograph.simulation import SimulationRunCompleted
    from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,build_registry_from_adcs_three_task_provider
    from astrograph.oag.native_bridge import from_native_frames,write_package_graph,project_registered_telemetry
    from astrograph.oag.kernel import EvidenceDiagnosisKernel
    from astrograph.oag.sensor_checks import sensor_check_tools
    from astrograph.oag.agent import AutonomousDiagnosisService,_apply_features
    from astrograph.orchestration.models import WorkflowContext,PermissionScope
    from astrograph.evaluation.three_task_replay import evaluate_three_task_replay
    root=a.project_root.resolve();samples=[3,8] if a.cohort=='development' else [4,9];endpoints=[30.,150.,300.]
    model=root/'models/adcs_v5_2_2/p2_innovation_v2.pt'
    old_package=root/'configs/datasets/adcs_v5_2_2/runtime_window_integrated_evidence.json'
    new_package=root/'configs/datasets/adcs_v5_2_2/runtime_sensor_checks.research.json'
    provider=load_adcs_three_task_torch_package(model,expected_sha256=digest(model),device='cpu')
    registry=build_registry_from_adcs_three_task_provider(provider)
    builders=[ConfiguredRuntimeEvidenceBuilder(f) for f in (old_package,new_package)]
    kernels=[EvidenceDiagnosisKernel(write_package_graph(f,a.output/('graph_'+str(i)))) for i,f in enumerate((old_package,new_package))]
    schema=json.loads((root/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text())['canonical_sha256']
    context=WorkflowContext(workflow_id='three-task-retrospective',run_id='replay',actor='diagnostic-agent',
        permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE])
    started=time.monotonic();inferences=[];nominal_cache={}
    with zipfile.ZipFile(a.dataset_zip) as z:
        dataset_root=next(n for n in z.namelist() if n.endswith('/dataset_manifest.json')).rsplit('/',1)[0]
        manifest=json.loads(z.read(dataset_root+'/dataset_manifest.json'))
        mission_map={s['scenario_id']:s['mission_id'] for s in manifest['operating_scenarios']}
        files=sorted(n for n in z.namelist() if n.endswith('/telemetry.csv') and any('/sample_%02d/'%j in n for j in samples))
        plan={'cohort':a.cohort,'samples':samples,'endpoints':endpoints,'window_length':20,'trajectory_count':len(files),
            'dataset_sha256':digest(a.dataset_zip),'model_sha256':digest(model),
            'packages':{f.name:digest(f) for f in (old_package,new_package)},'source_files':files,
            'source_sha256':{str(f.relative_to(root)):digest(f) for folder in ('src/astrograph/oag','src/astrograph/fusion','src/astrograph/evidence','src/astrograph/evaluation') for f in sorted((root/folder).glob('*.py'))},
            'script_sha256':digest(__file__),'branches':BRANCHES,'memory':'disabled/frozen; no memory gain claimed',
            'scope':'Retrospective used-data regression, no new confirmation claim. All branches same 20 samples and nominal window; no physical longer-history branch.',
            'extra_observables':'Sensor checks use accelerometer/quaternion and applied-torque channels beyond the model-selected 33; equal time window is not an identical-feature pure-fusion comparison.',
            'predeclared_experiment':'same model and thresholds; original vs specific research knowledge; zero vs eager-one vs planned-one check; no learned or tuned fusion weight'}
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        with (a.output/'inference.jsonl').open('w') as log:
            for number,name in enumerate(files):
                parts=Path(name[len(dataset_root)+1:]).parts
                mission=parts[1] if parts[0]=='nominal_baseline' else mission_map[parts[1]]
                nname=dataset_root+'/nominal_baseline/'+mission+'/'+parts[-2]+'/telemetry.csv'
                if nname not in nominal_cache:
                    nominal_cache[nname]=project_registered_telemetry(pd.read_csv(io.BytesIO(z.read(nname)),low_memory=False),schema_sha256=schema)
                nominal=nominal_cache[nname]
                source=project_registered_telemetry(pd.read_csv(io.BytesIO(z.read(name)),low_memory=False),schema_sha256=schema)
                opaque=hashlib.sha256(name.encode()).hexdigest()[:20]
                for endpoint in endpoints:
                    record={'source':name,'endpoint_s':endpoint,'case_id':opaque,'status':'SUCCEEDED','heads':None,'errors':{},'checks_executed':0}
                    for branch in BRANCHES:record[branch]={'kind':'UNAVAILABLE','ids':[]}
                    prefix=source[source.time_s<=endpoint+1e-8].copy();baseline=nominal[nominal.time_s<=endpoint+1e-8].copy()
                    # Model inference is isolated from optional graph preparation.
                    try:
                        heads=head_inference(prefix,provider=provider,registry=registry,episode_id='HEAD-'+opaque+str(endpoint))
                        record['heads']=heads
                        target=provider.class_map[heads['predicted_dataset_class']]
                        record['data_only']=({'kind':'CLASS','ids':[target]} if heads['detected'] else {'kind':'NORMAL','ids':[]})
                        window=prefix.iloc[-20:];paired=baseline[baseline.time_s.between(window.time_s.iloc[0],window.time_s.iloc[-1])]
                        same=window.to_csv(index=False)==paired.to_csv(index=False)
                        record['data_plus_reference']={'kind':'NORMAL','ids':[]} if same else record['data_only']
                        record['reference_exact_match']=same
                    except Exception as exc:
                        record['errors']['model']={'type':type(exc).__name__,'message':str(exc)}
                        record['status']='MODEL_UNAVAILABLE'
                    event=SimulationRunCompleted(event_id='EVENT-'+opaque,run_id='RUN-'+opaque,run_dir='/admitted/'+opaque,
                        nominal_run_dir='/admitted/reference',runtime_id='retrospective-window-replay')
                    for i in (0,1):
                        try:
                            data=from_native_frames(builder=builders[i],event=event,source=prefix,nominal=baseline,
                                provider=provider,registry=registry,review_evidence=True,decision_policy='evidence_aware')
                            if record['heads'] is None:raise AssertionError('model branch isolation mismatch')
                            if data.three_task_result.input_data_sha256 != record['heads']['input_data_sha256']:
                                raise AssertionError('unequal model input across branches')
                            if data.three_task_result.component_probabilities != record['heads']['component_probabilities']:
                                raise AssertionError('model head changed across equal-input branches')
                            result=kernels[i].evaluate(data)
                            if i==0:
                                record['dev2_single']=prediction(result['diagnosis'])
                                record['dev2_status']=result['diagnosis']['status']
                                continue
                            record['sensor_initial']=prediction(result['diagnosis'])
                            tools=sensor_check_tools(data,source=prefix,nominal=baseline)
                            # Eager comparator executes the same numerical tool on
                            # the same input, independent of the Agent decision.
                            eager_bundle=data.bundle
                            for tool in tools:
                                requested=tuple(sorted(tool.feature_definition_ids))
                                eager_bundle,_=_apply_features(eager_bundle,tool,requested,tool.handler(requested))
                            eager=kernels[i].evaluate(data,bundle=eager_bundle)
                            record['sensor_eager']=prediction(eager['diagnosis'])
                            # Construct fresh tools: no shared eager cache/counts.
                            agent=AutonomousDiagnosisService(kernels[i],tools=sensor_check_tools(data,source=prefix,nominal=baseline),max_checks=1)
                            result=agent.run(data,context=context)
                            record.update(sensor_agent=prediction(result['diagnosis']),sensor_working=working_prediction(result['diagnosis']),
                                agent_status=result['diagnosis']['status'],knowledge_verified=result['diagnosis']['knowledge_verified'],
                                checks_executed=result['checks_executed'],stopping_reason=result['stopping_reason'],
                                planning_trace=result['planning_trace'],reason_codes=result['diagnosis']['reason_codes'],
                                candidate_explanations=result['diagnosis']['candidate_explanations'],
                                evidence_bundle_sha256=result['evidence_bundle_sha256'])
                            # Save complete exemplars without reading labels.
                            if number<2 and endpoint==300:
                                (a.output/(opaque+'_agent.json')).write_text(json.dumps(result,indent=2)+'\n')
                        except Exception as exc:
                            record['errors']['old_knowledge' if i==0 else 'sensor_knowledge']={'type':type(exc).__name__,'message':str(exc)}
                            if record['status']=='SUCCEEDED':record['status']='PARTIAL'
                    inferences.append(record);log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                if (number+1)%20==0:print(f'{number+1}/{len(files)}, {time.monotonic()-started:.1f}s',flush=True)
        # LABEL JOIN BOUNDARY: no labels.csv opened before inference log closes.
        inference_hash=digest(a.output/'inference.jsonl')
        semantic_inference_hash=semantic_inference_digest(inferences)
        mapping=json.loads((root/'configs/datasets/adcs_v5_2_2/fault_class_map.json').read_text())['mappings']
        hierarchy=json.loads((root/'configs/datasets/adcs_v5_2_2/diagnosis_hierarchy.json').read_text())['fault_to_component']
        component_map={mapping[k]:v for k,v in hierarchy.items()};label_cache={};joined=[]
        for record in inferences:
            lp=record['source'].replace('/telemetry.csv','/labels.csv')
            if lp not in label_cache:label_cache[lp]=pd.read_csv(io.BytesIO(z.read(lp)))
            labels=label_cache[lp];row=labels.loc[np.isclose(labels.time_s,record['endpoint_s'],rtol=0,atol=1e-8)]
            if len(row)!=1:raise ValueError('ambiguous or missing endpoint label')
            label=row.iloc[0];active=bool(int(label['event_active']))
            target=mapping[str(label['diagnostic_class'])] if active else None
            joined.append({**record,'target':target,'target_component':component_map[target] if active else None,
                           'diagnostic_class':str(label['diagnostic_class']),'label_join_inference_sha256':inference_hash})
    report=evaluate_three_task_replay(joined,fault_to_component=component_map,dataset_class_to_fault=mapping,branches=BRANCHES)
    report.update(runtime_s=time.monotonic()-started,cohort=a.cohort,inference_sha256=inference_hash,semantic_inference_sha256=semantic_inference_hash,
        execution_status_counts=dict(Counter(r['status'] for r in joined)),
        agent_status_counts=dict(Counter(r.get('agent_status','UNAVAILABLE') for r in joined)),
        tool_calls=sum(r['checks_executed'] for r in joined),eager_tool_calls=sum(r['sensor_eager']['kind']!='UNAVAILABLE' for r in joined),
        model_weights_changed=False,fusion_weights_changed=False)
    # Whole-record per-window correction/harm accounting includes normal outputs.
    def correct(r,b):
        v=r[b]
        return (v['kind']=='NORMAL' and r['target'] is None) or (v['kind'] in {'CLASS','MODEL_ONLY','PREFERRED_UNCONFIRMED'} and v['ids']==[r['target']])
    report['net_changes']={}
    for ref in ('data_plus_reference','dev2_single','sensor_initial','sensor_eager'):
        corrected=sum(not correct(r,ref) and correct(r,'sensor_agent') for r in joined)
        harmed=sum(correct(r,ref) and not correct(r,'sensor_agent') for r in joined)
        report['net_changes'][ref]={'corrected':corrected,'harmed':harmed,'net':corrected-harmed}
    (a.output/'joined_evaluation.json').write_text(json.dumps(joined,indent=2,allow_nan=False)+'\n')
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('branches','model_heads')},indent=2),flush=True)

if __name__=='__main__':main()
