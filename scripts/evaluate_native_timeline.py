"""Fixed-grid native ADCS evaluation. Labels are loaded only after all inference."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
import torch
from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
from astrograph.simulation import SimulationRunCompleted
from astrograph.oag.native_bridge import from_native_frames,write_package_graph
from astrograph.oag.runtime import build_csv_oag_runtime,OAG_PIPELINE_ID
from astrograph.oag.timeline import build_evidence_timeline
from astrograph.runtime import RuntimeRequest
from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,build_registry_from_adcs_three_task_provider


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence-package',type=Path,default=None,
                   help='Defaults to the paired short-window RW mechanism package; use runtime_evidence.json to reproduce post42.')
    p.add_argument('--endpoints',type=int,nargs='+',default=list(range(20,301,20)))
    p.add_argument('--cohort-role',choices=['DEVELOPMENT','HELD_OUT'],default='DEVELOPMENT')
    p.add_argument('--disable-supported-tie-resolution',action='store_true')
    p.add_argument('--classification-policy',choices=['hard_component','unmasked','soft_component'],default='hard_component')
    p.add_argument('--mechanism-confirmation',action='store_true')
    p.add_argument('--degraded-mechanism-evidence',action='store_true',help='On missing observations, emit separate candidate-only evidence; model remains unavailable.')
    a=p.parse_args()
    if a.endpoints!=sorted(set(a.endpoints)) or min(a.endpoints)<20:
        p.error('endpoints must be distinct, increasing, and >=20 seconds')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted(a.dataset.glob('**/telemetry.csv'))
    package=a.evidence_package or root/'configs/datasets/adcs_v5_2_2/runtime_window_integrated_evidence.json'
    manifest=json.loads((a.dataset/'dataset_manifest.json').read_text())
    scenario_mission={s['scenario_id']:s['mission_id'] for s in manifest['operating_scenarios']}
    plan={'endpoints_s':a.endpoints,'telemetry_sha256':[sha(x) for x in paths], 'model_sha256':sha(a.model),
          'evidence_package_sha256':sha(package), 'llm_invoked':False,
          'labels_available_to_inference':False,'model_detection_threshold_changed':False,
          'cohort_role':a.cohort_role,'tie_policy_disabled':a.disable_supported_tie_resolution,
          'classification_policy':a.classification_policy,
          'mechanism_confirmation':a.mechanism_confirmation,
          'degraded_mechanism_evidence':a.degraded_mechanism_evidence,
          'source':'Evidence package and model frozen before this inference run; cohort role declared explicitly.'}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    provider=load_adcs_three_task_torch_package(a.model,expected_sha256=sha(a.model),device='cpu')
    if a.classification_policy != 'hard_component':
        from astrograph.integrations.adcs_three_task_torch import save_adcs_three_task_torch_package
        provider.diagnosis_selection={**provider.diagnosis_selection,'classification_policy':a.classification_policy}
        provider.manifest=provider.manifest.model_copy(update={'model_version':provider.manifest.model_version+'.'+a.classification_policy})
        variant_path=a.output/'policy-variant.pt'
        save_adcs_three_task_torch_package(provider,variant_path)
        provider=load_adcs_three_task_torch_package(variant_path,expected_sha256=sha(variant_path),device='cpu')
        plan['effective_model_sha256']=sha(variant_path)
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    registry=build_registry_from_adcs_three_task_provider(provider)
    builder=ConfiguredRuntimeEvidenceBuilder(package)
    graph=write_package_graph(package,a.output/'graph');runtime=build_csv_oag_runtime(graph)
    records=[]
    for i,path in enumerate(paths):
        relative=path.relative_to(a.dataset)
        # Pair mapping comes from folder layout only; opaque IDs enter inference.
        mission=relative.parts[1] if relative.parts[0]=='nominal_baseline' else scenario_mission[relative.parts[1]]
        nominal=a.dataset/'nominal_baseline'/mission/path.parent.name/'telemetry.csv'
        frame=pd.read_csv(path,low_memory=False);nom=pd.read_csv(nominal,low_memory=False);windows=[]
        case=a.output/f'case-{i:03d}';case.mkdir()
        for end in plan['endpoints_s']:
            event=SimulationRunCompleted(event_id=f'EV-{i}-{end}',run_id=f'RUN-{i}',run_dir='opaque-runtime-source',nominal_run_dir='opaque-runtime-reference',runtime_id='independent-timeline.v1')
            try:
                options={'resolve_supported_ties':False,'resolve_unsupported_top':False} if a.disable_supported_tie_resolution else {}
                options['mechanism_confirmation_enabled']=a.mechanism_confirmation
                data=from_native_frames(builder=builder,event=event,source=frame[frame.time_s<=end+1e-9],nominal=nom,provider=provider,registry=registry,review_evidence=True,**options)
                result=runtime.execute(RuntimeRequest(request_id=f'W-{i}-{end}',pipeline_id=OAG_PIPELINE_ID,payload=data.model_dump(mode='json'))).model_dump(mode='json')
                projection=data.three_task_result.evidence_projection
                fused=result['output'].get('fusion',{}).get('birrf_result',{})
                windows.append({'endpoint_s':end,'status':result['status'],'summary':result['output'].get('summary'),'error':result.get('error'),
                    'paired_reference_identical':projection.source_window_sha256==projection.nominal_window_sha256,
                    'raw_fusion_top_fault_id':fused.get('top_fault_id')})
                if end==300:(case/'input.json').write_text(data.model_dump_json(indent=2)+'\n')
            except Exception as e:
                failed={'endpoint_s':end,'status':'FAILED','error':str(e)}
                if a.degraded_mechanism_evidence:
                    from astrograph.oag.degraded import mechanism_only_fallback
                    schema=json.loads((root/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text())['canonical_sha256']
                    try:
                        failed['degraded_evidence']=mechanism_only_fallback(builder=builder,event=event,
                            source=frame[frame.time_s<=end+1e-9],nominal=nom,schema_sha256=schema,
                            window_length=int(provider.manifest.input_contract['window_length']))
                    except Exception as fallback_error:
                        failed['degraded_evidence_error']=str(fallback_error)
                windows.append(failed)
        timeline=build_evidence_timeline(windows)
        (case/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n');(case/'windows.json').write_text(json.dumps(windows,indent=2)+'\n')
        records.append({'case_id':case.name,'source_path':str(relative),'timeline':timeline})
        print(f'{i+1}/{len(paths)} {case.name}',flush=True)
    # Evaluation identity is joined after inference. No target enters a prompt.
    mapping={'adcs_rw_jamming':'ADCS_RW_JAM','adcs_rw_torque_authority_loss':'ADCS_RW_TORQUE_AUTHORITY_LOSS','adcs_rw_speed_sensor_fault':'ADCS_RW_SPEED_SENSOR_BIAS',
             'adcs_rw_motor_failure':'ADCS_RW_MOTOR_FAILURE','adcs_gyro_bias_step':'ADCS_GYRO_BIAS_DRIFT'}
    for r in records:
        parts=Path(r['source_path']).parts;target=None if parts[0]=='nominal_baseline' else mapping[parts[2]]
        r['evaluation_target']=target
        seen={x['fault_id'] for x in r['timeline']['historical_candidates']}
        r['target_ever_supported']=target in seen if target else not seen and r['timeline']['unavailable_windows']==0
        final_window=json.loads((a.output/r['case_id']/'windows.json').read_text())[-1]
        r['terminal_exact_match']=(r['timeline']['current']['state']!='DATA_UNAVAILABLE' and
            (r['timeline']['current']['candidate']==target if target else final_window.get('paired_reference_identical',False)))
    summary={'n_traces':len(records),'n_windows':sum(r['timeline']['scheduled_windows'] for r in records),
        'unavailable_windows':sum(r['timeline']['unavailable_windows'] for r in records),
        'terminal_exact_matches':sum(r['terminal_exact_match'] for r in records),
        'target_ever_supported_or_nominal_quiet':sum(r['target_ever_supported'] for r in records),
        'nominal_traces':sum(r['evaluation_target'] is None for r in records),
        'nominal_with_supported_alarm':sum(r['evaluation_target'] is None and bool(r['timeline']['historical_candidates']) for r in records),
        'by_target':{str(t):{'n':sum(r['evaluation_target']==t for r in records),'terminal_exact':sum(r['evaluation_target']==t and r['terminal_exact_match'] for r in records),'ever_supported':sum(r['evaluation_target']==t and r['target_ever_supported'] for r in records)} for t in [None,*mapping.values()]},
        'claim_boundary':'Declared cohort role; fixed model and evidence package during inference. Window success, rejection and abstention must be compared separately. No LLM invoked or performance gate promotion.'}
    (a.output/'evaluation.json').write_text(json.dumps({'summary':summary,'records':records},indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
