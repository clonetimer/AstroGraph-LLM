"""Actual MissionAgent normal-mode replay, not just the underlying kernel.

Prepared model outputs are generated from telemetry by the registered provider
in this process, then the real public MissionAgent facade executes diagnosis.
Labels are NOT read here. Every planned window has an output or failure record.
"""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from astrograph.agents.mission.models import AgentSettings,AgentRequest
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.storage import file_sha,sha
from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,build_registry_from_adcs_three_task_provider
from astrograph.oag.native_bridge import from_native_frames,write_package_graph,project_registered_telemetry
from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
from astrograph.simulation import SimulationRunCompleted
from evaluate_core_fusion_zip import prediction
from evaluate_three_task_agent_zip import head_inference

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-zip',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--name',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];records=json.loads((a.data/'test_rows.json').read_text())
    modelhash=file_sha(a.model);provider=load_adcs_three_task_torch_package(a.model,expected_sha256=modelhash)
    registry=build_registry_from_adcs_three_task_provider(provider)
    packet=root/'configs/datasets/adcs_v5_2_2/runtime_sensor_checks.research.json';builder=ConfiguredRuntimeEvidenceBuilder(packet)
    graph=write_package_graph(packet,a.output/'graph');schema=json.loads((root/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text())['canonical_sha256']
    admitted=a.output/'admitted';admitted.mkdir();source_cache={};settings=AgentSettings(project_root=str(root),read_roots=[str(root),str(a.output)],output_root=str(a.output/'tasks'))
    agent=MissionAgent(settings);start=time.perf_counter();nchecks=0;statuses={};mismatches=0
    prior=root/'governance/implementation/post48_dev3/used_test_final/inference.jsonl'
    old={}
    if a.name=='production' and prior.exists():
        for line in prior.read_text().splitlines():
            row=json.loads(line);old[(row['source'],float(row['endpoint_s']))]=row['sensor_agent']
    with zipfile.ZipFile(a.dataset_zip) as z, (a.output/'inference.jsonl').open('w') as stream:
        for i,row in enumerate(records):
            result={'id':row['id'],'source':row['source'],'endpoint_s':row['endpoint_s'],'sample_index':row['sample_index'],'paired_group':row['paired_group'],'valid':row['valid'],
                    'normal':{'kind':'UNAVAILABLE','ids':[]},'data_only':{'kind':'UNAVAILABLE','ids':[]},'status':'UNAVAILABLE','checks_executed':0}
            if row['valid']:
                begin=time.perf_counter()
                try:
                    frames=[];paths=[]
                    for role in ('source','nominal'):
                        key=(row[role],row['endpoint_s'])
                        if key not in source_cache:
                            df=project_registered_telemetry(pd.read_csv(io.BytesIO(z.read(row[role])),low_memory=False),schema_sha256=schema)
                            df=df[df.time_s<=row['endpoint_s']+1e-8].copy()
                            path=admitted/(hashlib.sha256(str(key).encode()).hexdigest()[:24]+'.csv');df.to_csv(path,index=False)
                            # Bind evidence to the exact CSV bytes the Mission tool will reload.
                            # Pandas default float parsing can differ by one ULP after serialization.
                            # Do not relax hash checks: construct the packet from that same reload.
                            source_cache[key]=(pd.read_csv(path,low_memory=False),path)
                        df,path=source_cache[key];frames.append(df);paths.append(path)
                    event=SimulationRunCompleted(event_id='E-'+row['id'],run_id='R-'+row['id'],run_dir='/admitted/'+row['id'],nominal_run_dir='/admitted/reference',runtime_id='normal-mode-benchmark')
                    data=from_native_frames(builder=builder,event=event,source=frames[0],nominal=frames[1],provider=provider,registry=registry,review_evidence=True,decision_policy='evidence_aware')
                    ip=admitted/(row['id']+'.json');ip.write_text(data.model_dump_json())
                    req=AgentRequest(task_id='normal-'+row['id'],work_mode='normal',artifacts={'input':str(ip),'graph_dir':str(graph),'source':str(paths[0]),'nominal':str(paths[1])})
                    mission=agent.run(req)
                    if mission.status not in {'SUCCEEDED','PARTIAL'}:raise ValueError(mission.output)
                    output=mission.output['diagnosis_result'];result['normal']=prediction(output['diagnosis'])
                    heads=head_inference(frames[0],provider=provider,registry=registry,episode_id='H-'+row['id'])
                    target=provider.class_map[heads['predicted_dataset_class']]
                    result['data_only']={'kind':'CLASS','ids':[target]} if heads['detected'] else {'kind':'NORMAL','ids':[]}
                    result.update(status=mission.status,checks_executed=output['checks_executed'],diagnosis_status=output['diagnosis']['status'],
                        stopped=output['stopping_reason'],knowledge_verified=output['diagnosis']['knowledge_verified'],
                        result_sha256=sha(mission.model_dump(mode='json')),mode_policy_sha256=mission.mode_receipt['policy_sha256'],elapsed_s=time.perf_counter()-begin,
                        detector_probability=heads['detection_probability'],detection_threshold=heads['detection_threshold'],
                        direct_component_probabilities=heads['direct_component_probabilities'],raw_class_probabilities=heads['raw_class_probabilities'])
                    nchecks+=output['checks_executed']
                    if old and old.get((row['source'],float(row['endpoint_s'])))!=result['normal']:mismatches+=1
                except Exception as e:
                    result.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:1000])
            statuses[result['status']]=statuses.get(result['status'],0)+1
            stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
            if (i+1)%50==0:print(a.name,i+1,len(records),round(time.perf_counter()-start,1),statuses,flush=True)
    receipt={'name':a.name,'model_sha256':modelhash,'protocol_sha256':file_sha(a.data/'protocol.json'),'work_mode':'normal','mission_facade_executed':True,
             'model_inference':'registered provider before bound-input mission call; not a fake model tool rerun',
             'mode_settings':settings.model_dump(mode='json'),'status_counts':statuses,'windows':len(records),'tool_calls':nchecks,
             'total_wall_s':time.perf_counter()-start,'labels_read':False,'memory':'disabled','knowledge_sha256':file_sha(packet),
             'production_vs_dev3_prediction_mismatches':mismatches if old else None,
             'inference_sha256':file_sha(a.output/'inference.jsonl'),'production_modified':False,
             'information':'paired nominal and additional sensor checks beyond 33 model channels; not equal-input pure model comparison'}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
if __name__=='__main__':main()
