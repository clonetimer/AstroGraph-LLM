from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,sys,hashlib
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from astrograph.oag.physical_diagnosis import diagnose_physical_complement
from scripts.evaluate_diagnostic_enhancement import effect_target
parser=argparse.ArgumentParser(description='Fixed five-second physical-channel diagnosis latency; labels joined after inference.')
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--workers',type=int,default=4)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1];data=args.dataset;out=args.output
out.mkdir(parents=True,exist_ok=False)
policy=root/'src/astrograph/oag/physical_diagnosis.py'
(out/'plan.json').write_text(json.dumps({'grid_s':5,'start_s':20,'end_s':300,'model_invoked':False,
    'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
    'telemetry_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob('**/telemetry.csv'))]},indent=2)+'\n')
manifest=json.loads((data/'dataset_manifest.json').read_text());missions={s['scenario_id']:s['mission_id'] for s in manifest['operating_scenarios']}
paths=sorted(data.glob('**/telemetry.csv'));ends=list(range(20,301,5))
def run(item):
 index,path=item;relative=path.relative_to(data);parts=relative.parts;mission=parts[1] if parts[0]=='nominal_baseline' else missions[parts[1]]
 s=pd.read_csv(path,low_memory=False);n=pd.read_csv(data/'nominal_baseline'/mission/path.parent.name/'telemetry.csv',low_memory=False)
 result=[{'endpoint_s':end,'diagnosis':diagnose_physical_complement(s[s.time_s<=end+1e-9],n)} for end in ends]
 (out/f'case-{index:03d}.json').write_text(json.dumps({'source':str(relative),'windows':result},indent=2)+'\n')
 return path,result
with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run,enumerate(paths)))
# Evaluation only, after all truth-free inference.
mapping={'adcs_rw_jamming':'ADCS_RW_JAM','adcs_rw_torque_authority_loss':'ADCS_RW_TORQUE_AUTHORITY_LOSS','adcs_rw_speed_sensor_fault':'ADCS_RW_SPEED_SENSOR_BIAS','adcs_rw_motor_failure':'ADCS_RW_TORQUE_AUTHORITY_LOSS','adcs_gyro_bias_step':'ADCS_GYRO_BIAS_DRIFT'}
rows=[];false_alerts=0;wrong_fault_windows=0
for path,windows in results:
 meta=json.loads(path.with_name('sample_metadata.json').read_text());onset=meta['event_start_s']
 for w in windows:
  p=w['diagnosis']['prediction'];active=onset is not None and w['endpoint_s']>=onset
  false_alerts+=int(not active and p['kind']=='CLASS')
  wrong_fault_windows+=int(active and p['kind']=='CLASS' and p['ids']!=[mapping[meta['condition_id']]])
 if onset is None:continue
 target=mapping[meta['condition_id']];active=[w for w in windows if w['endpoint_s']>=onset]
 decisive=[w for w in active if w['diagnosis']['prediction']['kind']=='CLASS']
 correct=[w for w in decisive if w['diagnosis']['prediction']['ids']==[target]]
 rows.append({'source':str(path.relative_to(data)),'target':target,'onset_s':onset,'family':meta['pair_seed'],
  'first_correct_s':correct[0]['endpoint_s'] if correct else None,
  'latency_s':correct[0]['endpoint_s']-onset if correct else None,
  'first_decision_correct':decisive[0]['diagnosis']['prediction']['ids']==[target] if decisive else None})
latencies=[r['latency_s'] for r in rows if r['latency_s'] is not None]
report={'grid_s':5,'fault_traces':len(rows),'correctly_diagnosed_traces':len(latencies),'censored_without_correct':sum(r['latency_s'] is None for r in rows),'latency_quantiles_s':np.percentile(latencies,[50,90,95,100]).tolist() if latencies else None,'first_decision_wrong_traces':sum(r['first_decision_correct'] is False for r in rows),'normal_or_pre_event_false_classifications':false_alerts,'wrong_fault_windows':wrong_fault_windows,'rows':rows,'claim_boundary':'Physical channel only, fixed five-second observation grid. Full model was not evaluated on this dense grid. Latencies include up to five seconds of scheduling quantization; failures are reported rather than removed.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='rows'})
