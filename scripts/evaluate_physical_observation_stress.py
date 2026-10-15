from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from astrograph.oag.physical_diagnosis import diagnose_physical_complement
from scripts.evaluate_diagnostic_enhancement import effect_target
parser=argparse.ArgumentParser(description='Bounded applied-torque noise sensitivity of the physical channel, without model imputation.')
parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();data=args.dataset
if args.output.exists():raise ValueError('output already exists')
args.output.parent.mkdir(parents=True,exist_ok=True)
manifest=json.loads((data/'dataset_manifest.json').read_text());missions={s['scenario_id']:s['mission_id'] for s in manifest['operating_scenarios']}
completed=[];rng=np.random.default_rng(4727)
for path in sorted(data.glob('**/telemetry.csv')):
 parts=path.relative_to(data).parts;mission=parts[1] if parts[0]=='nominal_baseline' else missions[parts[1]]
 s=pd.read_csv(path,low_memory=False);n=pd.read_csv(data/'nominal_baseline'/mission/path.parent.name/'telemetry.csv',low_memory=False)
 sn,nn=s.copy(),n.copy()
 for j in range(4):
  field=f'adcs.control.applied_torque_nm_{j}'
  sn[field]+=rng.uniform(-1e-6,1e-6,len(sn));nn[field]+=rng.uniform(-1e-6,1e-6,len(nn))
 for end in [40,120,200,300]:
  completed.append((str(path.relative_to(data)),end,{name:diagnose_physical_complement(src[src.time_s<=end],ref)['prediction'] for name,src,ref in [('clean',s,n),('bounded_noise',sn,nn)]}))
# Truth is only loaded after every prediction.
mapping={'adcs_rw_jamming':'ADCS_RW_JAM','adcs_rw_torque_authority_loss':'ADCS_RW_TORQUE_AUTHORITY_LOSS','adcs_rw_speed_sensor_fault':'ADCS_RW_SPEED_SENSOR_BIAS','adcs_rw_motor_failure':'ADCS_RW_TORQUE_AUTHORITY_LOSS','adcs_gyro_bias_step':'ADCS_GYRO_BIAS_DRIFT'}
counts={name:Counter() for name in ['clean','bounded_noise']}
for source,end,predictions in completed:
 meta=json.loads((data/source).with_name('sample_metadata.json').read_text());active=meta['event_start_s'] is not None and end>=meta['event_start_s'];target=mapping[meta['condition_id']] if active else None
 for name,p in predictions.items():
  counts[name].update({'windows':1,'fault_windows':int(active),'fault_correct':int(active and p=={'kind':'CLASS','ids':[target]}),'wrong_classifications':int(p['kind']=='CLASS' and p['ids']!=[target]),'normal_alert':int(not active and p['kind']=='CLASS'),'classifications':int(p['kind']=='CLASS')})
report={'assumed_error_bound_nm':1e-6,'distribution':'independent uniform within the stated bound, source and reference applied torque only','model_invoked':False,'metrics':counts,'claim_boundary':'Physical channel sensitivity probe; not evidence that the frozen neural model tolerates noise, nor flight sensor calibration.'}
args.output.write_text(json.dumps(report,indent=2)+'\n');print(report)
