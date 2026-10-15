import sys,pathlib,json,zipfile,csv,io,hashlib
import argparse
p=argparse.ArgumentParser();p.add_argument('--workspace',type=pathlib.Path,required=True);p.add_argument('--data-dir',type=pathlib.Path,required=True);args=p.parse_args()
raise ValueError('R21_SEEN_EVALUATION_PREDICTION_REGENERATION_BLOCKED: historical R20 runners disabled in R21; read frozen results')
ROOT=args.workspace.resolve();SRC=pathlib.Path(__file__).resolve().parents[2];DATA=args.data_dir.resolve();PROV=ROOT/'work/dxc_provenance';CACHE=ROOT/'work/esa_preprocessed';OUT=ROOT/'results/R20_EXTERNAL_EVALUATION_RECEIPTS_COMPLETION'
if DATA.is_relative_to(SRC) or ROOT.is_relative_to(SRC):raise ValueError('EXTERNAL_DATA_OR_WORKSPACE_INSIDE_SOURCE')
sys.path[:0]=[str(ROOT/'work/tool_dependencies'),str(SRC/'src')]
import numpy as np,pandas as pd
from astrograph.evaluation.external_dataset_contract import *
from astrograph.evaluation.ops_sat_external_validation import write
from astrograph.evaluation.esa_anomaly_adapter import event_category,require_evaluation_prerequisites
from astrograph.evaluation.esa_event_metrics import evaluate_events
from astrograph.evaluation.esa_statistical_predictor import fit_reference,score
gate=json.loads((OUT/'dxc_pipeline_gate_receipt.json').read_text());verify_seal(gate)
require_evaluation_prerequisites(ops_sat_passed=json.loads((ROOT/'results/R20_EXTERNAL_EVALUATION_RECEIPTS/ops_sat_external_evaluation_report.json').read_text())['train_test_group_overlap']==0,
 dxc_passed=gate['competition_all_cases_scored'] and gate['rematch_frozen_config_unchanged'] and gate['multi_fault_set_preserved'])
reference=json.loads((OUT/'esa_metric_official_reference_verification.json').read_text())
if reference['status']!='PASS':raise ValueError('ESA_REFERENCE_METRIC_VERIFICATION_REQUIRED')
config=json.loads((SRC/'configs/evaluation/r20_esa.json').read_text());config_sha=digest(config)
write(OUT/'esa_frozen_plan_receipt.json',seal({'plan':config,'config_sha256':config_sha,'prerequisite_gate_sha256':gate['receipt_sha256'],
 'gate_interpretation':'input/truth separation, provenance/admission and frozen scoring pipeline, not a minimum performance threshold; poor DXC performance retained',
 'predictor_source':file_identity(SRC/'src/astrograph/evaluation/esa_statistical_predictor.py')}))
source=DATA/'ESA-Mission1-lightweight.zip';source_before=file_identity(source)
derived=[{'relative_path':p.name,**file_identity(p)} for p in sorted(CACHE.glob('*.npz'))]
write(OUT/'esa_preprocessing_receipt.json',seal({'source_artifact':source_before,'derived_manifest':derived,'preprocessor_source':file_identity(ROOT/'work/read_esa_channels.py'),
 'transform':'5min median via pandas resample; restricted pickle class whitelist; no truth input','metadata':json.loads((CACHE/'metadata.json').read_text())}))
with zipfile.ZipFile(source) as z:
 labels_name=next(n for n in z.namelist() if n.endswith('/labels.csv'));types_name=next(n for n in z.namelist() if n.endswith('/anomaly_types.csv'))
 labels=list(csv.DictReader(io.StringIO(z.read(labels_name).decode('utf-8-sig'))));types={r['ID']:event_category(r['Category']) for r in csv.DictReader(io.StringIO(z.read(types_name).decode('utf-8-sig')))}
selected=[]
for r in labels:
 if r['Channel'] in config['channels']:
  selected.append({'event_id':r['ID'],'channel':r['Channel'],'start_ns':int(pd.Timestamp(r['StartTime']).value),'end_ns':int(pd.Timestamp(r['EndTime']).value),'category':types[r['ID']]})
metadata=json.loads((CACHE/'metadata.json').read_text());lower=min(r['start_ns'] for r in metadata);upper=max(r['end_ns'] for r in metadata)
envelopes={}
for r in selected:
 envelope=envelopes.setdefault(r['event_id'],[r['start_ns'],r['end_ns']]);envelope[0]=min(envelope[0],r['start_ns']);envelope[1]=max(envelope[1],r['end_ns'])
cuts=[]
for fraction in config['partition_duration_fractions']:
 cut=int(lower+(upper-lower)*fraction)
 while True:
  overlapping=[end for start,end in envelopes.values() if start<cut<=end]
  if not overlapping:break
  cut=max(overlapping)+1
 cuts.append(cut)
if any(b<=a for a,b in zip([lower]+cuts,cuts+[upper+1])):raise ValueError('ESA_EVENT_DISJOINT_BOUNDARIES_UNAVAILABLE')
boundaries=[lower]+cuts+[upper+1];partitions={};group_rows=[]
for aid,(a,b) in sorted(envelopes.items()):
 index=int(np.searchsorted(cuts,a,side='right'))
 if b>=boundaries[index+1]:raise ValueError('ESA_EVENT_GROUP_PARTITION_LEAKAGE')
 group_rows.append({'group_id':aid,'partition':config['partitions'][index],'start_ns':a,'end_ns':b})
manifest={'boundaries_ns':boundaries,'partition_ids':config['partitions'],'event_groups':group_rows,'config_sha256':config_sha,'scope_channels':config['channels']}
write(OUT/'esa_partition_manifest.json',manifest)
grid=None;combined=None;observed_any=None;fitted={};channel_counts={}
for channel in config['channels']:
 with np.load(CACHE/(channel+'.npz')) as payload:
  times=payload['time_ns']+int(300e9);values=payload['value'];observed=payload['observed']
  train=(times>=lower)&(times<cuts[0])&observed
  for r in selected:
   if r['channel']==channel:train&=~((times>=r['start_ns'])&(times<=r['end_ns']))
  fitted[channel]=fit_reference(values[train],relative_floor=config['scale_floor_relative'])
  scores=score({'time_ns':times,'value':values,'observed':observed,'channel':channel},fitted[channel]);alarm=(scores>=config['score_threshold'])&observed
  if grid is None:grid=times.copy();combined=alarm.copy();observed_any=observed.copy()
  else:
   if not np.array_equal(grid,times):raise ValueError('ESA_CHANNEL_GRID_MISMATCH')
   combined|=alarm;observed_any|=observed
  channel_counts[channel]={'MODEL_TRAIN_nominal_bins':int(train.sum()),'observed_bin_count':int(observed.sum())}
# Freeze inference-only binary series before joining evaluation labels.
prediction_file=OUT/'esa_frozen_predictions.npz';np.savez_compressed(prediction_file,time_ns=grid,alarm=combined,observed=observed_any)
prediction_identity=file_identity(prediction_file)
write(OUT/'esa_fit_receipt.json',seal({'parameters':fitted,'training_partition':'MODEL_TRAIN','channel_counts':channel_counts,'config_sha256':config_sha}))
for index,name in enumerate(config['partitions']):
 start,end=boundaries[index],boundaries[index+1]-1;idx=np.where((grid>=start)&(grid<=end))[0]
 if not len(idx):raise ValueError('ESA_PARTITION_HAS_NO_OBSERVATIONS')
 times=grid[idx];flags=combined[idx];available=observed_any[idx]
 # State starts at first decision in the partition. No carry across split.
 alarm_events=[];run_start=None
 for i,flag in enumerate(flags):
  if flag and run_start is None:run_start=int(times[i])
  if not flag and run_start is not None:alarm_events.append({'start_ns':run_start,'end_ns':int(times[i]),'end_closed':False});run_start=None
 if run_start is not None:alarm_events.append({'start_ns':run_start,'end_ns':end,'end_closed':True})
 truths=[r for r in selected if start<=r['start_ns']<=r['end_ns']<=end]
 metrics=evaluate_events(truths,alarm_events,full_range=(start,end),confirmation_duration_s=config['confirmation_duration_s'])
 nominal=available.copy()
 for r in truths:nominal&=~((times>=r['start_ns'])&(times<=r['end_ns']))
 metrics['sample_false_positive_rate']=float((flags&nominal).sum()/nominal.sum()) if nominal.any() else None
 metrics['sample_false_positive_rate_unit']='observed common-grid decision samples, not original telemetry rows'
 metrics['nominal_observed_decision_count']=int(nominal.sum());metrics['false_positive_decision_count']=int((flags&nominal).sum())
 partitions[name]={'time_range_ns':[start,end],'observed_decision_count':int(available.sum()),'unavailable_decision_count':int((~available).sum()),
  'scoped_truth_event_count':len({r['event_id'] for r in truths}),'metrics':metrics,'alarm_event_count':len(alarm_events)}
 write(OUT/('esa_'+name.lower()+'_alarm_events.json'),alarm_events)
semantics={'categories':{'Anomaly':'ANOMALY','Rare Event':'RARE_NOMINAL','Communication Gap':'COMMUNICATION_GAP'},'nominal':'unannotated complement within admitted evaluation time/channel scope',
 'truth_intervals':'closed; grouped by official ID across scoped channels','primary_metric_filter':'Anomaly; official EW treats rare/gap annotations as unselected events, auxiliary challenge counts retained'}
universe=['NOMINAL','ANOMALY','RARE_NOMINAL','COMMUNICATION_GAP'];raw_manifest=json.loads((ROOT/'results/R20_EXTERNAL_EVALUATION_RECEIPTS/esa_payload_manifest.json').read_text())
receipt=seal({'schema_version':'astrograph-external-evaluation-dataset-receipt.v1','dataset_id':'ESA_MISSION1_LIGHTWEIGHT_SIX_CHANNELS_V1','dataset_family':'ESA',
 'source_artifact_sha256':source_before['sha256'],'source_artifact_size_bytes':source_before['size'],'evaluation_task_id':config['evaluation_task_id'],'task_kind':'ANOMALY_EVENT_DETECTION',
 'label_semantics':semantics,'label_semantics_sha256':digest(semantics),'target_universe':universe,'target_universe_sha256':digest(universe),
 'group_policy':{'event_id':'official ID; envelope never crosses split','config_sha256':config_sha,'scope_channels':config['channels']},
 'truth_provenance':{'labels_member':labels_name,'types_member':types_name,'member_manifest_receipt_sha256':raw_manifest['receipt_sha256']},
 'payload_manifest_sha256':digest(raw_manifest['members']),'allowed_claims':['six-channel retrospective real-data event screening','new chronological group-disjoint holdout; not published ESA baseline reproduction'],
 'forbidden_claims':['full 76-channel coverage','formal fault diagnosis','calibrated probability','flight qualification']})
ExternalDatasetReceipt.model_validate(receipt);write(OUT/'esa_dataset_receipt.json',receipt)
snapshot=build_snapshot(DATA,[source],dataset_receipt=receipt,truth_paths=[source],partitions=manifest,groups=list(envelopes));write(OUT/'esa_external_snapshot_receipt.json',snapshot)
verify_snapshot(DATA,snapshot,dataset_receipt=receipt,partitions=manifest,groups=list(envelopes))
if file_identity(source)!=source_before:raise ValueError('ESA_RAW_SOURCE_DRIFT')
report=seal({'schema_version':'astrograph-esa-event-external-evaluation-report.v1',**task_identity(receipt),'status':'EXECUTED','source_receipt_sha256':receipt['receipt_sha256'],
 'prediction_artifact':prediction_identity,'snapshot_receipt_sha256':snapshot['receipt_sha256'],'raw_telemetry_rows_read':sum(r['raw_sample_count'] for r in metadata),
 'supplied_channels':config['channels'],'partitions':partitions,'event_group_overlap':0,'qualification':'NOT_FLIGHT_QUALIFIED',
 'warnings':['simple static reference; not R19 raw detector or ESA published algorithm','5min retrospective median, prediction at bin end; original sample-level performance not claimed',
 'missing input gives no alarm and is counted as unavailable, not certified nominal','split thresholds are engineering defaults frozen before evaluation; no probability fitting',
 'only six numerical channel payloads were supplied; labels outside scope excluded explicitly','scope starts/ends at supplied raw telemetry extrema; anonymized dataset timestamps, no real UTC operational claim'],
 'out_of_scope_label_intervals':len(labels)-len(selected),'config_sha256':config_sha},'report_sha256')
write(OUT/'esa_event_evaluation_report.json',report)
validation=seal({'schema_version':'astrograph-external-validation-receipt.v1',**task_identity(receipt),'dataset_receipt_sha256':receipt['receipt_sha256'],
 'snapshot_receipt_sha256':snapshot['receipt_sha256'],'predictions_sha256':prediction_identity['sha256'],'evaluation_report_sha256':report['report_sha256'],
 'track_id':'ESA_SIX_CHANNEL_MEDIAN_MAD','score_semantics':'UNCALIBRATED_ANOMALY_SCORE','status':'EXECUTED','qualification':'NOT_FLIGHT_QUALIFIED'})
ExternalValidationReceipt.model_validate(validation);write(OUT/'esa_external_validation_receipt.json',validation)
print('ESA external',partitions['EXTERNAL_VALIDATION'],flush=True)
