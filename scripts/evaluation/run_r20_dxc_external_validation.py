import sys,pathlib,json,zipfile,hashlib,time,collections,ctypes
import argparse
p=argparse.ArgumentParser();p.add_argument('--workspace',type=pathlib.Path,required=True);p.add_argument('--data-dir',type=pathlib.Path,required=True);args=p.parse_args()
raise ValueError('R21_SEEN_EVALUATION_PREDICTION_REGENERATION_BLOCKED: historical R20 runners disabled in R21; read frozen results')
ROOT=args.workspace.resolve();SRC=pathlib.Path(__file__).resolve().parents[2];DATA=args.data_dir.resolve();PROV=ROOT/'work/dxc_provenance';CACHE=ROOT/'work/esa_preprocessed';OUT=ROOT/'results/R20_EXTERNAL_EVALUATION_RECEIPTS_COMPLETION'
if DATA.is_relative_to(SRC) or ROOT.is_relative_to(SRC):raise ValueError('EXTERNAL_DATA_OR_WORKSPACE_INSIDE_SOURCE')
sys.path[:0]=[str(ROOT/'work/tool_dependencies'),str(SRC/'src')]
sys.path[:0]=[str(ROOT/'work/tool_dependencies'),str(SRC/'src')]
from astrograph.evaluation.external_dataset_contract import *
from astrograph.evaluation.ops_sat_external_validation import write
from astrograph.evaluation.nasa_adapt_adapter import *
from astrograph.evaluation.dxc_workbook_truth import *
from astrograph.evaluation.dxc_statistical_predictor import predict
from astrograph.evaluation.dxc_timeline_metrics import evaluate_case,summarize
from astrograph.evaluation.process_memory import observe_process_memory
OUT.mkdir(exist_ok=True);config=json.loads((SRC/'configs/evaluation/r20_dxc.json').read_text());config_identity=digest(config)
write(OUT/'dxc_frozen_plan_receipt.json',seal({'plan':config,'config_sha256':config_identity,'predictor_source':file_identity(SRC/'src/astrograph/evaluation/dxc_statistical_predictor.py'),'training':'none; sample annotations do not train/tune predictor'}))
catalogs={}
for version,path,names in [('09',DATA/'DXC09_ADAPTsystems_20081209.zip',{'full':'AdaptDXC/AdaptDXC.xml','lite':'AdaptDXC/AdaptLiteDXC.xml'}),('10',PROV/'DXC10_ADAPTsystems.zip',{'full':'ADAPT.xml','lite':'ADAPT-Lite.xml'})]:
 with zipfile.ZipFile(path) as z:
  for scope,n in names.items():catalogs[version+scope]=system_catalog(z.read(n))
write(OUT/'adapt_system_catalog.json',catalogs)
roles=[('DXC09_Industrial_Tier2SampleData_20090331.zip','dxc09_sample','NASA_ADAPT_DXC09_TIER2','DEVELOPMENT_ONLY','09full'),('DXC09_Industrial_Tier2CompetitionData20090330.zip','dxc09_competition','NASA_ADAPT_DXC09_TIER2','EXTERNAL_VALIDATION','09full'),('DXC09_Industrial_Tier2RematchNewData20090921.zip','dxc09_rematch','NASA_ADAPT_DXC09_TIER2','TEMPORAL_SHIFT_REPLICATION','09full'),('DXC10_DPI-SampleData.zip','dxc10_dpi','NASA_ADAPT_DXC10_DPI','DEVELOPMENT_ONLY','10lite'),('DXC10_DPII-SampleData.zip','dxc10_dpii','NASA_ADAPT_DXC10_DPII','DEVELOPMENT_ONLY','10full')]
seen={};reports={}
for filename,prefix,task,role,key in roles:
 catalog=catalogs[key];cases=[];predictions={};manifest=[]
 with zipfile.ZipFile(DATA/filename) as z:
  workbook_name=next(n for n in z.namelist() if n.endswith('.xls'));wb=z.read(workbook_name);annotations=read_workbook_bytes(wb)
  for n in sorted(z.namelist()):
   if not n.endswith('.scn'):continue
   content=z.read(n);sid=hashlib.sha256(content).hexdigest();name=pathlib.PurePosixPath(n).stem
   if sid in seen:raise ValueError('SCENARIO_ROLE_CONTENT_OVERLAP:'+name)
   seen[sid]=role;manifest.append({'relative_path':n,'size':len(content),'sha256':sid});case={'case_id':sid,'group_id':sid,'audit_member':n}
   try:
    online,scn_truth=parse_scenario(content.decode('utf-8-sig'));t0=time.perf_counter();prediction=predict(online,catalog,config);wall=time.perf_counter()-t0
    predictions[sid]=prediction;case['runtime']={'inference_wall_s':wall,**observe_process_memory()}
    annotation=annotations.get(name)
    if annotation is None:case.update(status='SCORING_BLOCKED_TRUTH_UNAVAILABLE',reason='No exact workbook annotation; no filename inference',scn_truth_audit=scn_truth)
    else:
     try:case['truth_agreement']=verify_injection_agreement(annotation,scn_truth)
     except ValueError as e:case['truth_agreement']={'status':'DISAGREEMENT_RETAINED','reason':str(e),'scn_truth':scn_truth}
     case['truth']=annotation;case['status']='TRUTH_ADMITTED_PENDING_FROZEN_PREDICTION_JOIN'
   except ValueError as e:case.update(status='PARSER_OR_INFERENCE_BLOCKED',reason=str(e));predictions[sid]={'status':'BLOCKED','reason':str(e)}
   cases.append(case)
 # Persist independent inference before evaluator joins any truth.
 write(OUT/(prefix+'_frozen_predictions.json'),seal({'predictions':predictions,'config_sha256':config_identity}))
 for case in cases:
  if case['status']=='TRUTH_ADMITTED_PENDING_FROZEN_PREDICTION_JOIN':
   case['metrics']=evaluate_case(case['truth'],predictions[case['case_id']]);case['status']='SCORED'
 scored=[c for c in cases if c['status']=='SCORED']
 report={'evaluation_task_id':task,'role':role,'eligible_case_count':len(cases),'status':'EXECUTED' if len(scored)==len(cases) else 'EXECUTED_WITH_BLOCKED_CASES',
  'case_status_counts':dict(collections.Counter(c['status'] for c in cases)),'multi_fault_cases':sum(len(c.get('truth',{}).get('true_fault_ids',[]))>1 for c in cases),
  'metrics':summarize(scored) if role!='DEVELOPMENT_ONLY' else None,'development_metrics_status':'withheld by sample-only adapter policy' if role=='DEVELOPMENT_ONLY' else None,
  'case_receipt_sha256':digest(cases),'predictions_sha256':digest(predictions),'config_sha256':config_identity,
  'qualification':'NOT_FLIGHT_QUALIFIED','warnings':['sensor-morphology hypotheses and only local adjacent-current hardware candidates; not a validated complete physics model',
    'baseline first 30 seconds is observation only, not guaranteed nominal','unknown units excluded and audited; no formal EvidenceFact or probability calibration',
    'workbook explicit truth is primary; SCN agreement checked independently and conflicts preserved','DXC10 supplied data are sample; no external-performance claim']}
 write(OUT/(prefix+'_case_evaluation_receipt.json'),seal({'cases':cases,'workbook_member':workbook_name,'workbook_sha256':hashlib.sha256(wb).hexdigest(),'config_sha256':config_identity}))
 write(OUT/(prefix+'_external_evaluation_report.json'),seal(report,'report_sha256'));reports[prefix]=report
 from astrograph.evaluation.dxc_receipt_writer import bind_role_evaluation
 report=bind_role_evaluation(DATA/filename,OUT,prefix,catalog);reports[prefix]=report
 print(prefix,report['case_status_counts'],'exact set',report['metrics'].get('exact_fault_set_match') if report['metrics'] else 'development only',flush=True)
 if digest(json.loads((SRC/'configs/evaluation/r20_dxc.json').read_text()))!=config_identity:raise ValueError('FROZEN_DIAGNOSTIC_CONFIG_DRIFT')
write(OUT/'dxc_pipeline_gate_receipt.json',seal({'case_content_role_overlap':0,'competition_all_cases_scored':reports['dxc09_competition']['case_status_counts']=={'SCORED':120},
 'rematch_frozen_config_unchanged':True,'dxc10_distinct_tasks':reports['dxc10_dpi']['evaluation_task_id']!=reports['dxc10_dpii']['evaluation_task_id'],
 'sample_performance_withheld':True,'multi_fault_set_preserved':True,'scope':'software/admission pipeline gate, no minimum performance threshold or qualification','qualification':'NOT_FLIGHT_QUALIFIED'}))
