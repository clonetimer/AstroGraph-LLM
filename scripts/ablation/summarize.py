"""Re-score all frozen inference records and export readable ablation tables."""
from __future__ import annotations
import argparse,sys,json,csv,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from common import *
from run_adcs import ARMS

LABELS={'data_only':'仅数据模型','data_reference':'数据模型＋健康参照一致性','no_checks':'完整机理与融合，不补证','full_adaptive':'完整机理与融合，按需补证','full_eager':'完整机理与融合，每窗检查',
'forward_score_off':'前向图谱排名分关闭','backward_score_off':'后向图谱排名分关闭','both_kg_scores_off':'两路图谱排名分均关闭（保留证据裁决）','data_score_off':'数据排名分关闭（保留数据候选）',
'lineage_cap_off':'同源贡献封顶关闭','kg_redundancy_cap_off':'KG排名冗余封顶关闭','no_executable_knowledge':'移除可执行知识（固定已获观测）','explicit_counter_rules_off':'移除显式COUNTER规则（ADCS无此角色，空操作对照）'}

def main(args):
 from astrograph.evaluation.three_task_replay import evaluate_three_task_replay,_fault_prediction
 base=args.results.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 mapping=read(ROOT/'configs/datasets/adcs_v5_2_2/fault_class_map.json')['mappings'];hierarchy=read(ROOT/'configs/datasets/adcs_v5_2_2/diagnosis_hierarchy.json')['fault_to_component'];comp={mapping[k]:v for k,v in hierarchy.items()}
 rows_out=[];delta_out=[];validation=[];support={}
 for cohort in ('adcs_development','adcs_used_test'):
  p=base/cohort;joined=read(p/'joined.json');old=read(p/'report.json');plan=read(p/'plan.json')
  assert sha_file(p/'inference.jsonl')==old['inference_sha256']
  fresh=evaluate_three_task_replay(joined,fault_to_component=comp,dataset_class_to_fault=mapping,branches=ARMS)
  assert fresh['branches']==old['branches'] and fresh['model_heads']==old['model_heads']
  assert all(r['label_join_inference_sha256']==old['inference_sha256'] for r in joined)
  validation.append({'cohort':cohort,'rescore_equal':True,'windows':len(joined),'inference_sha256':old['inference_sha256']})
  support[cohort]={a:dict(collections.Counter(r['details'].get(a,{}).get('status','UNAVAILABLE') for r in joined)) for a in ['no_checks','full_adaptive','full_eager','no_executable_knowledge']}
  for arm in ARMS:
   m=fresh['branches'][arm];active=[r for r in joined if r['target'] is not None]
   rs={'cohort':cohort,'arm':arm,'name':LABELS[arm],'windows':len(joined),'fault_windows':len(active),
       'detection_macro_f1_all':m['detection_all']['macro_f1'],'detection_macro_f1_executable':m['detection_executable']['macro_f1'],
       'localization_direct_head_macro_f1':fresh['model_heads']['localization_direct_head']['macro_f1'],
       'localization_system_macro_f1':m['localization_system_derived']['macro_f1'],'classification_macro_f1':m['classification_system']['macro_f1'],
       'correct_fault':m['classification_system']['correct'],'wrong_fault_class':sum(r[arm]['kind'] in {'CLASS','MODEL_ONLY'} and r[arm]['ids']!=[r['target']] for r in active),
       'missed_anomaly':sum(r[arm]['kind']=='NORMAL' for r in active),'unresolved_fault':m['classification_system']['abstained'],
       'fault_unavailable':m['classification_system']['unavailable'],'all_unavailable':m['detection_all']['unavailable'],
       'false_positive_normal':sum(r['target'] is None and r[arm]['kind'] in {'CLASS','MODEL_ONLY','SET'} for r in joined),
       'numeric_checks':old['checks_executed'] if arm=='full_adaptive' else (old['eager_calls'] if arm=='full_eager' else (0 if arm in {'data_only','data_reference','no_checks'} else None))}
   rows_out.append(rs)
   valid=[r for r in joined if arm in r['details'] and 'full_adaptive' in r['details']]
   def order(d):return sorted((x['fault_id'],x['final_rank']) for x in d['ranking'])
   delta_out.append({'cohort':cohort,'arm':arm,'fixed_prediction_changes':sum(_fault_prediction(r[arm])!=_fault_prediction(r['full_adaptive']) for r in joined),
       'supported_state_changes':sum(r['details'][arm]['status']!=r['details']['full_adaptive']['status'] for r in valid),
       'rrf_nomination_changes':sum(r['details'][arm]['top']!=r['details']['full_adaptive']['top'] for r in valid),
       'rank_order_changes':sum(order(r['details'][arm])!=order(r['details']['full_adaptive']) for r in valid),
       'ranking_record_changes':sum(r['details'][arm]['ranking']!=r['details']['full_adaptive']['ranking'] for r in valid)})
 for name,rows in [('adcs_metrics',rows_out),('adcs_deltas',delta_out)]:
  write(out/(name+'.json'),rows)
  with (out/(name+'.csv')).open('x',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write(out/'RESCORE_CHECK.json',{'checks':validation,'retrospective':True,'new_generation_requests':0})
 write(out/'support_states.json',support)
 numeric=read(base/'numeric/report.json')
 with (out/'numeric.csv').open('x',encoding='utf-8-sig',newline='') as f:
  fields=['case','arm','checks','admitted','selected','status','stop','sequence'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in numeric['rows']:w.writerow({k:(json.dumps(r[k],ensure_ascii=False) if k=='sequence' else r.get(k)) for k in fields})
 rags=read(base/'rag_factorial/report-001.json')['rows'];ragrows=[]
 for r in rags:
  q=r.get('quality',{});ragrows.append({'arm':r['cell']['planner'],'budget':r['cell']['max_optional_actions'],'repeat':r['cell']['repeat'],'status':r['status'],
   'actual_retrievals':q.get('retrieval',{}).get('executed_calls'),'new_texts':[x['new_texts'] for x in q.get('retrieval',{}).get('calls',[])],
   'selected':q.get('diagnosis',{}).get('selected'),'decision_status':q.get('diagnosis',{}).get('status'),
   'checks':q.get('checks',{}).get('executed'),'admitted':q.get('checks',{}).get('admitted_feature_result_count')})
 write(out/'rag_rows.json',ragrows)
 current=read(base/'archived_live/report.json');write(out/'archived_live_summary.json',current)
 print(canonical({'rescore':'IDENTICAL','metrics_rows':len(rows_out),'numeric_runs':len(numeric['rows']),'rag_completed':sum(r['status']!='NOT_RUN' for r in ragrows),'archived_verified':sum(g['verified'] for g in current['groups'].values())}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
