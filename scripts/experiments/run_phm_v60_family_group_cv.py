#!/usr/bin/env python3
from __future__ import annotations
import argparse, dataclasses, json, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'/'experiments'))
from astrograph.data_driven.phm_v60 import PHMDataset, build_dataset, load_records
from run_phm_v60_sat_comparison_ablation import train_fixed_family
H=ROOT/'configs/datasets/phm_v6_0/diagnosis_hierarchy.json'
def fold_records(records,fold):
 out=[]
 for r in records:
  rem=r.sample_index%5; sp='test' if rem==fold else ('validation' if rem==(fold+1)%5 else 'train'); out.append(dataclasses.replace(r,split=sp))
 return out
def metrics(r): return {k:float(r[k]['macro_f1']) for k in ['detection','localization','classification','end_to_end']}
def agg(rows,key):
 return {m:{'mean':float(np.mean([r[key][m] for r in rows])),'std':float(np.std([r[key][m] for r in rows])),'minimum':float(np.min([r[key][m] for r in rows])),'values':[r[key][m] for r in rows]} for m in ['detection','localization','classification','end_to_end']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
 mapping=json.loads(H.read_text())['fault_to_component']; records=load_records(args.dataset_root); data=build_dataset(args.dataset_root,mapping,progress=True)
 specs=[{'id':'svc-c10-k256','family':'svc','k':256,'C':10.0},{'id':'extra-trees-500','family':'extra_trees','n_estimators':500,'max_features':0.3,'min_samples_leaf':2},{'id':'hist-gb-k256','family':'hist_gb','k':256,'max_iter':250,'learning_rate':0.08,'max_leaf_nodes':31,'l2_regularization':1.0}]
 rows=[]
 for fold in range(5):
  fr=fold_records(records,fold); ds=PHMDataset(data.features,fr,data.components)
  row={'fold':fold,'test_groups':sorted({r.sample_index for r in fr if r.split=='test'}),'validation_groups':sorted({r.sample_index for r in fr if r.split=='validation'})}
  for s in specs:
   print('[fold]',fold,s['id'],flush=True); row[s['id']]=metrics(train_fixed_family(ds,mapping,s))
  rows.append(row)
 report={'schema_version':'astrograph-phm-v60-family-group-cv.v1','folds':rows,'aggregate':{s['id']:agg(rows,s['id']) for s in specs},'claim_boundary':'Post-activation trace-summary group CV; same known-onset retrospective boundary as dev12. Test fold not used for selection.'}
 args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');
 print(args.output_dir/'report.json')
if __name__=='__main__':main()
