#!/usr/bin/env python3
"""Route and stage-oracle ablation for the dev12 PHM retrospective cascade."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from astrograph.data_driven.phm_v60 import build_dataset, train_three_stage_adaptive, _aligned_probabilities, _route_class_probabilities, metric_bundle
H=ROOT/'configs/datasets/phm_v6_0/diagnosis_hierarchy.json'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dataset-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 mapping=json.loads(H.read_text())['fault_to_component']; ds=build_dataset(args.dataset_root,mapping,progress=True); res=train_three_stage_adaptive(ds,mapping); models=res['models']; cols=res['feature_columns']; rec=ds.records
 split=np.array([r.split for r in rec]); cond=np.array([r.condition_id for r in rec],dtype=str); fault=cond!='nominal'; te=split=='test'; truth=cond[te]; active=truth!='nominal'; det_truth=np.where(truth=='nominal','nominal','fault'); comps=sorted(set(ds.components[(split=='train')&fault])); classes=sorted(set(cond[(split=='train')&fault])); truecomp=ds.components[te]
 dp=_aligned_probabilities(models['detection'],ds.features.loc[te],cols['detection'],['nominal','fault'])[:,1]; th=res['selection']['detection']['threshold']; dpr=np.where(dp>=th,'fault','nominal'); lp=_aligned_probabilities(models['localization'],ds.features.loc[te],cols['localization'],comps); cp=_aligned_probabilities(models['classification'],ds.features.loc[te],cols['classification'],classes)
 out={'selected_route':res['selection']['classification']['route'],'selected_threshold':th,'routes':{},'stage_oracles':{},'claim_boundary':'Retrospective post-activation trace-summary ablation; oracle rows are diagnostic upper bounds, not deployable models.'}
 for route in ['raw','soft:0.25','soft:0.5','soft:1','hard']:
  cpr=_route_class_probabilities(cp,lp,classes,comps,mapping,route); e2e=np.where(dpr=='nominal','nominal',cpr); out['routes'][route]={'classification':metric_bundle(truth[active],cpr[active],classes),'end_to_end':metric_bundle(truth,e2e,['nominal',*classes])}
 route=res['selection']['classification']['route']; cpr=_route_class_probabilities(cp,lp,classes,comps,mapping,route)
 out['stage_oracles']['oracle_detection']={'end_to_end':metric_bundle(truth,np.where(det_truth=='nominal','nominal',cpr),['nominal',*classes])}
 out['stage_oracles']['oracle_classification']={'end_to_end':metric_bundle(truth,np.where(dpr=='nominal','nominal',truth),['nominal',*classes])}
 out['stage_oracles']['oracle_detection_and_classification']={'end_to_end':metric_bundle(truth,truth,['nominal',*classes])}
 oracle_lp=np.zeros_like(lp)
 for i,c in enumerate(truecomp):
  if c in comps: oracle_lp[i,comps.index(c)]=1
 hard_oracle=_route_class_probabilities(cp,oracle_lp,classes,comps,mapping,'hard'); e2e=np.where(dpr=='nominal','nominal',hard_oracle)
 out['stage_oracles']['oracle_localization_hard_route']={'classification':metric_bundle(truth[active],hard_oracle[active],classes),'end_to_end':metric_bundle(truth,e2e,['nominal',*classes])}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(args.output)
if __name__=='__main__': main()
