#!/usr/bin/env python3
"""Qualify frozen ADCS probabilities and bounded input-novelty OOD without test fitting."""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from astrograph.data_driven.adcs_posthoc import fit_posthoc_qualification_v2
from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package
from scripts.integration import run_adcs_v522_regression as adcs


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->int:
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset-root',type=Path,required=True)
 p.add_argument('--model',type=Path,default=ROOT/'models/adcs_v5_2_2/p2_innovation_v2.pt')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--max-train-reference',type=int,default=12000)
 p.add_argument('--max-stress-windows',type=int,default=2048)
 a=p.parse_args()
 msha=sha(a.model); provider=load_adcs_three_task_torch_package(a.model,expected_sha256=msha,device='cpu')
 rows=adcs.load_trajectories(a.dataset_root); audit=adcs.audit_split(rows)
 if audit['status']!='PASS':raise RuntimeError(audit)
 dev=[r for r in rows if r.split in {'train','validation'}]
 classes=list(provider.class_names);components=list(provider.component_names)
 windows=adcs.build_windows(dev,{x:i for i,x in enumerate(classes)},{x:i for i,x in enumerate(components)},feature_columns=provider.selected_channels)
 train=adcs.preprocess_windows(windows['train'],provider.preprocessor); val=adcs.preprocess_windows(windows['validation'],provider.preprocessor)
 import torch
 torch.set_num_threads(1); provider.model.eval(); probs=[[],[],[]]
 with torch.inference_mode():
  for st in range(0,len(val.x),2048):
   out=provider.model(torch.from_numpy(val.x[st:st+2048]))
   for i,logits in enumerate(out):probs[i].append(torch.softmax(logits,1).cpu().numpy())
 det_head,direct_comp,class_prob=[np.concatenate(x) for x in probs]
 summary=adcs.summary_features(val.x); ll=summary@provider.logistic_coefficient+float(provider.logistic_intercept); logistic=1/(1+np.exp(-np.clip(ll,-60,60)))
 blend=float(provider.detection_selection['detector_deep_blend_alpha']); det=blend*det_head[:,1]+(1-blend)*logistic
 ci={name:i for i,name in enumerate(components)}; derived=np.zeros_like(direct_comp)
 for j,name in enumerate(classes):derived[:,ci[provider.fault_to_component[name]]]+=class_prob[:,j]
 alpha=float(provider.diagnosis_selection['hierarchy_blend_alpha']); comp=alpha*direct_comp+(1-alpha)*derived; comp/=np.maximum(comp.sum(1,keepdims=True),1e-12)
 cond=np.zeros_like(class_prob);policy=provider.diagnosis_selection.get('classification_policy','hard_component')
 for i,pred in enumerate(comp.argmax(1)):
  if policy=='hard_component':
   allowed=np.array([ci[provider.fault_to_component[name]]==pred for name in classes]);cond[i]=np.where(allowed,class_prob[i],0)
  elif policy=='soft_component':
   cond[i]=class_prob[i]*np.array([comp[i,ci[provider.fault_to_component[name]]] for name in classes])
  else:cond[i]=class_prob[i]
  cond[i]/=max(float(cond[i].sum()),1e-12)
 joint=np.c_[1-det,det[:,None]*cond];joint/=np.maximum(joint.sum(1,keepdims=True),1e-12)
 sample_group={r.sample_id:r.trajectory_pair_id for r in dev}; groups=np.array([sample_group[s] for s in val.trajectory_ids])
 rng=np.random.default_rng(20260915);ids=np.arange(len(train.x));
 if len(ids)>a.max_train_reference:ids=np.sort(rng.choice(ids,a.max_train_reference,replace=False))
 # Stress windows are selected across the full validation partition only for evaluator-side OOD qualification.
 qx=val.x
 if len(qx)>a.max_stress_windows:qx=qx[np.sort(rng.choice(len(qx),a.max_stress_windows,replace=False))]
 mask=((np.arange(qx.shape[2])%3)==0)[None,None,:]
 sign=qx.copy();sign[:,:,::4]*=-1
 primary={
  'hidden_channel_value_roll_1':adcs.summary_features(np.roll(qx,1,axis=2)),
  'distributed_offset_5sigma':adcs.summary_features(np.clip(qx+mask*5.0,-10.0,10.0)),
 }
 secondary={
  'amplitude_x3':adcs.summary_features(np.clip(qx*3.0,-10.0,10.0)),
  'temporal_reverse':adcs.summary_features(qx[:,::-1,:].copy()),
  'window_constant_last_value':adcs.summary_features(np.repeat(qx[:,-1:,:],qx.shape[1],axis=1)),
  'sparse_sign_flip':adcs.summary_features(sign),
 }
 artifact=fit_posthoc_qualification_v2(
  model_artifact_sha256=msha,dataset_marker_sha256=sha(a.dataset_root/'sample_index.csv'),channel_names=provider.selected_channels,
  validation_group_ids=groups,validation_detection_truth=val.detection,validation_fault_truth=val.isolation,joint_probabilities=joint,
  train_summary=adcs.summary_features(train.x[ids]),validation_summary=summary,primary_stress_summaries=primary,secondary_stress_summaries=secondary,
 )
 payload=artifact.model_dump(mode='json');payload['provenance']={
  'dataset_root':str(a.dataset_root.resolve()),'split_audit':audit,'test_trajectory_count_loaded':0,'development_trajectory_count':len(dev),
  'train_window_count':len(train.x),'validation_window_count':len(val.x),'train_reference_window_count':len(ids),'stress_window_count':len(qx),
  'source_model_path':str(a.model.resolve()),'source_model_sha256':msha,
 }
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 c=artifact.joint_probability_calibration;o=artifact.ood_profile
 print(json.dumps({'output':str(a.output),'qualification_id':artifact.qualification_id,'qualified':artifact.formal_probability_and_ood_eligible,
 'uniform_alpha':c.final_uniform_alpha,'group_cv_nll_before':c.mean_nll_before,'group_cv_nll_after':c.mean_nll_after,
 'group_cv_brier_before':c.mean_brier_before,'group_cv_brier_after':c.mean_brier_after,'group_cv_ece_before':c.mean_ece_before,'group_cv_ece_after':c.mean_ece_after,
 'ood_id_false_reject_rate':o.validation_id_false_reject_rate,'ood_primary_stress_recall':o.primary_stress_recall,'ood_secondary_stress_recall':o.secondary_stress_recall},ensure_ascii=False,indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
