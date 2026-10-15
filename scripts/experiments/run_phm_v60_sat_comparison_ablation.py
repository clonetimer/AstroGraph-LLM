#!/usr/bin/env python3
"""Fixed-split comparison and ablation for PHM v6.0 retrospective trace screening.

Protocol:
- Frozen sample_index Train/Validation/Test groups from phm_v60.py.
- Candidate fitting uses Train only.
- Detection threshold / routing choices use Validation only.
- Test is evaluated after choices are frozen.
- Primary comparison uses the same post-activation trace-summary information boundary.
- Full-trace/pre-activation variants are explicitly labelled information-boundary ablations.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from astrograph.data_driven.phm_v60 import (
    PHMDataset, STATS, audit_split, build_dataset, is_allowed_channel, load_records,
    metric_bundle, train_three_stage, train_three_stage_adaptive,
    _aligned_probabilities, _route_class_probabilities, _finite_f_classif,
)

HIERARCHY = ROOT / 'configs/datasets/phm_v6_0/diagnosis_hierarchy.json'


def _usable(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if frame[c].notna().sum() >= 2 and frame[c].nunique(dropna=True) > 1]


def make_family_pipeline(frame: pd.DataFrame, spec: dict[str, Any], seed: int = 20260813):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.feature_selection import SelectKBest
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    usable = _usable(frame)
    fam = spec['family']
    k = min(int(spec.get('k', 256)), len(usable))
    if fam == 'logistic':
        p = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                      ('select', SelectKBest(_finite_f_classif, k=k)),
                      ('classify', LogisticRegression(max_iter=4000, class_weight='balanced', C=float(spec.get('C',1)), random_state=seed))])
    elif fam == 'svc':
        p = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                      ('select', SelectKBest(_finite_f_classif, k=k)),
                      ('classify', CalibratedClassifierCV(
                          estimator=SVC(
                              C=float(spec.get('C', 10)), class_weight='balanced',
                              random_state=seed,
                          ),
                          method='sigmoid', cv=5, ensemble=False,
                      ))])
    elif fam == 'extra_trees':
        p = Pipeline([('impute', SimpleImputer(strategy='median')),
                      ('classify', ExtraTreesClassifier(n_estimators=int(spec.get('n_estimators',500)), max_features=float(spec.get('max_features',0.3)), min_samples_leaf=int(spec.get('min_samples_leaf',2)), class_weight='balanced', random_state=seed, n_jobs=-1))])
    elif fam == 'random_forest':
        p = Pipeline([('impute', SimpleImputer(strategy='median')),
                      ('classify', RandomForestClassifier(n_estimators=int(spec.get('n_estimators',500)), max_features=spec.get('max_features','sqrt'), min_samples_leaf=int(spec.get('min_samples_leaf',1)), class_weight='balanced_subsample', random_state=seed, n_jobs=-1))])
    elif fam == 'hist_gb':
        p = Pipeline([('impute', SimpleImputer(strategy='median')), ('select', SelectKBest(_finite_f_classif, k=k)),
                      ('classify', HistGradientBoostingClassifier(max_iter=int(spec.get('max_iter',250)), learning_rate=float(spec.get('learning_rate',0.08)), max_leaf_nodes=int(spec.get('max_leaf_nodes',31)), l2_regularization=float(spec.get('l2_regularization',1.0)), random_state=seed))])
    else:
        raise ValueError(fam)
    return p, usable


def train_fixed_family(dataset: PHMDataset, mapping: dict[str,str], spec: dict[str,Any], *, route_selection: bool=True, threshold_selection: bool=True, fixed_route: str|None=None, fixed_threshold: float|None=None, seed: int=20260813) -> dict[str,Any]:
    records=dataset.records
    split=np.asarray([r.split for r in records]); conditions=np.asarray([r.condition_id for r in records],dtype=str)
    det_truth=np.where(conditions=='nominal','nominal','fault'); fault=conditions!='nominal'
    tr=split=='train'; va=split=='validation'; te=split=='test'; trf=tr&fault; vaf=va&fault
    # detection
    dm,dc=make_family_pipeline(dataset.features.loc[tr],spec,seed); dm.fit(dataset.features.loc[tr,dc],det_truth[tr])
    vp=_aligned_probabilities(dm,dataset.features.loc[va],dc,['nominal','fault'])[:,1]
    if fixed_threshold is not None: th=float(fixed_threshold)
    elif threshold_selection:
        choices=[]
        for x in np.linspace(.1,.9,17):
            pred=np.where(vp>=x,'fault','nominal'); score=metric_bundle(det_truth[va],pred,['nominal','fault'])['macro_f1']
            choices.append((score,-abs(float(x)-.5),float(x)))
        th=max(choices)[2]
    else: th=.5
    # localization
    comps=sorted(set(dataset.components[trf])); lm,lc=make_family_pipeline(dataset.features.loc[trf],spec,seed); lm.fit(dataset.features.loc[trf,lc],dataset.components[trf])
    # classification
    classes=sorted(set(conditions[trf])); cm,cc=make_family_pipeline(dataset.features.loc[trf],spec,seed); cm.fit(dataset.features.loc[trf,cc],conditions[trf])
    val_cp=_aligned_probabilities(cm,dataset.features.loc[va],cc,classes); val_lp=_aligned_probabilities(lm,dataset.features.loc[va],lc,comps)
    routes=['raw','soft:0.25','soft:0.5','soft:1','hard']
    if fixed_route is not None: route=fixed_route
    elif route_selection:
        rows=[]
        for idx,r in enumerate(routes):
            pr=_route_class_probabilities(val_cp,val_lp,classes,comps,mapping,r)
            sc=metric_bundle(conditions[vaf],pr[fault[va]],classes)['macro_f1']; rows.append((sc,-idx,r))
        route=max(rows)[2]
    else: route='raw'
    # test
    dp=_aligned_probabilities(dm,dataset.features.loc[te],dc,['nominal','fault'])[:,1]; dpr=np.where(dp>=th,'fault','nominal')
    lp=_aligned_probabilities(lm,dataset.features.loc[te],lc,comps); lpr=np.asarray(comps)[lp.argmax(1)]
    cp=_aligned_probabilities(cm,dataset.features.loc[te],cc,classes); cpr=_route_class_probabilities(cp,lp,classes,comps,mapping,route)
    truth=conditions[te]; true_comp=dataset.components[te]; active=truth!='nominal'; e2e=np.where(dpr=='nominal','nominal',cpr)
    return {
        'id':spec['id'],'spec':spec,'threshold':th,'route':route,
        'detection':metric_bundle(det_truth[te],dpr,['nominal','fault']),
        'localization':metric_bundle(true_comp[active],lpr[active],comps),
        'classification':metric_bundle(truth[active],cpr[active],classes),
        'end_to_end':metric_bundle(truth,e2e,['nominal',*classes]),
        'failure_attribution':{
            'fault_traces':int(active.sum()),'first_wrong_detection':int(np.sum(active&(dpr=='nominal'))),
            'first_wrong_localization':int(np.sum(active&(dpr=='fault')&(lpr!=true_comp))),
            'fully_correct_fault_traces':int(np.sum(active&(dpr=='fault')&(cpr==truth))),
        }
    }


def stat_subset(dataset: PHMDataset, keep_stats: set[str]) -> PHMDataset:
    cols=[c for c in dataset.features.columns if c.rsplit('__',1)[-1] in keep_stats]
    return PHMDataset(features=dataset.features[cols].copy(),records=dataset.records,components=dataset.components.copy())


def summarize_values(values: np.ndarray, prefix: str, out: dict[str,float]):
    values=values[np.isfinite(values)]
    for stat in STATS:
        key=f'{prefix}__{stat}'
        if not len(values): out[key]=math.nan
        elif stat=='mean': out[key]=float(np.mean(values))
        elif stat=='std': out[key]=float(np.std(values))
        elif stat=='min': out[key]=float(np.min(values))
        elif stat=='max': out[key]=float(np.max(values))
        elif stat=='delta': out[key]=float(values[-1]-values[0])
        else: out[key]=float(np.sqrt(np.mean(np.square(values))))


def build_boundary_variants(base: PHMDataset, *, progress=False) -> dict[str,PHMDataset]:
    rows_full=[]; rows_pre=[]
    for i,r in enumerate(base.records,1):
        f=pd.read_csv(r.telemetry_path,low_memory=False); t=pd.to_numeric(f['time_s'],errors='coerce').to_numpy(dtype=float)
        full={}; pre={}; premask=t<r.activation_time_s
        for name in f.columns:
            if not is_allowed_channel(str(name)): continue
            vals=pd.to_numeric(f[name],errors='coerce').to_numpy(dtype=float)
            summarize_values(vals,str(name),full); summarize_values(vals[premask],str(name),pre)
        rows_full.append(full); rows_pre.append(pre)
        if progress and i%100==0: print(f'[boundary] {i}/{len(base.records)}',flush=True)
    return {
        'post_activation':base,
        'full_trace':PHMDataset(pd.DataFrame(rows_full),base.records,base.components.copy()),
        'pre_activation':PHMDataset(pd.DataFrame(rows_pre),base.records,base.components.copy()),
    }


def slim_existing(result: dict[str,Any], name: str) -> dict[str,Any]:
    return {'id':name,'detection':result['detection'],'localization':result['localization'],
            'classification':result['classification_predicted_component'],'end_to_end':result['end_to_end_18_class'],
            'failure_attribution':result['failure_attribution'],'selection':result.get('selection')}


def render(report):
    rows=[]
    for r in report['comparison']:
        rows.append(f"| {r['id']} | {r['detection']['macro_f1']:.4f} | {r['localization']['macro_f1']:.4f} | {r['classification']['macro_f1']:.4f} | {r['end_to_end']['macro_f1']:.4f} |")
    abl=[]
    for r in report['feature_ablation']:
        abl.append(f"| {r['id']} | {r['feature_count']} | {r['detection']['macro_f1']:.4f} | {r['localization']['macro_f1']:.4f} | {r['classification']['macro_f1']:.4f} | {r['end_to_end']['macro_f1']:.4f} |")
    bd=[]
    for r in report['boundary_ablation']:
        bd.append(f"| {r['id']} | {r['detection']['macro_f1']:.4f} | {r['localization']['macro_f1']:.4f} | {r['classification']['macro_f1']:.4f} | {r['end_to_end']['macro_f1']:.4f} |")
    return f"""# PHM v6.0 sat fixed-split comparison and ablation\n\nThe primary table uses the same post-activation trace-summary information boundary. Train fits models; Validation selects thresholds/routes; Test is evaluated after freeze. Full/pre-trace rows are information-boundary ablations and are not interchangeable with unknown-onset causal-window PHM metrics.\n\n## Model comparison\n\n| model | detection | localization | classification | E2E-18 |\n|---|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Feature-statistic ablation\n\n| condition | features | detection | localization | classification | E2E-18 |\n|---|---:|---:|---:|---:|---:|\n{chr(10).join(abl)}\n\n## Information-boundary ablation\n\n| boundary | detection | localization | classification | E2E-18 |\n|---|---:|---:|---:|---:|\n{chr(10).join(bd)}\n"""


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--skip-boundary',action='store_true'); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); hierarchy=json.loads(HIERARCHY.read_text()); mapping=hierarchy['fault_to_component']
    records=load_records(args.dataset_root); audit=audit_split(records)
    if audit['status']!='PASS': raise RuntimeError(audit)
    started=time.time(); data=build_dataset(args.dataset_root,mapping,progress=True)
    legacy=train_three_stage(data,mapping); adaptive=train_three_stage_adaptive(data,mapping)
    comparison=[slim_existing(legacy,'legacy_lr'),slim_existing(adaptive,'adaptive_validation_selected')]
    family_specs=[
        {'id':'lr-k256','family':'logistic','k':256,'C':1.0},
        {'id':'svc-c10-k256','family':'svc','k':256,'C':10.0},
        {'id':'extra-trees-500','family':'extra_trees','n_estimators':500,'max_features':0.3,'min_samples_leaf':2},
        {'id':'random-forest-500','family':'random_forest','n_estimators':500,'max_features':'sqrt','min_samples_leaf':1},
        {'id':'hist-gb-k256','family':'hist_gb','k':256,'max_iter':250,'learning_rate':0.08,'max_leaf_nodes':31,'l2_regularization':1.0},
    ]
    for s in family_specs:
        print('[compare]',s['id'],flush=True); comparison.append(train_fixed_family(data,mapping,s))
    # Feature ablations use the fully selected dev12 family identities; model family is fixed, val may retune threshold.
    dspec=adaptive['selection']['detection']['candidate']; lspec=adaptive['selection']['localization']['candidate']; cspec=adaptive['selection']['classification']['candidate']; route=adaptive['selection']['classification']['route']
    # Implement mixed-family selected cascade via a small wrapper by temporarily train each stage separately.
    def mixed(ds:PHMDataset, id_:str):
        # custom copy of fixed family logic with three specs
        rec=ds.records; split=np.asarray([r.split for r in rec]); cond=np.asarray([r.condition_id for r in rec],dtype=str); fault=cond!='nominal'; det=np.where(fault,'fault','nominal'); tr=split=='train';va=split=='validation';te=split=='test';trf=tr&fault;vaf=va&fault
        dm,dc=make_family_pipeline(ds.features.loc[tr],dspec); dm.fit(ds.features.loc[tr,dc],det[tr]); vp=_aligned_probabilities(dm,ds.features.loc[va],dc,['nominal','fault'])[:,1]; choices=[]
        for th in np.linspace(.1,.9,17): choices.append((metric_bundle(det[va],np.where(vp>=th,'fault','nominal'),['nominal','fault'])['macro_f1'],-abs(float(th)-.5),float(th)))
        th=max(choices)[2]
        comps=sorted(set(ds.components[trf])); lm,lc=make_family_pipeline(ds.features.loc[trf],lspec); lm.fit(ds.features.loc[trf,lc],ds.components[trf])
        classes=sorted(set(cond[trf])); cm,cc=make_family_pipeline(ds.features.loc[trf],cspec); cm.fit(ds.features.loc[trf,cc],cond[trf])
        dp=_aligned_probabilities(dm,ds.features.loc[te],dc,['nominal','fault'])[:,1]; dpr=np.where(dp>=th,'fault','nominal'); lp=_aligned_probabilities(lm,ds.features.loc[te],lc,comps);lpr=np.asarray(comps)[lp.argmax(1)];cp=_aligned_probabilities(cm,ds.features.loc[te],cc,classes);cpr=_route_class_probabilities(cp,lp,classes,comps,mapping,route);truth=cond[te];tc=ds.components[te];active=truth!='nominal';e2e=np.where(dpr=='nominal','nominal',cpr)
        return {'id':id_,'feature_count':int(ds.features.shape[1]),'threshold':th,'route':route,'detection':metric_bundle(det[te],dpr,['nominal','fault']),'localization':metric_bundle(tc[active],lpr[active],comps),'classification':metric_bundle(truth[active],cpr[active],classes),'end_to_end':metric_bundle(truth,e2e,['nominal',*classes])}
    variants={
        'full':set(STATS), 'drop_mean':set(STATS)-{'mean'}, 'drop_std':set(STATS)-{'std'}, 'drop_minmax':set(STATS)-{'min','max'}, 'drop_delta':set(STATS)-{'delta'}, 'drop_rms':set(STATS)-{'rms'}, 'mean_std_only':{'mean','std'}, 'change_range_only':{'min','max','delta'},
    }
    feature=[]
    for name,stats in variants.items():
        print('[feature]',name,flush=True); feature.append(mixed(stat_subset(data,stats),name))
    boundary=[]
    if not args.skip_boundary:
        bvars=build_boundary_variants(data,progress=True)
        for name,ds in bvars.items():
            print('[boundary-fit]',name,flush=True); boundary.append(mixed(ds,name))
    report={'schema_version':'astrograph-phm-v60-sat-comparison-ablation.v1','split_audit':audit,'protocol':{'primary_information_boundary':'post_activation_trace_summary','selection':'Train fit; Validation threshold/route/family selection; Test once after freeze','feature_stats':list(STATS)},'comparison':comparison,'feature_ablation':feature,'boundary_ablation':boundary,'adaptive_selection':adaptive['selection'],'runtime_seconds':time.time()-started,'claim_boundary':'Synthetic PHM v6.0 retrospective trace screening. Post-activation results assume known activation boundary. Full/pre-trace ablations are diagnostic experiments, not online causal-window certification.'}
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(args.output_dir/'report.md').write_text(render(report),encoding='utf-8')
    print(args.output_dir/'report.json')
if __name__=='__main__': main()
