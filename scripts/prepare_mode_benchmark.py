"""Prepare frozen 33-channel causal windows; never fit using test data.

Train/validation: ten predeclared endpoints. Test: the previously used 30/150/300
endpoint cohort, explicitly retrospective. Labels are stored separately from
inference inputs. Training never reads test_labels.json.
"""
import argparse,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from astrograph.data_driven.preprocessing import fit_channel_preprocessor
from astrograph.oag.native_bridge import project_registered_telemetry

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    packet=torch.load(root/'models/adcs_v5_2_2/p2_innovation_v2.pt',weights_only=True,map_location='cpu')
    channels=packet['selected_channels'];classes=packet['class_names'];components=packet['component_names'];mapping=packet['class_map'];hierarchy=packet['fault_to_component']
    schema=json.loads((root/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text())['canonical_sha256']
    splits={'train':[0,1,2,5,6,7],'validation':[3,8],'test':[4,9]}
    plan={'schema_version':'astrograph-mode-benchmark.v1','dataset_sha256':digest(a.dataset_zip),'channels':channels,'window_length':20,
          'splits':splits,'endpoints':{'train':list(range(30,301,30)),'validation':list(range(30,301,30)),'test':[30,150,300]},
          'seeds':[20260913,20260914,20260915],'families':['residual_tcn','inception_time','timesnet_ref','moderntcn_ref','patchtst_ref','hist_gradient_boosting'],
          'epochs':40,'min_epochs':20,'patience':10,'batch_size':128,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,
          'checkpoint':'highest mean validation detection/localization/classification macro-F1; detector0.5 at selection',
          'detection_threshold':'validation macro-F1 grid 0.05..0.95 step0.05, tie closest0.5; not calibration',
          'classes':classes,'components':components,'class_map':mapping,'fault_to_component':hierarchy,
          'normal_mode':'unchanged production plus separate equal-budget candidate TCN normal replay',
          'information_tracks':{'controlled':'same33channels, no nominal or knowledge, same train-only scaling and heads',
                                'normal':'data model + approved KG + paired reference + additional checks, disclose unequal information'},
          'test_status':'Previously used test trajectories, retrospective; not independent confirmation',
          'fresh_model_retraining':True,'production_retrained':False,'shared_pair_groups':'global sample index; six train, two validation, two test base seeds',
          'claim_boundary':'Reference-oriented compact adaptations; not exact SOTA reproduction. No universal superiority inference.'}
    (a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    buckets={s:[] for s in splits}
    with zipfile.ZipFile(a.dataset_zip) as z:
        files=sorted(n for n in z.namelist() if n.endswith('/telemetry.csv'))
        dataset_root=next(n for n in z.namelist() if n.endswith('/dataset_manifest.json')).rsplit('/',1)[0]
        manifest=json.loads(z.read(dataset_root+'/dataset_manifest.json'));missions={s['scenario_id']:s['mission_id'] for s in manifest['operating_scenarios']}
        for i,name in enumerate(files):
            sample=int(re.search(r'/sample_(\d+)/',name).group(1));split=next(k for k,v in splits.items() if sample in v)
            df=project_registered_telemetry(pd.read_csv(io.BytesIO(z.read(name)),low_memory=False),schema_sha256=schema)
            labels=pd.read_csv(io.BytesIO(z.read(name.replace('/telemetry.csv','/labels.csv'))))
            rel=Path(name[len(dataset_root)+1:]).parts;mission=rel[1] if rel[0]=='nominal_baseline' else missions[rel[1]]
            nominal=dataset_root+'/nominal_baseline/'+mission+'/'+rel[-2]+'/telemetry.csv'
            for end in plan['endpoints'][split]:
                w=df.loc[df.time_s<=end+1e-8].iloc[-20:];x=w[channels].to_numpy(np.float32)
                if x.shape!=(20,len(channels)):raise ValueError('bad window shape')
                times=w.time_s.to_numpy(float)
                if not np.isfinite(times).all() or not (np.diff(times)>0).all():raise ValueError('bad time')
                row=labels.loc[np.isclose(labels.time_s,end,rtol=0,atol=1e-8)]
                if len(row)!=1:raise ValueError('ambiguous endpoint truth')
                row=row.iloc[0];active=int(row.event_active);f=classes.index(str(row.diagnostic_class)) if active else -1
                c=components.index(hierarchy[classes[f]]) if active else -1
                ident={'id':hashlib.sha256((name+'@'+str(end)).encode()).hexdigest()[:24], 'source':name,'nominal':nominal,'endpoint_s':end,
                       'sample_index':sample,'paired_group':mission+':'+str(sample),'valid':bool(np.isfinite(x).all())}
                buckets[split].append((ident,x,active,c,f))
            if (i+1)%100==0:print(i+1,'/',len(files),flush=True)
    train=np.stack([r[1] for r in buckets['train'] if r[0]['valid']]);pre=fit_channel_preprocessor(train,method='robust_clip')
    torch.save({'method':pre.method,**{k:torch.from_numpy(getattr(pre,k)) for k in ('impute','center','scale','lower','upper')},'normalized_clip':pre.normalized_clip},a.output/'preprocessor.pt')
    summary={}
    for split,rows in buckets.items():
        raw=np.stack([r[1] for r in rows]);valid=np.array([r[0]['valid'] for r in rows]);scaled=pre.transform(raw)
        # Invalid rows remain listed. They are never provided to a model or imputed into a scored prediction.
        feature={'x':scaled,'valid':valid,'ids':np.array([r[0]['id'] for r in rows])}
        if split!='test':feature.update(anomaly=np.array([r[2] for r in rows]),component=np.array([r[3] for r in rows]),fault=np.array([r[4] for r in rows]))
        np.savez_compressed(a.output/(split+'_features.npz'),**feature)
        (a.output/(split+'_rows.json')).write_text(json.dumps([r[0] for r in rows],indent=2)+'\n')
        if split=='test':
            (a.output/'test_labels.json').write_text(json.dumps([{'id':r[0]['id'],'anomaly':r[2],'component':r[3],'fault':r[4]} for r in rows],indent=2)+'\n')
        summary[split]={'windows':len(rows),'valid':int(valid.sum()),'invalid':int((~valid).sum()),'normal':sum(r[2]==0 for r in rows),'fault':sum(r[2]==1 for r in rows)}
    receipt={'counts':summary,'group_overlap':0,'preprocess_fit':'valid train windows only','channels_sha256':hashlib.sha256(json.dumps(channels).encode()).hexdigest(),
             'artifacts':{p.name:digest(p) for p in sorted(a.output.iterdir()) if p.is_file()}}
    (a.output/'dataset_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
