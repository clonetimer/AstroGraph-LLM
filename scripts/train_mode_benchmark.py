"""Train a single frozen-comparison worker. Test labels are NEVER read here."""
from __future__ import annotations
import argparse,copy,hashlib,json,time,platform
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from astrograph.evaluation.reference_backbones import make_reference_model,REFERENCES
from astrograph.agents.mission.storage import file_sha

def macro(y,p,n):return f1_score(y,p,labels=list(range(n)),average='macro',zero_division=0)

def weights(y,n):
    count=np.bincount(y,minlength=n).astype(float)
    if (count==0).any():raise ValueError('training class missing')
    w=np.sqrt(count.sum()/count);return torch.tensor(w/w.mean(),dtype=torch.float32)

def infer(model,x,batch=128):
    model.eval();all_=[[],[],[]]
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            out=model(torch.from_numpy(x[start:start+batch]))
            for j,v in enumerate(out):all_[j].append(torch.softmax(v,dim=-1).numpy())
    return [np.concatenate(a) for a in all_]

def validation_score(probs,y):
    a,c,f=y;active=a==1
    return float(np.mean([macro(a,probs[0].argmax(1),2),macro(c[active],probs[1][active].argmax(1),5),macro(f[active],probs[2][active].argmax(1),14)]))

def select_threshold(p,y):
    candidates=np.linspace(.05,.95,19)
    values=[(macro(y,(p>=t).astype(int),2),-abs(t-.5),float(t)) for t in candidates]
    return max(values)[2]

def summary(x):return np.concatenate([x.mean(1),x.std(1),x[:,-1],x[:,-1]-x[:,0]],axis=1)

def export_tcn(model,pre_path,threshold,root,out,seed):
    from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,save_adcs_three_task_torch_package
    from astrograph.data_driven.preprocessing import ChannelPreprocessor
    p=root/'models/adcs_v5_2_2/p2_innovation_v2.pt';base=load_adcs_three_task_torch_package(p,expected_sha256=file_sha(p))
    raw=torch.load(pre_path,weights_only=True)
    base.preprocessor=ChannelPreprocessor(method=raw['method'],**{k:raw[k].numpy() for k in ('impute','center','scale','lower','upper')},normalized_clip=raw['normalized_clip'])
    base.model=model;base.logistic_coefficient=np.zeros(4*len(base.selected_channels));base.logistic_intercept=0.
    base.detection_selection={'detector_deep_blend_alpha':1.,'detection_threshold':threshold,'alarm_policy_mode':'point'}
    base.diagnosis_selection={'hierarchy_blend_alpha':1.,'classification_policy':'unmasked'}
    base.manifest=base.manifest.model_copy(update={'model_id':'controlled-tcn-'+str(seed),'model_version':'post48.dev5.benchmark'})
    return save_adcs_three_task_torch_package(base,out/'normal-candidate.pt')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=list(REFERENCES),required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--threads',type=int,default=1)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.use_deterministic_algorithms(True)
    protocol=json.loads((args.data/'protocol.json').read_text())
    if args.seed not in protocol['seeds'] or args.family not in protocol['families']:raise ValueError('not in frozen protocol')
    prehash=file_sha(args.data/'preprocessor.pt');phash=file_sha(args.data/'protocol.json')
    sources={str(f):file_sha(f) for f in [Path(__file__),Path(__file__).resolve().parents[1]/'src/astrograph/evaluation/reference_backbones.py']}
    receipt={'family':args.family,'seed':args.seed,'protocol_sha256':phash,'preprocessor_sha256':prehash,'source_hashes':sources,
        'torch':torch.__version__,'python':platform.python_version(),'device':'cpu','reference':REFERENCES[args.family],
        'test_labels_read':False,'production_modified':False,'start_unix':time.time(),'thread_count':args.threads}
    (out/'started.json').write_text(json.dumps(receipt,indent=2)+'\n')
    tr=np.load(args.data/'train_features.npz',allow_pickle=False);va=np.load(args.data/'validation_features.npz',allow_pickle=False)
    tv=tr['valid'];vv=va['valid'];x=tr['x'][tv];vx=va['x'][vv]
    y=[tr[k][tv] for k in ('anomaly','component','fault')];vy=[va[k][vv] for k in ('anomaly','component','fault')]
    history=[];start=time.perf_counter()
    if args.family=='hist_gradient_boosting':
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        models=[];features=summary(x);vfeatures=summary(vx)
        for j,target in enumerate(y):
            active=np.ones(len(y[0]),bool) if j==0 else y[0]==1
            m=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=args.seed)
            m.fit(features[active],target[active]);models.append(m)
        probs=[m.predict_proba(vfeatures) for m in models];best=validation_score(probs,vy);best_epoch=None
        joblib.dump(models,out/'models.joblib');params=None
    else:
        model=make_reference_model(args.family,x.shape[2],5,14,x.shape[1]);params=sum(p.numel() for p in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
        lossweights=[weights(y[0],2),weights(y[1][y[0]==1],5),weights(y[2][y[0]==1],14)]
        tensors=[torch.from_numpy(a).long() for a in y];tx=torch.from_numpy(x)
        best=-1.;best_epoch=0;best_state=None;rng=np.random.default_rng(args.seed)
        for epoch in range(1,protocol['epochs']+1):
            model.train();total=0.;order=rng.permutation(len(x));begin=time.perf_counter()
            for k in range(0,len(x),protocol['batch_size']):
                index=order[k:k+protocol['batch_size']];target=[a[index] for a in tensors];logits=model(tx[index]);active=target[0]==1
                loss=torch.nn.functional.cross_entropy(logits[0],target[0],weight=lossweights[0])
                if active.any():
                    loss=loss+torch.nn.functional.cross_entropy(logits[1][active],target[1][active],weight=lossweights[1])+torch.nn.functional.cross_entropy(logits[2][active],target[2][active],weight=lossweights[2])
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(index)
            vprob=infer(model,vx,protocol['batch_size']);score=validation_score(vprob,vy)
            row={'epoch':epoch,'loss':total/len(x),'validation_selection_macro_f1':score,'elapsed_s':time.perf_counter()-begin};history.append(row)
            with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if score>best+1e-12:best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            if epoch%5==0:print(args.family,args.seed,epoch,round(score,5),'sec',round(row['elapsed_s'],1),flush=True)
            if epoch>=protocol['min_epochs'] and epoch-best_epoch>=protocol['patience']:break
        model.load_state_dict(best_state);probs=infer(model,vx,protocol['batch_size'])
        torch.save({'family':args.family,'state_dict':best_state,'protocol_sha256':phash,'seed':args.seed},out/'weights.pt')
    threshold=select_threshold(probs[0][:,1],vy[0]);train_s=time.perf_counter()-start
    np.savez_compressed(out/'validation_probabilities.npz',detector=probs[0],localizer=probs[1],classifier=probs[2],ids=va['ids'][vv])
    # Only features are loaded for test; predictions are persisted before any evaluator joins labels.
    te=np.load(args.data/'test_features.npz',allow_pickle=False);valid=te['valid'];test=te['x'][valid];begin=time.perf_counter()
    if args.family=='hist_gradient_boosting':pred=[m.predict_proba(summary(test)) for m in models]
    else:pred=infer(model,test,protocol['batch_size'])
    infer_s=time.perf_counter()-begin
    np.savez_compressed(out/'test_probabilities.npz',detector=pred[0],localizer=pred[1],classifier=pred[2],ids=te['ids'][valid],threshold=np.array(threshold))
    candidate=None
    if args.family=='residual_tcn':candidate=export_tcn(model,args.data/'preprocessor.pt',threshold,Path(__file__).resolve().parents[1],out,args.seed)
    receipt.update(end_unix=time.time(),training_s=train_s,test_inference_s=infer_s,test_inference_ms_per_window=infer_s/len(test)*1000,
        parameters=params,epochs_completed=len(history),selected_epoch=best_epoch,validation_selection_score=best,
        threshold=threshold,train_valid_windows=len(x),validation_valid_windows=len(vx),test_valid_windows=len(test),
        candidate_sha256=candidate,test_predictions_sha256=file_sha(out/'test_probabilities.npz'))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('FINISHED',args.family,args.seed,train_s,flush=True)
if __name__=='__main__':main()
