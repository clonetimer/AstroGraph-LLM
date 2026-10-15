"""Offline three-task candidate trainer using the existing shared residual TCN.

Training is a separately approved task. JSON + weights-only state_dict is NOT a
production ADCS package; promotion requires calibration and independent evaluation.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .storage import file_sha, sha

class TrainingManifest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    schema_version: Literal["astrograph-agent-training.v1"]="astrograph-agent-training.v1"
    data_path: str
    data_sha256: str=Field(pattern=r"^[0-9a-f]{64}$")
    channel_names: list[str]=Field(min_length=1,max_length=256)
    channel_units: list[str]
    fault_ids: list[str]=Field(min_length=1,max_length=128)
    component_ids: list[str]=Field(min_length=1,max_length=64)
    fault_to_component: dict[str,str]
    epochs: int=Field(default=5,ge=1,le=100)
    width: int=Field(default=16,ge=4,le=128)
    batch_size: int=Field(default=32,ge=1,le=512)
    learning_rate: float=Field(default=0.001,gt=0,le=0.1)
    seed: int=20260913

    @model_validator(mode="after")
    def dimensions(self):
        if len(self.channel_units)!=len(self.channel_names):raise ValueError("channel unit count mismatch")
        for values in (self.channel_names,self.component_ids,self.fault_ids):
            if len(values)!=len(set(values)) or any(not v.strip() for v in values):raise ValueError("empty or duplicate identity")
        if set(self.fault_to_component)!=set(self.fault_ids) or not set(self.fault_to_component.values())<=set(self.component_ids):
            raise ValueError("complete fault-to-component mapping required")
        if any(x.lower().startswith(("label","truth","fault_id","target")) for x in self.channel_names):
            raise ValueError("label columns cannot be model inputs")
        return self


def validate_dataset(manifest, path):
    if file_sha(path)!=manifest.data_sha256:raise ValueError("training dataset hash mismatch")
    expected={f"{p}_{s}" for p in ("train","validation") for s in ("x","anomaly","component","fault","group")}
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist())>256_000_000:raise ValueError("training archive exceeds expanded-size budget")
    with np.load(path,allow_pickle=False) as loaded:
        if set(loaded.files)!=expected:raise ValueError("training NPZ must contain exactly train/validation arrays; no hidden test inputs")
        data={k:loaded[k] for k in loaded.files}
    group_sets=[]
    mapping=np.array([manifest.component_ids.index(manifest.fault_to_component[f]) for f in manifest.fault_ids])
    for p in ("train","validation"):
        x=data[p+"_x"]
        if x.ndim!=3 or x.shape[0]<2 or x.shape[1]<4 or x.shape[2]!=len(manifest.channel_names) or x.size>8_000_000:
            raise ValueError("expected bounded [N,T,C] features")
        if not np.isfinite(x).all():raise ValueError("training features contain nonfinite observations; no silent imputation")
        for s in ("anomaly","component","fault","group"):
            if data[p+"_"+s].shape!=(len(x),):raise ValueError("label/group shape mismatch")
        a,c,f=(data[p+"_"+s] for s in ("anomaly","component","fault"))
        if any(not np.issubdtype(v.dtype,np.integer) for v in (a,c,f)):raise ValueError("integer task labels required")
        active=a==1
        if not set(a.tolist())<={0,1}:raise ValueError("binary anomaly labels required")
        if not set(f[active].tolist())<=set(range(len(mapping))) or not set(c[active].tolist())<=set(range(len(manifest.component_ids))):
            raise ValueError("unknown fault/component label")
        if (c[~active]!=-1).any() or (f[~active]!=-1).any() or not np.array_equal(c[active],mapping[f[active]]):
            raise ValueError("task label hierarchy mismatch")
        groups=data[p+"_group"]
        if groups.dtype.kind not in {"U","S"} or any(not str(g).strip() for g in groups):raise ValueError("explicit source groups required")
        group_sets.append(set(groups.tolist()))
    if group_sets[0]&group_sets[1]:raise ValueError("train/validation source-group leakage")
    if data["train_x"].shape[1:]!=data["validation_x"].shape[1:]:raise ValueError("window shape mismatch")
    if set(data["train_anomaly"].tolist())!={0,1} or set(data["train_fault"][data["train_anomaly"]==1].tolist())!=set(range(len(manifest.fault_ids))):
        raise ValueError("training data must cover normal and every configured fault")
    return data


def train_candidate(manifest_path,output):
    import torch
    from torch import nn
    from astrograph.data_driven.three_task_tcn import make_shared_residual_tcn,prepare_torch_device
    manifest_path=Path(manifest_path)
    manifest=TrainingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    data_path=Path(manifest.data_path)
    if not data_path.is_absolute():data_path=manifest_path.parent/data_path
    data=validate_dataset(manifest,data_path)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);prepare_torch_device("cpu",manifest.seed)
    x=data["train_x"].astype("float32")
    mean=x.mean(axis=(0,1));scale=x.std(axis=(0,1));scale[scale<1e-6]=1
    model=make_shared_residual_tcn(x.shape[2],len(manifest.component_ids),len(manifest.fault_ids),width=manifest.width)
    initial=sha({k:v.tolist() for k,v in model.state_dict().items()})
    optimizer=torch.optim.Adam(model.parameters(),lr=manifest.learning_rate)
    tx=torch.tensor((x-mean)/scale);ta=torch.tensor(data["train_anomaly"],dtype=torch.long)
    tc=torch.tensor(data["train_component"],dtype=torch.long);tf=torch.tensor(data["train_fault"],dtype=torch.long)
    losses=[]
    for _ in range(manifest.epochs):
        model.train();total=0
        for ids in torch.randperm(len(tx)).split(manifest.batch_size):
            logits=model(tx[ids]);active=ta[ids]==1
            loss=nn.functional.cross_entropy(logits[0],ta[ids])
            if active.any():
                loss=loss+nn.functional.cross_entropy(logits[1][active],tc[ids][active])+nn.functional.cross_entropy(logits[2][active],tf[ids][active])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=float(loss.detach())*len(ids)
        losses.append(total/len(tx))
    model.eval()
    with torch.no_grad():
        pred=model(torch.tensor((data["validation_x"].astype("float32")-mean)/scale))
        probabilities=[v.softmax(-1).numpy() for v in pred]
    from sklearn.metrics import f1_score,confusion_matrix
    metrics={}
    for i,(name,count) in enumerate((("detection",2),("localization",len(manifest.component_ids)),("classification",len(manifest.fault_ids)))):
        mask=np.ones(len(data["validation_x"]),dtype=bool) if i==0 else data["validation_anomaly"]==1
        truth=data["validation_"+("anomaly","component","fault")[i]][mask]
        predicted=probabilities[i][mask].argmax(-1)
        metrics[name]={"n":len(truth),"macro_f1":float(f1_score(truth,predicted,labels=list(range(count)),average="macro",zero_division=0)),
            "confusion_matrix":confusion_matrix(truth,predicted,labels=list(range(count))).tolist(),"scope":"INDEPENDENT_HEAD_ON_VALIDATION_NOT_SYSTEM_FUSION"}
    weights=out/"weights.pt";torch.save(model.state_dict(),weights)
    package={"schema_version":"astrograph-agent-tcn-candidate.v1","architecture":"existing_shared_residual_tcn",
        "width":manifest.width,"window_size":x.shape[1],"channel_names":manifest.channel_names,"channel_units":manifest.channel_units,
        "fault_ids":manifest.fault_ids,"component_ids":manifest.component_ids,"fault_to_component":manifest.fault_to_component,
        "mean":mean.tolist(),"scale":scale.tolist(),"weights_sha256":file_sha(weights),"training_manifest_sha256":file_sha(manifest_path),
        "training_data_sha256":manifest.data_sha256,"status":"REVIEW_REQUIRED","calibrated":False,"production_admitted":False,
        "initial_state_sha256":initial,"final_state_sha256":sha({k:v.tolist() for k,v in model.state_dict().items()})}
    (out/"manifest.json").write_text(json.dumps(package,ensure_ascii=False,indent=2), encoding='utf-8')
    report={"status":"REVIEW_REQUIRED","epochs":manifest.epochs,"losses":losses,"validation_metrics":metrics,
        "weights_changed":package["initial_state_sha256"]!=package["final_state_sha256"],"active_model_modified":False,
        "candidate_manifest":str(out/"manifest.json"),"candidate_manifest_sha256":file_sha(out/"manifest.json"),"group_overlap":0,
        "test_partition_used":False,"calibration_performed":False}
    (out/"training_report.json").write_text(json.dumps(report,indent=2), encoding='utf-8')
    return report


def infer_candidate(manifest_path,features_path):
    import torch
    from astrograph.data_driven.three_task_tcn import make_shared_residual_tcn
    manifest_path=Path(manifest_path);m=json.loads(manifest_path.read_text(encoding="utf-8"));weights=manifest_path.parent/"weights.pt"
    if m.get("schema_version")!="astrograph-agent-tcn-candidate.v1" or file_sha(weights)!=m["weights_sha256"]:
        raise ValueError("unrecognized or changed candidate package")
    with np.load(features_path,allow_pickle=False) as data:
        if set(data.files)!={"x","channel_names","channel_units"}:raise ValueError("inference must be truth-free and channel-bound")
        x=data["x"].astype("float32")
        if data["channel_names"].tolist()!=m["channel_names"] or data["channel_units"].tolist()!=m["channel_units"]:
            raise ValueError("inference channel order/unit mismatch")
    if x.ndim!=3 or x.shape[1:]!=(m["window_size"],len(m["channel_names"])) or x.size>8_000_000 or not np.isfinite(x).all():
        raise ValueError("invalid inference window")
    if not (4<=m["width"]<=128 and 4<=m["window_size"]<=10000 and 1<=len(m["fault_ids"])<=128
            and 1<=len(m["component_ids"])<=64 and 1<=len(m["channel_names"])<=256):
        raise ValueError("candidate dimensions outside allowed limits")
    if len(m["mean"])!=x.shape[2] or len(m["scale"])!=x.shape[2] or not np.isfinite(m["mean"]).all() or not np.isfinite(m["scale"]).all() or min(m["scale"])<=0:
        raise ValueError("invalid candidate normalization")
    torch.set_num_threads(1)
    model=make_shared_residual_tcn(x.shape[2],len(m["component_ids"]),len(m["fault_ids"]),width=m["width"])
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True));model.eval()
    with torch.no_grad():logits=model(torch.tensor((x-np.array(m["mean"],dtype="float32"))/np.array(m["scale"],dtype="float32")))
    return {"status":"SUCCEEDED","model_status":"OFFLINE_CANDIDATE_NOT_PRODUCTION","input_sha256":file_sha(Path(features_path)),
        "model_sha256":m["weights_sha256"],"score_semantics":"UNCALIBRATED_SOFTMAX",
        "heads":{name:{"classes":classes,"probabilities":v.softmax(-1).tolist()} for name,classes,v in zip(
            ["detection","localization","classification"],[["NORMAL","ANOMALY"],m["component_ids"],m["fault_ids"]],logits,strict=True)},
        "fusion_or_diagnostic_admission_performed":False}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("operation",choices=["train","infer"]);p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True);p.add_argument("--features")
    args=p.parse_args()
    if args.operation=="train":
        result=train_candidate(args.manifest,args.output)
        print(json.dumps(result))
    else:
        result=infer_candidate(args.manifest,args.features)
        Path(args.output).write_text(json.dumps(result), encoding='utf-8');print(json.dumps(result))
