"""Read-only manifests for external validation datasets.

The adapters intentionally avoid loading bundled trained models and avoid pickle
deserialization.  They materialize statistical grouping and contamination
boundaries before any experiment is run.
"""
from __future__ import annotations
import csv, hashlib, io, json, re, zipfile
from pathlib import Path
from typing import Any


def _file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def _member_sha(z:zipfile.ZipFile,name:str)->str:
    h=hashlib.sha256()
    with z.open(name) as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def build_smap_msl_manifest(path:Path)->dict[str,Any]:
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        label_name=next((n for n in names if n.endswith("labeled_anomalies.csv")),None)
        if label_name is None: raise ValueError("labeled_anomalies.csv missing")
        rows=list(csv.DictReader(io.TextIOWrapper(z.open(label_name),encoding="utf-8")))
        raw_train=sorted(n for n in names if "/train/" in n and n.endswith(".npy"))
        raw_test=sorted(n for n in names if "/test/" in n and n.endswith(".npy"))
        excluded=sorted(n for n in names if any(x in n for x in ("/models/","/y_hat/","/smoothed_errors/")))
        channels=sorted(set(Path(n).stem for n in raw_test))
        return {"schema_version":"astrograph-smap-msl-adapter-manifest.v1","archive_sha256":_file_sha(path),"label_member":label_name,"label_member_sha256":_member_sha(z,label_name),"channel_count":len(channels),"channels":channels,"label_row_count":len(rows),"raw_train_members":raw_train,"raw_test_members":raw_test,"excluded_contamination_members":excluded,"independent_unit":"CHANNEL_OR_PREDECLARED_EVENT_CLUSTER","claim_boundary":"Anomaly/localization external task only; not ADCS fault identification or RCA."}

_DXC_RE=re.compile(r"Exp_(\d+)([^.]*)\.(txt|mat|scn)$",re.I)
def build_dxc_manifest(paths:list[Path])->dict[str,Any]:
    experiments={}; archives=[]
    for path in paths:
        archives.append({"name":path.name,"sha256":_file_sha(path)})
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                m=_DXC_RE.fullmatch(Path(name).name)
                if not m: continue
                exp=m.group(1); variant=m.group(2); ext=m.group(3).lower()
                key=f"{path.stem}:{exp}"
                item=experiments.setdefault(key,{"archive":path.name,"experiment_id":exp,"representations":[],"variants":set()})
                item["representations"].append({"member":name,"format":ext})
                item["variants"].add(variant)
    clean=[]
    for key,item in sorted(experiments.items()):
        item["representations"]=sorted(item["representations"],key=lambda x:x["member"]); item["variants"]=sorted(item["variants"]); clean.append(item)
    return {"schema_version":"astrograph-dxc09-adapt-adapter-manifest.v1","archives":archives,"experiment_group_count":len(clean),"experiments":clean,"independent_unit":"EXPERIMENT_ID_WITH_ARCHIVE_VARIANT_PROVENANCE","claim_boundary":"TXT/MAT/SCN representations are not independent repeats; EPS testbed results do not establish ADCS flight validity."}


def build_esa_quarantine_plan(path:Path)->dict[str,Any]:
    nested=[]
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".zip") and "/channels/" in name:
                data=z.read(name)
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as nz:
                        members=nz.namelist(); dangerous=[]
                        for n in members:
                            low=n.lower()
                            head=nz.read(n)[:16]
                            looks_pickle=(len(head)>=2 and head[0]==0x80 and 0x02 <= head[1] <= 0x05) or head.startswith(b"(dp") or head.startswith(b"ccopy_reg")
                            if low.endswith((".pkl",".pickle",".joblib",".dill")) or looks_pickle:
                                dangerous.append(n)
                except zipfile.BadZipFile:
                    members=[]; dangerous=["INVALID_NESTED_ZIP"]
                nested.append({"member":name,"nested_member_count":len(members),"unsafe_serialized_members":dangerous})
    return {"schema_version":"astrograph-esa-quarantine-plan.v1","archive_sha256":_file_sha(path),"channel_archives":nested,"normal_runtime_deserialization_allowed":False,"required_conversion":"ONE_SHOT_NETWORK_ISOLATED_PICKLE_TO_PARQUET_OR_ARROW","claim_boundary":"No physical RCA may be inferred from anonymized channels without external semantic mapping."}

__all__=["build_smap_msl_manifest","build_dxc_manifest","build_esa_quarantine_plan"]
