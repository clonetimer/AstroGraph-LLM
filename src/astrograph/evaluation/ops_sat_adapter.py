"""Separate raw and official derived-feature OPS-SAT tracks and offline truth."""
from __future__ import annotations
import csv, hashlib, math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from .external_dataset_contract import digest, PARTITIONS

RAW_FIELDS={'channel','timestamp','value','sampling','segment'}
FEATURE_FIELDS=('duration','len','mean','var','std','kurtosis','skew','n_peaks','smooth10_n_peaks','smooth20_n_peaks','diff_peaks','diff2_peaks','diff_var','diff2_var','gaps_squared','len_weighted','var_div_duration','var_div_len')
TRUTH_FIELDS={'label','anomaly','train'}

def finite(value):
    number=float(value)
    if not math.isfinite(number):raise ValueError("NONFINITE_OBSERVATION")
    return number

def validate_online_fields(payload: dict, *, track: str):
    allowed=RAW_FIELDS if track=='raw' else set(FEATURE_FIELDS)|{'segment','channel','sampling'} if track=='feature' else set()
    if not allowed or set(payload)!=allowed:raise ValueError("LABEL_FIELD_OR_UNADMITTED_FIELD_IN_ONLINE_INPUT")

def truth_row(row):
    if row['anomaly'] not in {'0','1'} or row['train'] not in {'0','1'}:raise ValueError("LABEL_SEMANTICS_UNADMITTED")
    # Official modeling_examples uses numeric anomaly, never the auxiliary label.
    # The supplied raw export has label='anomaly' even for numeric nominal rows.
    return {'segment':row['segment'],'anomaly':int(row['anomaly']),'train':int(row['train'])}

def load_feature_track(path: Path):
    online={};truth={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or [])!=set(FEATURE_FIELDS)|{'segment','channel','sampling','anomaly','train'}:raise ValueError("FEATURE_SCHEMA_DRIFT")
        for row in reader:
            sid=row['segment']
            if sid in online:raise ValueError("SEGMENT_ID_DUPLICATE")
            truth[sid]=truth_row(row)
            payload={k:row[k] for k in ('segment','channel','sampling')}
            payload.update({k:finite(row[k]) for k in FEATURE_FIELDS})
            validate_online_fields(payload,track='feature');online[sid]=payload
    return online,truth

def load_raw_track(path: Path):
    online={};truth={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or [])!=RAW_FIELDS|TRUTH_FIELDS:raise ValueError("RAW_SCHEMA_DRIFT")
        for row in reader:
            sid=row['segment']; t=truth_row(row)
            if sid in truth and truth[sid]!=t:raise ValueError("SEGMENT_TRUTH_DRIFT")
            truth[sid]=t
            point={k:row[k] for k in RAW_FIELDS};point['value']=finite(point['value'])
            validate_online_fields(point,track='raw')
            points=online.setdefault(sid,[])
            if points and any(point[k]!=points[0][k] for k in ('channel','sampling')):raise ValueError("SEGMENT_CHANNEL_DRIFT")
            timestamp=datetime.fromisoformat(point['timestamp'].replace('Z','+00:00'))
            if timestamp.tzinfo is None:raise ValueError("TIMESTAMP_ZONE_REQUIRED")
            if points and timestamp<=datetime.fromisoformat(points[-1]['timestamp'].replace('Z','+00:00')):raise ValueError("TIMESTAMP_NOT_INCREASING")
            points.append(point)
    return online,truth

def build_partitions(truth: dict, config: dict):
    rows=[]
    for sid,t in sorted(truth.items()):
        if sid!=t['segment']:raise ValueError("GROUP_ID_MISMATCH")
        if t['train']==0:partition='EXTERNAL_VALIDATION'
        else:
            bucket=int(hashlib.sha256((config['hash_prefix']+sid).encode()).hexdigest()[:8],16)%100
            partition=next(p for upper,p in config['train_partition_upper_bounds'] if bucket<upper)
        rows.append({'group_id':sid,'partition':partition})
    result={'schema_version':'astrograph-ops-sat-partition-manifest.v1','config_sha256':digest(config),'groups':rows}
    verify_partitions(result,truth)
    return result

def verify_partitions(manifest,truth):
    rows=manifest['groups']; ids=[r['group_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(truth):raise ValueError("GROUP_PARTITION_LEAKAGE_OR_MISSING_GROUP")
    for row in rows:
        if row['partition'] not in PARTITIONS:raise ValueError("PARTITION_UNADMITTED")
        if (truth[row['group_id']]['train']==0)!=(row['partition']=='EXTERNAL_VALIDATION'):raise ValueError("OFFICIAL_EXTERNAL_GROUP_MOVED")
    for partition in PARTITIONS:
        labels={truth[r['group_id']]['anomaly'] for r in rows if r['partition']==partition}
        if labels!={0,1}:raise ValueError("PARTITION_CLASS_COVERAGE_INSUFFICIENT:"+partition)

def _raw_segment_score_core(points):
    """Truth-free OPS raw score core shared by governed R21/R22 wrappers."""
    from astrograph.integrations.multidimensional_diagnostics import MultiMetricWindowDetector
    if not points:
        raise ValueError("SEGMENT_BASELINE_OR_WINDOW_INSUFFICIENT")
    for point in points:validate_online_fields(point,track='raw')
    segment_ids={str(point['segment']) for point in points}
    channels={str(point['channel']) for point in points}
    sampling={str(point['sampling']) for point in points}
    if len(segment_ids)!=1 or len(channels)!=1 or len(sampling)!=1:
        raise ValueError("SEGMENT_IDENTITY_DRIFT")
    t0=datetime.fromisoformat(points[0]['timestamp'].replace('Z','+00:00'))
    samples=[{'t':(datetime.fromisoformat(p['timestamp'].replace('Z','+00:00'))-t0).total_seconds(),'values':{'value':p['value']}} for p in points]
    if any(b['t']<=a['t'] for a,b in zip(samples,samples[1:])):
        raise ValueError("TIMESTAMP_NOT_INCREASING")
    if len(samples)<24:raise ValueError("SEGMENT_BASELINE_OR_WINDOW_INSUFFICIENT")
    detector=MultiMetricWindowDetector.from_samples(samples,['value'],minimum_channels=1)
    n=max(8,int(len(samples)*.2));scores=[]
    for end in range(n+4,len(samples)+1,4):
        _,score,_=detector.detect(samples[max(n,end-16):end]);scores.append(score)
    if not scores:raise ValueError("SEGMENT_WINDOW_INSUFFICIENT")
    return max(scores)

def raw_segment_score(points):
    """R21 development wrapper around the truth-free raw score core."""
    from .data_exposure import guard_development
    guard_development('OPTIMIZE', group_ids=[points[0]['segment']] if points else [])
    return _raw_segment_score_core(points)
