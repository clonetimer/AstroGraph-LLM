"""Label-free causal local-change mechanism evidence for ADCS temporal localization."""
from __future__ import annotations
from enum import Enum
from functools import cached_property
import math
from typing import Mapping, Sequence
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from astrograph.assurance import canonical_sha256

_SHA256=r"^[0-9a-f]{64}$"; _TRJ=r"^TRJ-[0-9A-F]{20}$"; _GRP=r"^GRP-[0-9A-F]{16}$"

class MechanismBank(str,Enum):
    GYRO="GYRO"; REACTION_WHEEL="REACTION_WHEEL"; STAR_TRACKER="STAR_TRACKER"

BANK_CHANNELS={
    MechanismBank.GYRO:(
        "adcs.innovation.star_gyro_rate_residual_rad_s_x",
        "adcs.innovation.star_gyro_rate_residual_rad_s_y",
        "adcs.innovation.star_gyro_rate_residual_rad_s_z",
        "adcs.innovation.star_gyro_rate_residual_norm_rad_s",
    ),
    MechanismBank.REACTION_WHEEL:(
        "adcs.rw.measured_speed_rad_s_0","adcs.rw.measured_speed_rad_s_1",
        "adcs.rw.measured_speed_rad_s_2","adcs.rw.measured_speed_rad_s_3",
        "adcs.rw.speed_response_residual_nm_0","adcs.rw.speed_response_residual_nm_1",
        "adcs.rw.speed_response_residual_nm_2","adcs.rw.speed_response_residual_nm_3",
        "adcs.rw.speed_response_residual_norm_nm","adcs.rw.speed_sensor_invalid_count",
    ),
    MechanismBank.STAR_TRACKER:(
        "adcs.pointing_error_deg","adcs.innovation.star_sun_direction_deg",
        "adcs.innovation.star_earth_direction_deg","adcs.star_tracker.q0",
        "adcs.star_tracker.q1","adcs.star_tracker.q2","adcs.star_tracker.q3",
        "adcs.star_tracker.valid","adcs.star_tracker.solution_fresh",
    ),
}
MECHANISM_CHANNEL_ORDER=tuple(sorted({c for values in BANK_CHANNELS.values() for c in values}))

class ScoreAggregation(str,Enum):
    MAX="MAX"; Q90="Q90"; Q75="Q75"

class MechanismScoreConfiguration(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-mechanism-score-configuration.v1"
    window_points:int=Field(default=3,ge=2,le=32)
    warmup_points:int=Field(default=31,ge=5)
    channel_floor_multiplier:float=Field(gt=0)
    previous_window_spread_multiplier:float=Field(gt=0)
    aggregation:ScoreAggregation
    runtime_ground_truth_access:bool=False
    @field_validator("channel_floor_multiplier","previous_window_spread_multiplier")
    @classmethod
    def finite(cls,v):
        v=float(v)
        if not math.isfinite(v):raise ValueError("configuration value must be finite")
        return v
    @model_validator(mode="after")
    def safe(self):
        if self.window_points!=3:raise ValueError("B011 formal family fixes window_points=3")
        if self.runtime_ground_truth_access:raise ValueError("runtime truth access forbidden")
        return self
    @computed_field
    @cached_property
    def configuration_sha256(self)->str:
        return canonical_sha256(self.model_dump(mode="json",exclude={"configuration_sha256"}))

class MechanismTemporalPolicy(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-mechanism-temporal-policy.v1"
    policy_id:str
    score_configuration_sha256:str=Field(pattern=_SHA256)
    threshold_multiplier:float=Field(ge=1.0)
    persistence_points:int=Field(ge=2)
    notification_buffer_points:int=Field(default=3,ge=0)
    warmup_points:int=Field(default=31,ge=5)
    development_groups:tuple[int,...]=(3,8)
    excluded_groups:tuple[int,...]=(4,9)
    calibration_groups:tuple[int,...]=(0,1,2,5,6,7)
    runtime_ground_truth_access:bool=False
    @model_validator(mode="after")
    def safe(self):
        if set(self.development_groups)!={3,8} or set(self.excluded_groups)!={4,9}:raise ValueError("group boundary differs")
        if set(self.calibration_groups)&{3,4,8,9}:raise ValueError("calibration overlap")
        if self.runtime_ground_truth_access:raise ValueError("runtime truth access forbidden")
        return self
    @computed_field
    @cached_property
    def policy_sha256(self)->str:
        return canonical_sha256(self.model_dump(mode="json",exclude={"policy_sha256"}))

class FrozenMechanismTrace(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-mechanism-evidence-trace.v1"
    opaque_trajectory_id:str=Field(pattern=_TRJ)
    independent_group_id:str=Field(pattern=_GRP)
    development_group:int
    current_telemetry_sha256:str=Field(pattern=_SHA256)
    score_configuration_sha256:str=Field(pattern=_SHA256)
    time_s:tuple[float,...]=Field(min_length=2)
    bank_scores:Mapping[MechanismBank,tuple[float,...]]
    runtime_ground_truth_access:bool=False
    @model_validator(mode="after")
    def exact(self):
        if self.development_group in {4,9}:raise ValueError("groups 4/9 forbidden")
        if set(self.bank_scores)!=set(MechanismBank):raise ValueError("all banks required")
        if any(len(v)!=len(self.time_s) for v in self.bank_scores.values()):raise ValueError("score length differs")
        if self.runtime_ground_truth_access:raise ValueError("runtime truth access forbidden")
        return self
    @computed_field
    @cached_property
    def trace_sha256(self)->str:
        return canonical_sha256(self.model_dump(mode="json",exclude={"trace_sha256"}))

def candidate_score_configurations():
    return tuple(
        MechanismScoreConfiguration(
            window_points=3,warmup_points=31,channel_floor_multiplier=floor,
            previous_window_spread_multiplier=spread,aggregation=agg,
        )
        for floor in (0.03,0.1,0.3,1.0)
        for spread in (1.0,3.0)
        for agg in (ScoreAggregation.MAX,ScoreAggregation.Q90,ScoreAggregation.Q75)
    )

def candidate_policies():
    out=[]
    for config in candidate_score_configurations():
        for multiplier in (1.0,1.25,1.5,2.0,3.0):
            for persistence in (2,3,4,5,8):
                seed={"configuration_sha256":config.configuration_sha256,
                      "threshold_multiplier":multiplier,"persistence_points":persistence}
                out.append(MechanismTemporalPolicy(
                    policy_id="MEP-"+canonical_sha256(seed)[:20].upper(),
                    score_configuration_sha256=config.configuration_sha256,
                    threshold_multiplier=multiplier,persistence_points=persistence,
                ))
    return tuple(out)

def local_change_components(matrix:Sequence[Sequence[float|None]],window_points:int):
    x=np.asarray(matrix,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2*window_points:raise ValueError("matrix too short")
    windows=sliding_window_view(x,window_shape=window_points,axis=0)
    med=np.nanmedian(windows,axis=2)
    mad=np.nanmedian(np.abs(windows-med[:,:,None]),axis=2)*1.4826
    usable=med.shape[0]-window_points; start=2*window_points-1
    diff=np.zeros_like(x);spread=np.zeros_like(x)
    diff[start:start+usable]=np.abs(med[window_points:window_points+usable]-med[:usable])
    spread[start:start+usable]=mad[:usable]
    diff[~np.isfinite(diff)]=0;spread[~np.isfinite(spread)]=0
    return diff,spread

def bank_scores_from_components(difference,previous_spread,*,channel_order,channel_scale,configuration):
    order=tuple(channel_order);scale=np.asarray(channel_scale,dtype=np.float64)
    if difference.shape!=previous_spread.shape or difference.shape[1]!=len(order):raise ValueError("shape differs")
    den=np.maximum(previous_spread*configuration.previous_window_spread_multiplier,
                   scale[None,:]*configuration.channel_floor_multiplier)
    z=np.divide(difference,den,out=np.zeros_like(difference),where=np.isfinite(difference)&np.isfinite(den)&(den>0))
    z[:configuration.warmup_points]=0
    pos={c:i for i,c in enumerate(order)}
    q={ScoreAggregation.MAX:None,ScoreAggregation.Q90:.9,ScoreAggregation.Q75:.75}[configuration.aggregation]
    out={}
    for bank,channels in BANK_CHANNELS.items():
        values=z[:,[pos[c] for c in channels]]
        out[bank]=np.max(values,axis=1) if q is None else np.quantile(values,q,axis=1)
    return out

def local_change_bank_scores(matrix,*,channel_order,channel_scale,configuration):
    diff,spread=local_change_components(matrix,configuration.window_points)
    return bank_scores_from_components(diff,spread,channel_order=channel_order,
                                       channel_scale=channel_scale,configuration=configuration)

def select_scores(bank_scores,policy,anchors):
    runs={b:0 for b in MechanismBank};n=len(next(iter(bank_scores.values())))
    for i in range(policy.warmup_points,n):
        for bank in MechanismBank:
            hit=float(bank_scores[bank][i])>=float(anchors[bank])*policy.threshold_multiplier
            runs[bank]=runs[bank]+1 if hit else 0
            if runs[bank]>=policy.persistence_points:
                return {"status":"SELECTED","selected_index":i,"selected_time_s":float(i),
                        "notification_available_time_s":float(i+policy.notification_buffer_points),
                        "selected_bank":bank.value}
    return {"status":"NO_ALARM","selected_index":None,"selected_time_s":None,
            "notification_available_time_s":None,"selected_bank":None}
