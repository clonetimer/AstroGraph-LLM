from __future__ import annotations
from typing import Sequence
from pydantic import BaseModel, ConfigDict, Field
from astrograph.fusion import BiRRFConfig, FaultRanking, fuse_rankings

class FusionSensitivityScenario(BaseModel):
    model_config=ConfigDict(extra='forbid', frozen=True)
    scenario_id:str=Field(min_length=1)
    config:BiRRFConfig

class FusionSensitivityReport(BaseModel):
    model_config=ConfigDict(extra='forbid')
    schema_version:str='astrograph-birrf-sensitivity-audit.v1'
    baseline_scenario_id:str
    scenario_count:int
    baseline_top_fault_id:str|None
    top1_stability_rate:float
    top1_changed_scenarios:list[str]
    scenarios:list[dict]
    calibration_status:str='NOT_CALIBRATED_PERFORMANCE_STUDY'
    claim_boundary:str='Parameter sensitivity audit only; no accuracy, probability calibration, or qualification claim.'

def audit_birrf_sensitivity(data:FaultRanking, forward:FaultRanking, backward:FaultRanking, scenarios:Sequence[FusionSensitivityScenario], *, baseline_scenario_id:str)->FusionSensitivityReport:
    from .data_exposure import guard_development
    guard_development('TUNE')
    if not scenarios: raise ValueError('at least one sensitivity scenario is required')
    ids=[x.scenario_id for x in scenarios]
    if len(ids)!=len(set(ids)): raise ValueError('sensitivity scenario IDs must be unique')
    if baseline_scenario_id not in ids: raise ValueError('baseline scenario is absent')
    rows=[]
    for s in scenarios:
        result=fuse_rankings(data,forward,backward,config=s.config)
        order=[x.fault_id for x in sorted(result.candidates,key=lambda x:(x.final_rank,x.fault_id))]
        rows.append({'scenario_id':s.scenario_id,'top_fault_id':result.top_fault_id,'rank_order':order,
                     'configured_weights':result.metadata['configured_branch_weights'],
                     'effective_weights':result.metadata['effective_branch_weights'],
                     'smoothing':{'data':s.config.data_smoothing,'forward':s.config.forward_smoothing,'backward':s.config.backward_smoothing}})
    baseline=next(x for x in rows if x['scenario_id']==baseline_scenario_id)
    changed=[x['scenario_id'] for x in rows if x['top_fault_id']!=baseline['top_fault_id']]
    return FusionSensitivityReport(baseline_scenario_id=baseline_scenario_id,scenario_count=len(rows),baseline_top_fault_id=baseline['top_fault_id'],
        top1_stability_rate=(len(rows)-len(changed))/len(rows),top1_changed_scenarios=changed,scenarios=rows)

__all__=['FusionSensitivityScenario','FusionSensitivityReport','audit_birrf_sensitivity']
