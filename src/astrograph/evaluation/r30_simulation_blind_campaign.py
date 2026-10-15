"""R30 blind synthetic campaign for reduced-order simulation diagnosis.

The generator and diagnoser are deliberately separated by contracts.  The
prediction function receives only observed traces plus a fixed candidate
universe.  Hidden fault truth is joined only after prediction bytes are frozen.
The campaign is a protocol/generalization smoke under simulator/model mismatch;
it is not DXC external validation or flight evidence.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.simulation.adapt_reduced_order import (
    ADAPTLightBulbReducedOrderProvider,
    ADAPTResistiveLoadSpec,
    build_branch_boundary_condition,
)
from astrograph.simulation.diagnostic_provider import (
    SimulationHypothesisCandidate,
    build_simulation_trace,
    compare_counterfactual_diagnoses,
)


def _digest(value: Any) -> str:
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class R30BlindCase(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)
    case_id:str
    observed_time_s:tuple[float,...]
    observed_voltage_v:tuple[float,...]
    observed_current_a:tuple[float,...]
    pre_current_a:float=Field(gt=0.0)
    pre_voltage_v:float=Field(gt=0.0)
    hidden_fault_ids:tuple[str,...]
    generator_variant:str
    case_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def integrity(self):
        payload={k:v for k,v in self.model_dump(mode='json').items() if k!='case_sha256'}
        if _digest(payload)!=self.case_sha256: raise ValueError('R30_BLIND_CASE_SHA_MISMATCH')
        return self


class R30BlindPrediction(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)
    case_id:str
    predicted_fault_ids:tuple[str,...]
    best_diagnosis_id:str
    best_rmse:float=Field(ge=0.0)
    runner_semantics:str='TRUTH_FREE_FIXED_CANDIDATE_REDUCED_ORDER_SIMULATION'


class R30BlindCampaignReport(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)
    schema_version:str='astrograph-r30-synthetic-blind-simulation-campaign.v1'
    seed:int
    case_count:int=Field(ge=1)
    prediction_set_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    truth_set_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    exact_set_accuracy:float=Field(ge=0.0,le=1.0)
    fault_element_precision:float=Field(ge=0.0,le=1.0)
    fault_element_recall:float=Field(ge=0.0,le=1.0)
    fault_element_f1:float=Field(ge=0.0,le=1.0)
    single_fault_exact_accuracy:float=Field(ge=0.0,le=1.0)
    double_fault_exact_accuracy:float=Field(ge=0.0,le=1.0)
    status:str='SYNTHETIC_BLIND_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION'
    claim_boundary:str=(
        'Predictions are frozen before hidden synthetic truth is joined. Generator uses perturbed load parameters and measurement noise, '
        'but shares the same broad resistive-load physics family as the diagnoser. This is a simulator/model-mismatch protocol smoke, '
        'not independent DXC/mission validation, SOTA evidence, calibrated probability, HIL or flight qualification.'
    )


def _case(*,case_id:str,truth:tuple[str,...],rng:random.Random,loads:dict[str,float])->R30BlindCase:
    dt=0.5; onset=15.0
    times=tuple(round(i*dt,6) for i in range(81))
    # Generator deliberately differs from the diagnosis model: actual lamp power
    # varies ±12%, background contains a weak constant-current term, and sensors
    # include noise/bias.  These nuisance parameters are not exposed to diagnosis.
    actual={fid:p*rng.uniform(0.88,1.12) for fid,p in loads.items()}
    background_g=rng.uniform(0.018,0.035) # siemens-like A/V contribution
    background_const_a=rng.uniform(0.15,0.35)
    voltage_bias=rng.uniform(-0.6,0.6)
    current_bias=rng.uniform(-0.015,0.015)
    voltages=[];currents=[]
    for t in times:
        v=120.0 + 2.5*math.sin(0.19*t+rng.uniform(-0.02,0.02)) + rng.gauss(0,0.25) + voltage_bias
        active_power=sum(p for fid,p in actual.items() if not (t>=onset and fid in truth))
        cur=background_const_a + background_g*v + active_power/max(v,80.0) + current_bias + rng.gauss(0,0.012)
        voltages.append(v);currents.append(cur)
    pre=[i for i,t in enumerate(times) if onset-8.0<=t<onset]
    pre_i=sum(currents[i] for i in pre)/len(pre);pre_v=sum(voltages[i] for i in pre)/len(pre)
    post=[i for i,t in enumerate(times) if t>=onset]
    payload={
        'case_id':case_id,
        'observed_time_s':tuple(times[i] for i in post),
        'observed_voltage_v':tuple(voltages[i] for i in post),
        'observed_current_a':tuple(currents[i] for i in post),
        'pre_current_a':pre_i,'pre_voltage_v':pre_v,'hidden_fault_ids':truth,
        'generator_variant':'PERTURBED_RESISTIVE_LOAD_WITH_BACKGROUND_CONDUCTANCE_CONSTANT_CURRENT_AND_SENSOR_NOISE_V1',
    }
    return R30BlindCase(**payload,case_sha256=_digest(payload))


def generate_blind_cases(seed:int=3001)->tuple[R30BlindCase,...]:
    rng=random.Random(seed)
    loads={'LGT_A:FailedOff':25.0,'LGT_B:FailedOff':55.0,'LGT_C:FailedOff':60.0}
    truths=[(x,) for x in loads]+list(itertools.combinations(sorted(loads),2))
    cases=[]
    for rep in range(8):
        for truth in truths:
            cases.append(_case(case_id=f'R30-{rep:02d}-{len(cases):03d}',truth=tuple(sorted(truth)),rng=rng,loads=loads))
    return tuple(cases)


def _candidate_universe()->tuple[SimulationHypothesisCandidate,...]:
    ids=('LGT_A:FailedOff','LGT_B:FailedOff','LGT_C:FailedOff')
    rows=[]
    for size in (1,2):
        for faults in itertools.combinations(ids,size):
            rows.append(SimulationHypothesisCandidate(diagnosis_id='SIM-'+ '+'.join(x.split(':')[0] for x in faults),fault_ids=tuple(faults)))
    return tuple(rows)


def predict_cases_truth_free(cases:Sequence[R30BlindCase])->tuple[R30BlindPrediction,...]:
    loads=(
        ADAPTResistiveLoadSpec(fault_id='LGT_A:FailedOff',component_id='LGT_A',component_type='LightBulb25W',rated_power_w=25.0,current_sensor_id='I_BRANCH',voltage_sensor_id='V_BUS'),
        ADAPTResistiveLoadSpec(fault_id='LGT_B:FailedOff',component_id='LGT_B',component_type='LightBulb55W',rated_power_w=55.0,current_sensor_id='I_BRANCH',voltage_sensor_id='V_BUS'),
        ADAPTResistiveLoadSpec(fault_id='LGT_C:FailedOff',component_id='LGT_C',component_type='LightBulb60W',rated_power_w=60.0,current_sensor_id='I_BRANCH',voltage_sensor_id='V_BUS'),
    )
    candidates=_candidate_universe();out=[]
    for case in cases:
        # IMPORTANT: hidden_fault_ids is intentionally not consumed below.
        boundary=build_branch_boundary_condition(branch_id='SYNTH_BRANCH',current_sensor_id='I_BRANCH',voltage_sensor_id='V_BUS',time_s=case.observed_time_s,observed_voltage_v=case.observed_voltage_v,pre_current_a=case.pre_current_a,pre_voltage_v=case.pre_voltage_v,source_artifact_sha256=case.case_sha256)
        provider=ADAPTLightBulbReducedOrderProvider(boundary=boundary,load_specs=loads)
        observed=build_simulation_trace(trace_id='OBS-'+case.case_id,time_s=case.observed_time_s,channels={'I_BRANCH':case.observed_current_a},source_artifact_sha256=case.case_sha256,provenance_kind='R30_SYNTHETIC_BLIND_OBSERVED_TRACE')
        review=compare_counterfactual_diagnoses(observed=observed,candidates=candidates,provider=provider,channel_scales={'I_BRANCH':0.05},channels=['I_BRANCH'])
        best=next(x for x in review.comparisons if x.diagnosis_id==review.best_diagnosis_id)
        out.append(R30BlindPrediction(case_id=case.case_id,predicted_fault_ids=best.fault_ids,best_diagnosis_id=best.diagnosis_id,best_rmse=best.aggregate_normalized_rmse))
    return tuple(out)


def evaluate_frozen_predictions(*,cases:Sequence[R30BlindCase],predictions:Sequence[R30BlindPrediction],seed:int)->R30BlindCampaignReport:
    if [x.case_id for x in cases] != [x.case_id for x in predictions]: raise ValueError('R30_CASE_PREDICTION_IDENTITY_MISMATCH')
    pred_payload=[x.model_dump(mode='json') for x in predictions]
    truth_payload=[{'case_id':x.case_id,'hidden_fault_ids':x.hidden_fault_ids,'case_sha256':x.case_sha256} for x in cases]
    tp=fp=fn=exact=single_n=single_ok=double_n=double_ok=0
    for case,pred in zip(cases,predictions):
        t=set(case.hidden_fault_ids);p=set(pred.predicted_fault_ids)
        tp+=len(t&p);fp+=len(p-t);fn+=len(t-p);ok=t==p;exact+=int(ok)
        if len(t)==1: single_n+=1;single_ok+=int(ok)
        else: double_n+=1;double_ok+=int(ok)
    precision=tp/(tp+fp) if tp+fp else 0.0;recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return R30BlindCampaignReport(seed=seed,case_count=len(cases),prediction_set_sha256=_digest(pred_payload),truth_set_sha256=_digest(truth_payload),exact_set_accuracy=exact/len(cases),fault_element_precision=precision,fault_element_recall=recall,fault_element_f1=f1,single_fault_exact_accuracy=single_ok/single_n,double_fault_exact_accuracy=double_ok/double_n)


def run_blind_campaign(seed:int=3001)->tuple[R30BlindCampaignReport,tuple[R30BlindPrediction,...],tuple[R30BlindCase,...]]:
    cases=generate_blind_cases(seed)
    predictions=predict_cases_truth_free(cases)
    report=evaluate_frozen_predictions(cases=cases,predictions=predictions,seed=seed)
    return report,predictions,cases

__all__=['R30BlindCampaignReport','R30BlindCase','R30BlindPrediction','evaluate_frozen_predictions','generate_blind_cases','predict_cases_truth_free','run_blind_campaign']
