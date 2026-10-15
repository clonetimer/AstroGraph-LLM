"""Fail-closed contracts for a temporal-only sealed ADCS campaign.

This module deliberately separates a frozen temporal confirmatory design from
Route-A identification/KG claims.  It contains no model, fusion, KG, or
rejection-policy placeholders.  A campaign is executable only when the exact
Basilisk generator, fault mapping, frozen temporal stack, prediction runner,
materializer, evaluator, statistics plan, and one-time authorization are bound
by SHA-256.
"""
from __future__ import annotations
import hashlib, json, math, re
from enum import Enum
from functools import cached_property
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

_SHA=re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _binomial_tail_ge(successes:int,trials:int,p:float)->float:
    return sum(math.comb(trials,k)*(p**k)*((1-p)**(trials-k)) for k in range(successes,trials+1))


def _binomial_cdf_le(successes:int,trials:int,p:float)->float:
    return sum(math.comb(trials,k)*(p**k)*((1-p)**(trials-k)) for k in range(0,successes+1))


def clopper_pearson_lower_one_sided(successes:int,trials:int,alpha:float=.05)->float:
    if not 0<=successes<=trials or trials<1: raise ValueError("invalid binomial counts")
    if successes==0:return 0.0
    lo,hi=0.0,1.0
    for _ in range(120):
        mid=(lo+hi)/2
        if _binomial_tail_ge(successes,trials,mid)<alpha: lo=mid
        else: hi=mid
    return (lo+hi)/2


def clopper_pearson_upper_one_sided(successes:int,trials:int,alpha:float=.05)->float:
    if not 0<=successes<=trials or trials<1: raise ValueError("invalid binomial counts")
    if successes==trials:return 1.0
    lo,hi=0.0,1.0
    for _ in range(120):
        mid=(lo+hi)/2
        if _binomial_cdf_le(successes,trials,mid)>alpha: lo=mid
        else: hi=mid
    return (lo+hi)/2


class CampaignState(str,Enum):
    DRAFT="DRAFT"
    FROZEN_RUNTIME_BLOCKED="FROZEN_RUNTIME_BLOCKED"
    ARMED="ARMED"
    STARTED="STARTED"
    DATA_GENERATED="DATA_GENERATED"
    PREDICTIONS_FROZEN="PREDICTIONS_FROZEN"
    EVALUATED="EVALUATED"
    FAILED="FAILED"


class FaultConditionMapping(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    diagnostic_class:str=Field(min_length=1)
    generator_condition_id:str=Field(min_length=1)
    semantic_relation:str=Field(pattern=r"^(EXACT|MECHANISM_EQUIVALENT)$")
    rationale:str=Field(min_length=1)
    runtime_exposed:bool=False
    @model_validator(mode="after")
    def safe(self):
        if self.runtime_exposed:raise ValueError("fault mapping cannot be exposed to runtime")
        return self


class TemporalCampaignPrecisionPlan(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-temporal-campaign-precision-plan.v1"
    independent_group_count:int=Field(default=30,ge=10)
    mission_strata:tuple[str,...]
    groups_per_mission:int=Field(default=10,ge=5)
    fault_class_count:int=Field(default=14,ge=1)
    nominal_run_count:int=Field(default=30,ge=1)
    fault_run_count:int=Field(default=420,ge=1)
    total_run_count:int=Field(default=450,ge=1)
    alpha:float=Field(default=.05,gt=0,lt=.5)
    maximum_nominal_false_alarm_group_rate:float=Field(default=.10,ge=0,le=1)
    minimum_per_class_timely_rate:float=Field(default=.80,ge=0,le=1)
    minimum_per_class_successes:int=Field(default=28,ge=1)
    minimum_group_macro_lcb:float=Field(default=.90,ge=0,le=1)
    bootstrap_repetitions:int=Field(default=20000,ge=5000)
    bootstrap_seed:int=Field(default=14001,ge=0)
    independent_unit:str="MONTE_CARLO_PAIR_GROUP"
    @model_validator(mode="after")
    def exact(self):
        if self.independent_group_count!=len(self.mission_strata)*self.groups_per_mission:
            raise ValueError("group count does not equal mission strata x groups_per_mission")
        if self.nominal_run_count!=self.independent_group_count:
            raise ValueError("one nominal run is required per independent group")
        if self.fault_run_count!=self.independent_group_count*self.fault_class_count:
            raise ValueError("fault run count differs")
        if self.total_run_count!=self.nominal_run_count+self.fault_run_count:
            raise ValueError("total run count differs")
        if clopper_pearson_upper_one_sided(0,self.nominal_run_count,self.alpha)>self.maximum_nominal_false_alarm_group_rate:
            raise ValueError("group count is insufficient for zero-false-alarm precision target")
        if clopper_pearson_lower_one_sided(self.minimum_per_class_successes,self.independent_group_count,self.alpha)<self.minimum_per_class_timely_rate:
            raise ValueError("per-class success threshold does not meet lower-bound target")
        return self
    @computed_field
    @cached_property
    def zero_false_alarm_upper_bound(self)->float:
        return clopper_pearson_upper_one_sided(0,self.nominal_run_count,self.alpha)
    @computed_field
    @cached_property
    def all_group_success_lower_bound(self)->float:
        return clopper_pearson_lower_one_sided(self.independent_group_count,self.independent_group_count,self.alpha)
    @computed_field
    @cached_property
    def minimum_per_class_lower_bound(self)->float:
        return clopper_pearson_lower_one_sided(self.minimum_per_class_successes,self.independent_group_count,self.alpha)
    @computed_field
    @cached_property
    def plan_sha256(self)->str:
        return canonical_sha256(self.model_dump(mode="json",exclude={"plan_sha256"}))


class TemporalConfirmatoryAnalysisPlan(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-temporal-confirmatory-analysis-plan.v1"
    plan_id:str
    status:str=Field(pattern=r"^FROZEN$")
    campaign_id:str
    precision_plan:TemporalCampaignPrecisionPlan
    primary_endpoint:str="GROUP_MACRO_NOTIFY_WITHIN_20S"
    co_primary_safety_endpoints:tuple[str,...]=("ZERO_PRE_ONSET_SELECTIONS","NOMINAL_FALSE_ALARM_EXACT_UPPER_BOUND")
    timely_definition:str="selected_time>=active_start AND selected_time<=active_end AND notification_time-active_start<=20s"
    group_success_definition:str="fault timely rate >= 13/14 AND zero pre-onset AND nominal trajectory has no alarm"
    confirmatory_success_rule:str="all co-primary safety endpoints pass; cluster-bootstrap lower bound of group-macro timely rate >=0.90; every class has at least 28/30 timely trajectories"
    cluster_bootstrap_unit:str="MONTE_CARLO_PAIR_GROUP"
    multiplicity_policy:str="HIERARCHICAL_CO_PRIMARY_THEN_PER_CLASS"
    no_post_result_policy_change:bool=True
    groups_4_9_forbidden:bool=True
    one_time_evaluation:bool=True
    @model_validator(mode="after")
    def safe(self):
        if not all((self.no_post_result_policy_change,self.groups_4_9_forbidden,self.one_time_evaluation)):
            raise ValueError("confirmatory safety flags must all be true")
        return self
    @computed_field
    @cached_property
    def analysis_plan_sha256(self)->str:
        return canonical_sha256(self.model_dump(mode="json",exclude={"analysis_plan_sha256"}))


class SealedTemporalCampaignPreregistration(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-sealed-temporal-campaign-preregistration.v2"
    campaign_id:str=Field(min_length=1)
    status:CampaignState
    b013_source_zip_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    temporal_stack_freeze_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    sat_platform_archive_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    generator_entrypoint_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    generator_plan_preview_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    seed_plan_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    fault_mapping_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    precision_plan_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    analysis_plan_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    materializer_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    execution_gate_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    prediction_runner_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    temporal_calibration_bundle_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    runtime_admission_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    expected_bsk_distribution:str="bsk==2.11.0"
    expected_bsk_wheel_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    base_seed:int=Field(ge=0)
    seed_commitment_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    mission_strata:tuple[str,...]
    generator_conditions:tuple[str,...]
    independent_group_count:int=30
    total_run_count:int=450
    hidden_truth:bool=True
    telemetry_outcomes_accessed:bool=False
    design_preview_accessed:bool=True
    one_time_generation:bool=True
    one_time_evaluation:bool=True
    previously_accessed_campaign_ids:tuple[str,...]=()
    claim_boundary:str="Simulator-level temporal confirmation only; no fault-identification, KG, external, HIL, flight, safety or qualification claim."
    @model_validator(mode="after")
    def safe(self):
        if self.campaign_id in set(self.previously_accessed_campaign_ids):raise ValueError("campaign ID was previously accessed")
        if self.telemetry_outcomes_accessed:raise ValueError("campaign telemetry outcomes must remain untouched at preregistration")
        if not all((self.hidden_truth,self.one_time_generation,self.one_time_evaluation)):
            raise ValueError("sealed campaign requires hidden truth and one-time generation/evaluation")
        if self.status not in {CampaignState.FROZEN_RUNTIME_BLOCKED,CampaignState.ARMED}:
            raise ValueError("preregistration must be frozen or armed")
        return self
    @computed_field
    @cached_property
    def preregistration_sha256(self)->str:
        return canonical_sha256(self.model_dump(mode="json",exclude={"preregistration_sha256"}))


class OneTimeExecutionLedger(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-one-time-temporal-campaign-ledger.v1"
    campaign_id:str
    preregistration_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    authorization_token_sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    state:CampaignState=CampaignState.FROZEN_RUNTIME_BLOCKED
    generation_consumed:bool=False
    evaluation_consumed:bool=False
    blocked_attempt_count:int=Field(default=0,ge=0)
    execution_attempt_id:str|None=None
    generator_exit_code:int|None=None
    generated_dataset_manifest_sha256:str|None=Field(default=None,pattern=r"^[0-9a-f]{64}$")
    prediction_bundle_sha256:str|None=Field(default=None,pattern=r"^[0-9a-f]{64}$")
    truth_table_sha256:str|None=Field(default=None,pattern=r"^[0-9a-f]{64}$")
    result_sha256:str|None=Field(default=None,pattern=r"^[0-9a-f]{64}$")
    claim_boundary:str="A blocked runtime-admission attempt does not consume the one-time authorization."
    @model_validator(mode="after")
    def safe(self):
        if self.state==CampaignState.FROZEN_RUNTIME_BLOCKED and (self.generation_consumed or self.evaluation_consumed):
            raise ValueError("blocked campaign cannot consume authorization")
        if self.evaluation_consumed and not self.generation_consumed:
            raise ValueError("evaluation cannot precede generation")
        if self.generated_dataset_manifest_sha256 is not None and not self.generation_consumed:
            raise ValueError("dataset manifest cannot exist before generation")
        if self.result_sha256 is not None and not self.evaluation_consumed:
            raise ValueError("result cannot exist before evaluation")
        return self
    @computed_field
    @cached_property
    def ledger_sha256(self)->str:
        return canonical_sha256(self.model_dump(mode="json",exclude={"ledger_sha256"}))

__all__=["CampaignState","FaultConditionMapping","TemporalCampaignPrecisionPlan","TemporalConfirmatoryAnalysisPlan","SealedTemporalCampaignPreregistration","OneTimeExecutionLedger","clopper_pearson_lower_one_sided","clopper_pearson_upper_one_sided"]
