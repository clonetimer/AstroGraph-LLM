"""Outcome-blind operational stress plan for telemetry robustness evaluation."""
from __future__ import annotations
import hashlib,json,random
from pydantic import BaseModel,ConfigDict,Field,model_validator

class StressCase(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    case_id:str
    perturbation:str=Field(pattern=r"^(CHANNEL_DROPOUT|ADDITIVE_NOISE|TIME_JITTER|FIXED_DELAY|MISSING_BLOCK)$")
    parameters:dict[str,float|int|str|list[str]]

class OperationalStressPlan(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version:str="astrograph-operational-stress-plan.v1"
    plan_id:str
    seed:int
    cases:list[StressCase]
    truth_used_for_design:bool=False
    selected_on_groups:list[str]=Field(default_factory=list)
    forbidden_groups:list[str]=Field(default_factory=lambda:["4","9"])
    plan_sha256:str|None=None
    @model_validator(mode="after")
    def valid(self):
        if self.truth_used_for_design: raise ValueError("stress design must be outcome-blind")
        if set(self.selected_on_groups)&set(self.forbidden_groups): raise ValueError("post-hoc groups cannot select stress settings")
        ids=[c.case_id for c in self.cases]
        if len(ids)!=len(set(ids)): raise ValueError("duplicate stress case")
        exp=self.compute_sha256()
        if self.plan_sha256 is not None and self.plan_sha256!=exp: raise ValueError("plan_sha256 mismatch")
        return self
    def compute_sha256(self):
        p=self.model_dump(mode="json",exclude={"plan_sha256"})
        return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def default_outcome_blind_stress_plan(seed:int=20260829)->OperationalStressPlan:
    cases=[
      StressCase(case_id="DROP_ONE_CHANNEL",perturbation="CHANNEL_DROPOUT",parameters={"count":1}),
      StressCase(case_id="DROP_THREE_CHANNELS",perturbation="CHANNEL_DROPOUT",parameters={"count":3}),
      StressCase(case_id="NOISE_0P5SIGMA",perturbation="ADDITIVE_NOISE",parameters={"relative_train_sigma":0.5}),
      StressCase(case_id="NOISE_1P0SIGMA",perturbation="ADDITIVE_NOISE",parameters={"relative_train_sigma":1.0}),
      StressCase(case_id="JITTER_0P2S",perturbation="TIME_JITTER",parameters={"max_abs_s":0.2}),
      StressCase(case_id="DELAY_1S",perturbation="FIXED_DELAY",parameters={"delay_s":1.0}),
      StressCase(case_id="MISSING_5S",perturbation="MISSING_BLOCK",parameters={"duration_s":5.0}),
    ]
    return OperationalStressPlan(plan_id="ROUTE_A_OPERATIONAL_STRESS_V1",seed=seed,cases=cases,selected_on_groups=["3","8"])

__all__=["StressCase","OperationalStressPlan","default_outcome_blind_stress_plan"]
