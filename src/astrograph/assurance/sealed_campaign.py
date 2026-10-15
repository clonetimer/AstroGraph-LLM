"""Fail-closed preregistration receipt for future untouched simulator campaigns."""
from __future__ import annotations
import hashlib, json, re
from pydantic import BaseModel, ConfigDict, Field, model_validator
_SHA=re.compile(r"^[0-9a-f]{64}$")

class SealedCampaignPreregistration(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)
    schema_version:str="astrograph-sealed-campaign-preregistration.v1"
    campaign_id:str=Field(min_length=1)
    source_tree_sha256:str
    generator_sha256:str
    model_package_sha256:str
    temporal_policy_sha256:str
    evidence_package_sha256:str
    fusion_policy_sha256:str
    rejection_policy_sha256:str
    evaluator_sha256:str
    statistics_plan_sha256:str
    hidden_truth:bool=True
    one_time_evaluation:bool=True
    previously_accessed_campaign_ids:list[str]=Field(default_factory=list)
    status:str=Field(pattern=r"^(DRAFT|FROZEN)$")
    preregistration_sha256:str|None=None

    @model_validator(mode="after")
    def validate_all(self):
        for name in ("source_tree_sha256","generator_sha256","model_package_sha256","temporal_policy_sha256","evidence_package_sha256","fusion_policy_sha256","rejection_policy_sha256","evaluator_sha256","statistics_plan_sha256"):
            if not _SHA.fullmatch(getattr(self,name)): raise ValueError(f"{name} must be lowercase SHA-256")
        if self.campaign_id in set(self.previously_accessed_campaign_ids): raise ValueError("sealed campaign ID was previously accessed")
        if self.status=="FROZEN" and (not self.hidden_truth or not self.one_time_evaluation): raise ValueError("frozen confirmatory campaign requires hidden truth and one-time evaluation")
        exp=self.compute_sha256()
        if self.preregistration_sha256 is not None and self.preregistration_sha256!=exp: raise ValueError("preregistration_sha256 mismatch")
        return self
    def compute_sha256(self):
        p=self.model_dump(mode="json",exclude={"preregistration_sha256"})
        return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()

SealedCampaignPreregisration = SealedCampaignPreregistration

__all__=["SealedCampaignPreregistration","SealedCampaignPreregisration"]
