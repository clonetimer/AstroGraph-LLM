"""Explicit independence audit for scientific datasets and campaigns.

A new random seed is not automatically an independent generator.  The audit
separates implementation, scenario design, randomization, evaluator, and label
source, and prevents already accessed data from becoming confirmatory.
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator

class IndependenceStatus(str, Enum):
    INDEPENDENT="INDEPENDENT"
    PARTIAL="PARTIAL"
    SHARED="SHARED"
    UNKNOWN="UNKNOWN"

class DatasetIndependenceRecord(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)
    schema_version:str="astrograph-dataset-independence.v1"
    dataset_id:str=Field(min_length=1)
    generator_implementation:IndependenceStatus
    scenario_design:IndependenceStatus
    randomization:IndependenceStatus
    evaluator_implementation:IndependenceStatus
    label_source:IndependenceStatus
    previously_accessed:bool
    hidden_truth_until_prediction_freeze:bool
    notes:list[str]=Field(default_factory=list)
    confirmatory_ready:bool=False

    @model_validator(mode="after")
    def validate_ready(self):
        required=(self.generator_implementation,self.scenario_design,self.evaluator_implementation,self.label_source)
        should=(not self.previously_accessed and self.hidden_truth_until_prediction_freeze and all(x is IndependenceStatus.INDEPENDENT for x in required))
        if self.confirmatory_ready!=should:
            raise ValueError("confirmatory_ready must be derived from independent implementation/scenario/evaluator/label source, unseen status, and hidden truth")
        return self

__all__=["IndependenceStatus","DatasetIndependenceRecord"]
