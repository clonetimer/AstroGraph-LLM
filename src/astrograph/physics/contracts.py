from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.semantics import DimensionVector


class ParameterRole(str, Enum):
    FIXED = "FIXED"
    LEARNABLE = "LEARNABLE"


class StateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: str = Field(min_length=1)
    quantity_kind_id: str = Field(min_length=1)
    canonical_unit: str = Field(min_length=1)
    lower_bound: float
    upper_bound: float
    observation_port_role_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_bounds(self) -> "StateSpec":
        if self.lower_bound >= self.upper_bound:
            raise ValueError("state lower_bound must be less than upper_bound")
        return self


class InputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_id: str = Field(min_length=1)
    port_role_id: str = Field(min_length=1)
    quantity_kind_id: str = Field(min_length=1)
    canonical_unit: str = Field(min_length=1)
    required: bool = True


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_id: str = Field(min_length=1)
    dimension: DimensionVector
    canonical_unit: str = Field(min_length=1)
    role: ParameterRole
    lower_bound: float
    upper_bound: float
    initial_value: float

    @model_validator(mode="after")
    def valid_interval(self) -> "ParameterSpec":
        if self.lower_bound >= self.upper_bound:
            raise ValueError("parameter lower_bound must be less than upper_bound")
        if not self.lower_bound <= self.initial_value <= self.upper_bound:
            raise ValueError("parameter initial_value must be within physical bounds")
        return self


class ConservationLaw(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    law_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    input_ids: list[str] = Field(min_length=1)
    residual_tolerance: float = Field(gt=0.0)
    description: str = ""


class EventMode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    guard_expression: str = Field(min_length=1)
    transition: str = Field(min_length=1)


class ObservabilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_state_ids: list[str] = Field(min_length=1)
    minimum_samples: int = Field(ge=3)
    minimum_input_variance: float = Field(ge=0.0)


class DynamicsContract(BaseModel):
    """Auditable, dataset-neutral contract for one component dynamics family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-dynamics-contract.v1"
    contract_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    component_archetype_id: str = Field(min_length=1)
    equation_family: str = Field(min_length=1)
    states: list[StateSpec] = Field(min_length=1)
    inputs: list[InputSpec] = Field(min_length=1)
    parameters: list[ParameterSpec] = Field(min_length=1)
    conservation_laws: list[ConservationLaw] = Field(min_length=1)
    event_modes: list[EventMode] = Field(default_factory=list)
    observability: ObservabilitySpec
    stability_statement: str = Field(min_length=1)

    @model_validator(mode="after")
    def internally_consistent(self) -> "DynamicsContract":
        collections = {
            "state": [item.state_id for item in self.states],
            "input": [item.input_id for item in self.inputs],
            "parameter": [item.parameter_id for item in self.parameters],
            "event": [item.event_id for item in self.event_modes],
        }
        for label, identifiers in collections.items():
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate {label} identifiers")
        state_ids = set(collections["state"])
        input_ids = set(collections["input"])
        if not set(self.observability.observed_state_ids) <= state_ids:
            raise ValueError("observability references an unknown state")
        for law in self.conservation_laws:
            if law.state_id not in state_ids or not set(law.input_ids) <= input_ids:
                raise ValueError("conservation law references an unknown state or input")
        return self

    def parameter(self, parameter_id: str) -> ParameterSpec:
        for item in self.parameters:
            if item.parameter_id == parameter_id:
                return item
        raise KeyError(f"unknown parameter: {parameter_id}")


__all__ = [
    "ConservationLaw",
    "DynamicsContract",
    "EventMode",
    "InputSpec",
    "ObservabilitySpec",
    "ParameterRole",
    "ParameterSpec",
    "StateSpec",
]
