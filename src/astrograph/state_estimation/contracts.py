from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.physics import DynamicsContract
from astrograph.semantics import SemanticTelemetryFrame


class ParameterEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_id: str
    value: float
    canonical_unit: str
    lower_confidence: float
    upper_confidence: float
    physical_lower_bound: float
    physical_upper_bound: float

    @model_validator(mode="after")
    def ordered_intervals(self) -> "ParameterEstimate":
        if not self.physical_lower_bound <= self.value <= self.physical_upper_bound:
            raise ValueError("parameter estimate violates physical bounds")
        if not self.physical_lower_bound <= self.lower_confidence <= self.value:
            raise ValueError("invalid lower confidence bound")
        if not self.value <= self.upper_confidence <= self.physical_upper_bound:
            raise ValueError("invalid upper confidence bound")
        return self


class StateTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: str
    canonical_unit: str
    values: list[float]
    lower_confidence: list[float]
    upper_confidence: list[float]

    @model_validator(mode="after")
    def aligned(self) -> "StateTrajectory":
        if not len(self.values) == len(self.lower_confidence) == len(self.upper_confidence):
            raise ValueError("state trajectory and confidence arrays must align")
        return self


class ResidualSeries(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    residual_id: str
    canonical_unit: str
    values: list[float]
    root_mean_square: float = Field(ge=0.0)
    mean: float


class UncertaintySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal[
        "linearized_residual_covariance",
        "split_conformal_scaled_residual",
    ] = "linearized_residual_covariance"
    residual_standard_deviation: float = Field(ge=0.0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    calibrated: bool = False
    calibration_version: str = "NOT_CALIBRATED"
    state_interval_multiplier: float = Field(default=1.96, ge=0.0)
    parameter_interval_multiplier: float = Field(default=1.96, ge=0.0)


class ArchetypeUncertaintyCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_archetype_id: str = Field(min_length=1)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    state_interval_multiplier: float = Field(ge=0.0)
    parameter_interval_multiplier: float = Field(ge=0.0)
    calibration_trajectory_count: int = Field(gt=0)
    calibration_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conservation_method: Literal["GLS_CHANGE_POINT_PARAMETRIC_BOOTSTRAP"] = (
        "GLS_CHANGE_POINT_PARAMETRIC_BOOTSTRAP"
    )
    conservation_false_alarm_rate: float = Field(default=0.05, gt=0.0, lt=1.0)
    conservation_bootstrap_replicates: int = Field(default=256, ge=100)
    conservation_bootstrap_seed: int = Field(default=918273645, ge=0)
    conservation_mc_confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    conservation_change_point_fractions: list[float] = Field(min_length=3)
    conservation_development_episode_scores: list[float] = Field(min_length=10)

    @model_validator(mode="after")
    def valid_conservation_configuration(self) -> "ArchetypeUncertaintyCalibration":
        scores = self.conservation_development_episode_scores
        if any(score < 0.0 for score in scores):
            raise ValueError("conservation calibration scores must be non-negative")
        if scores != sorted(scores):
            raise ValueError("conservation calibration scores must be sorted")
        fractions = self.conservation_change_point_fractions
        if fractions != sorted(set(fractions)):
            raise ValueError("conservation change-point fractions must be sorted and unique")
        if any(fraction < 0.1 or fraction > 0.9 for fraction in fractions):
            raise ValueError("conservation change-point fractions must stay in [0.1, 0.9]")
        return self


class UncertaintyCalibrationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-uncertainty-calibration.v1"
    calibration_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    estimator_id: str = Field(min_length=1)
    estimator_version: str = Field(min_length=1)
    method: Literal["split_conformal_scaled_residual"] = "split_conformal_scaled_residual"
    formal_hidden_truth_used: Literal[False] = False
    synthetic_reference_used: Literal[True] = True
    random_seed: int = Field(ge=0)
    archetypes: list[ArchetypeUncertaintyCalibration] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_archetypes(self) -> "UncertaintyCalibrationProfile":
        identifiers = [item.component_archetype_id for item in self.archetypes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate uncertainty calibration archetypes")
        return self

    def for_archetype(self, component_archetype_id: str) -> ArchetypeUncertaintyCalibration | None:
        return next(
            (
                item
                for item in self.archetypes
                if item.component_archetype_id == component_archetype_id
            ),
            None,
        )


class ConstraintViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    index: int | None = None
    magnitude: float = Field(ge=0.0)
    limit: float = Field(ge=0.0)


class ConservationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal[
        "UNCALIBRATED_RAW_ONLY",
        "GLS_CHANGE_POINT_PARAMETRIC_BOOTSTRAP",
    ]
    raw_physical_tolerance: float = Field(gt=0.0)
    raw_violation_count: int = Field(ge=0)
    raw_violation_fraction: float = Field(ge=0.0, le=1.0)
    episode_score: float = Field(ge=0.0)
    empirical_p_value: float = Field(gt=0.0, le=1.0)
    p_value_upper_confidence: float = Field(gt=0.0, le=1.0)
    false_alarm_rate: float = Field(gt=0.0, lt=1.0)
    bootstrap_replicates: int = Field(ge=0)
    statistically_significant: bool
    calibration_version: str


class ModelReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    estimator_id: str
    estimator_version: str
    dynamics_contract_id: str
    dynamics_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identification_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    residual_model: Literal["NONE"] = "NONE"
    uncertainty_calibration_version: str = "NOT_CALIBRATED"
    uncertainty_calibration_sha256: str = Field(
        default="NONE", pattern=r"^(?:NONE|[0-9a-f]{64})$"
    )
    conservation_calibration_version: str = "NOT_CALIBRATED"
    production_slot: Literal["component_state_estimator"] = "component_state_estimator"
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False


class StateEstimateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-state-estimate.v1"
    component_archetype_id: str
    timestamps_s: list[float]
    states: list[StateTrajectory]
    parameters: list[ParameterEstimate]
    residuals: list[ResidualSeries]
    uncertainty: UncertaintySummary
    ood_score: float = Field(ge=0.0, le=1.0)
    constraint_violations: list[ConstraintViolation]
    conservation_assessment: ConservationAssessment
    receipt: ModelReceipt
    claim_boundary: str = (
        "Gray-box state and parameter estimate from public semantic telemetry; it is not a fault label, "
        "causal proof, safety decision, or formal acceptance result."
    )


class StateEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        frame: SemanticTelemetryFrame,
        contract: DynamicsContract,
    ) -> StateEstimateResult:
        raise NotImplementedError


__all__ = [
    "ConstraintViolation",
    "ConservationAssessment",
    "ArchetypeUncertaintyCalibration",
    "ModelReceipt",
    "ParameterEstimate",
    "ResidualSeries",
    "StateEstimateResult",
    "StateEstimator",
    "StateTrajectory",
    "UncertaintySummary",
    "UncertaintyCalibrationProfile",
]
