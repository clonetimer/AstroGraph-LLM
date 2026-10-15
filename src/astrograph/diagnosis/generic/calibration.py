from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.state_estimation import StateEstimateResult

from .contracts import FailureMechanism, OpenSetDecision


MECHANISM_FEATURE_NAMES = (
    "raw_signed_mean_over_rms",
    "raw_late_early_shift_over_rms",
    "raw_linear_slope_over_rms",
    "raw_lag1_correlation",
    "raw_peak_over_rms",
    "raw_positive_fraction",
    "whitened_signed_mean_over_rms",
    "whitened_late_early_shift_over_rms",
    "whitened_linear_slope_over_rms",
    "whitened_lag1_correlation",
    "whitened_peak_over_rms",
    "whitened_positive_fraction",
)


class MechanismFeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-mechanism-feature-vector.v1"] = (
        "astrograph-mechanism-feature-vector.v1"
    )
    feature_names: list[str]
    values: list[float]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False

    @model_validator(mode="after")
    def valid_fixed_schema(self) -> "MechanismFeatureVector":
        if tuple(self.feature_names) != MECHANISM_FEATURE_NAMES:
            raise ValueError("mechanism feature schema does not match the frozen order")
        if len(self.values) != len(MECHANISM_FEATURE_NAMES):
            raise ValueError("mechanism feature vector has the wrong dimension")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("mechanism feature vector must be finite")
        return self


class MechanismCalibrationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-mechanism-calibration.v1"] = (
        "astrograph-mechanism-calibration.v1"
    )
    calibration_id: str = Field(min_length=1)
    calibration_version: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_family: Literal["L2_REGULARIZED_MULTINOMIAL_LOGISTIC_REGRESSION"]
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_names: list[str]
    class_order: list[FailureMechanism]
    feature_mean: list[float]
    feature_scale: list[float]
    coefficients: list[list[float]]
    intercepts: list[float]
    temperature: float = Field(gt=0.0)
    class_centroids: list[list[float]]
    class_distance_thresholds: list[float]
    open_set_alpha: float = Field(gt=0.0, lt=1.0)
    development_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_case_count: int = Field(gt=0)
    calibration_case_count: int = Field(gt=0)
    formal_hidden_truth_used: Literal[False] = False
    synthetic_controlled_reference_used: Literal[True] = True

    @model_validator(mode="after")
    def aligned_model_arrays(self) -> "MechanismCalibrationArtifact":
        dimension = len(MECHANISM_FEATURE_NAMES)
        class_count = len(self.class_order)
        if self.feature_names != list(MECHANISM_FEATURE_NAMES):
            raise ValueError("calibration feature schema does not match the frozen order")
        if class_count < 2 or len(set(self.class_order)) != class_count:
            raise ValueError("mechanism classes must be unique")
        if len(self.feature_mean) != dimension or len(self.feature_scale) != dimension:
            raise ValueError("feature normalization arrays have the wrong dimension")
        if any(value <= 0.0 or not math.isfinite(value) for value in self.feature_scale):
            raise ValueError("feature scales must be finite and positive")
        matrix_fields = (self.coefficients, self.class_centroids)
        if any(len(matrix) != class_count for matrix in matrix_fields):
            raise ValueError("class matrix row count does not match class order")
        if any(len(row) != dimension for matrix in matrix_fields for row in matrix):
            raise ValueError("class matrix has the wrong feature dimension")
        if len(self.intercepts) != class_count:
            raise ValueError("intercept count does not match class order")
        if len(self.class_distance_thresholds) != class_count:
            raise ValueError("distance threshold count does not match class order")
        if any(value < 0.0 for value in self.class_distance_thresholds):
            raise ValueError("distance thresholds cannot be negative")
        return self


class CalibratedMechanismPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: FailureMechanism
    probability: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[FailureMechanism, float]
    open_set_decision: OpenSetDecision
    class_distance: float = Field(ge=0.0)
    class_distance_threshold: float = Field(ge=0.0)


def _series_features(values: list[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 4 or not bool(np.all(np.isfinite(array))):
        raise ValueError("residual feature extraction requires at least four finite values")
    rms = max(float(np.sqrt(np.mean(np.square(array)))), 1.0e-12)
    quarter = max(len(array) // 4, 1)
    centered_index = np.linspace(-1.0, 1.0, len(array))
    slope = float(np.linalg.lstsq(
        np.column_stack([centered_index, np.ones(len(array))]), array, rcond=None
    )[0][0])
    standard_deviation = float(np.std(array))
    lag_one = (
        float(np.corrcoef(array[:-1], array[1:])[0, 1])
        if standard_deviation > 1.0e-12
        else 0.0
    )
    if not math.isfinite(lag_one):
        lag_one = 0.0
    return (
        float(np.mean(array)) / rms,
        float(np.mean(array[-quarter:]) - np.mean(array[:quarter])) / rms,
        slope / rms,
        lag_one,
        float(np.max(np.abs(array))) / rms,
        float(np.mean(array > 0.0)),
    )


def extract_mechanism_features(
    state_estimate: StateEstimateResult,
) -> MechanismFeatureVector:
    residuals = {item.residual_id: item.values for item in state_estimate.residuals}
    try:
        raw = residuals["balance_equation"]
        whitened = residuals["balance_innovation_whitened"]
    except KeyError as exc:
        raise ValueError(f"required residual is missing: {exc.args[0]}") from exc
    values = [*_series_features(raw), *_series_features(whitened)]
    return MechanismFeatureVector(
        feature_names=list(MECHANISM_FEATURE_NAMES),
        values=values,
        input_sha256=state_estimate.receipt.identification_data_sha256,
    )


class CalibratedMechanismModel:
    def __init__(self, artifact: MechanismCalibrationArtifact) -> None:
        self.artifact = artifact
        self._mean = np.asarray(artifact.feature_mean, dtype=float)
        self._scale = np.asarray(artifact.feature_scale, dtype=float)
        self._coefficients = np.asarray(artifact.coefficients, dtype=float)
        self._intercepts = np.asarray(artifact.intercepts, dtype=float)
        self._centroids = np.asarray(artifact.class_centroids, dtype=float)
        self._distance_thresholds = np.asarray(
            artifact.class_distance_thresholds, dtype=float
        )

    def predict(
        self, features: MechanismFeatureVector
    ) -> CalibratedMechanismPrediction:
        values = (np.asarray(features.values, dtype=float) - self._mean) / self._scale
        logits = (self._coefficients @ values + self._intercepts) / self.artifact.temperature
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        index = int(np.argmax(probabilities))
        distance = float(np.linalg.norm(values - self._centroids[index]))
        threshold = float(self._distance_thresholds[index])
        return CalibratedMechanismPrediction(
            mechanism_id=self.artifact.class_order[index],
            probability=float(probabilities[index]),
            class_probabilities={
                mechanism: float(probabilities[class_index])
                for class_index, mechanism in enumerate(self.artifact.class_order)
            },
            open_set_decision=(
                OpenSetDecision.IN_DISTRIBUTION
                if distance <= threshold
                else OpenSetDecision.OOD
            ),
            class_distance=distance,
            class_distance_threshold=threshold,
        )


__all__ = [
    "CalibratedMechanismModel",
    "CalibratedMechanismPrediction",
    "MECHANISM_FEATURE_NAMES",
    "MechanismCalibrationArtifact",
    "MechanismFeatureVector",
    "extract_mechanism_features",
]
