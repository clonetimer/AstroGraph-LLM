from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.physics import DynamicsContract
from astrograph.semantics import SemanticTelemetryFrame

from .contracts import FailureMechanism, OpenSetDecision


SEGMENTED_FEATURE_NAMES = (
    "selected_split_fraction",
    "pre_fit_r2",
    "post_decomposition_r2",
    "parameter_shift_norm",
    "parameter_shift_alignment",
    "parameter_shift_primary_signed",
    "constant_disturbance_signed",
    "ramp_disturbance_signed",
    "post_residual_rms_ratio",
    "post_residual_mean_over_scale",
    "post_residual_slope_over_scale",
    "impulse_l1_concentration",
    "onset_window_energy_fraction",
    "lag1_autocorrelation",
    "max_periodic_autocorrelation_lag_2_20",
    "sign_change_fraction",
    "active_fraction_two_mad",
    "longest_active_run_fraction",
    "state_plateau_fraction",
    "state_jump_over_pre_scale",
    "input_residual_correlation",
    "selected_change_score_log1p",
)
CHANGE_FRACTIONS = tuple(float(value) for value in np.linspace(0.2, 0.8, 13))
EPSILON = 1.0e-12


class SegmentedMechanismFeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-segmented-mechanism-feature-vector.v1"] = (
        "astrograph-segmented-mechanism-feature-vector.v1"
    )
    feature_names: list[str]
    values: list[float]
    anomaly_score: float = Field(ge=0.0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False

    @model_validator(mode="after")
    def valid_fixed_schema(self) -> "SegmentedMechanismFeatureVector":
        if tuple(self.feature_names) != SEGMENTED_FEATURE_NAMES:
            raise ValueError("segmented mechanism feature schema does not match the frozen order")
        if len(self.values) != len(SEGMENTED_FEATURE_NAMES):
            raise ValueError("segmented mechanism feature vector has the wrong dimension")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("segmented mechanism feature vector must be finite")
        if not math.isfinite(self.anomaly_score):
            raise ValueError("segmented anomaly score must be finite")
        return self


class SegmentedMechanismCalibrationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-segmented-mechanism-calibration.v1"] = (
        "astrograph-segmented-mechanism-calibration.v1"
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
    anomaly_threshold: float = Field(ge=0.0)
    class_centers: list[list[float]]
    class_inverse_covariances: list[list[list[float]]]
    class_distance_thresholds: list[float]
    open_set_alpha: float = Field(gt=0.0, lt=1.0)
    anomaly_false_alarm_alpha: float = Field(gt=0.0, lt=1.0)
    covariance_shrinkage: float = Field(ge=0.0, le=1.0)
    covariance_eigenvalue_floor: float = Field(gt=0.0)
    development_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normal_calibration_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_case_count: int = Field(gt=0)
    calibration_case_count: int = Field(gt=0)
    normal_calibration_case_count: int = Field(gt=0)
    formal_hidden_truth_used: Literal[False] = False
    synthetic_controlled_reference_used: Literal[True] = True

    @model_validator(mode="after")
    def aligned_model_arrays(self) -> "SegmentedMechanismCalibrationArtifact":
        dimension = len(SEGMENTED_FEATURE_NAMES)
        class_count = len(self.class_order)
        if self.feature_names != list(SEGMENTED_FEATURE_NAMES):
            raise ValueError("segmented calibration feature schema differs from the frozen order")
        if class_count < 2 or len(set(self.class_order)) != class_count:
            raise ValueError("segmented mechanism classes must be unique")
        if len(self.feature_mean) != dimension or len(self.feature_scale) != dimension:
            raise ValueError("segmented feature normalization arrays have the wrong dimension")
        if any(value <= 0.0 or not math.isfinite(value) for value in self.feature_scale):
            raise ValueError("segmented feature scales must be finite and positive")
        if len(self.coefficients) != class_count or any(
            len(row) != dimension for row in self.coefficients
        ):
            raise ValueError("logistic coefficient matrix is not aligned")
        if len(self.intercepts) != class_count:
            raise ValueError("logistic intercept count is not aligned")
        if len(self.class_centers) != class_count or any(
            len(row) != dimension for row in self.class_centers
        ):
            raise ValueError("open-set class centers are not aligned")
        if len(self.class_inverse_covariances) != class_count or any(
            len(matrix) != dimension
            or any(len(row) != dimension for row in matrix)
            for matrix in self.class_inverse_covariances
        ):
            raise ValueError("open-set inverse covariance matrices are not aligned")
        if len(self.class_distance_thresholds) != class_count or any(
            value < 0.0 for value in self.class_distance_thresholds
        ):
            raise ValueError("open-set distance thresholds are not aligned")
        return self


class SegmentedMechanismPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anomalous: bool
    anomaly_score: float = Field(ge=0.0)
    anomaly_threshold: float = Field(ge=0.0)
    mechanism_id: FailureMechanism
    probability: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[FailureMechanism, float]
    open_set_decision: OpenSetDecision
    class_distance: float = Field(ge=0.0)
    class_distance_threshold: float = Field(ge=0.0)


def _signal(
    frame: SemanticTelemetryFrame, contract: DynamicsContract, port_role_id: str
) -> np.ndarray:
    matches = [
        item
        for item in frame.signals
        if item.component_archetype_id == contract.component_archetype_id
        and item.port_role_id == port_role_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one public semantic signal for {port_role_id}, found {len(matches)}"
        )
    if any(value is None for value in matches[0].values):
        raise ValueError(f"required public semantic signal {port_role_id} contains missing values")
    return np.asarray(matches[0].values, dtype=float)


def _rms(values: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.square(values)))), EPSILON)


def _robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    return max(mad, _rms(values) if mad <= EPSILON else mad, EPSILON)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) <= EPSILON or float(np.std(right)) <= EPSILON:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _physical_arrays(
    frame: SemanticTelemetryFrame, contract: DynamicsContract
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = _signal(frame, contract, contract.states[0].observation_port_role_id)
    time = np.asarray(frame.timestamps_s, dtype=float)
    dt = np.diff(time)
    derivative = np.diff(state) / dt
    if contract.equation_family == "bounded_integral_balance":
        power = _signal(frame, contract, contract.inputs[0].port_role_id)
        design = (0.5 * (power[:-1] + power[1:])).reshape(-1, 1)
    elif contract.equation_family == "first_order_thermal_balance":
        heat = _signal(frame, contract, contract.inputs[0].port_role_id)
        ambient = _signal(frame, contract, contract.inputs[1].port_role_id)
        heat_mid = 0.5 * (heat[:-1] + heat[1:])
        state_mid = 0.5 * (state[:-1] + state[1:])
        ambient_mid = 0.5 * (ambient[:-1] + ambient[1:])
        design = np.column_stack([heat_mid, -(state_mid - ambient_mid)])
    else:
        raise ValueError(f"unsupported frozen equation family: {contract.equation_family}")
    column_scale = np.maximum(np.sqrt(np.mean(np.square(design), axis=0)), EPSILON)
    return state, derivative, design / column_scale


def _selected_segment(
    derivative: np.ndarray, design: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float]:
    candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    length = len(derivative)
    parameter_count = design.shape[1]
    for fraction in CHANGE_FRACTIONS:
        split = int(round(fraction * length))
        split = min(max(split, parameter_count + 3), length - parameter_count - 3)
        pre_design = design[:split]
        pre_derivative = derivative[:split]
        pre_coefficients = np.linalg.lstsq(pre_design, pre_derivative, rcond=None)[0]
        residual = derivative - design @ pre_coefficients
        pre_scale = _robust_scale(residual[:split])
        post_residual = residual[split:]
        ramp = np.linspace(0.0, 1.0, len(post_residual))
        post_design = np.column_stack(
            [design[split:], np.ones(len(post_residual)), ramp]
        )
        decomposition = np.linalg.lstsq(post_design, post_residual, rcond=None)[0]
        remaining = post_residual - post_design @ decomposition
        reduction = max(
            0.0,
            float(post_residual @ post_residual - remaining @ remaining),
        )
        change_score = reduction / max(pre_scale * pre_scale, EPSILON)
        candidates.append(
            (
                change_score,
                split,
                pre_coefficients,
                residual,
                decomposition,
                pre_scale,
            )
        )
    # max() retains the earliest grid point because the negative split is the secondary key.
    chosen = max(candidates, key=lambda item: (item[0], -item[1]))
    return chosen[1], chosen[2], chosen[3], chosen[4], chosen[5], chosen[0]


def extract_segmented_mechanism_features(
    frame: SemanticTelemetryFrame,
    contract: DynamicsContract,
) -> SegmentedMechanismFeatureVector:
    if frame.labels_used or frame.truth_used:
        raise ValueError("segmented diagnosis accepts public semantic telemetry only")
    if frame.signals[0].component_archetype_id != contract.component_archetype_id:
        raise ValueError("semantic frame and dynamics contract archetypes differ")
    state, derivative, design = _physical_arrays(frame, contract)
    if len(derivative) < max(contract.observability.minimum_samples - 1, 20):
        raise ValueError("insufficient samples for frozen segmented evidence")
    split, pre_coefficients, residual, decomposition, pre_scale, change_score = (
        _selected_segment(derivative, design)
    )
    parameter_count = design.shape[1]
    parameter_shift = decomposition[:parameter_count]
    constant = float(decomposition[parameter_count])
    ramp = float(decomposition[parameter_count + 1])
    pre_derivative = derivative[:split]
    pre_remaining = residual[:split]
    centered_sst = float(
        np.sum(np.square(pre_derivative - float(np.mean(pre_derivative))))
    )
    pre_r2 = 1.0 - float(pre_remaining @ pre_remaining) / max(centered_sst, EPSILON)
    post_residual = residual[split:]
    unit_ramp = np.linspace(-1.0, 1.0, len(post_residual))
    post_slope = float(
        np.linalg.lstsq(
            np.column_stack([unit_ramp, np.ones(len(unit_ramp))]),
            post_residual,
            rcond=None,
        )[0][0]
    )
    post_design = np.column_stack(
        [
            design[split:],
            np.ones(len(post_residual)),
            np.linspace(0.0, 1.0, len(post_residual)),
        ]
    )
    post_remaining = post_residual - post_design @ decomposition
    post_sse = float(post_residual @ post_residual)
    post_decomposition_r2 = max(
        0.0, min(1.0, 1.0 - float(post_remaining @ post_remaining) / max(post_sse, EPSILON))
    )
    pre_rms = _rms(pre_remaining)
    parameter_norm = float(np.linalg.norm(parameter_shift))
    pre_parameter_norm = float(np.linalg.norm(pre_coefficients))
    alignment = (
        float(parameter_shift @ pre_coefficients)
        / (parameter_norm * pre_parameter_norm)
        if parameter_norm > EPSILON and pre_parameter_norm > EPSILON
        else 0.0
    )
    l1 = max(float(np.sum(np.abs(residual))), EPSILON)
    energy = max(float(residual @ residual), EPSILON)
    window_start = max(0, split - 2)
    window_end = min(len(residual), split + 3)
    lag_one = _correlation(residual[:-1], residual[1:])
    periodic = max(
        (
            abs(_correlation(residual[:-lag], residual[lag:]))
            for lag in range(2, min(20, len(residual) - 2) + 1)
        ),
        default=0.0,
    )
    signs = np.sign(residual)
    nonzero_pairs = (signs[:-1] != 0.0) & (signs[1:] != 0.0)
    sign_changes = (
        float(np.mean(signs[:-1][nonzero_pairs] != signs[1:][nonzero_pairs]))
        if bool(np.any(nonzero_pairs))
        else 0.0
    )
    active = np.abs(residual) > 2.0 * pre_scale
    state_difference = np.diff(state)
    pre_state_difference = state_difference[:split]
    state_scale = _robust_scale(pre_state_difference)
    plateau = np.abs(state_difference) <= state_scale
    input_residual_correlation = max(
        (
            abs(_correlation(post_residual, design[split:, column]))
            for column in range(parameter_count)
        ),
        default=0.0,
    )
    values = [
        split / len(residual),
        max(-1.0, min(1.0, pre_r2)),
        post_decomposition_r2,
        parameter_norm / pre_scale,
        max(-1.0, min(1.0, alignment)),
        float(parameter_shift[0]) / pre_scale,
        constant / pre_scale,
        ramp / pre_scale,
        _rms(post_residual) / pre_rms,
        float(np.mean(post_residual)) / pre_scale,
        post_slope / pre_scale,
        float(np.max(np.abs(residual))) / l1,
        float(np.sum(np.square(residual[window_start:window_end]))) / energy,
        lag_one,
        periodic,
        sign_changes,
        float(np.mean(active)),
        _longest_true_run(active) / len(residual),
        float(np.mean(plateau)),
        float(np.max(np.abs(state_difference))) / state_scale,
        input_residual_correlation,
        math.log1p(change_score),
    ]
    max_residual_z = float(np.max(np.abs(residual))) / pre_scale
    plateau_denominator = max(1, math.ceil(0.05 * len(residual)))
    plateau_z = _longest_true_run(plateau) / plateau_denominator
    anomaly_score = max(change_score, max_residual_z * max_residual_z, plateau_z * plateau_z)
    return SegmentedMechanismFeatureVector(
        feature_names=list(SEGMENTED_FEATURE_NAMES),
        values=values,
        anomaly_score=anomaly_score,
        input_sha256=frame.semantic_data_sha256,
    )


class CalibratedSegmentedMechanismModel:
    """One L2 logistic class head with a non-classifying conformal envelope."""

    def __init__(self, artifact: SegmentedMechanismCalibrationArtifact) -> None:
        self.artifact = artifact
        self._mean = np.asarray(artifact.feature_mean, dtype=float)
        self._scale = np.asarray(artifact.feature_scale, dtype=float)
        self._coefficients = np.asarray(artifact.coefficients, dtype=float)
        self._intercepts = np.asarray(artifact.intercepts, dtype=float)
        self._centers = np.asarray(artifact.class_centers, dtype=float)
        self._inverse_covariances = np.asarray(
            artifact.class_inverse_covariances, dtype=float
        )

    def predict(
        self, features: SegmentedMechanismFeatureVector
    ) -> SegmentedMechanismPrediction:
        standardized = (np.asarray(features.values, dtype=float) - self._mean) / self._scale
        logits = (
            self._coefficients @ standardized + self._intercepts
        ) / self.artifact.temperature
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        index = int(np.argmax(probabilities))
        difference = standardized - self._centers[index]
        distance = math.sqrt(
            max(
                0.0,
                float(
                    difference
                    @ self._inverse_covariances[index]
                    @ difference
                ),
            )
            / len(SEGMENTED_FEATURE_NAMES)
        )
        threshold = float(self.artifact.class_distance_thresholds[index])
        return SegmentedMechanismPrediction(
            anomalous=features.anomaly_score > self.artifact.anomaly_threshold,
            anomaly_score=features.anomaly_score,
            anomaly_threshold=self.artifact.anomaly_threshold,
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
    "CalibratedSegmentedMechanismModel",
    "CHANGE_FRACTIONS",
    "SEGMENTED_FEATURE_NAMES",
    "SegmentedMechanismCalibrationArtifact",
    "SegmentedMechanismFeatureVector",
    "SegmentedMechanismPrediction",
    "extract_segmented_mechanism_features",
]
