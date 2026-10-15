from __future__ import annotations

import hashlib
import math
from importlib.resources import files
from statistics import NormalDist

import numpy as np

from astrograph.physics import DynamicsContract
from astrograph.semantics import SemanticSignal, SemanticTelemetryFrame

from . import numerical
from .contracts import (
    ConstraintViolation,
    ConservationAssessment,
    ModelReceipt,
    ParameterEstimate,
    ResidualSeries,
    StateEstimateResult,
    StateEstimator,
    StateTrajectory,
    UncertaintySummary,
    ArchetypeUncertaintyCalibration,
    UncertaintyCalibrationProfile,
)


class StateEstimationError(ValueError):
    pass


def _contract_sha256(contract: DynamicsContract) -> str:
    return hashlib.sha256(contract.model_dump_json(exclude_none=False).encode("utf-8")).hexdigest()


def _signal(frame: SemanticTelemetryFrame, contract: DynamicsContract, port_role_id: str) -> np.ndarray:
    matches = [
        item
        for item in frame.signals
        if item.component_archetype_id == contract.component_archetype_id
        and item.port_role_id == port_role_id
    ]
    if len(matches) != 1:
        raise StateEstimationError(
            f"expected exactly one semantic signal for port role {port_role_id}, found {len(matches)}"
        )
    values = matches[0].values
    if any(value is None for value in values):
        raise StateEstimationError(f"missing values are not allowed for required port role {port_role_id}")
    return np.asarray(values, dtype=float)


def _bounded(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _parameter(
    contract: DynamicsContract,
    parameter_id: str,
    value: float,
    standard_error: float,
    interval_multiplier: float,
) -> ParameterEstimate:
    spec = contract.parameter(parameter_id)
    estimate = _bounded(value, spec.lower_bound, spec.upper_bound)
    radius = interval_multiplier * max(float(standard_error), 0.0)
    return ParameterEstimate(
        parameter_id=parameter_id,
        value=estimate,
        canonical_unit=spec.canonical_unit,
        lower_confidence=_bounded(estimate - radius, spec.lower_bound, estimate),
        upper_confidence=_bounded(estimate + radius, estimate, spec.upper_bound),
        physical_lower_bound=spec.lower_bound,
        physical_upper_bound=spec.upper_bound,
    )


def _residual(residual_id: str, unit: str, values: np.ndarray) -> ResidualSeries:
    return ResidualSeries(
        residual_id=residual_id,
        canonical_unit=unit,
        values=[float(value) for value in values],
        root_mean_square=float(np.sqrt(np.mean(np.square(values)))),
        mean=float(np.mean(values)),
    )


def _trajectory(
    state_id: str,
    unit: str,
    values: np.ndarray,
    sigma: float,
    interval_multiplier: float,
) -> StateTrajectory:
    radius = interval_multiplier * sigma
    return StateTrajectory(
        state_id=state_id,
        canonical_unit=unit,
        values=[float(value) for value in values],
        lower_confidence=[float(value - radius) for value in values],
        upper_confidence=[float(value + radius) for value in values],
    )


def _linear_covariance(design: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    return numerical.linear_covariance(design, residuals)


def _first_difference_cholesky(dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return numerical.first_difference_cholesky(dt)


def _lower_bidiagonal_solve(
    values: np.ndarray,
    diagonal: np.ndarray,
    subdiagonal: np.ndarray,
) -> np.ndarray:
    return numerical.lower_bidiagonal_solve(values, diagonal, subdiagonal)


def _gls_balance_fit(
    derivative: np.ndarray,
    design: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return numerical.gls_balance_fit(derivative, design, dt)


def conservation_episode_score(
    whitened_innovation: np.ndarray,
    whitened_design: np.ndarray,
    cholesky_diagonal: np.ndarray,
    cholesky_subdiagonal: np.ndarray,
    change_point_fractions: list[float],
) -> tuple[float, np.ndarray]:
    return numerical.conservation_episode_score(
        whitened_innovation,
        whitened_design,
        cholesky_diagonal,
        cholesky_subdiagonal,
        change_point_fractions,
    )


class GrayBoxStateEstimator(StateEstimator):
    """Single production baseline: known balance equations plus bounded identification."""

    estimator_id = "astrograph.gray-box.bounded-balance"
    estimator_version = "1.2.0"

    def __init__(
        self,
        calibration_profile: UncertaintyCalibrationProfile | None = None,
        *,
        use_frozen_calibration: bool = True,
    ) -> None:
        if calibration_profile is not None and use_frozen_calibration:
            raise ValueError("explicit calibration requires use_frozen_calibration=False")
        self._calibration_sha256 = "NONE"
        if calibration_profile is not None:
            payload = calibration_profile.model_dump_json(exclude_none=False).encode("utf-8")
            self._calibration_sha256 = hashlib.sha256(payload).hexdigest()
            self._calibration = calibration_profile
        elif use_frozen_calibration:
            resource = files("astrograph.resources").joinpath(
                "gray_box_uncertainty_calibration.v1.json"
            )
            raw = resource.read_bytes()
            self._calibration_sha256 = hashlib.sha256(raw).hexdigest()
            self._calibration = UncertaintyCalibrationProfile.model_validate_json(raw)
        else:
            self._calibration = None

        if self._calibration is not None and (
            self._calibration.estimator_id != self.estimator_id
            or self._calibration.estimator_version != self.estimator_version
        ):
            raise ValueError("uncertainty calibration does not match estimator identity")

    def _calibration_for(
        self, component_archetype_id: str
    ) -> ArchetypeUncertaintyCalibration | None:
        if self._calibration is None:
            return None
        return self._calibration.for_archetype(component_archetype_id)

    @staticmethod
    def _conservation_assessment(
        *,
        raw_tolerance: float,
        raw_violation_count: int,
        residual_count: int,
        whitened_innovation: np.ndarray,
        whitened_design: np.ndarray,
        cholesky_diagonal: np.ndarray,
        cholesky_subdiagonal: np.ndarray,
        calibration: ArchetypeUncertaintyCalibration | None,
        calibration_version: str,
    ) -> tuple[ConservationAssessment, float]:
        raw_fraction = raw_violation_count / max(residual_count, 1)
        change_point_fractions = (
            calibration.conservation_change_point_fractions
            if calibration is not None
            else [float(value) for value in np.linspace(0.2, 0.8, 13)]
        )
        episode_score, directions = conservation_episode_score(
            whitened_innovation,
            whitened_design,
            cholesky_diagonal,
            cholesky_subdiagonal,
            change_point_fractions,
        )
        if calibration is None:
            assessment = ConservationAssessment(
                method="UNCALIBRATED_RAW_ONLY",
                raw_physical_tolerance=raw_tolerance,
                raw_violation_count=raw_violation_count,
                raw_violation_fraction=raw_fraction,
                episode_score=episode_score,
                empirical_p_value=1.0,
                p_value_upper_confidence=1.0,
                false_alarm_rate=0.05,
                bootstrap_replicates=0,
                statistically_significant=False,
                calibration_version="NOT_CALIBRATED",
            )
            return assessment, min(1.0, raw_fraction)
        if directions.shape[1] == 0:
            assessment = ConservationAssessment(
                method=calibration.conservation_method,
                raw_physical_tolerance=raw_tolerance,
                raw_violation_count=raw_violation_count,
                raw_violation_fraction=raw_fraction,
                episode_score=episode_score,
                empirical_p_value=1.0,
                p_value_upper_confidence=1.0,
                false_alarm_rate=calibration.conservation_false_alarm_rate,
                bootstrap_replicates=calibration.conservation_bootstrap_replicates,
                statistically_significant=False,
                calibration_version=calibration_version,
            )
            return assessment, 0.0
        replicates = calibration.conservation_bootstrap_replicates
        rng = np.random.default_rng(calibration.conservation_bootstrap_seed)
        null_derivative = rng.normal(size=(replicates, residual_count))
        design_pseudoinverse = np.linalg.pinv(whitened_design)
        null_residual = null_derivative - (
            null_derivative @ design_pseudoinverse.T
        ) @ whitened_design.T
        degrees = max(residual_count - whitened_design.shape[1], 1)
        null_scale = np.sqrt(np.sum(np.square(null_residual), axis=1) / degrees)
        null_scores = np.max(np.abs(null_residual @ directions), axis=1) / np.maximum(
            null_scale, 1.0e-15
        )
        exceedances = int(np.count_nonzero(null_scores >= episode_score))
        p_value = float((1 + exceedances) / (replicates + 1))

        # A finite bootstrap must not turn Monte Carlo noise into an alarm.  The one-sided Wilson
        # upper bound keeps the declared 0.05 false-alarm rate unchanged while requiring positive
        # evidence that the simulated tail probability is below it.
        z_value = NormalDist().inv_cdf(calibration.conservation_mc_confidence_level)
        observed_rate = exceedances / replicates
        denominator = 1.0 + z_value * z_value / replicates
        p_value_upper = min(
            1.0,
            (
                observed_rate
                + z_value * z_value / (2.0 * replicates)
                + z_value
                * math.sqrt(
                    observed_rate * (1.0 - observed_rate) / replicates
                    + z_value * z_value / (4.0 * replicates * replicates)
                )
            )
            / denominator,
        )
        significant = p_value_upper <= calibration.conservation_false_alarm_rate
        assessment = ConservationAssessment(
            method=calibration.conservation_method,
            raw_physical_tolerance=raw_tolerance,
            raw_violation_count=raw_violation_count,
            raw_violation_fraction=raw_fraction,
            episode_score=episode_score,
            empirical_p_value=p_value,
            p_value_upper_confidence=p_value_upper,
            false_alarm_rate=calibration.conservation_false_alarm_rate,
            bootstrap_replicates=replicates,
            statistically_significant=significant,
            calibration_version=calibration_version,
        )
        return assessment, 1.0 - p_value_upper

    @staticmethod
    def _uncertainty(
        sigma: float,
        calibration: ArchetypeUncertaintyCalibration | None,
        calibration_version: str,
    ) -> UncertaintySummary:
        if calibration is None:
            return UncertaintySummary(residual_standard_deviation=sigma)
        return UncertaintySummary(
            method="split_conformal_scaled_residual",
            residual_standard_deviation=sigma,
            confidence_level=calibration.confidence_level,
            calibrated=True,
            calibration_version=calibration_version,
            state_interval_multiplier=calibration.state_interval_multiplier,
            parameter_interval_multiplier=calibration.parameter_interval_multiplier,
        )

    def estimate(
        self,
        frame: SemanticTelemetryFrame,
        contract: DynamicsContract,
    ) -> StateEstimateResult:
        if frame.labels_used or frame.truth_used:
            raise StateEstimationError("state estimation accepts public semantic telemetry only")
        if len(frame.timestamps_s) < contract.observability.minimum_samples:
            raise StateEstimationError("insufficient samples for the dynamics observability contract")
        dispatch = {
            "bounded_integral_balance": self._estimate_integral_balance,
            "first_order_thermal_balance": self._estimate_thermal_balance,
            "dual_node_thermal_balance": self._estimate_dual_node_thermal,
            "first_order_actuator_lag": self._estimate_actuator_lag,
            "inertial_spin_balance": self._estimate_inertial_spin,
        }
        try:
            return dispatch[contract.equation_family](frame, contract)
        except KeyError as exc:
            raise StateEstimationError(
                f"unsupported equation family: {contract.equation_family}"
            ) from exc

    def _receipt(self, frame: SemanticTelemetryFrame, contract: DynamicsContract) -> ModelReceipt:
        return ModelReceipt(
            estimator_id=self.estimator_id,
            estimator_version=self.estimator_version,
            dynamics_contract_id=contract.contract_id,
            dynamics_contract_sha256=_contract_sha256(contract),
            identification_data_sha256=frame.semantic_data_sha256,
            uncertainty_calibration_version=(
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED"
            ),
            uncertainty_calibration_sha256=self._calibration_sha256,
            conservation_calibration_version=(
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED"
            ),
        )

    def _estimate_integral_balance(
        self,
        frame: SemanticTelemetryFrame,
        contract: DynamicsContract,
    ) -> StateEstimateResult:
        state_spec = contract.states[0]
        calibration = self._calibration_for(contract.component_archetype_id)
        state_multiplier = calibration.state_interval_multiplier if calibration else 1.96
        parameter_multiplier = calibration.parameter_interval_multiplier if calibration else 1.96
        state = _signal(frame, contract, state_spec.observation_port_role_id)
        power = _signal(frame, contract, contract.inputs[0].port_role_id)
        time = np.asarray(frame.timestamps_s, dtype=float)
        dt = np.diff(time)
        power_mid = 0.5 * (power[:-1] + power[1:])
        if float(np.var(power_mid)) < contract.observability.minimum_input_variance:
            raise StateEstimationError("input excitation is below the observability contract")
        derivative = np.diff(state) / dt

        # Gap-1 efficiency branch: when BOTH charge/discharge efficiency params
        # are declared, the storage model is d(storage)/dt = P * eff(P)/C with
        # eff = charge_eff (P>=0) or 1/discharge_eff (P<0).  This is piecewise
        # linear in P, so we fit a two-column design [P_charge, P_discharge]
        # and recover C from the charge slope and discharge_eff from the ratio
        # of the discharge slope to the charge slope.  Legacy contracts without
        # the efficiency params keep the old single-slope path (regression).
        has_eff = ("charge_efficiency" in [p.parameter_id for p in contract.parameters]
                   and "discharge_efficiency" in [p.parameter_id for p in contract.parameters])
        if has_eff:
            p_charge = np.where(power_mid >= 0.0, power_mid, 0.0)
            p_discharge = np.where(power_mid < 0.0, power_mid, 0.0)
            design = np.column_stack((p_charge, p_discharge))
            if np.linalg.matrix_rank(design) < 2:
                raise StateEstimationError(
                    "storage efficiency identification requires both charge and discharge excitation"
                )
        else:
            design = power_mid.reshape(-1, 1)
        (
            coefficients,
            derivative_residual,
            whitened_innovation,
            whitened_design,
            cholesky_diagonal,
            cholesky_subdiagonal,
        ) = _gls_balance_fit(derivative, design, dt)
        covariance = _linear_covariance(whitened_design, whitened_innovation)
        if has_eff:
            inv_c_charge, inv_c_discharge = (float(item) for item in coefficients)
            # d(storage)/dt = P*charge_eff/C (charge) or P/discharge_eff/C (discharge).
            # Linearised slopes: charge slope = charge_eff/C, discharge slope = 1/(discharge_eff*C).
            # charge_eff is FIXED (known physical constant, default 0.98), so:
            #   C = charge_eff / charge_slope
            #   discharge_eff = 1 / (discharge_slope * C)
            # This resolves the 3-unknown / 2-equation underdetermination by fixing
            # the coulombic charging efficiency, which the generator does treat as
            # a constant (near 1).
            charge_eff_spec = contract.parameter("charge_efficiency")
            charge_eff_known = float(charge_eff_spec.initial_value)
            if inv_c_charge <= 0.0 or inv_c_discharge <= 0.0:
                raise StateEstimationError("identified storage capacity is non-positive")
            capacity = charge_eff_known / inv_c_charge
            capacity_se = math.sqrt(max(float(covariance[0, 0]), 0.0)) / (inv_c_charge * inv_c_charge)
            capacity_parameter = _parameter(contract, "energy_capacity", capacity, capacity_se, parameter_multiplier)
            capacity = capacity_parameter.value
            raw_discharge_eff = 1.0 / (inv_c_discharge * capacity)
            se_c1 = math.sqrt(max(float(covariance[1, 1]), 0.0))
            discharge_eff_se = abs(raw_discharge_eff) * math.sqrt((se_c1 / inv_c_discharge) ** 2)
            discharge_eff_parameter = _parameter(
                contract, "discharge_efficiency", raw_discharge_eff, discharge_eff_se, parameter_multiplier
            )
            discharge_eff = discharge_eff_parameter.value
            parameter = capacity_parameter
        else:
            inverse_capacity = float(coefficients[0])
            if inverse_capacity <= 0.0:
                raise StateEstimationError("identified storage capacity is non-positive")
            capacity = 1.0 / inverse_capacity
            inverse_capacity_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
            capacity_se = inverse_capacity_se / (inverse_capacity * inverse_capacity)
            parameter = _parameter(
                contract, "energy_capacity", capacity, capacity_se, parameter_multiplier
            )
            capacity = parameter.value

        predicted = np.empty_like(state)
        predicted[0] = state[0]
        eff_discharge = discharge_eff if has_eff else 1.0
        for index, step in enumerate(dt):
            if has_eff:
                eff = 1.0 / eff_discharge if power_mid[index] < 0.0 else 1.0
                predicted[index + 1] = predicted[index] + step * power_mid[index] * eff / capacity
            else:
                predicted[index + 1] = predicted[index] + step * power_mid[index] / capacity
            predicted[index + 1] = np.clip(
                predicted[index + 1], state_spec.lower_bound, state_spec.upper_bound
            )
        state_residual = state - predicted
        sigma = float(np.std(state_residual, ddof=1)) if len(state_residual) > 1 else 0.0
        law = contract.conservation_laws[0]
        violations = [
            ConstraintViolation(
                code="CONSERVATION_RESIDUAL_EXCEEDED",
                index=index + 1,
                magnitude=abs(float(value)),
                limit=law.residual_tolerance,
            )
            for index, value in enumerate(derivative_residual)
            if abs(float(value)) > law.residual_tolerance
        ]
        assessment, ood_score = self._conservation_assessment(
            raw_tolerance=law.residual_tolerance,
            raw_violation_count=len(violations),
            residual_count=len(derivative_residual),
            whitened_innovation=whitened_innovation,
            whitened_design=whitened_design,
            cholesky_diagonal=cholesky_diagonal,
            cholesky_subdiagonal=cholesky_subdiagonal,
            calibration=calibration,
            calibration_version=(
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED"
            ),
        )
        return StateEstimateResult(
            component_archetype_id=contract.component_archetype_id,
            timestamps_s=frame.timestamps_s,
            states=[
                _trajectory(
                    state_spec.state_id,
                    state_spec.canonical_unit,
                    predicted,
                    sigma,
                    state_multiplier,
                )
            ],
            parameters=[parameter, discharge_eff_parameter] if has_eff else [parameter],
            residuals=[
                _residual("state_reconstruction", state_spec.canonical_unit, state_residual),
                _residual("balance_equation", "1/s", derivative_residual),
                _residual("balance_innovation_whitened", "1/s", whitened_innovation),
            ],
            uncertainty=self._uncertainty(
                sigma,
                calibration,
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED",
            ),
            ood_score=ood_score,
            constraint_violations=violations,
            conservation_assessment=assessment,
            receipt=self._receipt(frame, contract),
        )

    def _estimate_thermal_balance(
        self,
        frame: SemanticTelemetryFrame,
        contract: DynamicsContract,
    ) -> StateEstimateResult:
        state_spec = contract.states[0]
        calibration = self._calibration_for(contract.component_archetype_id)
        state_multiplier = calibration.state_interval_multiplier if calibration else 1.96
        parameter_multiplier = calibration.parameter_interval_multiplier if calibration else 1.96
        temperature = _signal(frame, contract, state_spec.observation_port_role_id)
        heat = _signal(frame, contract, contract.inputs[0].port_role_id)
        ambient = _signal(frame, contract, contract.inputs[1].port_role_id)
        time = np.asarray(frame.timestamps_s, dtype=float)
        dt = np.diff(time)
        heat_mid = 0.5 * (heat[:-1] + heat[1:])
        temperature_mid = 0.5 * (temperature[:-1] + temperature[1:])
        ambient_mid = 0.5 * (ambient[:-1] + ambient[1:])
        design = np.column_stack((heat_mid, ambient_mid - temperature_mid))
        if np.linalg.matrix_rank(design) < 2:
            raise StateEstimationError("thermal inputs do not identify capacity and conductance")
        derivative = np.diff(temperature) / dt
        (
            coefficients,
            derivative_residual,
            whitened_innovation,
            whitened_design,
            cholesky_diagonal,
            cholesky_subdiagonal,
        ) = _gls_balance_fit(derivative, design, dt)
        inverse_capacity, cooling_rate = (float(item) for item in coefficients)
        if inverse_capacity <= 0.0 or cooling_rate <= 0.0:
            raise StateEstimationError("identified thermal parameters are non-positive")
        capacity = 1.0 / inverse_capacity
        conductance = cooling_rate / inverse_capacity
        covariance = _linear_covariance(whitened_design, whitened_innovation)
        se_a = math.sqrt(max(float(covariance[0, 0]), 0.0))
        capacity_se = se_a / (inverse_capacity * inverse_capacity)
        gradient = np.asarray([-cooling_rate / (inverse_capacity**2), 1.0 / inverse_capacity])
        conductance_se = math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
        capacity_estimate = _parameter(
            contract, "thermal_capacity", capacity, capacity_se, parameter_multiplier
        )
        conductance_estimate = _parameter(
            contract,
            "thermal_conductance",
            conductance,
            conductance_se,
            parameter_multiplier,
        )
        capacity = capacity_estimate.value
        conductance = conductance_estimate.value

        predicted = np.empty_like(temperature)
        predicted[0] = temperature[0]
        for index, step in enumerate(dt):
            left = capacity / step + conductance / 2.0
            right = (
                predicted[index] * (capacity / step - conductance / 2.0)
                + heat_mid[index]
                + conductance * ambient_mid[index]
            )
            predicted[index + 1] = right / left
        state_residual = temperature - predicted
        sigma = float(np.std(state_residual, ddof=1)) if len(state_residual) > 1 else 0.0
        law = contract.conservation_laws[0]
        violations = [
            ConstraintViolation(
                code="CONSERVATION_RESIDUAL_EXCEEDED",
                index=index + 1,
                magnitude=abs(float(value)),
                limit=law.residual_tolerance,
            )
            for index, value in enumerate(derivative_residual)
            if abs(float(value)) > law.residual_tolerance
        ]
        assessment, ood_score = self._conservation_assessment(
            raw_tolerance=law.residual_tolerance,
            raw_violation_count=len(violations),
            residual_count=len(derivative_residual),
            whitened_innovation=whitened_innovation,
            whitened_design=whitened_design,
            cholesky_diagonal=cholesky_diagonal,
            cholesky_subdiagonal=cholesky_subdiagonal,
            calibration=calibration,
            calibration_version=(
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED"
            ),
        )
        return StateEstimateResult(
            component_archetype_id=contract.component_archetype_id,
            timestamps_s=frame.timestamps_s,
            states=[
                _trajectory(
                    state_spec.state_id,
                    state_spec.canonical_unit,
                    predicted,
                    sigma,
                    state_multiplier,
                )
            ],
            parameters=[capacity_estimate, conductance_estimate],
            residuals=[
                _residual("state_reconstruction", state_spec.canonical_unit, state_residual),
                _residual("balance_equation", "K/s", derivative_residual),
                _residual("balance_innovation_whitened", "K/s", whitened_innovation),
            ],
            uncertainty=self._uncertainty(
                sigma,
                calibration,
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED",
            ),
            ood_score=ood_score,
            constraint_violations=violations,
            conservation_assessment=assessment,
            receipt=self._receipt(frame, contract),
        )

    def _estimate_dual_node_thermal(
        self,
        frame: SemanticTelemetryFrame,
        contract: DynamicsContract,
    ) -> StateEstimateResult:
        """Two-node (bus + battery) thermal balance — gap-2 closure.

        Generator physics::
            C_bus·dT_bus/dt   = P_net − G_rad·(T_bus−T_sink) − G_cond·(T_bus−T_batt)
            C_batt·dT_batt/dt =                    + G_cond·(T_bus−T_batt)

        Linearised as a 2x2 system in the shared GLS machinery:
            dT_bus/dt  = (1/C_bus)·P_net − (G_rad/C_bus)·(T_bus−T_sink)
                         − (G_cond/C_bus)·(T_bus−T_batt)
            dT_batt/dt = (G_cond/C_batt)·(T_bus−T_batt)

        We fit the bus equation first (3 unknown coefficients) and then the
        battery equation (1 unknown, G_cond/C_batt) using the identified
        conductance, yielding C_bus, G_rad, G_cond, C_batt.
        """
        bus_spec = contract.states[0]
        batt_spec = contract.states[1]
        calibration = self._calibration_for(contract.component_archetype_id)
        state_multiplier = calibration.state_interval_multiplier if calibration else 1.96
        parameter_multiplier = calibration.parameter_interval_multiplier if calibration else 1.96
        t_bus = _signal(frame, contract, bus_spec.observation_port_role_id)
        t_batt = _signal(frame, contract, batt_spec.observation_port_role_id)
        heat = _signal(frame, contract, contract.inputs[0].port_role_id)
        ambient = _signal(frame, contract, contract.inputs[1].port_role_id)
        time = np.asarray(frame.timestamps_s, dtype=float)
        dt = np.diff(time)
        if float(np.var(heat[:-1])) < contract.observability.minimum_input_variance:
            raise StateEstimationError("thermal input excitation is below the observability contract")

        # Forward-point values: the GLS design must align with the discrete
        # derivative (T[i+1]-T[i])/dt = rhs at time i.  Using midpoint values
        # here mismatches any forward-Euler latent and turns coefficients
        # negative on continuously-varying inputs, so the design uses the
        # forward-point states and inputs.
        heat_f = heat[:-1]
        t_bus_f = t_bus[:-1]
        t_batt_f = t_batt[:-1]
        ambient_f = ambient[:-1]
        coupling = t_bus_f - t_batt_f
        radiator = t_bus_f - ambient_f

        # Bus equation: dT_bus/dt = a·P_net − b·radiator − c·coupling, where
        # a=1/C_bus, b=G_rad/C_bus, c=G_cond/C_bus.  The three columns span very
        # different scales (heat ~1e1, radiator ~1e1, coupling ~1e0, but the
        # coefficient inv_c_bus ~1e-5), so the design matrix is column-scaled
        # before the GLS fit and the coefficients are rescaled back afterwards.
        d_bus = np.diff(t_bus) / dt
        design_bus = np.column_stack((heat_f, -radiator, -coupling))
        if np.linalg.matrix_rank(design_bus) < 3:
            raise StateEstimationError(
                "dual-node thermal inputs do not identify bus capacity, radiator and conductance"
            )
        col_scale = np.std(design_bus, axis=0)
        col_scale = np.where(col_scale < 1.0e-12, 1.0, col_scale)
        scaled_design_bus = design_bus / col_scale
        (coef_bus, resid_bus, innov_bus, design_w_bus, chole_bus_diag, chole_bus_subdiag) = _gls_balance_fit(
            d_bus, scaled_design_bus, dt
        )
        inv_c_bus, g_rad_over_c_bus, g_cond_over_c_bus = (float(x) for x in coef_bus / col_scale)
        if inv_c_bus <= 0.0 or g_rad_over_c_bus < 0.0 or g_cond_over_c_bus < 0.0:
            raise StateEstimationError("identified dual-node thermal parameters are non-positive")
        c_bus = 1.0 / inv_c_bus
        g_rad = g_rad_over_c_bus * c_bus
        g_cond = g_cond_over_c_bus * c_bus
        # Covariance on the scaled design; convert back to the unscaled coefficients.
        cov_bus = _linear_covariance(design_w_bus, innov_bus)
        cov_bus = cov_bus / (col_scale[:, None] * col_scale[None, :])

        # Battery equation: dT_batt/dt = (G_cond/C_batt)·coupling.
        d_batt = np.diff(t_batt) / dt
        design_batt = coupling.reshape(-1, 1)
        if np.linalg.matrix_rank(design_batt) < 1:
            raise StateEstimationError("dual-node thermal has no bus-battery coupling")
        (coef_batt, resid_batt, innov_batt, design_w_batt, chole_batt_diag, chole_batt_subdiag) = _gls_balance_fit(
            d_batt, design_batt, dt
        )
        g_cond_over_c_batt = float(coef_batt[0])
        if g_cond_over_c_batt <= 0.0:
            raise StateEstimationError("identified battery capacity is non-positive")
        c_batt = g_cond / g_cond_over_c_batt
        cov_batt = _linear_covariance(design_w_batt, innov_batt)

        # SE propagation for the four parameters.
        se_inv_c_bus = math.sqrt(max(float(cov_bus[0, 0]), 0.0))
        c_bus_se = se_inv_c_bus / (inv_c_bus * inv_c_bus)
        grad_rad = np.asarray([g_rad_over_c_bus / inv_c_bus**2, -1.0 / inv_c_bus, 0.0])
        g_rad_se = math.sqrt(max(float(grad_rad @ cov_bus @ grad_rad), 0.0))
        grad_cond = np.asarray([g_cond_over_c_bus / inv_c_bus**2, 0.0, -1.0 / inv_c_bus])
        g_cond_se = math.sqrt(max(float(grad_cond @ cov_bus @ grad_cond), 0.0))
        g_over_c_se = math.sqrt(max(float(cov_batt[0, 0]), 0.0))
        c_batt_se = math.sqrt(
            (g_cond_se / g_cond_over_c_batt) ** 2
            + (g_cond * g_over_c_se / (g_cond_over_c_batt**2)) ** 2
        )

        c_bus_param = _parameter(contract, "thermal_capacity_bus", c_bus, c_bus_se, parameter_multiplier)
        c_batt_param = _parameter(contract, "thermal_capacity_battery", c_batt, c_batt_se, parameter_multiplier)
        g_cond_param = _parameter(contract, "thermal_conductance", g_cond, g_cond_se, parameter_multiplier)
        g_rad_param = _parameter(contract, "radiator_conductance", g_rad, g_rad_se, parameter_multiplier)
        c_bus = c_bus_param.value
        c_batt = c_batt_param.value
        g_cond = g_cond_param.value
        g_rad = g_rad_param.value

        # Reconstruct both nodes (semi-implicit trapezoid).
        pred_bus = np.empty_like(t_bus)
        pred_batt = np.empty_like(t_batt)
        pred_bus[0] = t_bus[0]
        pred_batt[0] = t_batt[0]
        for index, step in enumerate(dt):
            g = g_rad / c_bus
            k = g_cond / c_bus
            pred_bus[index + 1] = (
                pred_bus[index]
                + step * (heat_f[index] + g * ambient_f[index] + k * pred_batt[index]) / (1.0 + step * (g + k))
            )
            kb = g_cond / c_batt
            pred_batt[index + 1] = (
                pred_batt[index] + step * kb * pred_bus[index]
            ) / (1.0 + step * kb)
        state_residual_bus = t_bus - pred_bus
        state_residual_batt = t_batt - pred_batt
        sigma_bus = float(np.std(state_residual_bus, ddof=1)) if len(state_residual_bus) > 1 else 0.0
        sigma_batt = float(np.std(state_residual_batt, ddof=1)) if len(state_residual_batt) > 1 else 0.0

        # Combined conservation assessment uses the bus innovation.
        law = contract.conservation_laws[0]
        violations = [
            ConstraintViolation(
                code="CONSERVATION_RESIDUAL_EXCEEDED",
                index=index + 1,
                magnitude=abs(float(value)),
                limit=law.residual_tolerance,
            )
            for index, value in enumerate(resid_bus)
            if abs(float(value)) > law.residual_tolerance
        ]
        assessment, ood_score = self._conservation_assessment(
            raw_tolerance=law.residual_tolerance,
            raw_violation_count=len(violations),
            residual_count=len(resid_bus),
            whitened_innovation=innov_bus,
            whitened_design=design_w_bus,
            cholesky_diagonal=chole_bus_diag,
            cholesky_subdiagonal=chole_bus_subdiag,
            calibration=calibration,
            calibration_version=(
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED"
            ),
        )
        return StateEstimateResult(
            component_archetype_id=contract.component_archetype_id,
            timestamps_s=frame.timestamps_s,
            states=[
                _trajectory(bus_spec.state_id, bus_spec.canonical_unit, pred_bus, sigma_bus, state_multiplier),
                _trajectory(batt_spec.state_id, batt_spec.canonical_unit, pred_batt, sigma_batt, state_multiplier),
            ],
            parameters=[c_bus_param, c_batt_param, g_cond_param, g_rad_param],
            residuals=[
                _residual("bus_reconstruction", bus_spec.canonical_unit, state_residual_bus),
                _residual("battery_reconstruction", batt_spec.canonical_unit, state_residual_batt),
                _residual("bus_balance_equation", "K/s", resid_bus),
                _residual("battery_balance_equation", "K/s", resid_batt),
                _residual("bus_innovation_whitened", "K/s", innov_bus),
            ],
            uncertainty=self._uncertainty(
                max(sigma_bus, sigma_batt),
                calibration,
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED",
            ),
            ood_score=ood_score,
            constraint_violations=violations,
            conservation_assessment=assessment,
            receipt=self._receipt(frame, contract),
        )

    def _estimate_actuator_lag(
        self,
        frame: SemanticTelemetryFrame,
        contract: DynamicsContract,
    ) -> StateEstimateResult:
        """First-order actuator lag: d(response)/dt = (gain·command − response)/τ.

        Linearised as d(response)/dt = (gain/τ)·command − (1/τ)·response, so the
        coefficients come from the shared GLS machinery. Identifies the gain and
        time constant, then reconstructs the response trajectory for comparison
        with the measured response.
        """
        state_spec = contract.states[0]
        calibration = self._calibration_for(contract.component_archetype_id)
        state_multiplier = calibration.state_interval_multiplier if calibration else 1.96
        parameter_multiplier = calibration.parameter_interval_multiplier if calibration else 1.96
        response = _signal(frame, contract, state_spec.observation_port_role_id)
        command = _signal(frame, contract, contract.inputs[0].port_role_id)
        time = np.asarray(frame.timestamps_s, dtype=float)
        dt = np.diff(time)
        if float(np.var(command[:-1])) < contract.observability.minimum_input_variance:
            raise StateEstimationError("command input excitation is below the observability contract")
        command_mid = 0.5 * (command[:-1] + command[1:])
        response_mid = 0.5 * (response[:-1] + response[1:])
        derivative = np.diff(response) / dt
        # Design: [command_mid, -response_mid]; coefficients [gain/τ, 1/τ].
        design = np.column_stack((command_mid, -response_mid))
        if np.linalg.matrix_rank(design) < 2:
            raise StateEstimationError("command/response inputs do not identify gain and time constant")
        (
            coefficients,
            derivative_residual,
            whitened_innovation,
            whitened_design,
            cholesky_diagonal,
            cholesky_subdiagonal,
        ) = _gls_balance_fit(derivative, design, dt)
        gain_over_tau, inv_tau = (float(item) for item in coefficients)
        if gain_over_tau <= 0.0 or inv_tau <= 0.0:
            raise StateEstimationError("identified actuator parameters are non-positive")
        time_constant = 1.0 / inv_tau
        gain = gain_over_tau * time_constant
        covariance = _linear_covariance(whitened_design, whitened_innovation)
        se_tau_inv = math.sqrt(max(float(covariance[1, 1]), 0.0))
        tau_se = se_tau_inv / (inv_tau * inv_tau)
        gradient = np.asarray([time_constant, gain_over_tau])
        gain_se = math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
        gain_estimate = _parameter(
            contract, "actuator_gain", gain, gain_se, parameter_multiplier
        )
        tau_estimate = _parameter(
            contract,
            "actuator_time_constant",
            time_constant,
            tau_se,
            parameter_multiplier,
        )
        gain = gain_estimate.value
        time_constant = tau_estimate.value

        # Reconstruct response with implicit Euler for dy/dt = (gain·u − y)/τ.
        predicted = np.empty_like(response)
        predicted[0] = response[0]
        for index, step in enumerate(dt):
            alpha = step / time_constant
            predicted[index + 1] = (
                predicted[index] + alpha * gain * command_mid[index]
            ) / (1.0 + alpha)
            predicted[index + 1] = np.clip(
                predicted[index + 1], state_spec.lower_bound, state_spec.upper_bound
            )
        state_residual = response - predicted
        sigma = float(np.std(state_residual, ddof=1)) if len(state_residual) > 1 else 0.0
        law = contract.conservation_laws[0]
        violations = [
            ConstraintViolation(
                code="CONSERVATION_RESIDUAL_EXCEEDED",
                index=index + 1,
                magnitude=abs(float(value)),
                limit=law.residual_tolerance,
            )
            for index, value in enumerate(derivative_residual)
            if abs(float(value)) > law.residual_tolerance
        ]
        assessment, ood_score = self._conservation_assessment(
            raw_tolerance=law.residual_tolerance,
            raw_violation_count=len(violations),
            residual_count=len(derivative_residual),
            whitened_innovation=whitened_innovation,
            whitened_design=whitened_design,
            cholesky_diagonal=cholesky_diagonal,
            cholesky_subdiagonal=cholesky_subdiagonal,
            calibration=calibration,
            calibration_version=(
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED"
            ),
        )
        return StateEstimateResult(
            component_archetype_id=contract.component_archetype_id,
            timestamps_s=frame.timestamps_s,
            states=[
                _trajectory(
                    state_spec.state_id,
                    state_spec.canonical_unit,
                    predicted,
                    sigma,
                    state_multiplier,
                )
            ],
            parameters=[gain_estimate, tau_estimate],
            residuals=[
                _residual("state_reconstruction", state_spec.canonical_unit, state_residual),
                _residual("command_response_balance", "N*m/s", derivative_residual),
                _residual("command_response_innovation_whitened", "N*m/s", whitened_innovation),
            ],
            uncertainty=self._uncertainty(
                sigma,
                calibration,
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED",
            ),
            ood_score=ood_score,
            constraint_violations=violations,
            conservation_assessment=assessment,
            receipt=self._receipt(frame, contract),
        )

    def _estimate_inertial_spin(
        self,
        frame: SemanticTelemetryFrame,
        contract: DynamicsContract,
    ) -> StateEstimateResult:
        """Reaction-wheel spin balance: J·dω/dt = T − b·ω.

        Linearised as dω/dt = (1/J)·T − (b/J)·ω, so the shared GLS machinery
        fits coefficients [1/J, b/J] against the design matrix [T, −ω], then
        recovers J = 1/c0 and b = c1/c0.  Reconstructs the wheel spin rate with
        semi-implicit Euler for comparison with the measured speed channel.
        """
        state_spec = contract.states[0]
        calibration = self._calibration_for(contract.component_archetype_id)
        state_multiplier = calibration.state_interval_multiplier if calibration else 1.96
        parameter_multiplier = calibration.parameter_interval_multiplier if calibration else 1.96
        angular_rate = _signal(frame, contract, state_spec.observation_port_role_id)
        command = _signal(frame, contract, contract.inputs[0].port_role_id)
        time = np.asarray(frame.timestamps_s, dtype=float)
        dt = np.diff(time)
        if float(np.var(command[:-1])) < contract.observability.minimum_input_variance:
            raise StateEstimationError("command input excitation is below the observability contract")
        command_mid = 0.5 * (command[:-1] + command[1:])
        angular_rate_mid = 0.5 * (angular_rate[:-1] + angular_rate[1:])
        derivative = np.diff(angular_rate) / dt
        # Design: [command_mid, -angular_rate_mid]; coefficients [1/J, b/J].
        design = np.column_stack((command_mid, -angular_rate_mid))
        if np.linalg.matrix_rank(design) < 2:
            raise StateEstimationError(
                "command/speed inputs do not identify inertia and damping"
            )
        (
            coefficients,
            derivative_residual,
            whitened_innovation,
            whitened_design,
            cholesky_diagonal,
            cholesky_subdiagonal,
        ) = _gls_balance_fit(derivative, design, dt)
        inverse_inertia, damping_over_inertia = (float(item) for item in coefficients)
        if inverse_inertia <= 0.0 or damping_over_inertia < 0.0:
            raise StateEstimationError("identified spin parameters are non-positive")
        moment_of_inertia = 1.0 / inverse_inertia
        damping_coefficient = damping_over_inertia / inverse_inertia
        covariance = _linear_covariance(whitened_design, whitened_innovation)
        se_inv_inertia = math.sqrt(max(float(covariance[0, 0]), 0.0))
        inertia_se = se_inv_inertia / (inverse_inertia * inverse_inertia)
        gradient = np.asarray(
            [-damping_over_inertia / (inverse_inertia**2), 1.0 / inverse_inertia]
        )
        damping_se = math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
        inertia_estimate = _parameter(
            contract,
            "moment_of_inertia",
            moment_of_inertia,
            inertia_se,
            parameter_multiplier,
        )
        damping_estimate = _parameter(
            contract,
            "damping_coefficient",
            damping_coefficient,
            damping_se,
            parameter_multiplier,
        )
        moment_of_inertia = inertia_estimate.value
        damping_coefficient = damping_estimate.value

        # Reconstruct spin rate with semi-implicit Euler for dω/dt = (T − b·ω)/J.
        predicted = np.empty_like(angular_rate)
        predicted[0] = angular_rate[0]
        for index, step in enumerate(dt):
            predicted[index + 1] = (
                predicted[index] + step * command_mid[index] / moment_of_inertia
            ) / (1.0 + step * damping_coefficient / moment_of_inertia)
            predicted[index + 1] = np.clip(
                predicted[index + 1], state_spec.lower_bound, state_spec.upper_bound
            )
        state_residual = angular_rate - predicted
        sigma = float(np.std(state_residual, ddof=1)) if len(state_residual) > 1 else 0.0
        law = contract.conservation_laws[0]
        violations = [
            ConstraintViolation(
                code="CONSERVATION_RESIDUAL_EXCEEDED",
                index=index + 1,
                magnitude=abs(float(value)),
                limit=law.residual_tolerance,
            )
            for index, value in enumerate(derivative_residual)
            if abs(float(value)) > law.residual_tolerance
        ]
        assessment, ood_score = self._conservation_assessment(
            raw_tolerance=law.residual_tolerance,
            raw_violation_count=len(violations),
            residual_count=len(derivative_residual),
            whitened_innovation=whitened_innovation,
            whitened_design=whitened_design,
            cholesky_diagonal=cholesky_diagonal,
            cholesky_subdiagonal=cholesky_subdiagonal,
            calibration=calibration,
            calibration_version=(
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED"
            ),
        )
        return StateEstimateResult(
            component_archetype_id=contract.component_archetype_id,
            timestamps_s=frame.timestamps_s,
            states=[
                _trajectory(
                    state_spec.state_id,
                    state_spec.canonical_unit,
                    predicted,
                    sigma,
                    state_multiplier,
                )
            ],
            parameters=[inertia_estimate, damping_estimate],
            residuals=[
                _residual("state_reconstruction", state_spec.canonical_unit, state_residual),
                _residual("spin_balance", "rad/s^2", derivative_residual),
                _residual("spin_innovation_whitened", "rad/s^2", whitened_innovation),
            ],
            uncertainty=self._uncertainty(
                sigma,
                calibration,
                self._calibration.version if self._calibration is not None else "NOT_CALIBRATED",
            ),
            ood_score=ood_score,
            constraint_violations=violations,
            conservation_assessment=assessment,
            receipt=self._receipt(frame, contract),
        )


class ProductionEstimatorRegistry:
    """Fail closed: one and only one production estimator per archetype."""

    def __init__(self) -> None:
        self._estimators: dict[str, StateEstimator] = {}

    def register(self, component_archetype_id: str, estimator: StateEstimator) -> None:
        if component_archetype_id in self._estimators:
            raise ValueError(
                f"production estimator already registered for {component_archetype_id}"
            )
        self._estimators[component_archetype_id] = estimator

    def get(self, component_archetype_id: str) -> StateEstimator:
        try:
            return self._estimators[component_archetype_id]
        except KeyError as exc:
            raise KeyError(
                f"no production estimator registered for {component_archetype_id}"
            ) from exc

    def manifest(self) -> dict[str, str]:
        return {
            archetype: getattr(estimator, "estimator_id", type(estimator).__name__)
            for archetype, estimator in sorted(self._estimators.items())
        }


__all__ = [
    "conservation_episode_score",
    "GrayBoxStateEstimator",
    "ProductionEstimatorRegistry",
    "StateEstimationError",
]
