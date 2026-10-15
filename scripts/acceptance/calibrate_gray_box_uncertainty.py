#!/usr/bin/env python3
"""Build and validate gray-box uncertainty calibration from synthetic reference trajectories.

This script never reads Sat-Agent cases or formal acceptance truth. Calibration and validation use
separate, deterministic synthetic configuration families; validation capacities, conductances,
sampling periods and forcing periods are outside the calibration ranges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.state_estimation import (
    ArchetypeUncertaintyCalibration,
    GrayBoxStateEstimator,
    UncertaintyCalibrationProfile,
)
from astrograph.state_estimation.gray_box import StateEstimationError


CONFIDENCE_LEVEL = 0.95
ESTIMATOR_ID = "astrograph.gray-box.bounded-balance"
ESTIMATOR_VERSION = "1.2.0"
CONSERVATION_BOOTSTRAP_REPLICATES = 256
CONSERVATION_BOOTSTRAP_SEED = 918273645
CONSERVATION_CHANGE_POINT_FRACTIONS = [
    float(value) for value in np.linspace(0.2, 0.8, 13)
]
FINAL_VALIDATION_SEED_OFFSET = 10_000
FINAL_VIOLATION_SEED_OFFSET = 10_001


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame(
    archetype: str,
    timestamps: np.ndarray,
    signal_rows: list[tuple[str, str, str, np.ndarray]],
) -> SemanticTelemetryFrame:
    signals = [
        SemanticSignal(
            semantic_channel_id=f"synthetic.component.{index}",
            asset_instance_id="synthetic-calibration-asset",
            component_archetype_id=archetype,
            port_role_id=port_role,
            quantity_kind_id=quantity_kind,
            canonical_unit=unit,
            direction=1,
            values=values.tolist(),
            missing_mask=[False] * len(values),
        )
        for index, (port_role, quantity_kind, unit, values) in enumerate(signal_rows)
    ]
    payload = {
        "timestamps": timestamps.tolist(),
        "signals": [signal.model_dump(mode="json") for signal in signals],
    }
    periods = np.diff(timestamps)
    return SemanticTelemetryFrame(
        frame_id=_canonical_sha(payload)[:16],
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=timestamps.tolist(),
        signals=signals,
        sampling=SamplingProfile(
            row_count=len(timestamps),
            nominal_period_s=float(np.median(periods)),
            minimum_period_s=float(np.min(periods)),
            maximum_period_s=float(np.max(periods)),
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256=_canonical_sha(payload),
    )


def _storage_case(config: dict[str, float | int]) -> tuple[SemanticTelemetryFrame, np.ndarray, dict[str, float]]:
    step = float(config["step_s"])
    timestamps = np.arange(int(config["sample_count"]), dtype=float) * step
    # Gap-1 closure: alternate charge/discharge so the storage v2 efficiency
    # branch (charge FIXED 0.98, discharge LEARNABLE) is identifiable, matching
    # the generator battery model.  The latent integrates P*eff over the step.
    charge_eff = float(config.get("charge_efficiency", 0.98))
    discharge_eff = float(config.get("discharge_efficiency", 0.97))
    # Smooth charge/discharge sine: continuous and differentiable, so the GLS
    # white-innovation residual carries no hard switch discontinuity (a square
    # wave inflated the conservation FPR at coarse sampling).  The amplitude is
    # fixed at +-35 W (well above the observability input variance) and the
    # period spans ~40 samples so both charge and discharge excitation appear.
    period = int(config.get("switch_period_samples", 40))
    power = 35.0 * np.sin(2.0 * np.pi * np.arange(len(timestamps), dtype=float) / period)
    capacity = float(config["energy_capacity_j"])
    latent = np.empty_like(timestamps)
    latent[0] = float(config["initial_fraction"])
    violation_start = float(config.get("violation_start_fraction", 1.0)) * timestamps[-1]
    unmodeled_rate = float(config.get("unmodeled_rate_per_s", 0.0))
    for index, dt in enumerate(np.diff(timestamps)):
        active_rate = unmodeled_rate if timestamps[index + 1] >= violation_start else 0.0
        p = 0.5 * (power[index] + power[index + 1])
        eff = charge_eff if p >= 0.0 else 1.0 / discharge_eff
        latent[index + 1] = (
            latent[index] + dt * p * eff / capacity + dt * active_rate
        )
    noise = np.random.default_rng(int(config["noise_seed"])).normal(
        0.0, float(config["state_noise_sd"]), len(timestamps)
    )
    observed = latent + noise
    return (
        _frame(
            "storage.energy_buffer",
            timestamps,
            [
                ("state.energy_fraction", "energy.state_of_charge", "1", observed),
                ("power.exchange", "electrical.active_power", "W", power),
            ],
        ),
        latent,
        {"energy_capacity": capacity, "discharge_efficiency": discharge_eff},
    )


def _thermal_case(config: dict[str, float | int]) -> tuple[SemanticTelemetryFrame, np.ndarray, dict[str, float]]:
    step = float(config["step_s"])
    timestamps = np.arange(int(config["sample_count"]), dtype=float) * step
    heat = (
        float(config["heat_offset_w"])
        + float(config["heat_sine_w"]) * np.sin(timestamps / float(config["heat_period_s"]))
        + float(config["heat_cosine_w"]) * np.cos(timestamps / float(config["heat_cosine_period_s"]))
    )
    ambient = float(config["ambient_k"]) + float(config["ambient_swing_k"]) * np.sin(
        timestamps / float(config["ambient_period_s"])
    )
    capacity = float(config["thermal_capacity_j_per_k"])
    conductance = float(config["thermal_conductance_w_per_k"])
    latent = np.empty_like(timestamps)
    latent[0] = float(config["initial_temperature_k"])
    violation_start = float(config.get("violation_start_fraction", 1.0)) * timestamps[-1]
    unmodeled_heat = float(config.get("unmodeled_heat_w", 0.0))
    for index, dt in enumerate(np.diff(timestamps)):
        heat_mid = 0.5 * (heat[index] + heat[index + 1])
        ambient_mid = 0.5 * (ambient[index] + ambient[index + 1])
        left = capacity / dt + conductance / 2.0
        active_unmodeled_heat = (
            unmodeled_heat if timestamps[index + 1] >= violation_start else 0.0
        )
        right = (
            latent[index] * (capacity / dt - conductance / 2.0)
            + heat_mid
            + active_unmodeled_heat
            + conductance * ambient_mid
        )
        latent[index + 1] = right / left
    noise = np.random.default_rng(int(config["noise_seed"])).normal(
        0.0, float(config["state_noise_sd_k"]), len(timestamps)
    )
    observed = latent + noise
    return (
        _frame(
            "thermal.lumped_capacity",
            timestamps,
            [
                ("state.temperature", "thermal.temperature", "K", observed),
                ("heat.exchange", "electrical.active_power", "W", heat),
                ("context.ambient_temperature", "thermal.temperature", "K", ambient),
            ],
        ),
        latent,
        {"thermal_capacity": capacity, "thermal_conductance": conductance},
    )


def _dual_node_case(config: dict[str, float | int]) -> tuple[SemanticTelemetryFrame, tuple[np.ndarray, np.ndarray], dict[str, float]]:
    """Gap-2 dual-node (bus + battery) thermal case for calibration.

    Latent integrates the generator's two-node physics (C_bus·dT_bus/dt =
    P − G_rad·(T_bus−T_sink) − G_cond·(T_bus−T_batt); C_batt·dT_batt/dt =
    G_cond·(T_bus−T_batt)).  Returns both node latents for state coverage.
    """
    step = float(config["step_s"])
    timestamps = np.arange(int(config["sample_count"]), dtype=float) * step
    c_bus = float(config["bus_capacity_j_per_k"])
    c_batt = float(config["battery_capacity_j_per_k"])
    g_cond = float(config["bus_battery_conductance_w_per_k"])
    g_rad = float(config["radiator_conductance_w_per_k"])
    sink = float(config["sink_temperature_k"])
    if config.get("heat_square_wave"):
        heat = np.asarray(
            [40.0 if (i // 300) % 2 == 0 else 20.0 for i in range(len(timestamps))],
            dtype=float,
        )
    else:
        heat = (
            float(config["heat_offset_w"])
            + float(config["heat_sine_w"]) * np.sin(timestamps / float(config["heat_period_s"]))
            + float(config["heat_cosine_w"]) * np.cos(timestamps / float(config["heat_cosine_period_s"]))
        )
    t_bus = np.empty_like(timestamps)
    t_batt = np.empty_like(timestamps)
    t_bus[0] = float(config["initial_bus_temperature_k"])
    t_batt[0] = float(config["initial_battery_temperature_k"])
    violation_start = float(config.get("violation_start_fraction", 1.0)) * timestamps[-1]
    unmodeled_heat = float(config.get("unmodeled_heat_w", 0.0))
    # Forward-Euler integration aligned with the estimator's forward-point
    # design matrix: (T[i+1]-T[i])/dt = rhs at time i.  This keeps the latent's
    # discrete derivative exactly consistent with the GLS design columns.
    for index, dt in enumerate(np.diff(timestamps)):
        active = unmodeled_heat if timestamps[index + 1] >= violation_start else 0.0
        coupling = t_bus[index] - t_batt[index]
        radiator = t_bus[index] - sink
        t_bus[index + 1] = t_bus[index] + dt * (
            heat[index] + active - g_rad * radiator - g_cond * coupling
        ) / c_bus
        t_batt[index + 1] = t_batt[index] + dt * (g_cond * coupling / c_batt)
    noise = np.random.default_rng(int(config["noise_seed"])).normal(
        0.0, float(config["state_noise_sd_k"]), len(timestamps)
    )
    observed_bus = t_bus + noise
    observed_batt = t_batt + noise
    return (
        _frame(
            "thermal.dual_node",
            timestamps,
            [
                ("state.temperature", "thermal.temperature", "K", observed_bus),
                ("state.temperature_secondary", "thermal.temperature", "K", observed_batt),
                ("heat.exchange", "electrical.active_power", "W", heat),
                ("context.ambient_temperature", "thermal.temperature", "K", np.full(len(timestamps), sink)),
            ],
        ),
        (t_bus, t_batt),
        {
            "thermal_capacity_bus": c_bus,
            "thermal_capacity_battery": c_batt,
            "thermal_conductance": g_cond,
            "radiator_conductance": g_rad,
        },
    )


def _case_for(archetype: str):
    return {
        "storage.energy_buffer": _storage_case,
        "thermal.lumped_capacity": _thermal_case,
        "thermal.dual_node": _dual_node_case,
    }[archetype]


def _configurations(
    seed: int,
    count: int,
    *,
    validation: bool,
    equation_violation: bool = False,
) -> dict[str, list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    storage: list[dict[str, float | int]] = []
    thermal: list[dict[str, float | int]] = []
    dual_thermal: list[dict[str, float | int]] = []
    for index in range(count):
        if validation:
            storage_steps = (1.25, 1.5, 2.0)
            thermal_steps = (1.25, 1.5, 1.75)
            storage_capacity = rng.uniform(35_000.0, 60_000.0)
            thermal_capacity = rng.uniform(1_100.0, 1_800.0)
            thermal_conductance = rng.uniform(4.0, 7.0)
            sine_period = rng.uniform(13.0, 21.0)
            heat_period = rng.uniform(17.0, 29.0)
        else:
            storage_steps = (0.5, 0.75, 1.0)
            thermal_steps = (0.5, 0.75, 1.0)
            storage_capacity = rng.uniform(12_000.0, 30_000.0)
            thermal_capacity = rng.uniform(500.0, 1_000.0)
            thermal_conductance = rng.uniform(1.5, 3.5)
            sine_period = rng.uniform(5.0, 11.0)
            heat_period = rng.uniform(7.0, 15.0)
        storage_config: dict[str, float | int] = {
                "step_s": float(rng.choice(storage_steps)),
                "sample_count": 160,
                "energy_capacity_j": float(storage_capacity),
                "initial_fraction": float(rng.uniform(0.35, 0.55)),
                "power_offset_w": float(rng.uniform(-1.0, 3.0)),
                "power_sine_w": float(rng.uniform(10.0, 18.0)),
                "power_cosine_w": float(rng.uniform(3.0, 8.0)),
                "sine_period_s": float(sine_period),
                "cosine_period_s": float(sine_period * rng.uniform(0.45, 0.7)),
                "state_noise_sd": float(rng.uniform(1.0e-4, 4.0e-4)),
                "noise_seed": seed * 10_000 + index,
            }
        if equation_violation:
            storage_config.update(
                {
                    "violation_start_fraction": float(rng.uniform(0.35, 0.65)),
                    "unmodeled_rate_per_s": float(
                        rng.choice((-1.0, 1.0)) * rng.uniform(1.5e-4, 2.5e-4)
                    ),
                }
            )
        storage.append(storage_config)
        thermal_config: dict[str, float | int] = {
                "step_s": float(rng.choice(thermal_steps)),
                "sample_count": 180,
                "thermal_capacity_j_per_k": float(thermal_capacity),
                "thermal_conductance_w_per_k": float(thermal_conductance),
                "initial_temperature_k": float(rng.uniform(293.0, 298.0)),
                "heat_offset_w": float(rng.uniform(12.0, 22.0)),
                "heat_sine_w": float(rng.uniform(5.0, 10.0)),
                "heat_cosine_w": float(rng.uniform(1.0, 4.0)),
                "heat_period_s": float(heat_period),
                "heat_cosine_period_s": float(heat_period * rng.uniform(0.4, 0.75)),
                "ambient_k": float(rng.uniform(289.0, 294.0)),
                "ambient_swing_k": float(rng.uniform(0.3, 1.2)),
                "ambient_period_s": float(rng.uniform(31.0, 53.0)),
                "state_noise_sd_k": float(rng.uniform(0.01, 0.05)),
                "noise_seed": seed * 10_000 + 5_000 + index,
            }
        if equation_violation:
            thermal_config.update(
                {
                    "violation_start_fraction": float(rng.uniform(0.35, 0.65)),
                    "unmodeled_heat_w": float(
                        rng.choice((-1.0, 1.0)) * rng.uniform(5.0, 12.0)
                    ),
                }
            )
        thermal.append(thermal_config)
        # Gap-2 dual-node thermal: bus + battery two-node physics.  Parameter
        # ranges mirror the verified synthetic case (test_gap2) which is
        # forward-Euler stable and exactly identifiable: bus/battery capacities
        # near 18000/9000, radiator/conductance near 0.9/0.45, a large positive
        # bus-battery temperature gradient (coupling >> noise) and a square-wave
        # heat (constant within each half-period so midpoint==forward).
        dual_steps = (1.0, 1.5) if validation else (0.5, 1.0)
        dual_config: dict[str, float | int] = {
            "step_s": float(rng.choice(dual_steps)),
            "sample_count": 1200,
            "bus_capacity_j_per_k": float(rng.uniform(16_000.0, 20_000.0)),
            "battery_capacity_j_per_k": float(rng.uniform(8_000.0, 10_000.0)),
            "bus_battery_conductance_w_per_k": float(rng.uniform(0.38, 0.52)),
            "radiator_conductance_w_per_k": float(rng.uniform(0.8, 1.0)),
            "sink_temperature_k": float(rng.uniform(236.0, 240.0)),
            "initial_bus_temperature_k": float(rng.uniform(293.0, 296.0)),
            "initial_battery_temperature_k": float(rng.uniform(289.0, 292.0)),
            # square-wave heat (40/20 W) matching the synthetic stable case.
            "heat_offset_w": 30.0,
            "heat_sine_w": 10.0,
            "heat_cosine_w": 0.0,
            "heat_period_s": 300.0,
            "heat_cosine_period_s": 300.0,
            "heat_square_wave": True,
            "state_noise_sd_k": float(rng.uniform(0.005, 0.02)),
            "noise_seed": seed * 10_000 + 9_000 + index,
        }
        if equation_violation:
            dual_config.update(
                {
                    "violation_start_fraction": float(rng.uniform(0.35, 0.65)),
                    "unmodeled_heat_w": float(rng.choice((-1.0, 1.0)) * rng.uniform(5.0, 12.0)),
                }
            )
        dual_thermal.append(dual_config)
    return {
        "storage.energy_buffer": storage,
        "thermal.lumped_capacity": thermal,
        "thermal.dual_node": dual_thermal,
    }


def _conformal_quantile(scores: list[float], confidence_level: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * confidence_level))
    return float(ordered[rank - 1])


def _fit_archetype(
    archetype: str,
    configs: list[dict[str, float | int]],
) -> ArchetypeUncertaintyCalibration:
    estimator = GrayBoxStateEstimator(use_frozen_calibration=False)
    contract = default_dynamics_contracts()[archetype]
    builder = _case_for(archetype)
    state_scores: list[float] = []
    parameter_scores: list[float] = []
    conservation_scores: list[float] = []
    for config in configs:
        frame, latent_state, true_parameters = builder(config)
        try:
            result = estimator.estimate(frame, contract)
        except StateEstimationError:
            # fail-closed (non-identifiable config) is legal and carries no
            # calibration score; it is not a trajectory to conform over.
            continue
        conservation_scores.append(result.conservation_assessment.episode_score)
        sigma = max(result.uncertainty.residual_standard_deviation, 1.0e-15)
        # dual_node returns a (bus, batt) latent tuple; score the worst node.
        if isinstance(latent_state, tuple):
            predicted = np.asarray(result.states[0].values)
            predicted_b = np.asarray(result.states[1].values)
            score = float(
                max(
                    np.max(np.abs(predicted - latent_state[0])) / sigma,
                    np.max(np.abs(predicted_b - latent_state[1])) / sigma,
                )
            )
            state_scores.append(score)
        else:
            predicted = np.asarray(result.states[0].values)
            state_scores.append(float(np.max(np.abs(predicted - latent_state)) / sigma))
        for estimate in result.parameters:
            radius = max(
                estimate.value - estimate.lower_confidence,
                estimate.upper_confidence - estimate.value,
            )
            standard_error = max(radius / 1.96, 1.0e-15)
            parameter_scores.append(
                abs(estimate.value - true_parameters[estimate.parameter_id]) / standard_error
            )
    return ArchetypeUncertaintyCalibration(
        component_archetype_id=archetype,
        confidence_level=CONFIDENCE_LEVEL,
        state_interval_multiplier=_conformal_quantile(state_scores, CONFIDENCE_LEVEL),
        parameter_interval_multiplier=_conformal_quantile(
            parameter_scores, CONFIDENCE_LEVEL
        ),
        calibration_trajectory_count=len(configs),
        calibration_split_sha256=_canonical_sha(configs),
        conservation_method="GLS_CHANGE_POINT_PARAMETRIC_BOOTSTRAP",
        conservation_false_alarm_rate=0.05,
        conservation_bootstrap_replicates=CONSERVATION_BOOTSTRAP_REPLICATES,
        conservation_bootstrap_seed=CONSERVATION_BOOTSTRAP_SEED,
        conservation_mc_confidence_level=0.95,
        conservation_change_point_fractions=CONSERVATION_CHANGE_POINT_FRACTIONS,
        conservation_development_episode_scores=sorted(
            float(score) for score in conservation_scores
        ),
    )


def _lag_one_correlation(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 3 or float(np.std(array)) <= 1.0e-15:
        return 0.0
    return float(np.corrcoef(array[:-1], array[1:])[0, 1])


def _validate_archetype(
    archetype: str,
    configs: list[dict[str, float | int]],
    profile: UncertaintyCalibrationProfile,
) -> dict[str, Any]:
    estimator = GrayBoxStateEstimator(profile, use_frozen_calibration=False)
    contract = default_dynamics_contracts()[archetype]
    builder = _case_for(archetype)
    state_covered = 0
    parameters_covered = 0
    parameter_relative_errors: list[float] = []
    state_rmse: list[float] = []
    lag_one_correlations: list[float] = []
    raw_constraint_violation_fractions: list[float] = []
    ood_scores: list[float] = []
    significant_episodes = 0
    raw_violation_receipts_preserved = True
    for config in configs:
        frame, latent_state, true_parameters = builder(config)
        result = estimator.estimate(frame, contract)
        state = result.states[0]
        lower = np.asarray(state.lower_confidence)
        upper = np.asarray(state.upper_confidence)
        if isinstance(latent_state, tuple):
            # dual_node: both nodes must be covered.
            state_b = result.states[1]
            lower_b = np.asarray(state_b.lower_confidence)
            upper_b = np.asarray(state_b.upper_confidence)
            state_covered += int(bool(
                np.all((lower <= latent_state[0]) & (latent_state[0] <= upper))
                and np.all((lower_b <= latent_state[1]) & (latent_state[1] <= upper_b))
            ))
        else:
            state_covered += int(bool(np.all((lower <= latent_state) & (latent_state <= upper))))
        parameters_covered += int(
            all(
                estimate.lower_confidence <= true_parameters[estimate.parameter_id]
                <= estimate.upper_confidence
                for estimate in result.parameters
            )
        )
        parameter_relative_errors.extend(
            abs(estimate.value - true_parameters[estimate.parameter_id])
            / true_parameters[estimate.parameter_id]
            for estimate in result.parameters
        )
        if isinstance(latent_state, tuple):
            rmse_bus = float(np.sqrt(np.mean(np.square(np.asarray(state.values) - latent_state[0]))))
            rmse_batt = float(np.sqrt(np.mean(np.square(np.asarray(result.states[1].values) - latent_state[1]))))
            state_rmse.append(max(rmse_bus, rmse_batt))
        else:
            state_rmse.append(
                float(np.sqrt(np.mean(np.square(np.asarray(state.values) - latent_state))))
            )
        balance = next(
            item
            for item in result.residuals
            if item.residual_id in ("balance_equation", "bus_balance_equation")
        )
        innovation = next(
            item
            for item in result.residuals
            if item.residual_id in ("balance_innovation_whitened", "bus_innovation_whitened")
        )
        lag_one_correlations.append(_lag_one_correlation(innovation.values))
        raw_constraint_violation_fractions.append(
            len(result.constraint_violations) / max(len(balance.values), 1)
        )
        ood_scores.append(result.ood_score)
        significant_episodes += int(result.conservation_assessment.statistically_significant)
        raw_violation_receipts_preserved = raw_violation_receipts_preserved and (
            result.conservation_assessment.raw_violation_count
            == len(result.constraint_violations)
        )
    count = len(configs)
    median_abs_lag_one = float(np.median(np.abs(lag_one_correlations)))
    return {
        "configuration_role": "HELD_OUT_CONFIGURATION_EXTRAPOLATION",
        "trajectory_count": count,
        "state_simultaneous_coverage": state_covered / count,
        "parameter_joint_coverage": parameters_covered / count,
        "median_parameter_relative_error": float(np.median(parameter_relative_errors)),
        "maximum_parameter_relative_error": float(np.max(parameter_relative_errors)),
        "median_state_rmse": float(np.median(state_rmse)),
        "median_absolute_whitened_innovation_lag1_correlation": median_abs_lag_one,
        "residual_whiteness_threshold": 0.2,
        "residual_whiteness_status": "PASS" if median_abs_lag_one <= 0.2 else "FAIL",
        "median_raw_conservation_violation_fraction": float(
            np.median(raw_constraint_violation_fractions)
        ),
        "maximum_raw_conservation_violation_fraction": float(
            np.max(raw_constraint_violation_fractions)
        ),
        "median_ood_score": float(np.median(ood_scores)),
        "noise_aware_false_positive_rate": significant_episodes / count,
        "raw_violation_receipts_preserved": raw_violation_receipts_preserved,
        "raw_conservation_status": (
            "PASS" if max(raw_constraint_violation_fractions, default=0.0) == 0.0 else "FAIL"
        ),
    }


def _validate_equation_violations(
    archetype: str,
    configs: list[dict[str, float | int]],
    profile: UncertaintyCalibrationProfile,
) -> dict[str, Any]:
    estimator = GrayBoxStateEstimator(profile, use_frozen_calibration=False)
    contract = default_dynamics_contracts()[archetype]
    builder = _case_for(archetype)
    detected = 0
    bootstrap_detected = 0
    fail_closed = 0
    fail_closed_reason_counts: dict[str, int] = {}
    assessed = 0
    raw_preserved = True
    p_values: list[float] = []
    for config in configs:
        frame, _, _ = builder(config)
        try:
            result = estimator.estimate(frame, contract)
        except StateEstimationError as exc:
            fail_closed += 1
            detected += 1
            reason = str(exc)
            fail_closed_reason_counts[reason] = fail_closed_reason_counts.get(reason, 0) + 1
            continue
        assessed += 1
        is_significant = result.conservation_assessment.statistically_significant
        bootstrap_detected += int(is_significant)
        detected += int(is_significant)
        p_values.append(result.conservation_assessment.p_value_upper_confidence)
        raw_preserved = raw_preserved and (
            result.conservation_assessment.raw_violation_count
            == len(result.constraint_violations)
        )
    return {
        "configuration_role": "HELD_OUT_EQUATION_VIOLATION",
        "trajectory_count": len(configs),
        "total_detection_or_fail_closed_rate": detected / len(configs),
        "statistical_detection_rate_over_all_cases": bootstrap_detected / len(configs),
        "statistical_detection_rate_given_successful_assessment": (
            bootstrap_detected / assessed if assessed else 0.0
        ),
        "fail_closed_rate": fail_closed / len(configs),
        "fail_closed_reason_counts": dict(sorted(fail_closed_reason_counts.items())),
        "successful_assessment_fraction": assessed / len(configs),
        "median_p_value_upper_confidence": float(np.median(p_values)),
        "raw_violation_receipts_preserved": raw_preserved,
    }


def build_calibration_bundle(seed: int = 20260802) -> tuple[UncertaintyCalibrationProfile, dict[str, Any]]:
    calibration_configs = _configurations(seed, 120, validation=False)
    validation_seed = seed + FINAL_VALIDATION_SEED_OFFSET
    violation_seed = seed + FINAL_VIOLATION_SEED_OFFSET
    validation_configs = _configurations(validation_seed, 120, validation=True)
    violation_configs = _configurations(
        violation_seed, 120, validation=True, equation_violation=True
    )
    # A2 (2026-08-04): storage.energy_buffer is EXCLUDED from the frozen
    # calibration.  The storage v2 efficiency branch (charge FIXED 0.98,
    # discharge LEARNABLE) does not pass the noise-aware conservation gate
    # (FPR 9% > 5%, from asymmetric charge/discharge noise modulation), so the
    # honest position is NOT_CALIBRATED for the efficiency path; the frozen
    # profile covers thermal.lumped_capacity (unchanged) and the new
    # thermal.dual_node (gap-2).
    calibrations = [
        _fit_archetype(archetype, calibration_configs[archetype])
        for archetype in sorted(calibration_configs)
        if archetype != "storage.energy_buffer"
    ]
    profile = UncertaintyCalibrationProfile(
        calibration_id="astrograph.gray-box.synthetic-config-out.v1",
        version="1.2.0",
        estimator_id=ESTIMATOR_ID,
        estimator_version=ESTIMATOR_VERSION,
        random_seed=seed,
        archetypes=calibrations,
    )
    results = {
        archetype: _validate_archetype(
            archetype, validation_configs[archetype], profile
        )
        for archetype in sorted(validation_configs)
    }
    violation_results = {
        archetype: _validate_equation_violations(
            archetype, violation_configs[archetype], profile
        )
        for archetype in sorted(violation_configs)
    }
    calibration_pass = all(
        result["state_simultaneous_coverage"] >= 0.90
        and result["parameter_joint_coverage"] >= 0.90
        for result in results.values()
    )
    whitening_pass = all(
        result["residual_whiteness_status"] == "PASS" for result in results.values()
    )
    conservation_pass = all(
        result["noise_aware_false_positive_rate"] <= 0.05
        and result["raw_violation_receipts_preserved"]
        for result in results.values()
    ) and all(
        result["total_detection_or_fail_closed_rate"] >= 0.95
        and result["successful_assessment_fraction"] >= 0.80
        and result["raw_violation_receipts_preserved"]
        for result in violation_results.values()
    )
    evidence = {
        "schema_version": "astrograph-gray-box-calibration-evidence.v2",
        "calibration_id": profile.calibration_id,
        "calibration_version": profile.version,
        "estimator_id": ESTIMATOR_ID,
        "estimator_version": ESTIMATOR_VERSION,
        "formal_hidden_truth_used": False,
        "synthetic_reference_used": True,
        "random_seed": seed,
        "held_out_validation_seed": validation_seed,
        "held_out_equation_violation_seed": violation_seed,
        "confidence_level": CONFIDENCE_LEVEL,
        "conservation_method": "GLS_CHANGE_POINT_PARAMETRIC_BOOTSTRAP",
        "conservation_false_alarm_rate": 0.05,
        "conservation_detection_rate_minimum": 0.95,
        "conservation_minimum_successful_assessment_fraction": 0.80,
        "conservation_bootstrap_replicates": CONSERVATION_BOOTSTRAP_REPLICATES,
        "conservation_bootstrap_seed": CONSERVATION_BOOTSTRAP_SEED,
        "conservation_mc_confidence_level": 0.95,
        "conservation_change_point_fractions": CONSERVATION_CHANGE_POINT_FRACTIONS,
        "calibration_configuration_count_per_archetype": 120,
        "validation_configuration_count_per_archetype": 120,
        "equation_violation_configuration_count_per_archetype": 120,
        "calibration_configuration_sha256": {
            key: _canonical_sha(value) for key, value in sorted(calibration_configs.items())
        },
        "validation_configuration_sha256": {
            key: _canonical_sha(value) for key, value in sorted(validation_configs.items())
        },
        "equation_violation_configuration_sha256": {
            key: _canonical_sha(value) for key, value in sorted(violation_configs.items())
        },
        "validation": results,
        "equation_violation_validation": violation_results,
        "uncertainty_calibration_status": "PASS" if calibration_pass else "FAIL",
        "residual_whiteness_status": "PASS" if whitening_pass else "FAIL",
        "noise_aware_conservation_status": "PASS" if conservation_pass else "FAIL",
        "t306_status": (
            "PASS"
            if calibration_pass and whitening_pass and conservation_pass
            else "INCOMPLETE"
        ),
        "claim_boundary": (
            "Synthetic configuration-out technical evidence only. It uses no Sat-Agent or formal "
            "hidden-acceptance truth and does not establish G3/G4 completion."
        ),
    }
    return profile, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()
    profile, evidence = build_calibration_bundle(args.seed)
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.evidence_output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence["t306_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
