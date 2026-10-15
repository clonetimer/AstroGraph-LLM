"""Tests for the speed-inertia gray-box estimator family.

Covers the ``actuator.spinning_rotor`` archetype / ``inertial_spin_balance``
equation family: J·dω/dt = T − b·ω, identifying the wheel moment of inertia J
and the viscous damping coefficient b.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.state_estimation import GrayBoxStateEstimator
from astrograph.state_estimation.gray_box import StateEstimationError


def _frame(
    archetype: str,
    timestamps: np.ndarray,
    signals: list[tuple[str, str, str, list[float]]],
) -> SemanticTelemetryFrame:
    semantic_signals = [
        SemanticSignal(
            semantic_channel_id=f"asset.component.{index}",
            asset_instance_id="asset-under-test",
            component_archetype_id=archetype,
            port_role_id=port_role,
            quantity_kind_id=quantity_kind,
            canonical_unit=unit,
            direction=1,
            values=values,
            missing_mask=[False] * len(values),
        )
        for index, (port_role, quantity_kind, unit, values) in enumerate(signals)
    ]
    payload = {
        "timestamps": timestamps.tolist(),
        "signals": [item.model_dump(mode="json") for item in semantic_signals],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SemanticTelemetryFrame(
        frame_id="test-inertial-spin-frame",
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=list(timestamps),
        signals=semantic_signals,
        sampling=SamplingProfile(
            row_count=len(timestamps),
            nominal_period_s=float(np.median(np.diff(timestamps))),
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256=digest,
        labels_used=False,
        truth_used=False,
        claim_boundary="test only",
    )


def _synthetic_wheel(
    *,
    moment_of_inertia: float = 0.1,
    damping_coefficient: float = 1.0e-5,
    n: int = 200,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate J·dω/dt = T − b·ω with a time-varying, positive torque.

    The torque is a slow positive sine offset so the spin rate stays inside the
    contract's physical range [-1e5, 1e5] rad/s (a wheel may spin in reverse to
    dump momentum, hence the symmetric bounds matching the abs() saturation
    guard) and the per-step dynamics change slowly enough that the midpoint
    design has negligible discretisation bias.
    """
    time = np.arange(n) * dt
    command = 0.01 + 0.009 * np.sin(0.02 * time)
    rate = np.empty(n)
    rate[0] = 50.0
    for i in range(1, n):
        rate[i] = (
            rate[i - 1]
            + dt * (command[i - 1] - damping_coefficient * rate[i - 1]) / moment_of_inertia
        )
    return time, command, rate


def _estimator() -> GrayBoxStateEstimator:
    return GrayBoxStateEstimator(use_frozen_calibration=False)


class TestInertialSpinContract:
    def test_contract_registered(self):
        contracts = default_dynamics_contracts()
        assert "actuator.spinning_rotor" in contracts
        c = contracts["actuator.spinning_rotor"]
        assert c.equation_family == "inertial_spin_balance"
        assert c.contract_id == "astrograph.physics.actuator.spinning-rotor.v1"
        assert c.parameters[0].parameter_id == "moment_of_inertia"
        assert c.parameters[0].canonical_unit == "kg*m^2"
        assert c.parameters[1].parameter_id == "damping_coefficient"
        assert c.observability.minimum_samples == 12


class TestInertialSpinEstimation:
    def test_recovers_J_and_b(self):
        j_true, b_true = 0.1, 1.0e-5
        time, command, rate = _synthetic_wheel(
            moment_of_inertia=j_true, damping_coefficient=b_true
        )
        frame = _frame(
            "actuator.spinning_rotor",
            time,
            [
                ("state.rate", "mechanical.angular_rate", "rad/s", rate.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.spinning_rotor"]
        result = _estimator().estimate(frame, contract)

        inertia_est = next(
            p for p in result.parameters if p.parameter_id == "moment_of_inertia"
        )
        damping_est = next(
            p for p in result.parameters if p.parameter_id == "damping_coefficient"
        )
        assert inertia_est.value == pytest.approx(j_true, rel=0.01), (
            f"J={inertia_est.value}"
        )
        assert damping_est.value == pytest.approx(b_true, rel=0.05), (
            f"b={damping_est.value}"
        )

    def test_parameter_physical_bounds_respected(self):
        time, command, rate = _synthetic_wheel(moment_of_inertia=0.05, damping_coefficient=5.0e-6)
        frame = _frame(
            "actuator.spinning_rotor",
            time,
            [
                ("state.rate", "mechanical.angular_rate", "rad/s", rate.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.spinning_rotor"]
        result = _estimator().estimate(frame, contract)
        for p in result.parameters:
            assert p.value >= p.physical_lower_bound
            assert p.value <= p.physical_upper_bound

    def test_state_reconstruction_small_residual(self):
        time, command, rate = _synthetic_wheel(moment_of_inertia=0.08, damping_coefficient=2.0e-5)
        frame = _frame(
            "actuator.spinning_rotor",
            time,
            [
                ("state.rate", "mechanical.angular_rate", "rad/s", rate.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.spinning_rotor"]
        result = _estimator().estimate(frame, contract)
        residual = next(r for r in result.residuals if r.residual_id == "state_reconstruction")
        scale = float(np.max(np.abs(rate)))
        assert residual.root_mean_square / max(scale, 1e-9) < 0.05, (
            f"rms={residual.root_mean_square} scale={scale}"
        )

    def test_flat_command_rejected(self):
        """No torque excitation → observability contract must fail closed."""
        time, _, rate = _synthetic_wheel()
        flat_command = np.ones_like(time) * 0.01
        frame = _frame(
            "actuator.spinning_rotor",
            time,
            [
                ("state.rate", "mechanical.angular_rate", "rad/s", rate.tolist()),
                ("command.effort", "mechanical.torque", "N*m", flat_command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.spinning_rotor"]
        with pytest.raises(StateEstimationError):
            _estimator().estimate(frame, contract)

    def test_rank_deficient_command_speed_collinear_rejected(self):
        """Rank-deficiency gate (P2-3): command and speed collinear but each
        non-constant → the [T_mid, −ω_mid] design matrix has rank < 2, so the
        estimator must fail closed with StateEstimationError instead of
        emitting a parameter pair.

        Collinearity: T = A·exp(λt) and ω = K·T.  Because ω ∝ ∫T dt (the
        exponential integrates to itself up to a scale), every midpoint pair
        satisfies ω_mid = K·T_mid, i.e. the two design columns are exactly
        proportional.  The input-variance gate is deliberately NOT what trips
        first — var(T) is well above the observability floor — so the test
        exercises the explicit rank check.
        """
        n, dt = 120, 0.1
        time = np.arange(n) * dt
        lam, amp, scale = 0.05, 0.01, 1000.0
        command = amp * np.exp(lam * time)
        rate = scale * command  # exactly collinear with command, but non-constant
        frame = _frame(
            "actuator.spinning_rotor",
            time,
            [
                ("state.rate", "mechanical.angular_rate", "rad/s", rate.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.spinning_rotor"]
        # The excitation gate is satisfied (var > minimum_input_variance), so the
        # failure must come from the design-matrix rank check.
        assert np.var(command[:-1]) >= contract.observability.minimum_input_variance
        with pytest.raises(StateEstimationError):
            _estimator().estimate(frame, contract)

    def test_labels_rejected(self):
        time, command, rate = _synthetic_wheel()
        frame = _frame(
            "actuator.spinning_rotor",
            time,
            [
                ("state.rate", "mechanical.angular_rate", "rad/s", rate.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.spinning_rotor"]
        bad = frame.model_copy(update={"labels_used": True})
        with pytest.raises(StateEstimationError):
            _estimator().estimate(bad, contract)
