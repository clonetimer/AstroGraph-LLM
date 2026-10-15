"""Tests for the third gray-box estimator: first-order actuator lag."""

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
        frame_id="test-actuator-frame",
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


def _synthetic_actuator(*, gain: float = 0.9, tau: float = 2.0, n: int = 200, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a first-order actuator lag with a varying command.

    d(response)/dt = (gain·command − response)/τ. Command is a slow sine.
    """
    time = np.arange(n) * dt
    command = 1.0 + 0.5 * np.sin(0.05 * time)  # gentle, positive command
    response = np.empty(n)
    response[0] = 0.0
    for i in range(1, n):
        response[i] = response[i - 1] + dt * (gain * command[i - 1] - response[i - 1]) / tau
    return time, command, response


def _estimator() -> GrayBoxStateEstimator:
    return GrayBoxStateEstimator(use_frozen_calibration=False)


class TestActuatorContract:
    def test_contract_registered(self):
        contracts = default_dynamics_contracts()
        assert "actuator.controlled_effort" in contracts
        c = contracts["actuator.controlled_effort"]
        assert c.equation_family == "first_order_actuator_lag"
        assert c.parameters[0].parameter_id == "actuator_gain"
        assert c.parameters[1].parameter_id == "actuator_time_constant"


class TestActuatorEstimation:
    def test_recovers_gain_and_tau(self):
        gain_true, tau_true = 0.9, 2.0
        time, command, response = _synthetic_actuator(gain=gain_true, tau=tau_true)
        frame = _frame(
            "actuator.controlled_effort",
            time,
            [
                ("response.effort", "mechanical.torque", "N*m", response.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.controlled_effort"]
        result = _estimator().estimate(frame, contract)

        gain_est = next(p for p in result.parameters if p.parameter_id == "actuator_gain")
        tau_est = next(p for p in result.parameters if p.parameter_id == "actuator_time_constant")
        # Gain should be near 0.9 (bounded [0, 10]); tau near 2.0.
        assert gain_est.value == pytest.approx(gain_true, abs=0.15), f"gain={gain_est.value}"
        assert tau_est.value == pytest.approx(tau_true, abs=0.5), f"tau={tau_est.value}"

    def test_parameter_physical_bounds_respected(self):
        time, command, response = _synthetic_actuator(gain=0.5, tau=1.0)
        frame = _frame(
            "actuator.controlled_effort",
            time,
            [
                ("response.effort", "mechanical.torque", "N*m", response.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.controlled_effort"]
        result = _estimator().estimate(frame, contract)
        for p in result.parameters:
            assert p.value >= p.physical_lower_bound
            assert p.value <= p.physical_upper_bound

    def test_state_reconstruction_small_residual(self):
        time, command, response = _synthetic_actuator(gain=0.8, tau=3.0)
        frame = _frame(
            "actuator.controlled_effort",
            time,
            [
                ("response.effort", "mechanical.torque", "N*m", response.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.controlled_effort"]
        result = _estimator().estimate(frame, contract)
        residual = next(r for r in result.residuals if r.residual_id == "state_reconstruction")
        # Normalised RMS should be small relative to response scale.
        scale = float(np.max(np.abs(response)))
        assert residual.root_mean_square / max(scale, 1e-9) < 0.05, f"rms={residual.root_mean_square} scale={scale}"

    def test_flat_command_rejected(self):
        """No command excitation → observability contract must fail closed."""
        time, _, response = _synthetic_actuator(gain=0.9, tau=2.0)
        flat_command = np.ones_like(time) * 1.0
        frame = _frame(
            "actuator.controlled_effort",
            time,
            [
                ("response.effort", "mechanical.torque", "N*m", response.tolist()),
                ("command.effort", "mechanical.torque", "N*m", flat_command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.controlled_effort"]
        with pytest.raises(StateEstimationError):
            _estimator().estimate(frame, contract)

    def test_labels_rejected(self):
        time, command, response = _synthetic_actuator()
        frame = _frame(
            "actuator.controlled_effort",
            time,
            [
                ("response.effort", "mechanical.torque", "N*m", response.tolist()),
                ("command.effort", "mechanical.torque", "N*m", command.tolist()),
            ],
        )
        contract = default_dynamics_contracts()["actuator.controlled_effort"]
        bad = frame.model_copy(update={"labels_used": True})
        with pytest.raises(StateEstimationError):
            _estimator().estimate(bad, contract)
