from __future__ import annotations

import numpy as np
import pytest

from astrograph.experiments.so3_star_gyro_temporal import (
    FORMAL_HORIZON_SETS,
    SO3StarGyroContract,
    SO3StarGyroPolicy,
    calibrate_nominal_anchors,
    candidate_policies,
    generate_so3_residuals,
    quaternion_geodesic_deg,
    quaternion_multiply,
    rotation_vector_to_quaternion,
    select_so3_star_gyro,
)


def base_arrays(n: int = 50):
    time = np.arange(n, dtype=float)
    q = np.zeros((n, 4), dtype=float)
    q[:, 0] = 1.0
    gyro = np.zeros((n, 3), dtype=float)
    validity = np.ones((n, 6), dtype=float)
    return time, q, gyro, validity


def test_candidate_grid_is_frozen_and_unique():
    contract = SO3StarGyroContract()
    policies = candidate_policies(contract)
    assert len(FORMAL_HORIZON_SETS) == 8
    assert len(policies) == 224
    assert len({policy.policy_sha256 for policy in policies}) == 224
    assert all(policy.threshold_multiplier >= 1.05 for policy in policies)


def test_geodesic_is_quaternion_sign_invariant():
    q = np.array([[0.9238795325, 0.3826834324, 0.0, 0.0]])
    assert quaternion_geodesic_deg(q, -q)[0] == pytest.approx(0.0, abs=1e-9)


def test_geodesic_returns_known_ninety_degree_rotation():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    ninety = rotation_vector_to_quaternion(np.array([[np.pi / 2.0, 0.0, 0.0]]))
    assert quaternion_geodesic_deg(identity, ninety)[0] == pytest.approx(90.0, abs=1e-8)


def test_constant_rate_propagation_has_near_zero_residual():
    contract = SO3StarGyroContract()
    time, q, gyro, validity = base_arrays()
    gyro[:, 2] = 0.001
    for index in range(1, len(time)):
        increment = rotation_vector_to_quaternion(np.array([[0.0, 0.0, 0.001]]))[0]
        q[index] = quaternion_multiply(q[index - 1], increment)
    residuals = generate_so3_residuals(time, q, gyro, validity, contract)
    assert np.nanmax(residuals[1]) < 1e-6
    assert np.nanmax(residuals[8]) < 1e-6


def test_attitude_step_produces_causal_horizon_evidence():
    contract = SO3StarGyroContract()
    time, q, gyro, validity = base_arrays()
    step = rotation_vector_to_quaternion(np.array([[0.0, 0.0, np.deg2rad(0.1)]]))[0]
    q[35:] = step
    residuals = generate_so3_residuals(time, q, gyro, validity, contract)
    assert np.nanmax(residuals[8][35:43]) > 0.09
    assert np.nanmax(residuals[8][:35]) == pytest.approx(0.0, abs=1e-12)


def test_invalid_star_or_gyro_breaks_horizon_support():
    contract = SO3StarGyroContract()
    time, q, gyro, validity = base_arrays()
    validity[34, 0] = 0.0
    residuals = generate_so3_residuals(time, q, gyro, validity, contract)
    assert np.isnan(residuals[8][35])


def test_policy_rejects_posthoc_boundary_or_unknown_horizon():
    contract = SO3StarGyroContract()
    with pytest.raises(ValueError):
        SO3StarGyroPolicy(
            policy_id="bad",
            contract_sha256=contract.contract_sha256,
            horizon_points=(3,),
            threshold_multiplier=2.0,
            persistence_points=2,
        )
    with pytest.raises(ValueError):
        SO3StarGyroPolicy(
            policy_id="bad",
            contract_sha256=contract.contract_sha256,
            horizon_points=(8,),
            threshold_multiplier=2.0,
            persistence_points=2,
            development_groups=(4, 9),
        )


def test_calibration_and_selection_are_deterministic():
    contract = SO3StarGyroContract()
    time, q, gyro, validity = base_arrays()
    nominal = generate_so3_residuals(time, q, gyro, validity, contract)
    # Create small positive nominal residuals so maximum anchors are well-defined.
    nominal = {h: np.where(np.isfinite(v), v + 1e-4, v) for h, v in nominal.items()}
    anchors = calibrate_nominal_anchors((nominal, nominal), contract)
    fault = {h: np.array(v, copy=True) for h, v in nominal.items()}
    fault[8][35:39] = anchors[8] * 3.0
    policy = SO3StarGyroPolicy(
        policy_id="p",
        contract_sha256=contract.contract_sha256,
        horizon_points=(8,),
        threshold_multiplier=2.0,
        persistence_points=4,
    )
    result = select_so3_star_gyro(fault, anchors, policy)
    assert result["status"] == "SELECTED"
    assert result["selected_index"] == 38
