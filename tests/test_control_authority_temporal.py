from __future__ import annotations

import numpy as np
import pytest

from astrograph.experiments.control_authority_temporal import (
    ControlAuthorityContract,
    ControlAuthorityPolicy,
    candidate_policies,
    generate_control_authority_score,
    select_control_authority,
)


def arrays(length: int = 60):
    return (
        np.zeros((length, 4), dtype=float),
        np.zeros((length, 4), dtype=float),
        np.zeros((length, 4), dtype=float),
        np.zeros((length, 4), dtype=float),
    )


def test_candidate_family_is_fixed():
    contract = ControlAuthorityContract()
    policies = candidate_policies(contract)
    assert len(policies) == 60
    assert all(policy.threshold >= 0.5 for policy in policies)
    assert all(policy.minimum_hits <= policy.evidence_window_points for policy in policies)


def test_open_torque_path_produces_unit_delivery_loss():
    command, residual, commanded_current, applied_current = arrays()
    command[35:, 1] = 0.002
    residual[35:, 1] = 0.002
    score, valid, source = generate_control_authority_score(
        command,
        residual,
        commanded_current,
        applied_current,
        ControlAuthorityContract(),
    )
    assert np.allclose(score[35:], 1.0)
    assert valid[35:].all()
    assert np.all(source[35:] == 1)


def test_current_delivery_loss_is_activity_gated():
    command, residual, commanded_current, applied_current = arrays()
    commanded_current[35:, 2] = 0.10
    applied_current[35:, 2] = 0.01
    score, valid, source = generate_control_authority_score(
        command,
        residual,
        commanded_current,
        applied_current,
        ControlAuthorityContract(),
    )
    assert np.allclose(score[35:], 0.9)
    assert valid[35:].all()
    assert np.all(source[35:] == 2)


def test_no_command_means_unobservable_not_fault():
    command, residual, commanded_current, applied_current = arrays()
    residual[35:, 0] = 1.0
    score, valid, source = generate_control_authority_score(
        command,
        residual,
        commanded_current,
        applied_current,
        ControlAuthorityContract(),
    )
    assert np.all(score == 0.0)
    assert not valid.any()
    assert np.all(source == 0)


def test_rolling_supported_hits_trigger_without_label():
    time = np.arange(60, dtype=float)
    score = np.zeros(60, dtype=float)
    valid = np.zeros(60, dtype=bool)
    source = np.zeros(60, dtype=np.uint8)
    for index in (35, 37, 38):
        score[index] = 1.0
        valid[index] = True
        source[index] = 1
    contract = ControlAuthorityContract()
    policy = ControlAuthorityPolicy(
        policy_id="test",
        contract_sha256=contract.contract_sha256,
        threshold=0.9,
        evidence_window_points=5,
        minimum_hits=3,
    )
    selected = select_control_authority(time, score, valid, source, policy)
    assert selected["status"] == "SELECTED"
    assert selected["selected_index"] == 38
    assert selected["notification_available_time_s"] == 41.0


def test_groups_4_9_cannot_be_reintroduced():
    contract = ControlAuthorityContract()
    with pytest.raises(ValueError, match="boundary"):
        ControlAuthorityPolicy(
            policy_id="bad",
            contract_sha256=contract.contract_sha256,
            threshold=0.9,
            evidence_window_points=5,
            minimum_hits=2,
            development_groups=(4, 9),
        )
