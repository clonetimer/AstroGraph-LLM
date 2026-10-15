"""AI-Native Phase 5: model lifecycle state machine tests (deploy->upgrade->rollback)."""
from __future__ import annotations

import pytest

from astrograph.data_driven.lifecycle import (
    AlwaysPassGate, LifecycleStage, ModelLifecycleManager, RejectingGate,
)


def test_deploy_requires_staging_and_human_approval():
    manager = ModelLifecycleManager(AlwaysPassGate())
    with pytest.raises(ValueError):
        manager.deploy('model-a', '1.0.0', human_approved=True)
    manager.stage('model-a', '1.0.0')
    with pytest.raises(ValueError):
        manager.deploy('model-a', '1.0.0', human_approved=False)
    state = manager.deploy('model-a', '1.0.0', human_approved=True)
    assert state.stage == LifecycleStage.DEPLOYED
    assert state.regression_gate_passed is True
    assert manager.deployed_version('model-a') == '1.0.0'


def test_gate_failure_blocks_deployment():
    manager = ModelLifecycleManager(RejectingGate())
    manager.stage('model-a', '1.0.0')
    state = manager.deploy('model-a', '1.0.0', human_approved=True)
    assert state.stage == LifecycleStage.FAILED
    assert state.regression_gate_passed is False
    assert manager.deployed_version('model-a') is None


def test_upgrade_records_rollback_target():
    manager = ModelLifecycleManager(AlwaysPassGate())
    manager.stage('model-a', '1.0.0')
    manager.deploy('model-a', '1.0.0', human_approved=True)
    manager.stage('model-a', '2.0.0')
    state = manager.deploy('model-a', '2.0.0', human_approved=True)
    assert manager.deployed_version('model-a') == '2.0.0'
    deploy_event = [e for e in state.history if e.to_stage == LifecycleStage.DEPLOYED][-1]
    assert deploy_event.rollback_target == '1.0.0'


def test_rollback_restores_previous_deployed_version():
    manager = ModelLifecycleManager(AlwaysPassGate())
    manager.stage('model-a', '1.0.0')
    manager.deploy('model-a', '1.0.0', human_approved=True)
    manager.stage('model-a', '2.0.0')
    manager.deploy('model-a', '2.0.0', human_approved=True)
    rolled = manager.rollback('model-a', human_approved=True)
    assert rolled.stage == LifecycleStage.ROLLED_BACK
    assert manager.deployed_version('model-a') == '1.0.0'


def test_rollback_requires_human_approval():
    manager = ModelLifecycleManager(AlwaysPassGate())
    manager.stage('model-a', '1.0.0')
    manager.deploy('model-a', '1.0.0', human_approved=True)
    with pytest.raises(ValueError):
        manager.rollback('model-a', human_approved=False)
    assert manager.deployed_version('model-a') == '1.0.0'


def test_full_deploy_upgrade_rollback_drill():
    """Acceptance-criterion 2: one complete deploy->upgrade->rollback drill."""
    manager = ModelLifecycleManager(AlwaysPassGate())
    manager.stage('adcs-v522', 'v1')
    manager.deploy('adcs-v522', 'v1', human_approved=True)
    manager.stage('adcs-v522', 'v2')
    manager.deploy('adcs-v522', 'v2', human_approved=True)
    manager.rollback('adcs-v522', human_approved=True)
    assert manager.deployed_version('adcs-v522') == 'v1'
    v1 = manager.state('adcs-v522', 'v1')
    assert v1.stage == LifecycleStage.DEPLOYED
    v2 = manager.state('adcs-v522', 'v2')
    assert v2.stage == LifecycleStage.ROLLED_BACK
    # Audit trail is append-only and ordered.
    events = [event for event in v1.history + v2.history]
    assert len(events) >= 5
    assert [event.event_id for event in events] == sorted(event.event_id for event in events)


def test_unknown_state_raises():
    manager = ModelLifecycleManager(AlwaysPassGate())
    with pytest.raises(KeyError):
        manager.state('nope', '1.0.0')