from __future__ import annotations

import time
from enum import Enum
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field


class LifecycleStage(str, Enum):
    DRAFT = 'DRAFT'
    STAGED = 'STAGED'
    DEPLOYED = 'DEPLOYED'
    ROLLED_BACK = 'ROLLED_BACK'
    FAILED = 'FAILED'


class RegressionGateDecision(str, Enum):
    PASS = 'PASS'
    FAIL = 'FAIL'


class LifecycleEvent(BaseModel):
    """One auditable lifecycle transition record."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-model-lifecycle-event.v1'
    event_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    decision: RegressionGateDecision | None = None
    gate_evidence: str = ''
    rollback_target: str | None = None
    reason: str = Field(default='')
    emitted_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class LifecycleState(BaseModel):
    """Persistent lifecycle state of one model version."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-model-lifecycle-state.v1'
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    stage: LifecycleStage = LifecycleStage.DRAFT
    deployed_at: str = ''
    regression_gate_passed: bool | None = None
    regression_evidence: str = ''
    history: list[LifecycleEvent] = Field(default_factory=list)


class RegressionGate(Protocol):
    """Deterministic gate deciding whether a staged model may deploy."""

    def check(self, model_id: str, model_version: str) -> RegressionGateDecision:
        ...


class ModelLifecycleManager:
    """Deploy -> upgrade -> rollback state machine with a mandatory regression gate.

    Discipline:
    - A model may reach DEPLOYED only after the regression gate returns PASS.
    - Deploying a new version is an upgrade; the previous deployed version is the
      rollback target and is recorded in the event history.
    - rollback restores the previous DEPLOYED version's state; human decision is
      required (the caller supplies ``human_approved``) - nothing auto-rolls-back.
    - The gate is advisory-writable only through this manager; registry promotion
      remains behind the existing human gate (DEV_GOAL).
    """

    def __init__(self, gate: RegressionGate, *, registry: object | None = None) -> None:
        self._gate = gate
        self._registry = registry
        self._states: dict[tuple[str, str], LifecycleState] = {}
        self._deployed: dict[str, str] = {}  # model_id -> current deployed version
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f'LIFECYCLE-{self._event_counter:06d}'

    def _append(self, state: LifecycleState, event: LifecycleEvent) -> None:
        state.history.append(event)
        state.stage = event.to_stage

    def stage(self, model_id: str, model_version: str) -> LifecycleState:
        """Move DRAFT -> STAGED (a candidate is placed under the regression gate)."""
        key = (model_id, model_version)
        state = self._states.setdefault(key, LifecycleState(model_id=model_id, model_version=model_version))
        if state.stage != LifecycleStage.DRAFT:
            raise ValueError(f'cannot stage {model_id}@{model_version} from {state.stage.value}')
        event = LifecycleEvent(
            event_id=self._next_event_id(), model_id=model_id, model_version=model_version,
            from_stage=LifecycleStage.DRAFT, to_stage=LifecycleStage.STAGED,
            reason='candidate staged for regression gate',
        )
        self._append(state, event)
        return state

    def deploy(self, model_id: str, model_version: str, *, human_approved: bool) -> LifecycleState:
        """Run the regression gate and, on PASS + human approval, promote to DEPLOYED.

        The previously deployed version (if any) becomes the rollback target.
        """
        key = (model_id, model_version)
        state = self._states.get(key)
        if state is None or state.stage != LifecycleStage.STAGED:
            raise ValueError(f'cannot deploy {model_id}@{model_version}: not staged')
        decision = self._gate.check(model_id, model_version)
        state.regression_gate_passed = decision == RegressionGateDecision.PASS
        if decision == RegressionGateDecision.FAIL:
            event = LifecycleEvent(
                event_id=self._next_event_id(), model_id=model_id, model_version=model_version,
                from_stage=LifecycleStage.STAGED, to_stage=LifecycleStage.FAILED,
                decision=decision, gate_evidence='regression gate FAIL',
                reason='regression gate rejected the candidate',
            )
            self._append(state, event)
            return state
        if not human_approved:
            raise ValueError(f'deploy of {model_id}@{model_version} requires explicit human approval')
        previous = self._deployed.get(model_id)
        event = LifecycleEvent(
            event_id=self._next_event_id(), model_id=model_id, model_version=model_version,
            from_stage=LifecycleStage.STAGED, to_stage=LifecycleStage.DEPLOYED,
            decision=decision, gate_evidence='regression gate PASS + human approval',
            rollback_target=previous, reason='deployment approved',
        )
        self._append(state, event)
        state.deployed_at = event.emitted_at
        self._deployed[model_id] = model_version
        return state

    def rollback(self, model_id: str, *, human_approved: bool) -> LifecycleState:
        """Restore the previous deployed version (human-gated)."""
        current = self._deployed.get(model_id)
        if current is None:
            raise ValueError(f'no deployed version for {model_id}')
        state = self._states.get((model_id, current))
        if state is None or state.stage != LifecycleStage.DEPLOYED:
            raise ValueError(f'current version {model_id}@{current} is not DEPLOYED')
        if not human_approved:
            raise ValueError('rollback requires explicit human approval')
        event = LifecycleEvent(
            event_id=self._next_event_id(), model_id=model_id, model_version=current,
            from_stage=LifecycleStage.DEPLOYED, to_stage=LifecycleStage.ROLLED_BACK,
            reason='human-approved rollback',
        )
        self._append(state, event)
        # Restore previous deployed version if it is still staged/deployed in history.
        previous = next(
            (e.rollback_target for e in reversed(state.history)
             if e.rollback_target is not None and e.to_stage == LifecycleStage.DEPLOYED),
            None,
        )
        if previous is not None and (previous_key := (model_id, previous)) in self._states:
            self._deployed[model_id] = previous
            prev_state = self._states[previous_key]
            if prev_state.stage != LifecycleStage.DEPLOYED:
                restore = LifecycleEvent(
                    event_id=self._next_event_id(), model_id=model_id, model_version=previous,
                    from_stage=prev_state.stage, to_stage=LifecycleStage.DEPLOYED,
                    reason='restored as rollback target after human-approved rollback',
                )
                self._append(prev_state, restore)
        return state

    def state(self, model_id: str, model_version: str) -> LifecycleState:
        key = (model_id, model_version)
        if key not in self._states:
            raise KeyError(f'unknown lifecycle state: {model_id}@{model_version}')
        return self._states[key]

    def deployed_version(self, model_id: str) -> str | None:
        return self._deployed.get(model_id)


class AlwaysPassGate:
    """Deterministic PASS gate for tests and dry runs."""

    def check(self, model_id: str, model_version: str) -> RegressionGateDecision:
        return RegressionGateDecision.PASS


class RejectingGate:
    """Deterministic FAIL gate for tests."""

    def check(self, model_id: str, model_version: str) -> RegressionGateDecision:
        return RegressionGateDecision.FAIL


__all__ = [
    'LifecycleStage', 'RegressionGateDecision', 'LifecycleEvent', 'LifecycleState',
    'RegressionGate', 'ModelLifecycleManager', 'AlwaysPassGate', 'RejectingGate',
]