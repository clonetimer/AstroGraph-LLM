"""Discrete state-consistency diagnosis for DXC/ADAPT.

R25's continuous residual path intentionally admitted only numeric sensors with
verified engineering units.  ADAPT also exposes bool PositionSensor instances
without engineering units.  Those values are *state observations*, not
continuous engineering measurements, and are therefore admitted through a
separate contract rather than weakening the numeric-unit gate.

The branch is deliberately narrow:

* only catalog-declared ``PositionSensor`` instances with bool data type;
* the position sensor must have exactly one direct engineering-graph neighbor;
* nominal command-associated transitions are suppressed only through an exact
  component-instance command binding;
* uncommanded closed->open transitions map only to an admitted ``StuckOpen`` or
  ``FailedOpen`` mode on the adjacent component;
* uncommanded open->closed transitions map only to admitted ``StuckClosed``.

This is deterministic consistency evidence for development diagnosis.  It is
not a calibrated probability and does not create a formal EvidenceFact.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCDiscreteConsistencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    startup_s: float = Field(ge=0.0, default=30.0)
    command_suppression_s: float = Field(gt=0.0, default=1.0)
    position_sensor_component_type: str = "PositionSensor"
    open_state_value: bool = False


class DXCDiscreteFaultEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    time_s: float
    sensor_component_id: str
    controlled_component_id: str
    previous_state: bool
    observed_state: bool
    predicted_fault_id: str
    reason_code: str
    source_ids: tuple[str, ...]


class DXCDiscreteConsistencyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-dxc-discrete-consistency-result.v1"
    events: tuple[DXCDiscreteFaultEvent, ...]
    predicted_fault_ids: tuple[str, ...]
    first_event_time_s: float | None = None
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_semantics: str = "DISCRETE_STATE_CONSISTENCY_NOT_PROBABILITY"
    formal_evidence_eligible: bool = False
    claim_boundary: str = (
        "Exact XML topology and command bindings plus bool PositionSensor transitions. "
        "The branch does not relax continuous-sensor unit admission, does not establish posterior probability, "
        "and does not create formal EvidenceFact records."
    )

    @model_validator(mode="after")
    def integrity(self) -> "DXCDiscreteConsistencyResult":
        payload = {
            "events": [item.model_dump(mode="json") for item in self.events],
            "predicted_fault_ids": self.predicted_fault_ids,
            "first_event_time_s": self.first_event_time_s,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC discrete consistency result_sha256 mismatch")
        return self


def _position_bindings(catalog: Mapping[str, Any], config: DXCDiscreteConsistencyConfig) -> dict[str, str]:
    components = {str(item["component_id"]): item for item in catalog.get("components", [])}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in catalog.get("connections", []):
        left = str(edge.get("c1") or "")
        right = str(edge.get("c2") or "")
        if left and right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    result: dict[str, str] = {}
    for component_id, component in components.items():
        if str(component.get("component_type") or "") != config.position_sensor_component_type:
            continue
        if str(component.get("data_type") or "").lower() != "bool":
            raise ValueError(f"DXC_DISCRETE_POSITION_SENSOR_NOT_BOOL:{component_id}")
        neighbors = sorted(adjacency.get(component_id, set()))
        if len(neighbors) != 1:
            raise ValueError(f"DXC_DISCRETE_POSITION_SENSOR_TOPOLOGY_AMBIGUOUS:{component_id}")
        target = neighbors[0]
        if target not in components:
            raise ValueError(f"DXC_DISCRETE_POSITION_SENSOR_TARGET_UNKNOWN:{component_id}")
        result[component_id] = target
    if not result:
        raise ValueError("DXC_DISCRETE_POSITION_SENSOR_BINDINGS_EMPTY")
    return result


def _fault_for_transition(
    component: Mapping[str, Any],
    previous: bool,
    observed: bool,
    *,
    open_state_value: bool,
) -> tuple[str | None, str | None]:
    modes = set(map(str, component.get("fault_modes") or []))
    previous_open = previous == open_state_value
    observed_open = observed == open_state_value
    if not previous_open and observed_open:
        if "StuckOpen" in modes:
            return "StuckOpen", "UNCOMMANDED_CLOSED_TO_OPEN_STUCK_OPEN"
        if "FailedOpen" in modes:
            return "FailedOpen", "UNCOMMANDED_CLOSED_TO_OPEN_FAILED_OPEN"
    if previous_open and not observed_open and "StuckClosed" in modes:
        return "StuckClosed", "UNCOMMANDED_OPEN_TO_CLOSED_STUCK_CLOSED"
    return None, None


def diagnose_discrete_state_consistency(
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    config: DXCDiscreteConsistencyConfig | None = None,
) -> DXCDiscreteConsistencyResult:
    """Return deterministic uncommanded component-state inconsistencies."""
    from .data_exposure import guard_prediction_generation

    guard_prediction_generation()
    cfg = config or DXCDiscreteConsistencyConfig()
    bindings = _position_bindings(catalog, cfg)
    components = {str(item["component_id"]): item for item in catalog.get("components", [])}
    commands_by_component: dict[str, set[str]] = defaultdict(set)
    command_target: dict[str, str] = {}
    for binding in catalog.get("command_bindings", []):
        command_id = str(binding.get("command_id") or "")
        component_id = str(binding.get("component_id") or "")
        if not command_id or not component_id:
            continue
        if command_id in command_target and command_target[command_id] != component_id:
            raise ValueError(f"DXC_DISCRETE_COMMAND_BINDING_AMBIGUOUS:{command_id}")
        command_target[command_id] = component_id
        commands_by_component[component_id].add(command_id)

    last_position: dict[str, bool] = {}
    last_command_time_s: dict[str, float] = {}
    events: list[DXCDiscreteFaultEvent] = []
    previous_tick: int | None = None

    for record in records:
        kind = str(record.get("kind") or "")
        tick = int(record.get("time_tick", -1))
        if tick < 0 or (previous_tick is not None and tick < previous_tick):
            raise ValueError("DXC_DISCRETE_TIME_INVALID")
        previous_tick = tick
        time_s = tick / 1000.0
        if kind == "command":
            command_id = str(record.get("command_id") or "")
            target = command_target.get(command_id)
            if target is not None:
                last_command_time_s[target] = time_s
            continue
        if kind != "sensors":
            raise ValueError("DXC_DISCRETE_UNADMITTED_RECORD_KIND")
        values = record.get("values") or {}
        for sensor_id, target_id in bindings.items():
            if sensor_id not in values:
                continue
            observed = values[sensor_id]
            if not isinstance(observed, bool):
                raise ValueError(f"DXC_DISCRETE_POSITION_SENSOR_VALUE_NOT_BOOL:{sensor_id}")
            if sensor_id not in last_position:
                last_position[sensor_id] = observed
                continue
            previous = last_position[sensor_id]
            last_position[sensor_id] = observed
            if previous == observed or time_s < cfg.startup_s:
                continue
            command_time = last_command_time_s.get(target_id)
            if command_time is not None and 0.0 <= time_s - command_time <= cfg.command_suppression_s:
                continue
            mode, reason = _fault_for_transition(
                components[target_id],
                previous,
                observed,
                open_state_value=cfg.open_state_value,
            )
            if mode is None or reason is None:
                continue
            events.append(DXCDiscreteFaultEvent(
                time_s=time_s,
                sensor_component_id=sensor_id,
                controlled_component_id=target_id,
                previous_state=previous,
                observed_state=observed,
                predicted_fault_id=f"{target_id}:{mode}",
                reason_code=reason,
                source_ids=(sensor_id, target_id),
            ))

    faults = tuple(sorted({item.predicted_fault_id for item in events}))
    first = min((item.time_s for item in events), default=None)
    payload = {
        "events": [item.model_dump(mode="json") for item in events],
        "predicted_fault_ids": faults,
        "first_event_time_s": first,
    }
    return DXCDiscreteConsistencyResult(
        events=tuple(events),
        predicted_fault_ids=faults,
        first_event_time_s=first,
        result_sha256=_digest(payload),
    )


__all__ = [
    "DXCDiscreteConsistencyConfig",
    "DXCDiscreteConsistencyResult",
    "DXCDiscreteFaultEvent",
    "diagnose_discrete_state_consistency",
]
