"""DXC reduced-order physical simulation consistency for unresolved plant hypotheses.

R29 adds a narrow deterministic core escape above R28.  When R28 intentionally
returns an unresolved physical-response case, this module may evaluate explicit
ADAPT ``LightBulb*W:FailedOff`` candidates using a candidate-specific numerical
branch-current simulation.  Candidate generation is derived from the admitted
engineering graph and component ratings; no fault truth, case ID, or LLM input
is consumed.

The initial physical scope is intentionally small.  It is a reduced-order
resistive-load model with observed bus voltage as an exogenous boundary
condition, plus local temperature and relay-position consistency constraints.
It is not VirtualADAPT, a full network solver, or a posterior probability model.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.simulation.adapt_reduced_order import (
    ADAPTLightBulbReducedOrderProvider,
    ADAPTResistiveLoadSpec,
    build_branch_boundary_condition,
)
from astrograph.simulation.diagnostic_provider import (
    SimulationHypothesisCandidate,
    build_simulation_trace,
    compare_counterfactual_diagnoses,
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCReducedOrderPhysicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    pre_window_s: float = Field(gt=0.0, default=10.0)
    post_window_s: float = Field(gt=0.0, default=10.0)
    temperature_late_start_s: float = Field(gt=0.0, default=30.0)
    temperature_late_window_s: float = Field(gt=0.0, default=30.0)
    max_topology_hops: int = Field(ge=2, default=5)
    max_failed_bulbs_per_branch: int = Field(ge=1, default=3)
    min_voltage_rms_v: float = Field(gt=0.0, default=80.0)
    max_voltage_relative_change: float = Field(gt=0.0, default=0.08)
    min_current_drop_a: float = Field(gt=0.0, default=0.05)
    min_current_drop_z: float = Field(gt=0.0, default=6.0)
    current_noise_floor_a: float = Field(gt=0.0, default=0.01)
    temperature_noise_floor_f: float = Field(gt=0.0, default=0.5)
    max_power_relative_error: float = Field(gt=0.0, default=0.60)
    min_temperature_cooling_z: float = Field(gt=0.0, default=3.0)
    command_guard_s: float = Field(gt=0.0, default=2.0)
    unresolved_reason_required: str = "SENSOR_OFFSET_UNCORROBORATED_PHYSICAL_RESPONSE_UNRESOLVED"


class DXCReducedOrderCandidateFit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    diagnosis_id: str = Field(min_length=1)
    fault_ids: tuple[str, ...] = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    current_sensor_id: str = Field(min_length=1)
    voltage_sensor_id: str = Field(min_length=1)
    total_rated_power_w: float = Field(gt=0.0)
    observed_current_drop_a: float = Field(gt=0.0)
    expected_current_drop_a: float = Field(gt=0.0)
    current_drop_z: float = Field(gt=0.0)
    power_relative_error: float = Field(ge=0.0)
    normalized_current_rmse: float = Field(ge=0.0)
    normalized_current_mae: float = Field(ge=0.0)
    temperature_supported_fault_ids: tuple[str, ...] = ()
    temperature_contradiction_codes: tuple[str, ...] = ()
    position_contradiction_codes: tuple[str, ...] = ()
    command_contradiction_codes: tuple[str, ...] = ()
    simulation_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_status: str = Field(min_length=1)
    score_semantics: str = "REDUCED_ORDER_TRACE_DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY"


class DXCReducedOrderPhysicsDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-reduced-order-physics-decision.v1"
    selected_fault_ids: tuple[str, ...]
    base_fault_ids: tuple[str, ...]
    base_unresolved_reason_codes: tuple[str, ...]
    candidate_fits: tuple[DXCReducedOrderCandidateFit, ...] = ()
    selected_diagnosis_id: str | None = None
    application_status: str = Field(min_length=1)
    reason_codes: tuple[str, ...] = ()
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_semantics: str = "REDUCED_ORDER_ELECTRICAL_LOAD_TIME_SERIES_FORWARD_SIMULATION"
    model_class: str = "ADAPT_RESISTIVE_LOAD_BRANCH_REDUCED_ORDER_V1"
    numerical_time_series_simulation: bool = True
    continuous_network_solver: bool = False
    agent_required: bool = False
    formal_evidence_eligible: bool = False
    qualification: str = "DEVELOPMENT_ONLY_REQUIRES_NEW_HOLDOUT"
    claim_boundary: str = (
        "Truth-blind reduced-order numerical simulation over an admitted ADAPT AC branch. "
        "Observed voltage is an exogenous boundary condition; candidate FailedOff loads are removed by rated power. "
        "This is not VirtualADAPT/Simulink, a full continuous network solver, posterior probability, formal EvidenceFact, "
        "or release authority. R28 remains the retained fallback outside the narrow modeled scope."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCReducedOrderPhysicsDecision":
        if self.continuous_network_solver or self.agent_required or self.formal_evidence_eligible:
            raise ValueError("R29 reduced-order decision cannot claim full solver, Agent dependency, or formal evidence")
        payload = {
            "selected_fault_ids": self.selected_fault_ids,
            "base_fault_ids": self.base_fault_ids,
            "base_unresolved_reason_codes": self.base_unresolved_reason_codes,
            "candidate_fits": [item.model_dump(mode="json") for item in self.candidate_fits],
            "selected_diagnosis_id": self.selected_diagnosis_id,
            "application_status": self.application_status,
            "reason_codes": self.reason_codes,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC reduced-order physics result_sha256 mismatch")
        return self


def _component_maps(catalog: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, set[str]]]:
    components = {str(item["component_id"]): item for item in catalog.get("components", [])}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in catalog.get("connections", []):
        left = str(edge.get("c1") or "")
        right = str(edge.get("c2") or "")
        if left and right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return components, adjacency


def _nearest(
    components: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, set[str]],
    start: str,
    predicate,
    max_hops: int,
) -> list[tuple[int, str]]:
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    found: list[tuple[int, str]] = []
    while queue:
        node, distance = queue.popleft()
        if distance > 0 and predicate(components.get(node, {})):
            found.append((distance, node))
        if distance >= max_hops:
            continue
        for neighbor in sorted(adjacency.get(node, set())):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return sorted(found)


def _series(records: Sequence[Mapping[str, Any]], sensor_id: str) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []
    for record in records:
        if str(record.get("kind") or "") != "sensors":
            continue
        value = (record.get("values") or {}).get(sensor_id)
        if value is None or isinstance(value, bool):
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"DXC_R29_NONFINITE_SENSOR:{sensor_id}")
        times.append(float(record["time_tick"]) / 1000.0)
        values.append(number)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _bool_state(records: Sequence[Mapping[str, Any]], sensor_id: str, time_s: float) -> bool | None:
    result: bool | None = None
    for record in records:
        if str(record.get("kind") or "") != "sensors":
            continue
        timestamp = float(record["time_tick"]) / 1000.0
        if timestamp > time_s + 1.0:
            break
        value = (record.get("values") or {}).get(sensor_id)
        if isinstance(value, bool):
            result = value
    return result


def _robust_scale(values: np.ndarray, floor: float) -> float:
    if len(values) < 3:
        return float(floor)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    return max(mad, float(np.std(values)), float(floor))


def _local_position_sensor(
    component_id: str,
    components: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, set[str]],
) -> str | None:
    wires = [node for node in adjacency.get(component_id, set()) if components.get(node, {}).get("component_type") == "Wire"]
    candidates: set[str] = set()
    for wire in wires:
        relays = [node for node in adjacency.get(wire, set()) if components.get(node, {}).get("component_type") == "Relay"]
        for relay in relays:
            candidates.update(
                node
                for node in adjacency.get(relay, set())
                if components.get(node, {}).get("component_type") == "PositionSensor"
            )
    return sorted(candidates)[0] if len(candidates) == 1 else None


def _related_relay_ids(
    component_id: str,
    components: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, set[str]],
) -> tuple[str, ...]:
    wires = [node for node in adjacency.get(component_id, set()) if components.get(node, {}).get("component_type") == "Wire"]
    relays = {
        node
        for wire in wires
        for node in adjacency.get(wire, set())
        if components.get(node, {}).get("component_type") == "Relay"
    }
    return tuple(sorted(relays))


def _load_specs(
    catalog: Mapping[str, Any],
    config: DXCReducedOrderPhysicsConfig,
) -> tuple[ADAPTResistiveLoadSpec, ...]:
    components, adjacency = _component_maps(catalog)
    result: list[ADAPTResistiveLoadSpec] = []
    for component_id, component in sorted(components.items()):
        match = re.fullmatch(r"LightBulb([0-9]+)W", str(component.get("component_type") or ""))
        if match is None or "FailedOff" not in set(map(str, component.get("fault_modes", []))):
            continue
        current = _nearest(
            components,
            adjacency,
            component_id,
            lambda item: item.get("kind") == "sensor" and item.get("unit") == "Amps",
            config.max_topology_hops,
        )
        voltage = _nearest(
            components,
            adjacency,
            component_id,
            lambda item: item.get("kind") == "sensor" and item.get("unit") == "VAC",
            config.max_topology_hops,
        )
        if not current or not voltage:
            continue
        direct_temp = sorted(
            node
            for node in adjacency.get(component_id, set())
            if components.get(node, {}).get("component_type") == "TemperatureSensor"
        )
        result.append(ADAPTResistiveLoadSpec(
            fault_id=f"{component_id}:FailedOff",
            component_id=component_id,
            component_type=str(component.get("component_type")),
            rated_power_w=float(match.group(1)),
            current_sensor_id=current[0][1],
            voltage_sensor_id=voltage[0][1],
            local_temperature_sensor_id=direct_temp[0] if len(direct_temp) == 1 else None,
            local_position_sensor_id=_local_position_sensor(component_id, components, adjacency),
        ))
    return tuple(result)


def _branch_observation(
    *,
    records: Sequence[Mapping[str, Any]],
    current_sensor_id: str,
    voltage_sensor_id: str,
    onset_s: float,
    config: DXCReducedOrderPhysicsConfig,
) -> dict[str, Any] | None:
    current_t, current_v = _series(records, current_sensor_id)
    voltage_t, voltage_v = _series(records, voltage_sensor_id)
    pre_current = current_v[(current_t >= onset_s - config.pre_window_s) & (current_t < onset_s)]
    post_mask = (current_t >= onset_s) & (current_t < onset_s + config.post_window_s)
    post_current = current_v[post_mask]
    post_times = current_t[post_mask]
    pre_voltage = voltage_v[(voltage_t >= onset_s - config.pre_window_s) & (voltage_t < onset_s)]
    if len(pre_current) < 5 or len(post_current) < 5 or len(pre_voltage) < 5:
        return None
    post_voltage = np.interp(post_times, voltage_t, voltage_v)
    pre_i = float(np.median(pre_current))
    post_i = float(np.median(post_current))
    pre_v = float(np.median(pre_voltage))
    post_v = float(np.median(post_voltage))
    current_scale = _robust_scale(pre_current, config.current_noise_floor_a)
    current_drop = pre_i - post_i
    return {
        "time_s": post_times,
        "observed_current_a": post_current,
        "observed_voltage_v": post_voltage,
        "pre_current_a": pre_i,
        "pre_voltage_v": pre_v,
        "post_voltage_v": post_v,
        "current_drop_a": current_drop,
        "current_drop_z": current_drop / current_scale,
        "current_scale_a": current_scale,
        "voltage_relative_change": abs(post_v - pre_v) / max(abs(pre_v), 1e-9),
    }


def _temperature_cooling(
    records: Sequence[Mapping[str, Any]],
    sensor_id: str,
    onset_s: float,
    config: DXCReducedOrderPhysicsConfig,
) -> tuple[float, float] | None:
    times, values = _series(records, sensor_id)
    before = values[(times >= onset_s - config.pre_window_s) & (times < onset_s)]
    later = values[
        (times >= onset_s + config.temperature_late_start_s)
        & (times < onset_s + config.temperature_late_start_s + config.temperature_late_window_s)
    ]
    if len(before) < 5 or len(later) < 5:
        return None
    pre = float(np.median(before))
    late = float(np.median(later))
    scale = _robust_scale(before, config.temperature_noise_floor_f)
    cooling = pre - late
    return cooling / scale, cooling


def _command_near_component(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    component_id: str,
    onset_s: float,
    config: DXCReducedOrderPhysicsConfig,
) -> tuple[str, ...]:
    components, adjacency = _component_maps(catalog)
    relay_ids = set(_related_relay_ids(component_id, components, adjacency))
    command_ids = {
        str(row.get("command_id"))
        for row in catalog.get("command_bindings", [])
        if str(row.get("component_id")) in relay_ids
    }
    found = []
    for record in records:
        if str(record.get("kind") or "") != "command":
            continue
        if str(record.get("command_id")) not in command_ids:
            continue
        timestamp = float(record["time_tick"]) / 1000.0
        if abs(timestamp - onset_s) <= config.command_guard_s:
            found.append(str(record.get("command_id")))
    return tuple(sorted(set(found)))


def rank_dxc_reduced_order_physics(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    base_fault_ids: Sequence[str],
    base_unresolved_reason_codes: Sequence[str],
    first_alarm_time_s: float | None,
    config: DXCReducedOrderPhysicsConfig | None = None,
) -> DXCReducedOrderPhysicsDecision:
    """Apply a narrow reduced-order physical resolver to an R28 decision.

    R29 is intentionally selective.  Resolved R28 diagnoses pass through.  The
    physical resolver is admitted only for the generic R28 unresolved physical-
    response disposition and only when an alarm time is available.  This keeps
    the new provider from becoming a broad post-contact reranker.
    """
    cfg = config or DXCReducedOrderPhysicsConfig()
    base = tuple(sorted(set(map(str, base_fault_ids))))
    unresolved = tuple(sorted(set(map(str, base_unresolved_reason_codes))))

    def finish(
        *,
        selected: tuple[str, ...],
        fits: Sequence[DXCReducedOrderCandidateFit],
        selected_diagnosis_id: str | None,
        status: str,
        reasons: Sequence[str],
    ) -> DXCReducedOrderPhysicsDecision:
        fit_tuple = tuple(fits)
        reason_tuple = tuple(sorted(set(map(str, reasons))))
        payload = {
            "selected_fault_ids": selected,
            "base_fault_ids": base,
            "base_unresolved_reason_codes": unresolved,
            "candidate_fits": [item.model_dump(mode="json") for item in fit_tuple],
            "selected_diagnosis_id": selected_diagnosis_id,
            "application_status": status,
            "reason_codes": reason_tuple,
        }
        return DXCReducedOrderPhysicsDecision(
            selected_fault_ids=selected,
            base_fault_ids=base,
            base_unresolved_reason_codes=unresolved,
            candidate_fits=fit_tuple,
            selected_diagnosis_id=selected_diagnosis_id,
            application_status=status,
            reason_codes=reason_tuple,
            result_sha256=_digest(payload),
        )

    if base:
        return finish(
            selected=base,
            fits=(),
            selected_diagnosis_id=None,
            status="PASSTHROUGH_R28_RESOLVED",
            reasons=("R28_ALREADY_RESOLVED_NO_R29_RERANK",),
        )
    if cfg.unresolved_reason_required not in unresolved:
        return finish(
            selected=base,
            fits=(),
            selected_diagnosis_id=None,
            status="NOT_APPLICABLE",
            reasons=("R28_UNRESOLVED_REASON_OUTSIDE_R29_SCOPE",),
        )
    if first_alarm_time_s is None or not math.isfinite(float(first_alarm_time_s)):
        return finish(
            selected=base,
            fits=(),
            selected_diagnosis_id=None,
            status="UNRESOLVED",
            reasons=("R29_ALARM_TIME_UNAVAILABLE",),
        )

    onset = float(first_alarm_time_s)
    specs = _load_specs(catalog, cfg)
    if not specs:
        return finish(selected=base, fits=(), selected_diagnosis_id=None, status="UNRESOLVED", reasons=("R29_NO_ADMITTED_LOAD_MODEL",))
    by_branch: dict[tuple[str, str], list[ADAPTResistiveLoadSpec]] = defaultdict(list)
    for spec in specs:
        by_branch[(spec.current_sensor_id, spec.voltage_sensor_id)].append(spec)

    components, _ = _component_maps(catalog)
    accepted: list[DXCReducedOrderCandidateFit] = []
    rejected: list[DXCReducedOrderCandidateFit] = []
    input_sha = _digest({
        "records": list(records),
        "catalog_xml_sha256": catalog.get("xml_sha256"),
        "onset_s": onset,
    })

    for (current_sensor, voltage_sensor), branch_specs in sorted(by_branch.items()):
        obs = _branch_observation(
            records=records,
            current_sensor_id=current_sensor,
            voltage_sensor_id=voltage_sensor,
            onset_s=onset,
            config=cfg,
        )
        if obs is None:
            continue
        if obs["pre_voltage_v"] < cfg.min_voltage_rms_v:
            continue
        if obs["voltage_relative_change"] > cfg.max_voltage_relative_change:
            continue
        if obs["current_drop_a"] < cfg.min_current_drop_a or obs["current_drop_z"] < cfg.min_current_drop_z:
            continue

        active_specs: list[ADAPTResistiveLoadSpec] = []
        temp_evidence: dict[str, tuple[float, float] | None] = {}
        position_state: dict[str, bool | None] = {}
        command_near: dict[str, tuple[str, ...]] = {}
        for spec in branch_specs:
            position_state[spec.fault_id] = (
                _bool_state(records, spec.local_position_sensor_id, onset)
                if spec.local_position_sensor_id
                else None
            )
            if position_state[spec.fault_id] is not False:
                active_specs.append(spec)
            temp_evidence[spec.fault_id] = (
                _temperature_cooling(records, spec.local_temperature_sensor_id, onset, cfg)
                if spec.local_temperature_sensor_id
                else None
            )
            command_near[spec.fault_id] = _command_near_component(
                records=records,
                catalog=catalog,
                component_id=spec.component_id,
                onset_s=onset,
                config=cfg,
            )
        if not active_specs:
            continue

        boundary = build_branch_boundary_condition(
            branch_id=f"{current_sensor}|{voltage_sensor}",
            current_sensor_id=current_sensor,
            voltage_sensor_id=voltage_sensor,
            time_s=obs["time_s"],
            observed_voltage_v=obs["observed_voltage_v"],
            pre_current_a=obs["pre_current_a"],
            pre_voltage_v=obs["pre_voltage_v"],
            source_artifact_sha256=input_sha,
        )
        provider = ADAPTLightBulbReducedOrderProvider(boundary=boundary, load_specs=active_specs)
        observed_trace = build_simulation_trace(
            trace_id=f"R29-OBS-{current_sensor}-{voltage_sensor}",
            time_s=obs["time_s"],
            channels={current_sensor: obs["observed_current_a"]},
            source_artifact_sha256=input_sha,
            provenance_kind="HOST_ADMITTED_DXC_OBSERVED_BRANCH_TRACE",
        )

        candidate_meta: dict[str, dict[str, Any]] = {}
        candidates: list[SimulationHypothesisCandidate] = []
        for cardinality in range(1, min(cfg.max_failed_bulbs_per_branch, len(active_specs)) + 1):
            for subset in itertools.combinations(active_specs, cardinality):
                fault_ids = tuple(sorted(item.fault_id for item in subset))
                diagnosis_id = f"R29-{_digest({'fault_ids': fault_ids})[:16].upper()}"
                total_power = sum(item.rated_power_w for item in subset)
                expected_drop = total_power / obs["pre_voltage_v"]
                relative_error = abs(obs["current_drop_a"] - expected_drop) / max(expected_drop, 1e-9)
                temp_supported: list[str] = []
                temp_contradictions: list[str] = []
                position_contradictions: list[str] = []
                command_contradictions: list[str] = []
                selected_set = set(fault_ids)
                for spec in active_specs:
                    evidence = temp_evidence.get(spec.fault_id)
                    if evidence is not None:
                        cooling_z, _cooling = evidence
                        if spec.fault_id in selected_set:
                            if cooling_z >= cfg.min_temperature_cooling_z:
                                temp_supported.append(spec.fault_id)
                            else:
                                temp_contradictions.append(f"EXPECTED_COOLING_NOT_OBSERVED:{spec.component_id}")
                        elif cooling_z >= cfg.min_temperature_cooling_z:
                            temp_contradictions.append(f"UNMODELED_COOLING_SIBLING:{spec.component_id}")
                    if spec.fault_id in selected_set and position_state.get(spec.fault_id) is False:
                        position_contradictions.append(f"LOAD_BRANCH_POSITION_OPEN:{spec.component_id}")
                    if spec.fault_id in selected_set and command_near.get(spec.fault_id):
                        command_contradictions.append(
                            f"RELATED_RELAY_COMMAND_NEAR_ONSET:{spec.component_id}:{','.join(command_near[spec.fault_id])}"
                        )
                candidate_meta[diagnosis_id] = {
                    "fault_ids": fault_ids,
                    "total_power_w": total_power,
                    "expected_drop_a": expected_drop,
                    "relative_error": relative_error,
                    "temp_supported": tuple(sorted(temp_supported)),
                    "temp_contradictions": tuple(sorted(temp_contradictions)),
                    "position_contradictions": tuple(sorted(position_contradictions)),
                    "command_contradictions": tuple(sorted(command_contradictions)),
                }
                candidates.append(SimulationHypothesisCandidate(
                    diagnosis_id=diagnosis_id,
                    fault_ids=fault_ids,
                ))

        review = compare_counterfactual_diagnoses(
            observed=observed_trace,
            candidates=candidates,
            provider=provider,
            channel_scales={current_sensor: obs["current_scale_a"]},
            channels=(current_sensor,),
        )
        comparison_by_id = {item.diagnosis_id: item for item in review.comparisons}
        for diagnosis_id, meta in candidate_meta.items():
            comparison = comparison_by_id[diagnosis_id]
            contradictions = (
                meta["temp_contradictions"]
                + meta["position_contradictions"]
                + meta["command_contradictions"]
            )
            supported = meta["relative_error"] <= cfg.max_power_relative_error and not contradictions
            fit = DXCReducedOrderCandidateFit(
                diagnosis_id=diagnosis_id,
                fault_ids=meta["fault_ids"],
                branch_id=boundary.branch_id,
                current_sensor_id=current_sensor,
                voltage_sensor_id=voltage_sensor,
                total_rated_power_w=meta["total_power_w"],
                observed_current_drop_a=obs["current_drop_a"],
                expected_current_drop_a=meta["expected_drop_a"],
                current_drop_z=obs["current_drop_z"],
                power_relative_error=meta["relative_error"],
                normalized_current_rmse=comparison.aggregate_normalized_rmse,
                normalized_current_mae=comparison.aggregate_normalized_mae,
                temperature_supported_fault_ids=meta["temp_supported"],
                temperature_contradiction_codes=meta["temp_contradictions"],
                position_contradiction_codes=meta["position_contradictions"],
                command_contradiction_codes=meta["command_contradictions"],
                simulation_trace_sha256=comparison.simulation_trace_sha256,
                fit_status="SUPPORTED" if supported else "REJECTED",
            )
            (accepted if supported else rejected).append(fit)

    accepted.sort(key=lambda item: (
        item.power_relative_error,
        item.normalized_current_rmse,
        len(item.fault_ids),
        item.fault_ids,
    ))
    rejected.sort(key=lambda item: (
        item.power_relative_error,
        item.normalized_current_rmse,
        len(item.fault_ids),
        item.fault_ids,
    ))
    fits = tuple(accepted + rejected)
    if not accepted:
        return finish(
            selected=base,
            fits=fits,
            selected_diagnosis_id=None,
            status="UNRESOLVED",
            reasons=("R29_NO_PHYSICALLY_SUPPORTED_LOAD_CANDIDATE",),
        )

    # Conservative admission: if more than one candidate survives the physical
    # constraints, do not manufacture certainty from a small numerical distance
    # difference on a post-contact development set.
    if len(accepted) != 1:
        return finish(
            selected=base,
            fits=fits,
            selected_diagnosis_id=None,
            status="UNRESOLVED_AMBIGUOUS",
            reasons=("R29_MULTIPLE_PHYSICALLY_SUPPORTED_LOAD_CANDIDATES",),
        )
    best = accepted[0]
    return finish(
        selected=best.fault_ids,
        fits=fits,
        selected_diagnosis_id=best.diagnosis_id,
        status="RESOLVED_BY_REDUCED_ORDER_PHYSICS",
        reasons=("R29_UNIQUE_PHYSICALLY_SUPPORTED_LOAD_CANDIDATE",),
    )


__all__ = [
    "DXCReducedOrderCandidateFit",
    "DXCReducedOrderPhysicsConfig",
    "DXCReducedOrderPhysicsDecision",
    "rank_dxc_reduced_order_physics",
]
