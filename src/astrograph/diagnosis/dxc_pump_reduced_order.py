"""R31 reduced-order water-pump mechanism consistency for DXC/ADAPT.

This module is a deterministic *corroborator* above the existing diagnosis
chain.  It does not silently replace an already selected pump mode.  Instead it
simulates both public ADAPT WaterPump modes (``FailedOff`` and ``FlowBlocked``)
from host-admitted telemetry and reports whether the physical best-fit mode
supports or contradicts the current core diagnosis.

The model is intentionally reduced order:

* rated pump power and nominal flow are extracted from the public ADAPT XML;
* branch current is modeled from rated pump power plus a residual background
  branch load under observed bus voltage;
* flow is modeled as a first-order decay with a nuisance time constant fitted
  on a fixed sample-time-derived grid;
* ``FlowBlocked`` also fits a bounded post-block flow asymptote, while
  ``FailedOff`` fixes the asymptote at zero.

No fault truth, case identity, LLM choice, or release authority enters this
core.  The fitted nuisance parameters are trace-fit parameters, not calibrated
physical constants or posterior probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.simulation.adapt_pump_reduced_order import (
    ADAPTWaterPumpFaultSpec,
    ADAPTWaterPumpReducedOrderProvider,
    ADAPTWaterPumpTypeParameters,
    build_pump_boundary_condition,
)
from astrograph.simulation.diagnostic_provider import (
    SimulationHypothesisCandidate,
    build_simulation_trace,
    compare_counterfactual_diagnoses,
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCWaterPumpPhysicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    pre_window_s: float = Field(gt=0.0, default=8.0)
    post_window_s: float = Field(gt=0.0, default=8.0)
    max_topology_hops: int = Field(ge=2, default=5)
    current_noise_floor_a: float = Field(gt=0.0, default=0.02)
    flow_noise_floor_gph: float = Field(gt=0.0, default=4.0)
    current_relative_scale: float = Field(gt=0.0, default=0.25)
    flow_relative_scale: float = Field(gt=0.0, default=0.10)
    flow_tau_sample_multipliers: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    max_flowblocked_asymptote_ratio: float = Field(gt=0.0, lt=1.0, default=0.95)
    command_guard_s: float = Field(gt=0.0, default=2.0)

    @model_validator(mode="after")
    def config_integrity(self) -> "DXCWaterPumpPhysicsConfig":
        if not self.flow_tau_sample_multipliers or any(value <= 0.0 for value in self.flow_tau_sample_multipliers):
            raise ValueError("R31_PUMP_TAU_MULTIPLIERS_INVALID")
        return self


class DXCWaterPumpCandidateFit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    diagnosis_id: str = Field(min_length=1)
    fault_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    fault_mode: str = Field(pattern=r"^(FailedOff|FlowBlocked)$")
    current_sensor_id: str = Field(min_length=1)
    voltage_sensor_id: str = Field(min_length=1)
    flow_sensor_id: str = Field(min_length=1)
    fitted_flow_tau_s: float = Field(gt=0.0)
    fitted_flow_asymptote_ratio: float = Field(ge=0.0, le=0.95)
    normalized_current_rmse: float = Field(ge=0.0)
    normalized_flow_rmse: float = Field(ge=0.0)
    aggregate_normalized_rmse: float = Field(ge=0.0)
    simulation_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_status: str = Field(min_length=1)
    score_semantics: str = "REDUCED_ORDER_CURRENT_FLOW_TRACE_DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY"


class DXCWaterPumpPhysicsDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-waterpump-reduced-order-decision.v1"
    selected_fault_ids: tuple[str, ...]
    base_fault_ids: tuple[str, ...]
    physics_preferred_fault_ids: tuple[str, ...] = ()
    candidate_fits: tuple[DXCWaterPumpCandidateFit, ...] = ()
    application_status: str = Field(min_length=1)
    reason_codes: tuple[str, ...] = ()
    supports_base_mode: bool | None = None
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_class: str = "ADAPT_WATERPUMP_ELECTRICAL_HYDRAULIC_REDUCED_ORDER_V1"
    simulation_semantics: str = "REDUCED_ORDER_COUPLED_CURRENT_FLOW_TIME_SERIES_FORWARD_SIMULATION"
    numerical_time_series_simulation: bool = True
    continuous_network_solver: bool = False
    auto_override_enabled: bool = False
    agent_required: bool = False
    formal_evidence_eligible: bool = False
    qualification: str = "DEVELOPMENT_CORROBORATOR_REQUIRES_NEW_HOLDOUT"
    claim_boundary: str = (
        "Truth-blind reduced-order WaterPump current/flow consistency review. R31 reports whether the physical model "
        "corroborates the existing pump-mode diagnosis but does not auto-override it. Fitted flow nuisance parameters "
        "are trace-fit quantities, not calibrated mechanism constants, probabilities, formal EvidenceFacts, or release authority."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCWaterPumpPhysicsDecision":
        if self.continuous_network_solver or self.auto_override_enabled or self.agent_required or self.formal_evidence_eligible:
            raise ValueError("R31_PUMP_AUTHORITY_OR_SOLVER_CLAIM_FORBIDDEN")
        payload = {
            "selected_fault_ids": self.selected_fault_ids,
            "base_fault_ids": self.base_fault_ids,
            "physics_preferred_fault_ids": self.physics_preferred_fault_ids,
            "candidate_fits": [item.model_dump(mode="json") for item in self.candidate_fits],
            "application_status": self.application_status,
            "reason_codes": self.reason_codes,
            "supports_base_mode": self.supports_base_mode,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("R31_PUMP_DECISION_SHA256_MISMATCH")
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


def _nearest(components, adjacency, start: str, predicate, max_hops: int) -> list[tuple[int, str]]:
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
            raise ValueError(f"R31_PUMP_NONFINITE_SENSOR:{sensor_id}")
        times.append(float(record["time_tick"]) / 1000.0)
        values.append(number)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _robust_scale(values: np.ndarray, floor: float) -> float:
    if len(values) < 3:
        return float(floor)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    return max(mad, float(np.std(values)), float(floor))


def _pump_topology(catalog: Mapping[str, Any], component_id: str, config: DXCWaterPumpPhysicsConfig) -> dict[str, Any]:
    components, adjacency = _component_maps(catalog)
    component = components.get(component_id)
    if component is None or str(component.get("component_type") or "") != "WaterPump":
        raise ValueError("R31_PUMP_COMPONENT_NOT_WATERPUMP")
    flow = _nearest(
        components, adjacency, component_id,
        lambda item: item.get("kind") == "sensor" and item.get("unit") == "GPH",
        config.max_topology_hops,
    )
    current = _nearest(
        components, adjacency, component_id,
        lambda item: item.get("kind") == "sensor" and item.get("unit") == "Amps",
        config.max_topology_hops,
    )
    voltage = _nearest(
        components, adjacency, component_id,
        lambda item: item.get("kind") == "sensor" and item.get("unit") == "VAC",
        config.max_topology_hops,
    )
    relays = _nearest(
        components, adjacency, component_id,
        lambda item: item.get("component_type") == "Relay",
        config.max_topology_hops,
    )
    if not flow or not current or not voltage:
        raise ValueError("R31_PUMP_REQUIRED_TOPOLOGY_OBSERVABILITY_MISSING")
    relay_ids = tuple(node for _, node in relays)
    commands = tuple(sorted({
        str(binding.get("command_id"))
        for binding in catalog.get("command_bindings", [])
        if str(binding.get("component_id")) in relay_ids and binding.get("command_id")
    }))
    return {
        "flow_sensor_id": flow[0][1],
        "current_sensor_id": current[0][1],
        "voltage_sensor_id": voltage[0][1],
        "relay_ids": relay_ids,
        "command_ids": commands,
    }


def _command_near_transition(records: Sequence[Mapping[str, Any]], command_ids: Sequence[str], transition_time_s: float, guard_s: float) -> bool:
    command_set = set(command_ids)
    for record in records:
        if str(record.get("kind") or "") != "command" or str(record.get("command_id") or "") not in command_set:
            continue
        timestamp = float(record["time_tick"]) / 1000.0
        if abs(timestamp - transition_time_s) <= guard_s:
            return True
    return False


def _fit_flow_hypothesis(
    *,
    mode: str,
    time_s: np.ndarray,
    observed_flow: np.ndarray,
    transition_time_s: float,
    pre_flow_gph: float,
    max_asymptote_ratio: float,
    tau_multipliers: Sequence[float],
) -> tuple[float, float, float]:
    if len(time_s) < 2 or pre_flow_gph <= 0.0:
        raise ValueError("R31_PUMP_FLOW_FIT_INSUFFICIENT_DATA")
    deltas = np.diff(time_s)
    positive = deltas[deltas > 0]
    if not len(positive):
        raise ValueError("R31_PUMP_FLOW_TIMELINE_INVALID")
    dt = float(np.median(positive))
    elapsed = np.maximum(time_s - float(transition_time_s), 0.0)
    normalized = observed_flow / pre_flow_gph
    best: tuple[float, float, float] | None = None
    for multiplier in tau_multipliers:
        tau = max(dt * float(multiplier), 1e-6)
        relaxation = np.exp(-elapsed / tau)
        if mode == "FailedOff":
            alpha = 0.0
        elif mode == "FlowBlocked":
            weight = 1.0 - relaxation
            denominator = float(np.dot(weight, weight))
            if denominator <= 1e-12:
                alpha = max_asymptote_ratio
            else:
                alpha = float(np.dot(normalized - relaxation, weight) / denominator)
                alpha = min(max(alpha, 0.0), max_asymptote_ratio)
        else:
            raise ValueError("R31_PUMP_UNSUPPORTED_MODE")
        predicted = alpha + (1.0 - alpha) * relaxation
        rmse = float(np.sqrt(np.mean((normalized - predicted) ** 2)))
        candidate = (rmse, tau, alpha)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[2], best[0]


def evaluate_dxc_water_pump_physics(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    type_parameters: ADAPTWaterPumpTypeParameters,
    base_fault_ids: Sequence[str],
    transition_time_s: float | None,
    config: DXCWaterPumpPhysicsConfig | None = None,
) -> DXCWaterPumpPhysicsDecision:
    cfg = config or DXCWaterPumpPhysicsConfig()
    base = tuple(sorted(set(map(str, base_fault_ids))))

    def build_decision(**kwargs) -> DXCWaterPumpPhysicsDecision:
        payload = {
            "selected_fault_ids": kwargs["selected_fault_ids"],
            "base_fault_ids": base,
            "physics_preferred_fault_ids": kwargs.get("physics_preferred_fault_ids", ()),
            "candidate_fits": [item.model_dump(mode="json") for item in kwargs.get("candidate_fits", ())],
            "application_status": kwargs["application_status"],
            "reason_codes": kwargs.get("reason_codes", ()),
            "supports_base_mode": kwargs.get("supports_base_mode"),
        }
        return DXCWaterPumpPhysicsDecision(
            selected_fault_ids=kwargs["selected_fault_ids"],
            base_fault_ids=base,
            physics_preferred_fault_ids=kwargs.get("physics_preferred_fault_ids", ()),
            candidate_fits=kwargs.get("candidate_fits", ()),
            application_status=kwargs["application_status"],
            reason_codes=kwargs.get("reason_codes", ()),
            supports_base_mode=kwargs.get("supports_base_mode"),
            result_sha256=_digest(payload),
        )

    if type_parameters.source_xml_sha256 != str(catalog.get("xml_sha256") or ""):
        raise ValueError("R31_PUMP_XML_PARAMETER_CATALOG_IDENTITY_MISMATCH")
    pump_faults = [fault for fault in base if fault.split(":", 1)[0] and fault.rsplit(":", 1)[-1] in {"FailedOff", "FlowBlocked"}]
    components = {str(item["component_id"]): item for item in catalog.get("components", [])}
    pump_faults = [fault for fault in pump_faults if components.get(fault.split(":", 1)[0], {}).get("component_type") == "WaterPump"]
    if len(base) != 1 or len(pump_faults) != 1:
        return build_decision(
            selected_fault_ids=base,
            application_status="PASSTHROUGH_NON_SINGLE_PUMP_DIAGNOSIS",
            reason_codes=("R31_WATERPUMP_CORROBORATOR_NOT_APPLICABLE",),
            supports_base_mode=None,
        )
    if transition_time_s is None or not math.isfinite(float(transition_time_s)):
        return build_decision(
            selected_fault_ids=base,
            application_status="PUMP_PHYSICS_NOT_EVALUATED_TRANSITION_UNAVAILABLE",
            reason_codes=("R31_TRANSITION_TIME_UNAVAILABLE",),
            supports_base_mode=None,
        )

    base_fault = pump_faults[0]
    component_id, base_mode = base_fault.split(":", 1)
    topology = _pump_topology(catalog, component_id, cfg)
    if _command_near_transition(records, topology["command_ids"], float(transition_time_s), cfg.command_guard_s):
        return build_decision(
            selected_fault_ids=base,
            application_status="PUMP_PHYSICS_NONDIAGNOSTIC_COMMAND_NEAR_TRANSITION",
            reason_codes=("R31_RELATED_COMMAND_NEAR_TRANSITION",),
            supports_base_mode=None,
        )

    current_t, current_v = _series(records, topology["current_sensor_id"])
    voltage_t, voltage_v = _series(records, topology["voltage_sensor_id"])
    flow_t, flow_v = _series(records, topology["flow_sensor_id"])
    onset = float(transition_time_s)
    pre_current = current_v[(current_t >= onset - cfg.pre_window_s) & (current_t < onset)]
    pre_voltage = voltage_v[(voltage_t >= onset - cfg.pre_window_s) & (voltage_t < onset)]
    pre_flow = flow_v[(flow_t >= onset - cfg.pre_window_s) & (flow_t < onset)]
    post_mask = (flow_t >= onset) & (flow_t <= onset + cfg.post_window_s)
    post_time = flow_t[post_mask]
    post_flow = flow_v[post_mask]
    if min(len(pre_current), len(pre_voltage), len(pre_flow), len(post_time)) < 4:
        return build_decision(
            selected_fault_ids=base,
            application_status="PUMP_PHYSICS_NOT_EVALUATED_INSUFFICIENT_DATA",
            reason_codes=("R31_PUMP_REQUIRED_WINDOW_INSUFFICIENT",),
            supports_base_mode=None,
        )
    post_current = np.interp(post_time, current_t, current_v)
    post_voltage = np.interp(post_time, voltage_t, voltage_v)
    pre_i = float(np.median(pre_current))
    pre_v = float(np.median(pre_voltage))
    pre_f = float(np.median(pre_flow))
    if pre_v <= 0.0 or pre_f <= 0.0:
        return build_decision(
            selected_fault_ids=base,
            application_status="PUMP_PHYSICS_NOT_EVALUATED_INVALID_OPERATING_POINT",
            reason_codes=("R31_PUMP_PRESTATE_NOT_ENERGIZED_OR_FLOWING",),
            supports_base_mode=None,
        )

    source_sha = _digest({
        "catalog_xml_sha256": catalog.get("xml_sha256"),
        "component_id": component_id,
        "transition_time_s": onset,
        "record_count": len(records),
        "records_sha256": _digest(list(records)),
    })
    boundary = build_pump_boundary_condition(
        branch_id=f"PUMP-BRANCH-{component_id}",
        current_sensor_id=topology["current_sensor_id"],
        voltage_sensor_id=topology["voltage_sensor_id"],
        flow_sensor_id=topology["flow_sensor_id"],
        transition_time_s=onset,
        time_s=post_time,
        observed_voltage_v=post_voltage,
        pre_current_a=pre_i,
        pre_voltage_v=pre_v,
        pre_flow_gph=pre_f,
        source_artifact_sha256=source_sha,
    )
    fault_specs: list[ADAPTWaterPumpFaultSpec] = []
    flow_fit_meta: dict[str, tuple[float, float, float]] = {}
    for mode in ("FailedOff", "FlowBlocked"):
        tau, alpha, flow_fit = _fit_flow_hypothesis(
            mode=mode,
            time_s=post_time,
            observed_flow=post_flow,
            transition_time_s=onset,
            pre_flow_gph=pre_f,
            max_asymptote_ratio=cfg.max_flowblocked_asymptote_ratio,
            tau_multipliers=cfg.flow_tau_sample_multipliers,
        )
        fault_id = f"{component_id}:{mode}"
        flow_fit_meta[fault_id] = (tau, alpha, flow_fit)
        fault_specs.append(ADAPTWaterPumpFaultSpec(
            fault_id=fault_id,
            component_id=component_id,
            fault_mode=mode,
            rated_power_w=type_parameters.rated_power_w,
            nominal_flow_gph=type_parameters.nominal_flow_gph,
            current_sensor_id=topology["current_sensor_id"],
            voltage_sensor_id=topology["voltage_sensor_id"],
            flow_sensor_id=topology["flow_sensor_id"],
            flow_asymptote_ratio=alpha,
            flow_tau_s=tau,
        ))

    provider = ADAPTWaterPumpReducedOrderProvider(boundary=boundary, fault_specs=fault_specs)
    observed = build_simulation_trace(
        trace_id=f"OBS-R31-{component_id}-{_digest(source_sha)[:12]}",
        time_s=post_time,
        channels={
            topology["current_sensor_id"]: post_current,
            topology["flow_sensor_id"]: post_flow,
        },
        source_artifact_sha256=source_sha,
        provenance_kind="R31_HOST_ADMITTED_DXC_WATERPUMP_OBSERVED_TRACE",
    )
    candidates = tuple(
        SimulationHypothesisCandidate(diagnosis_id=f"R31-{component_id}-{mode}", fault_ids=(f"{component_id}:{mode}",))
        for mode in ("FailedOff", "FlowBlocked")
    )
    rated_current = type_parameters.rated_power_w / pre_v
    current_scale = max(_robust_scale(pre_current, cfg.current_noise_floor_a), rated_current * cfg.current_relative_scale)
    flow_scale = max(_robust_scale(pre_flow, cfg.flow_noise_floor_gph), pre_f * cfg.flow_relative_scale)
    review = compare_counterfactual_diagnoses(
        observed=observed,
        candidates=candidates,
        provider=provider,
        channel_scales={topology["current_sensor_id"]: current_scale, topology["flow_sensor_id"]: flow_scale},
        channels=[topology["current_sensor_id"], topology["flow_sensor_id"]],
    )
    fits: list[DXCWaterPumpCandidateFit] = []
    for comparison in review.comparisons:
        fault_id = comparison.fault_ids[0]
        mode = fault_id.rsplit(":", 1)[1]
        tau, alpha, _ = flow_fit_meta[fault_id]
        fits.append(DXCWaterPumpCandidateFit(
            diagnosis_id=comparison.diagnosis_id,
            fault_id=fault_id,
            component_id=component_id,
            fault_mode=mode,
            current_sensor_id=topology["current_sensor_id"],
            voltage_sensor_id=topology["voltage_sensor_id"],
            flow_sensor_id=topology["flow_sensor_id"],
            fitted_flow_tau_s=tau,
            fitted_flow_asymptote_ratio=alpha,
            normalized_current_rmse=comparison.per_channel[topology["current_sensor_id"]]["normalized_rmse"],
            normalized_flow_rmse=comparison.per_channel[topology["flow_sensor_id"]]["normalized_rmse"],
            aggregate_normalized_rmse=comparison.aggregate_normalized_rmse,
            simulation_trace_sha256=comparison.simulation_trace_sha256,
            fit_status="BEST_PHYSICAL_FIT" if comparison.diagnosis_id == review.best_diagnosis_id else "ALTERNATE_PHYSICAL_FIT",
        ))
    fits.sort(key=lambda item: (item.aggregate_normalized_rmse, item.diagnosis_id))
    best_faults = next(item.fault_ids for item in candidates if item.diagnosis_id == review.best_diagnosis_id)
    supports = best_faults == (base_fault,)
    status = "PUMP_PHYSICS_CORROBORATES_BASE_MODE" if supports else "PUMP_PHYSICS_CONTRADICTS_BASE_MODE_NO_AUTO_OVERRIDE"
    reason_codes = (
        "R31_CURRENT_FLOW_REDUCED_ORDER_BEST_FIT_MATCHES_BASE" if supports
        else "R31_CURRENT_FLOW_REDUCED_ORDER_BEST_FIT_DIFFERS_FROM_BASE",
    )
    return build_decision(
        selected_fault_ids=base,
        physics_preferred_fault_ids=best_faults,
        candidate_fits=tuple(fits),
        application_status=status,
        reason_codes=reason_codes,
        supports_base_mode=supports,
    )


__all__ = [
    "DXCWaterPumpCandidateFit",
    "DXCWaterPumpPhysicsConfig",
    "DXCWaterPumpPhysicsDecision",
    "evaluate_dxc_water_pump_physics",
]
