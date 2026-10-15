"""DXC qualitative component-mode forward-response consistency model.

R28 moves beyond R27's local expected-response rules by representing a small,
explicit set of component/fault state transitions and comparing the response
that each modeled transition predicts with admitted telemetry.  The model is
truth-blind at runtime and can execute with the Agent completely disabled.

This is deliberately *not* a continuous electrical simulator and not a
probabilistic diagnosis model.  It is a bounded qualitative forward model for
fault modes for which the public ADAPT catalog plus telemetry provide a
well-defined response contract.  Unsupported modes pass through to the R27
candidate rather than receiving invented physics.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCForwardResponseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    startup_s: float = Field(ge=0.0, default=30.0)
    offset_startup_guard_s: float = Field(ge=0.0, default=60.0)
    command_guard_s: float = Field(gt=0.0, default=2.0)
    offset_pre_samples: int = Field(ge=8, default=20)
    offset_post_samples: int = Field(ge=8, default=20)
    offset_candidate_steps: int = Field(ge=1, default=12)
    offset_min_z: float = Field(gt=0.0, default=20.0)
    offset_min_relative_step: float = Field(gt=0.0, default=0.20)
    offset_std_ratio_min: float = Field(gt=0.0, default=0.35)
    offset_std_ratio_max: float = Field(gt=0.0, default=3.0)
    offset_persistence_z: float = Field(gt=0.0, default=12.0)
    stuck_previous_samples: int = Field(ge=4, default=12)
    stuck_min_suffix_samples: int = Field(ge=8, default=20)
    stuck_absolute_range_floor: float = Field(gt=0.0, default=0.005)
    stuck_relative_tolerance: float = Field(gt=0.0, default=1e-6)
    stuck_step_z: float = Field(gt=0.0, default=8.0)
    light_pre_window_s: float = Field(gt=0.0, default=10.0)
    light_late_start_s: float = Field(gt=0.0, default=30.0)
    light_min_cooling: float = Field(gt=0.0, default=5.0)
    light_peer_separation: float = Field(gt=0.0, default=10.0)
    shared_light_drop_relative: float = Field(gt=0.0, default=0.08)
    max_direct_numeric_sensor_faults: int = Field(ge=1, default=2)


class DXCModeTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(min_length=1)
    component_type: str = Field(min_length=1)
    fault_mode: str = Field(min_length=1)
    from_mode: str = "NOMINAL_OR_CURRENT_MODE"
    to_mode: str = Field(min_length=1)
    transition_kind: str = Field(min_length=1)
    expected_response_codes: tuple[str, ...]


class DXCForwardResponseFit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    fault_id: str = Field(min_length=3)
    fit_score: float = Field(ge=0.0)
    fit_status: str = Field(min_length=1)
    transition: DXCModeTransition
    observed_response_codes: tuple[str, ...]
    source_component_ids: tuple[str, ...] = ()
    metrics: dict[str, float | str | bool] = Field(default_factory=dict)
    score_semantics: str = "DETERMINISTIC_FORWARD_MODEL_FIT_NOT_PROBABILITY"


class DXCForwardResponseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-forward-response-decision.v1"
    selected_fault_ids: tuple[str, ...]
    base_fault_ids: tuple[str, ...]
    fits: tuple[DXCForwardResponseFit, ...]
    vetoed_fault_ids: tuple[str, ...] = ()
    unresolved_reason_codes: tuple[str, ...] = ()
    modeled_fault_ids: tuple[str, ...] = ()
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_semantics: str = "QUALITATIVE_COMPONENT_MODE_FORWARD_SIMULATION"
    score_semantics: str = "DETERMINISTIC_FORWARD_MODEL_FIT_NOT_PROBABILITY"
    continuous_physics_simulator: bool = False
    agent_required: bool = False
    formal_evidence_eligible: bool = False
    claim_boundary: str = (
        "Truth-blind qualitative state-transition/forward-response consistency model. "
        "It is not a continuous electrical simulator, posterior probability, formal EvidenceFact, "
        "or release authority. Unmodeled modes retain R27 behavior unless a modeled response "
        "explicitly refines or contradicts it."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCForwardResponseDecision":
        if self.continuous_physics_simulator or self.agent_required or self.formal_evidence_eligible:
            raise ValueError("R28 qualitative forward decision cannot claim continuous simulation, Agent dependency, or formal evidence")
        payload = {
            "selected_fault_ids": self.selected_fault_ids,
            "base_fault_ids": self.base_fault_ids,
            "fits": [item.model_dump(mode="json") for item in self.fits],
            "vetoed_fault_ids": self.vetoed_fault_ids,
            "unresolved_reason_codes": self.unresolved_reason_codes,
            "modeled_fault_ids": self.modeled_fault_ids,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC forward-response result_sha256 mismatch")
        return self


def _catalog_maps(catalog: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, set[str]]]:
    components = {str(item["component_id"]): item for item in catalog.get("components", [])}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in catalog.get("connections", []):
        left = str(edge.get("c1") or "")
        right = str(edge.get("c2") or "")
        if left and right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return components, adjacency


def _numeric_series(records: Sequence[Mapping[str, Any]], sensor_id: str) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []
    for record in records:
        if str(record.get("kind") or "") != "sensors":
            continue
        row = record.get("values") or {}
        value = row.get(sensor_id)
        if value is None or isinstance(value, bool):
            continue
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"DXC_FORWARD_NONFINITE_SENSOR:{sensor_id}")
        times.append(float(record["time_tick"]) / 1000.0)
        values.append(number)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _command_times(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [float(row["time_tick"]) / 1000.0 for row in records if str(row.get("kind") or "") == "command"],
        dtype=float,
    )


def _transition_for(component: Mapping[str, Any], fault_mode: str) -> DXCModeTransition:
    component_id = str(component.get("component_id") or "")
    component_type = str(component.get("component_type") or "UNKNOWN")
    kind = str(component.get("kind") or "")
    if kind == "sensor" and fault_mode == "Stuck":
        return DXCModeTransition(
            component_id=component_id,
            component_type=component_type,
            fault_mode=fault_mode,
            to_mode="STUCK_OBSERVATION",
            transition_kind="SENSOR_OBSERVATION_FAULT",
            expected_response_codes=("OWN_CHANNEL_ABRUPT_CHANGE", "OWN_CHANNEL_PERSISTENT_FREEZE"),
        )
    if kind == "sensor" and fault_mode == "Offset":
        return DXCModeTransition(
            component_id=component_id,
            component_type=component_type,
            fault_mode=fault_mode,
            to_mode="OFFSET_OBSERVATION",
            transition_kind="SENSOR_OBSERVATION_FAULT",
            expected_response_codes=("OWN_CHANNEL_ADDITIVE_STEP", "POST_STEP_DYNAMICS_RETAINED", "STEP_PERSISTS"),
        )
    if component_type == "LightBulb25W" and fault_mode == "FailedOff":
        return DXCModeTransition(
            component_id=component_id,
            component_type=component_type,
            fault_mode=fault_mode,
            to_mode="OFF",
            transition_kind="PLANT_COMPONENT_MODE_TRANSITION",
            expected_response_codes=("LOCAL_TEMPERATURE_COOLING", "SHARED_LIGHT_OUTPUT_DROP"),
        )
    return DXCModeTransition(
        component_id=component_id,
        component_type=component_type,
        fault_mode=fault_mode,
        to_mode=fault_mode,
        transition_kind="UNMODELED_PASSTHROUGH",
        expected_response_codes=("NO_R28_FORWARD_MODEL",),
    )


def _stuck_fit(
    *, records: Sequence[Mapping[str, Any]], component: Mapping[str, Any], config: DXCForwardResponseConfig
) -> DXCForwardResponseFit | None:
    sensor_id = str(component["component_id"])
    times, values = _numeric_series(records, sensor_id)
    n = len(values)
    if n < config.stuck_previous_samples + config.stuck_min_suffix_samples + 1:
        return None
    suffix_min = np.minimum.accumulate(values[::-1])[::-1]
    suffix_max = np.maximum.accumulate(values[::-1])[::-1]
    for index in range(config.stuck_previous_samples, n - config.stuck_min_suffix_samples):
        if times[index] < config.startup_s:
            continue
        previous = values[index - config.stuck_previous_samples:index]
        tolerance = max(abs(float(np.mean(previous))) * config.stuck_relative_tolerance, 1e-6)
        if float(suffix_max[index] - suffix_min[index]) > tolerance:
            continue
        if float(np.ptp(previous)) <= max(config.stuck_absolute_range_floor, 20.0 * tolerance):
            continue
        center = float(np.median(previous))
        scale = max(
            float(np.std(previous)),
            1.4826 * float(np.median(np.abs(previous - center))),
            abs(center) * 0.002,
            1e-3,
        )
        post = values[index:index + min(20, n - index)]
        step_z = abs(float(np.median(post)) - center) / scale
        if step_z < config.stuck_step_z:
            continue
        transition = _transition_for(component, "Stuck")
        return DXCForwardResponseFit(
            fault_id=f"{sensor_id}:Stuck",
            fit_score=40.0 + step_z,
            fit_status="SUPPORTED",
            transition=transition,
            observed_response_codes=("ABRUPT_CHANGE_OBSERVED", "PERSISTENT_SUFFIX_FREEZE_OBSERVED"),
            source_component_ids=(sensor_id,),
            metrics={"onset_time_s": float(times[index]), "step_z": float(step_z), "suffix_range": float(suffix_max[index] - suffix_min[index])},
        )
    return None


def _offset_fit(
    *, records: Sequence[Mapping[str, Any]], component: Mapping[str, Any], config: DXCForwardResponseConfig, commands: np.ndarray
) -> DXCForwardResponseFit | None:
    sensor_id = str(component["component_id"])
    times, values = _numeric_series(records, sensor_id)
    n = len(values)
    pre_n = config.offset_pre_samples
    post_n = config.offset_post_samples
    if n < pre_n + post_n + 1:
        return None
    jumps = np.abs(np.diff(values))
    indices = np.argsort(jumps)[::-1][: config.offset_candidate_steps] + 1
    best: tuple[float, float, float, float] | None = None
    for index in indices:
        if index < pre_n or index + post_n > n or times[index] < config.offset_startup_guard_s:
            continue
        if commands.size and float(np.min(np.abs(commands - times[index]))) <= config.command_guard_s:
            continue
        before = values[index - pre_n:index]
        after = values[index:index + post_n]
        center = float(np.median(before))
        after_center = float(np.median(after))
        step = after_center - center
        relative = abs(step) / max(abs(center), 1e-3)
        scale = max(
            1.4826 * float(np.median(np.abs(before - center))),
            float(np.std(before)),
            abs(center) * 0.002,
            1e-3,
        )
        z_value = abs(step) / scale
        if z_value < config.offset_min_z or relative < config.offset_min_relative_step:
            continue
        std_ratio = float(np.std(after)) / max(float(np.std(before)), 1e-6)
        if not config.offset_std_ratio_min <= std_ratio <= config.offset_std_ratio_max:
            continue
        later = values[index + post_n:min(n, index + post_n + 40)]
        if len(later) < post_n:
            continue
        later_shift = float(np.median(later)) - center
        if np.sign(later_shift) != np.sign(step) or abs(later_shift) / scale < config.offset_persistence_z:
            continue
        candidate = (float(times[index]), float(z_value), float(relative), float(std_ratio))
        if best is None or candidate[1] > best[1]:
            best = candidate
    if best is None:
        return None
    transition = _transition_for(component, "Offset")
    return DXCForwardResponseFit(
        fault_id=f"{sensor_id}:Offset",
        fit_score=30.0 + best[1],
        fit_status="SUPPORTED",
        transition=transition,
        observed_response_codes=("ADDITIVE_STEP_OBSERVED", "POST_STEP_VARIABILITY_RETAINED", "STEP_PERSISTED"),
        source_component_ids=(sensor_id,),
        metrics={"onset_time_s": best[0], "step_z": best[1], "relative_step": best[2], "post_pre_std_ratio": best[3]},
    )


def _direct_sensor_fits(
    *, records: Sequence[Mapping[str, Any]], components: Mapping[str, Mapping[str, Any]], config: DXCForwardResponseConfig
) -> list[DXCForwardResponseFit]:
    commands = _command_times(records)
    result: list[DXCForwardResponseFit] = []
    stuck_components: set[str] = set()
    for component_id, component in components.items():
        if component.get("kind") != "sensor":
            continue
        modes = set(map(str, component.get("fault_modes", [])))
        if "Stuck" in modes:
            fit = _stuck_fit(records=records, component=component, config=config)
            if fit is not None:
                result.append(fit)
                stuck_components.add(component_id)
    for component_id, component in components.items():
        if component.get("kind") != "sensor" or component_id in stuck_components:
            continue
        if "Offset" not in set(map(str, component.get("fault_modes", []))):
            continue
        fit = _offset_fit(records=records, component=component, config=config, commands=commands)
        if fit is not None:
            result.append(fit)
    return sorted(result, key=lambda item: (-item.fit_score, item.fault_id))


def _light_failed_off_fits(
    *,
    records: Sequence[Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, set[str]],
    base_fault_ids: Sequence[str],
    first_alarm_time_s: float | None,
    config: DXCForwardResponseConfig,
) -> list[DXCForwardResponseFit]:
    base_components = {str(value).split(":", 1)[0] for value in base_fault_ids}
    if not any(components.get(cid, {}).get("component_type") == "LightBulb25W" for cid in base_components):
        return []
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for component_id, component in components.items():
        if component.get("component_type") != "LightBulb25W":
            continue
        wires = [node for node in adjacency.get(component_id, set()) if components.get(node, {}).get("component_type") == "Wire"]
        temps = [node for node in adjacency.get(component_id, set()) if components.get(node, {}).get("component_type") == "TemperatureSensor"]
        if len(wires) == 1 and len(temps) == 1:
            groups[wires[0]].append((component_id, temps[0]))
    alarm = max(config.startup_s, float(first_alarm_time_s or config.startup_s))
    result: list[DXCForwardResponseFit] = []
    for wire_id, members in groups.items():
        if not any(component_id in base_components for component_id, _ in members):
            continue
        light_sensors = sorted({
            node
            for component_id, _ in members
            for node in adjacency.get(component_id, set())
            if components.get(node, {}).get("component_type") == "LightSensor"
        })
        shared_drop = False
        light_drop_value = 0.0
        for sensor_id in light_sensors:
            times, values = _numeric_series(records, sensor_id)
            before = values[(times >= alarm - config.light_pre_window_s) & (times < alarm)]
            later = values[times >= alarm + config.light_late_start_s]
            if len(before) < 5 or len(later) < 10:
                continue
            baseline = float(np.median(before))
            relative = (float(np.median(later[-20:])) - baseline) / max(abs(baseline), 1e-3)
            if relative < -config.shared_light_drop_relative:
                shared_drop = True
                light_drop_value = min(light_drop_value, relative)
        if not shared_drop:
            continue
        deltas: list[tuple[str, str, float]] = []
        for component_id, temp_id in members:
            times, values = _numeric_series(records, temp_id)
            before = values[(times >= alarm - config.light_pre_window_s) & (times < alarm)]
            later = values[times >= alarm + config.light_late_start_s]
            if len(before) < 5 or len(later) < 10:
                continue
            delta = float(np.median(later[-20:]) - np.median(before))
            deltas.append((component_id, temp_id, delta))
        if len(deltas) < 2:
            continue
        warmest = max(delta for _, _, delta in deltas)
        cooling = [item for item in deltas if item[2] < -config.light_min_cooling and warmest - item[2] >= config.light_peer_separation]
        if not 1 <= len(cooling) <= len(members) - 1:
            continue
        for component_id, temp_id, delta in cooling:
            transition = _transition_for(components[component_id], "FailedOff")
            result.append(DXCForwardResponseFit(
                fault_id=f"{component_id}:FailedOff",
                fit_score=50.0 + abs(delta) + abs(light_drop_value) * 10.0,
                fit_status="SUPPORTED",
                transition=transition,
                observed_response_codes=("LOCAL_TEMPERATURE_COOLING_OBSERVED", "SHARED_LIGHT_OUTPUT_DROP_OBSERVED"),
                source_component_ids=tuple(sorted({component_id, temp_id, wire_id, *light_sensors})),
                metrics={"temperature_late_delta": delta, "shared_light_relative_delta": float(light_drop_value)},
            ))
    return sorted(result, key=lambda item: (-item.fit_score, item.fault_id))


def rank_dxc_forward_response(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    base_fault_ids: Sequence[str],
    first_alarm_time_s: float | None,
    config: DXCForwardResponseConfig | None = None,
) -> DXCForwardResponseDecision:
    """Refine an R27 fault set with a bounded qualitative forward model.

    No truth labels or fault-injection times are consumed.  The R27 result is a
    prior engineering candidate set; R28 may refine it only when one of the
    explicitly modeled state transitions has a matching or contradicting
    response.  Unsupported modes pass through unchanged.
    """
    cfg = config or DXCForwardResponseConfig()
    components, adjacency = _catalog_maps(catalog)
    base = tuple(sorted(set(map(str, base_fault_ids))))
    selected = list(base)
    vetoed: set[str] = set()
    unresolved: list[str] = []
    fits: list[DXCForwardResponseFit] = []

    direct = _direct_sensor_fits(records=records, components=components, config=cfg)
    lights = _light_failed_off_fits(
        records=records,
        components=components,
        adjacency=adjacency,
        base_fault_ids=base,
        first_alarm_time_s=first_alarm_time_s,
        config=cfg,
    )
    fits.extend(direct)
    fits.extend(lights)

    # A modeled multi-bulb OFF response supersedes only sibling 25W light
    # candidates; unrelated plant/discrete hypotheses remain untouched.
    if lights:
        modeled_light_components = {
            cid
            for cid, component in components.items()
            if component.get("component_type") == "LightBulb25W"
        }
        retained = [fault for fault in selected if fault.split(":", 1)[0] not in modeled_light_components]
        vetoed.update(fault for fault in selected if fault.split(":", 1)[0] in modeled_light_components)
        selected = retained + [item.fault_id for item in lights]

    def is_preserved_fault(fault_id: str) -> bool:
        component = components.get(fault_id.split(":", 1)[0], {})
        return component.get("kind") != "sensor" or component.get("component_type") == "PositionSensor"

    preserved = [fault for fault in selected if is_preserved_fault(fault)]
    numeric_sensor = [fault for fault in selected if not is_preserved_fault(fault)]
    stuck_fits = [item for item in direct if item.transition.fault_mode == "Stuck"]
    offset_fits = [item for item in direct if item.transition.fault_mode == "Offset"]

    if direct:
        if preserved:
            # Beside a plant/discrete diagnosis, only a high-specificity freeze
            # signature is allowed to add a numeric sensor fault.  Offset steps
            # may be physical consequences of the plant event.
            direct_selected = [item.fault_id for item in stuck_fits[: cfg.max_direct_numeric_sensor_faults]]
        else:
            direct_selected = [
                item.fault_id
                for item in sorted(stuck_fits + offset_fits, key=lambda item: (-item.fit_score, item.fault_id))[
                    : cfg.max_direct_numeric_sensor_faults
                ]
            ]
        if direct_selected:
            vetoed.update(numeric_sensor)
            selected = preserved + direct_selected

    # A sensor-only Offset prediction that is not reproduced by the explicit
    # observation-fault forward model is not converted into a plant diagnosis.
    # It becomes unresolved, preventing a coherent physical response from being
    # mislabeled as a sensor fault without inventing the failed load identity.
    preserved = [fault for fault in selected if is_preserved_fault(fault)]
    if not preserved and selected and all(fault.endswith(":Offset") for fault in selected):
        confirmed_offsets = {item.fault_id for item in offset_fits}
        if not (set(selected) & confirmed_offsets):
            vetoed.update(selected)
            selected = []
            unresolved.append("SENSOR_OFFSET_UNCORROBORATED_PHYSICAL_RESPONSE_UNRESOLVED")

    selected_faults = tuple(sorted(set(selected)))
    fits_tuple = tuple(sorted(fits, key=lambda item: (-item.fit_score, item.fault_id, item.transition.transition_kind)))
    modeled_fault_ids = tuple(sorted({item.fault_id for item in fits_tuple}))
    payload = {
        "selected_fault_ids": selected_faults,
        "base_fault_ids": base,
        "fits": [item.model_dump(mode="json") for item in fits_tuple],
        "vetoed_fault_ids": tuple(sorted(vetoed - set(selected_faults))),
        "unresolved_reason_codes": tuple(sorted(set(unresolved))),
        "modeled_fault_ids": modeled_fault_ids,
    }
    return DXCForwardResponseDecision(
        selected_fault_ids=selected_faults,
        base_fault_ids=base,
        fits=fits_tuple,
        vetoed_fault_ids=tuple(sorted(vetoed - set(selected_faults))),
        unresolved_reason_codes=tuple(sorted(set(unresolved))),
        modeled_fault_ids=modeled_fault_ids,
        result_sha256=_digest(payload),
    )


__all__ = [
    "DXCForwardResponseConfig",
    "DXCForwardResponseDecision",
    "DXCForwardResponseFit",
    "DXCModeTransition",
    "rank_dxc_forward_response",
]
