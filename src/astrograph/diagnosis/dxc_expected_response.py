"""Deterministic DXC component-mode / expected-response consistency ranking.

This is a core diagnosis primitive, not an Agent tool.  It consumes already
admitted DXC records, an engineering catalog, and candidate faults produced by
upstream deterministic branches.  It can re-rank/relabel candidates using
component-mode response semantics while preserving the legacy candidates for
audit.

R27 intentionally keeps this logic separate from EvidenceFact authority:
selected candidates are engineering diagnosis hypotheses, not probabilities or
formal evidence.  The development rules were created after contact with the
DXC09 Sample development corpus and therefore require new holdout validation
before promotion to a production-default diagnosis branch.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCExpectedResponseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    startup_s: float = Field(gt=0.0, default=30.0)
    discrete_topology_hops: int = Field(ge=1, default=3)
    discrete_corroboration_z: float = Field(gt=0.0, default=4.0)
    discrete_corroboration_relative_change: float = Field(ge=0.0, default=0.01)
    discrete_required_corroborating_sensors: int = Field(ge=1, default=2)
    discrete_secondary_plateau_window_s: float = Field(gt=0.0, default=0.75)
    response_window_s: float = Field(gt=0.0, default=5.0)
    plateau_previous_samples: int = Field(ge=4, default=12)
    plateau_min_suffix_samples: int = Field(ge=8, default=20)
    plateau_absolute_range_floor: float = Field(gt=0.0, default=0.005)
    plateau_relative_tolerance: float = Field(gt=0.0, default=1e-6)
    offset_shift_z: float = Field(gt=0.0, default=8.0)
    fan_overspeed_relative: float = Field(gt=0.0, default=0.08)
    fan_failed_off_relative: float = Field(gt=0.0, default=0.70)
    fan_underspeed_relative: float = Field(gt=0.0, default=0.08)
    pump_failed_off_final_ratio: float = Field(gt=0.0, lt=1.0, default=0.20)
    pump_flow_blocked_final_ratio: float = Field(gt=0.0, lt=1.0, default=0.80)
    inverter_drop_relative: float = Field(gt=0.0, lt=1.0, default=0.50)
    max_supported_sensor_faults: int = Field(ge=1, default=2)


class DXCExpectedResponseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    fault_id: str = Field(min_length=3)
    score: float
    source: str
    reason_codes: tuple[str, ...]
    source_component_ids: tuple[str, ...] = ()


class DXCExpectedResponseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-expected-response-decision.v1"
    selected_fault_ids: tuple[str, ...]
    candidates: tuple[DXCExpectedResponseCandidate, ...]
    continuous_fault_ids: tuple[str, ...]
    discrete_fault_ids: tuple[str, ...]
    used_discrete_state_branch: bool
    reclassified_position_sensor_count: int = Field(ge=0)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_semantics: str = "DETERMINISTIC_EXPECTED_RESPONSE_PRIORITY_NOT_PROBABILITY"
    formal_evidence_eligible: bool = False
    claim_boundary: str = (
        "Deterministic component-mode/response consistency ranking over admitted telemetry and engineering topology. "
        "Scores are ordering weights only; this result is not posterior probability, formal EvidenceFact, or release authority."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCExpectedResponseDecision":
        if self.formal_evidence_eligible:
            raise ValueError("DXC expected-response decision cannot be formal evidence")
        payload = {
            "selected_fault_ids": self.selected_fault_ids,
            "candidates": [item.model_dump(mode="json") for item in self.candidates],
            "continuous_fault_ids": self.continuous_fault_ids,
            "discrete_fault_ids": self.discrete_fault_ids,
            "used_discrete_state_branch": self.used_discrete_state_branch,
            "reclassified_position_sensor_count": self.reclassified_position_sensor_count,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC expected-response result_sha256 mismatch")
        return self


def _catalog_maps(catalog: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, set[str]]]:
    components = {str(item["component_id"]): item for item in catalog.get("components", [])}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in catalog.get("connections", []):
        left, right = str(edge.get("c1") or ""), str(edge.get("c2") or "")
        if left and right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return components, adjacency


def _distances(adjacency: Mapping[str, set[str]], start: str, max_hops: int) -> dict[str, int]:
    found = {start: 0}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        if found[current] >= max_hops:
            continue
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor not in found:
                found[neighbor] = found[current] + 1
                queue.append(neighbor)
    return found


def _numeric_series(records: Sequence[Mapping[str, Any]], sensor_id: str) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []
    for record in records:
        if str(record.get("kind") or "") != "sensors":
            continue
        row = record.get("values") or {}
        if sensor_id not in row or isinstance(row[sensor_id], bool):
            continue
        value = float(row[sensor_id])
        if not np.isfinite(value):
            raise ValueError(f"DXC_EXPECTED_RESPONSE_NONFINITE_SENSOR:{sensor_id}")
        times.append(float(record["time_tick"]) / 1000.0)
        values.append(value)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _window_response(records: Sequence[Mapping[str, Any]], sensor_id: str, time_s: float, window_s: float) -> dict[str, float] | None:
    times, values = _numeric_series(records, sensor_id)
    if values.size < 4:
        return None
    pre = values[(times >= time_s - window_s) & (times < time_s)]
    post = values[(times >= time_s) & (times < time_s + window_s)]
    if pre.size < 2 or post.size < 2:
        return None
    pre_mean = float(np.mean(pre))
    post_mean = float(np.mean(post))
    scale = max(float(np.std(pre)), abs(pre_mean) * 0.002, 1e-3)
    return {
        "z": abs(post_mean - pre_mean) / scale,
        "relative_change": (post_mean - pre_mean) / max(abs(pre_mean), 1e-3),
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "pre_range": float(np.ptp(pre)),
        "post_range": float(np.ptp(post)),
    }


def _base_stats(records: Sequence[Mapping[str, Any]], sensor_id: str, startup_s: float) -> dict[str, Any] | None:
    times, values = _numeric_series(records, sensor_id)
    if values.size < 40:
        return None
    startup = values[times < startup_s]
    if startup.size < 20:
        startup = values[: min(60, max(20, values.size // 8))]
    later = values[startup.size :]
    if later.size < 20:
        return None
    center = float(np.median(startup))
    mad = float(np.median(np.abs(startup - center)))
    scale = max(1.4826 * mad, abs(center) * 0.002, 1e-3)
    return {
        "center": center,
        "scale": scale,
        "minimum": float(np.min(later)),
        "maximum": float(np.max(later)),
        "final": float(np.median(values[-20:])),
        "values": values,
        "times": times,
    }


def _suffix_plateau_onset(
    records: Sequence[Mapping[str, Any]], sensor_id: str, config: DXCExpectedResponseConfig
) -> float | None:
    times, values = _numeric_series(records, sensor_id)
    n = config.plateau_previous_samples
    if values.size < n + config.plateau_min_suffix_samples:
        return None
    for index in range(n, values.size - config.plateau_min_suffix_samples):
        if times[index] < config.startup_s:
            continue
        previous = values[index - n : index]
        suffix = values[index:]
        tolerance = max(abs(float(np.mean(previous))) * config.plateau_relative_tolerance, 1e-6)
        if float(np.ptp(suffix)) <= tolerance and float(np.ptp(previous)) > max(
            config.plateau_absolute_range_floor, 20.0 * tolerance
        ):
            return float(times[index])
    return None


def _continuous_sensor_candidates(
    records: Sequence[Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
    continuous_fault_ids: Sequence[str],
    config: DXCExpectedResponseConfig,
) -> list[DXCExpectedResponseCandidate]:
    result: list[DXCExpectedResponseCandidate] = []
    for fault_id in continuous_fault_ids:
        if ":" not in fault_id:
            continue
        component_id, _ = fault_id.split(":", 1)
        component = components.get(component_id, {})
        if component.get("kind") != "sensor" or component.get("unit_status") != "VERIFIED_FROM_XML":
            continue
        plateau = _suffix_plateau_onset(records, component_id, config)
        if plateau is not None and "Stuck" in component.get("fault_modes", []):
            result.append(DXCExpectedResponseCandidate(
                fault_id=f"{component_id}:Stuck",
                score=25.0,
                source="CONTINUOUS_SENSOR_MODE_CONSISTENCY",
                reason_codes=("PERSISTENT_SUFFIX_PLATEAU",),
                source_component_ids=(component_id,),
            ))
            continue
        if "Offset" not in component.get("fault_modes", []):
            continue
        stats = _base_stats(records, component_id, config.startup_s)
        if stats is None:
            continue
        shift_z = max(
            abs(stats["minimum"] - stats["center"]),
            abs(stats["maximum"] - stats["center"]),
        ) / stats["scale"]
        if shift_z >= config.offset_shift_z:
            result.append(DXCExpectedResponseCandidate(
                fault_id=f"{component_id}:Offset",
                score=20.0 + min(shift_z, 50.0) / 50.0,
                source="CONTINUOUS_SENSOR_MODE_CONSISTENCY",
                reason_codes=("PERSISTENT_SHIFT_WITHOUT_SUFFIX_PLATEAU",),
                source_component_ids=(component_id,),
            ))
    return result


def _expected_physical_candidates(
    records: Sequence[Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, set[str]],
    continuous_fault_ids: Sequence[str],
    config: DXCExpectedResponseConfig,
) -> list[DXCExpectedResponseCandidate]:
    result: list[DXCExpectedResponseCandidate] = []
    continuous_components = [fault.split(":", 1)[0] for fault in continuous_fault_ids if ":" in fault]

    def append(fault_id: str, score: float, source: str, reasons: tuple[str, ...], ids: tuple[str, ...]) -> None:
        result.append(DXCExpectedResponseCandidate(
            fault_id=fault_id, score=score, source=source, reason_codes=reasons, source_component_ids=ids
        ))

    for fault_id in continuous_fault_ids:
        if ":" not in fault_id:
            continue
        component_id, mode = fault_id.split(":", 1)
        component = components.get(component_id, {})
        component_type = str(component.get("component_type") or "")
        if component.get("kind") == "sensor":
            stats = _base_stats(records, component_id, config.startup_s)
            if stats is None:
                continue
            up = (stats["maximum"] - stats["center"]) / max(abs(stats["center"]), 1.0)
            down = (stats["minimum"] - stats["center"]) / max(abs(stats["center"]), 1.0)
            for neighbor in sorted(adjacency.get(component_id, set())):
                target = components.get(neighbor, {})
                target_type = str(target.get("component_type") or "")
                if component_type == "SpeedTransmitter" and target_type == "LargeFan":
                    selected: str | None = None
                    if up > config.fan_overspeed_relative and up >= abs(down):
                        selected = "OverSpeed"
                    elif down < -config.fan_failed_off_relative:
                        selected = "FailedOff"
                    elif down < -config.fan_underspeed_relative:
                        selected = "UnderSpeed"
                    if selected in target.get("fault_modes", []):
                        append(
                            f"{neighbor}:{selected}", 30.0 + max(up, abs(down)), "DIRECT_EXPECTED_RESPONSE",
                            ("SPEED_TRANSMITTER_DIRECTIONAL_RESPONSE",), (component_id, neighbor),
                        )
                if component_type == "FlowTransmitter" and target_type == "WaterPump":
                    values = stats["values"]
                    high = float(np.quantile(values, 0.9))
                    if high <= 1e-6:
                        continue
                    ratio = float(np.median(values[-20:])) / high
                    selected = None
                    score = 0.0
                    if ratio < config.pump_failed_off_final_ratio:
                        selected, score = "FailedOff", 35.0 + (1.0 - ratio)
                    elif ratio < config.pump_flow_blocked_final_ratio:
                        selected, score = "FlowBlocked", 28.0 + (1.0 - ratio)
                    if selected in target.get("fault_modes", []):
                        append(
                            f"{neighbor}:{selected}", score, "DIRECT_EXPECTED_RESPONSE",
                            ("FLOW_TRANSMITTER_OPERATING_TO_FINAL_RESPONSE",), (component_id, neighbor),
                        )
        elif component_type == "LargeFan":
            for neighbor in sorted(adjacency.get(component_id, set())):
                if components.get(neighbor, {}).get("component_type") != "SpeedTransmitter":
                    continue
                stats = _base_stats(records, neighbor, config.startup_s)
                if stats is None:
                    continue
                up = (stats["maximum"] - stats["center"]) / max(abs(stats["center"]), 1.0)
                down = (stats["minimum"] - stats["center"]) / max(abs(stats["center"]), 1.0)
                selected = None
                if up > config.fan_overspeed_relative and up >= abs(down):
                    selected = "OverSpeed"
                elif down < -config.fan_failed_off_relative:
                    selected = "FailedOff"
                elif down < -config.fan_underspeed_relative:
                    selected = "UnderSpeed"
                if selected in component.get("fault_modes", []):
                    append(
                        f"{component_id}:{selected}", 30.0 + max(up, abs(down)), "COMPONENT_MODE_EXPECTED_RESPONSE",
                        ("FAN_SPEED_MODE_DIRECTION",), (component_id, neighbor),
                    )
        elif component_type == "LightBulb25W" and mode == "FailedOff":
            # The existing continuous branch has already associated this physical
            # component with the response.  R27 gives the physical load candidate
            # authority over an adjacent temperature-sensor Offset explanation,
            # but does not invent which bulb failed when the branch candidate is absent.
            append(
                fault_id, 30.0, "EXISTING_PHYSICAL_COMPONENT_CANDIDATE",
                ("PHYSICAL_LOAD_CANDIDATE_PREFERRED_OVER_SENSOR_OFFSET",), (component_id,),
            )

    # Inverter FailedOff is admitted only when the continuous branch is already
    # reacting to AC-side sensors and at least one inverter-adjacent voltage or
    # frequency sensor shows a severe collapse.  This avoids unconditional
    # topology expansion.
    if any(str(components.get(cid, {}).get("component_type") or "").startswith("AC") for cid in continuous_components):
        for component_id, component in components.items():
            if component.get("component_type") != "Inverter" or "FailedOff" not in component.get("fault_modes", []):
                continue
            evidence = 0
            source_ids: list[str] = [component_id]
            for wire in sorted(adjacency.get(component_id, set())):
                for sensor_id in sorted(adjacency.get(wire, set())):
                    sensor_type = components.get(sensor_id, {}).get("component_type")
                    if sensor_type not in {"ACVoltageSensor", "ACFrequencyTransmitter"}:
                        continue
                    stats = _base_stats(records, sensor_id, config.startup_s)
                    if stats is None:
                        continue
                    drop = (stats["minimum"] - stats["center"]) / max(abs(stats["center"]), 1.0)
                    if drop < -config.inverter_drop_relative:
                        evidence += 1
                        source_ids.append(sensor_id)
            if evidence:
                append(
                    f"{component_id}:FailedOff", 25.0 + evidence, "INVERTER_EXPECTED_RESPONSE",
                    ("INVERTER_ADJACENT_AC_COLLAPSE",), tuple(source_ids),
                )

    # Deduplicate by fault_id, retaining the strongest deterministic explanation.
    by_fault: dict[str, DXCExpectedResponseCandidate] = {}
    for item in result:
        previous = by_fault.get(item.fault_id)
        if previous is None or (item.score, item.source, item.fault_id) > (previous.score, previous.source, previous.fault_id):
            by_fault[item.fault_id] = item
    return sorted(by_fault.values(), key=lambda item: (-item.score, item.fault_id, item.source))


def _global_plateaus(
    records: Sequence[Mapping[str, Any]], components: Mapping[str, Mapping[str, Any]], config: DXCExpectedResponseConfig
) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for component_id, component in components.items():
        if component.get("kind") != "sensor" or component.get("unit_status") != "VERIFIED_FROM_XML":
            continue
        if "Stuck" not in component.get("fault_modes", []):
            continue
        onset = _suffix_plateau_onset(records, component_id, config)
        if onset is not None:
            rows.append((onset, component_id))
    return sorted(rows)


def rank_dxc_expected_response(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    continuous_fault_ids: Sequence[str],
    discrete_events: Sequence[Mapping[str, Any]],
    discrete_fault_ids: Sequence[str],
    discrete_first_event_time_s: float | None,
    config: DXCExpectedResponseConfig | None = None,
) -> DXCExpectedResponseDecision:
    """Return deterministic component-mode candidates without reading truth labels."""
    cfg = config or DXCExpectedResponseConfig()
    components, adjacency = _catalog_maps(catalog)
    continuous_faults = tuple(sorted(set(map(str, continuous_fault_ids))))
    discrete_faults = tuple(sorted(set(map(str, discrete_fault_ids))))
    candidates: list[DXCExpectedResponseCandidate] = []
    selected: list[str] = []
    reclassified = 0

    if discrete_faults:
        for event in discrete_events:
            sensor_id = str(event["sensor_component_id"])
            target_id = str(event["controlled_component_id"])
            predicted_fault_id = str(event["predicted_fault_id"])
            event_time = float(event["time_s"])
            strong: set[str] = set()
            for component_id in _distances(adjacency, target_id, cfg.discrete_topology_hops):
                component = components.get(component_id, {})
                if component_id == sensor_id or component.get("kind") != "sensor":
                    continue
                if component.get("unit_status") != "VERIFIED_FROM_XML":
                    continue
                response = _window_response(records, component_id, event_time, cfg.response_window_s)
                if response is None:
                    continue
                if response["z"] >= cfg.discrete_corroboration_z and abs(response["relative_change"]) >= cfg.discrete_corroboration_relative_change:
                    strong.add(component_id)
            if len(strong) >= cfg.discrete_required_corroborating_sensors:
                candidate = DXCExpectedResponseCandidate(
                    fault_id=predicted_fault_id,
                    score=40.0 + len(strong),
                    source="DISCRETE_PHYSICAL_CORROBORATION",
                    reason_codes=("POSITION_STATE_CHANGE_WITH_MULTI_SENSOR_PHYSICAL_RESPONSE",),
                    source_component_ids=tuple(sorted({sensor_id, target_id, *strong})),
                )
            else:
                position = components.get(sensor_id, {})
                if "Stuck" in position.get("fault_modes", []):
                    candidate = DXCExpectedResponseCandidate(
                        fault_id=f"{sensor_id}:Stuck",
                        score=40.0,
                        source="DISCRETE_SENSOR_RECLASSIFICATION",
                        reason_codes=("POSITION_STATE_CHANGE_NOT_PHYSICALLY_CORROBORATED",),
                        source_component_ids=(sensor_id, target_id),
                    )
                    reclassified += 1
                else:
                    candidate = DXCExpectedResponseCandidate(
                        fault_id=predicted_fault_id,
                        score=35.0,
                        source="DISCRETE_UNRESOLVED_FALLBACK",
                        reason_codes=("POSITION_SENSOR_HAS_NO_ADMITTED_STUCK_MODE",),
                        source_component_ids=(sensor_id, target_id),
                    )
            candidates.append(candidate)
            selected.append(candidate.fault_id)

        if reclassified and continuous_faults:
            # R26 continuous output is bounded to at most a very small set.  When
            # a position observation is itself reclassified as faulty, retain one
            # independent continuous hypothesis to support multi-fault cases.
            candidates.append(DXCExpectedResponseCandidate(
                fault_id=continuous_faults[0],
                score=25.0,
                source="CONTINUOUS_SECONDARY_AFTER_POSITION_RECLASSIFICATION",
                reason_codes=("INDEPENDENT_CONTINUOUS_HYPOTHESIS_RETAINED",),
                source_component_ids=(continuous_faults[0].split(":", 1)[0],),
            ))
            selected.append(continuous_faults[0])
        elif reclassified == 0 and discrete_first_event_time_s is not None:
            near = [
                (onset, component_id)
                for onset, component_id in _global_plateaus(records, components, cfg)
                if abs(onset - discrete_first_event_time_s) <= cfg.discrete_secondary_plateau_window_s
                and component_id not in {fault.split(":", 1)[0] for fault in selected}
            ]
            if near:
                onset, component_id = near[0]
                fault_id = f"{component_id}:Stuck"
                candidates.append(DXCExpectedResponseCandidate(
                    fault_id=fault_id,
                    score=24.0,
                    source="COINCIDENT_SECONDARY_STUCK_RESPONSE",
                    reason_codes=("SUFFIX_PLATEAU_COINCIDENT_WITH_DISCRETE_EVENT",),
                    source_component_ids=(component_id,),
                ))
                selected.append(fault_id)
    else:
        physical = _expected_physical_candidates(records, components, adjacency, continuous_faults, cfg)
        sensor = _continuous_sensor_candidates(records, components, continuous_faults, cfg)
        candidates.extend(physical)
        candidates.extend(sensor)
        if physical and (not sensor or physical[0].score >= sensor[0].score):
            selected = [physical[0].fault_id]
        elif sensor:
            selected = [item.fault_id for item in sensor[: cfg.max_supported_sensor_faults]]
        elif not continuous_faults:
            plateau = _global_plateaus(records, components, cfg)
            max_time_s = max((float(row.get("time_tick", 0)) / 1000.0 for row in records), default=0.0)
            early = [(time_s, component_id) for time_s, component_id in plateau if time_s < 0.9 * max_time_s]
            if len(early) == 1:
                time_s, component_id = early[0]
                fault_id = f"{component_id}:Stuck"
                candidates.append(DXCExpectedResponseCandidate(
                    fault_id=fault_id,
                    score=18.0,
                    source="UNIQUE_PERSISTENT_SUFFIX_PLATEAU",
                    reason_codes=("UNIQUE_EARLY_SUFFIX_PLATEAU_WITH_NO_CONTINUOUS_CANDIDATE",),
                    source_component_ids=(component_id,),
                ))
                selected = [fault_id]
        if not selected and continuous_faults:
            physical_fallback = [
                fault for fault in continuous_faults
                if components.get(fault.split(":", 1)[0], {}).get("kind") != "sensor"
            ]
            selected = [physical_fallback[0] if physical_fallback else continuous_faults[0]]
            candidates.append(DXCExpectedResponseCandidate(
                fault_id=selected[0],
                score=1.0,
                source="R26_CONTINUOUS_FALLBACK",
                reason_codes=("NO_STRONGER_EXPECTED_RESPONSE_RULE",),
                source_component_ids=(selected[0].split(":", 1)[0],),
            ))

    selected_faults = tuple(sorted(set(selected)))
    # Deduplicate candidate rows for a stable audit surface.
    best: dict[tuple[str, str], DXCExpectedResponseCandidate] = {}
    for item in candidates:
        key = (item.fault_id, item.source)
        previous = best.get(key)
        if previous is None or item.score > previous.score:
            best[key] = item
    candidate_rows = tuple(sorted(best.values(), key=lambda item: (-item.score, item.fault_id, item.source)))
    payload = {
        "selected_fault_ids": selected_faults,
        "candidates": [item.model_dump(mode="json") for item in candidate_rows],
        "continuous_fault_ids": continuous_faults,
        "discrete_fault_ids": discrete_faults,
        "used_discrete_state_branch": bool(discrete_faults),
        "reclassified_position_sensor_count": reclassified,
    }
    return DXCExpectedResponseDecision(
        selected_fault_ids=selected_faults,
        candidates=candidate_rows,
        continuous_fault_ids=continuous_faults,
        discrete_fault_ids=discrete_faults,
        used_discrete_state_branch=bool(discrete_faults),
        reclassified_position_sensor_count=reclassified,
        result_sha256=_digest(payload),
    )


__all__ = [
    "DXCExpectedResponseCandidate",
    "DXCExpectedResponseConfig",
    "DXCExpectedResponseDecision",
    "rank_dxc_expected_response",
]
