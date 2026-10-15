"""Command-aware residual + engineering-graph diagnosis for DXC/ADAPT development.

R20's DXC baseline applied a generic local waveform detector and then guessed
sensor-local fault modes.  This module introduces a bounded hybrid path:

1. an offline-admitted one-step residual model consumes previous telemetry and
   explicit command-instance state;
2. a fixed startup residual normalization adapts the model to one scenario;
3. large residual sensors are projected through the engineering system graph;
4. structural conflicts are emitted to the existing minimal-diagnosis engine;
5. a deterministic graph-support ranker produces a small engineering hypothesis
   set while retaining the full minimal-diagnosis result for audit.

The residual model is development-only and the output is not calibrated
probability or formal EvidenceFact.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from astrograph.diagnosis.model_based import MinimalDiagnosisResult, enumerate_minimal_diagnoses
from astrograph.knowledge.system_model_graph import (
    ObservationMismatch,
    SystemModelGraph,
    build_system_model_graph,
    structural_conflicts_from_mismatches,
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCHybridResidualConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    ridge_alpha: float = Field(gt=0.0, default=10.0)
    startup_calibration_s: float = Field(gt=0.0, default=30.0)
    residual_z_threshold: float = Field(gt=0.0, default=8.0)
    persistence_decisions: int = Field(ge=1, default=1)
    top_residual_sensors: int = Field(ge=1, default=4)
    support_topology_hops: int = Field(ge=0, default=3)
    conflict_topology_hops: int = Field(ge=0, default=1)
    top_components: int = Field(ge=1, default=2)
    command_context_s: float = Field(gt=0.0, default=10.0)
    include_recent_command_in_structural_conflict: bool = False
    max_diagnosis_cardinality: int = Field(ge=1, default=4)
    max_diagnosis_candidates: int = Field(ge=1, default=128)
    max_search_nodes: int = Field(ge=1, default=100_000)


@dataclass(frozen=True)
class DXCNumericSequence:
    time_s: np.ndarray
    sensor_values: np.ndarray
    command_states: np.ndarray
    sensor_ids: tuple[str, ...]
    command_ids: tuple[str, ...]


@dataclass(frozen=True)
class DXCHybridResidualModel:
    sensor_ids: tuple[str, ...]
    command_ids: tuple[str, ...]
    ridge_alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: np.ndarray
    training_residual_center: np.ndarray
    training_residual_scale: np.ndarray
    model_sha256: str


def hybrid_model_artifact(model: DXCHybridResidualModel) -> dict[str, Any]:
    payload = {
        "schema_version": "astrograph-dxc-hybrid-residual-model.v1",
        "sensor_ids": list(model.sensor_ids),
        "command_ids": list(model.command_ids),
        "ridge_alpha": float(model.ridge_alpha),
        "x_mean": model.x_mean.tolist(),
        "x_scale": model.x_scale.tolist(),
        "y_mean": model.y_mean.tolist(),
        "y_scale": model.y_scale.tolist(),
        "ridge_coef": model.ridge_coef.tolist(),
        "ridge_intercept": model.ridge_intercept.tolist(),
        "training_residual_center": model.training_residual_center.tolist(),
        "training_residual_scale": model.training_residual_scale.tolist(),
        "model_sha256": model.model_sha256,
        "score_semantics": "STANDARDIZED_FORECAST_RESIDUAL_NOT_PROBABILITY",
        "qualification": "DEVELOPMENT_ONLY_NOT_FLIGHT_QUALIFIED",
    }
    return payload


def hybrid_model_from_artifact(payload: Mapping[str, Any]) -> DXCHybridResidualModel:
    required = {
        "schema_version", "sensor_ids", "command_ids", "ridge_alpha", "x_mean", "x_scale",
        "y_mean", "y_scale", "ridge_coef", "ridge_intercept", "training_residual_center",
        "training_residual_scale", "model_sha256", "score_semantics", "qualification",
    }
    if set(payload) != required or payload.get("schema_version") != "astrograph-dxc-hybrid-residual-model.v1":
        raise ValueError("DXC_HYBRID_MODEL_ARTIFACT_SCHEMA_INVALID")
    model = DXCHybridResidualModel(
        sensor_ids=tuple(map(str, payload["sensor_ids"])),
        command_ids=tuple(map(str, payload["command_ids"])),
        ridge_alpha=float(payload["ridge_alpha"]),
        x_mean=np.asarray(payload["x_mean"], dtype=float),
        x_scale=np.asarray(payload["x_scale"], dtype=float),
        y_mean=np.asarray(payload["y_mean"], dtype=float),
        y_scale=np.asarray(payload["y_scale"], dtype=float),
        ridge_coef=np.asarray(payload["ridge_coef"], dtype=float),
        ridge_intercept=np.asarray(payload["ridge_intercept"], dtype=float),
        training_residual_center=np.asarray(payload["training_residual_center"], dtype=float),
        training_residual_scale=np.asarray(payload["training_residual_scale"], dtype=float),
        model_sha256=str(payload["model_sha256"]),
    )
    canonical = {
        "sensor_ids": model.sensor_ids, "command_ids": model.command_ids, "alpha": model.ridge_alpha,
        "x_mean": model.x_mean.tolist(), "x_scale": model.x_scale.tolist(),
        "y_mean": model.y_mean.tolist(), "y_scale": model.y_scale.tolist(),
        "ridge_coef": model.ridge_coef.tolist(), "ridge_intercept": model.ridge_intercept.tolist(),
        "residual_center": model.training_residual_center.tolist(),
        "residual_scale": model.training_residual_scale.tolist(),
    }
    if _digest(canonical) != model.model_sha256:
        raise ValueError("DXC_HYBRID_MODEL_ARTIFACT_IDENTITY_DRIFT")
    return model


class DXCHybridPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-dxc-hybrid-model-based-prediction.v1"
    predicted_fault_ids: tuple[str, ...]
    first_alarm_time_s: float | None = None
    final_alarm_time_s: float | None = None
    final_component_ids: tuple[str, ...] = ()
    minimal_diagnosis: MinimalDiagnosisResult | None = None
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_semantics: str = "STANDARDIZED_FORECAST_RESIDUAL_AND_GRAPH_SUPPORT_NOT_PROBABILITY"
    qualification: str = "DEVELOPMENT_ONLY_NOT_FLIGHT_QUALIFIED"
    claim_boundary: str = (
        "Hybrid development predictor. Residuals are command-aware one-step engineering scores; structural graph "
        "support and bounded minimal diagnoses constrain hypotheses but do not prove causality or probability."
    )


def _admitted_ids(catalog: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sensors = tuple(sorted(
        str(c["component_id"])
        for c in catalog.get("components", [])
        if c.get("kind") == "sensor" and c.get("unit_status") == "VERIFIED_FROM_XML"
    ))
    commands = tuple(sorted(str(row["command_id"]) for row in catalog.get("command_bindings", [])))
    if not sensors:
        raise ValueError("DXC_HYBRID_NO_ADMITTED_NUMERIC_SENSORS")
    return sensors, commands


def extract_numeric_sequence(records: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any]) -> DXCNumericSequence:
    sensors, commands = _admitted_ids(catalog)
    command_state = {command: 0.0 for command in commands}
    times: list[float] = []
    values: list[list[float]] = []
    states: list[list[float]] = []
    for record in records:
        kind = record.get("kind")
        if kind == "command":
            command_id = str(record.get("command_id"))
            if command_id in command_state:
                command_state[command_id] = 1.0 if bool(record.get("value")) else 0.0
            continue
        if kind != "sensors":
            raise ValueError("DXC_HYBRID_UNADMITTED_RECORD_KIND")
        row = record.get("values") or {}
        numeric: list[float] = []
        usable = True
        for sensor in sensors:
            value = row.get(sensor)
            if value is None or isinstance(value, bool):
                usable = False
                break
            number = float(value)
            if not np.isfinite(number):
                raise ValueError("DXC_HYBRID_NONFINITE_SENSOR")
            numeric.append(number)
        if not usable:
            continue
        times.append(float(record["time_tick"]) / 1000.0)
        values.append(numeric)
        states.append([command_state[command] for command in commands])
    if len(times) < 3:
        raise ValueError("DXC_HYBRID_SEQUENCE_TOO_SHORT")
    return DXCNumericSequence(
        time_s=np.asarray(times, dtype=float),
        sensor_values=np.asarray(values, dtype=float),
        command_states=np.asarray(states, dtype=float),
        sensor_ids=sensors,
        command_ids=commands,
    )


def build_training_pairs(sequences: Sequence[DXCNumericSequence]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    for sequence in sequences:
        current = (sequence.sensor_ids, sequence.command_ids)
        if signature is None:
            signature = current
        elif signature != current:
            raise ValueError("DXC_HYBRID_TRAINING_SCHEMA_DRIFT")
        for index in range(1, len(sequence.time_s)):
            xs.append(np.r_[sequence.sensor_values[index - 1], sequence.command_states[index]])
            ys.append(sequence.sensor_values[index])
    if not xs:
        raise ValueError("DXC_HYBRID_TRAINING_EMPTY")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_hybrid_residual_model(
    sequences: Sequence[DXCNumericSequence],
    *,
    alpha: float = 10.0,
) -> DXCHybridResidualModel:
    from .data_exposure import guard_development
    guard_development("FIT")
    if not sequences or alpha <= 0:
        raise ValueError("DXC_HYBRID_FIT_INVALID")
    x, y = build_training_pairs(sequences)
    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(y)
    xn = x_scaler.transform(x)
    yn = y_scaler.transform(y)
    ridge = Ridge(alpha=float(alpha)).fit(xn, yn)
    residual = yn - ridge.predict(xn)
    center = np.median(residual, axis=0)
    mad = np.median(np.abs(residual - center), axis=0)
    scale = np.maximum(1.4826 * mad, 0.05)
    payload = {
        "sensor_ids": sequences[0].sensor_ids,
        "command_ids": sequences[0].command_ids,
        "alpha": float(alpha),
        "x_mean": x_scaler.mean_.tolist(),
        "x_scale": x_scaler.scale_.tolist(),
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
        "ridge_coef": np.asarray(ridge.coef_).tolist(),
        "ridge_intercept": np.asarray(ridge.intercept_).tolist(),
        "residual_center": center.tolist(),
        "residual_scale": scale.tolist(),
    }
    return DXCHybridResidualModel(
        sensor_ids=sequences[0].sensor_ids,
        command_ids=sequences[0].command_ids,
        ridge_alpha=float(alpha),
        x_mean=np.asarray(x_scaler.mean_, dtype=float),
        x_scale=np.asarray(x_scaler.scale_, dtype=float),
        y_mean=np.asarray(y_scaler.mean_, dtype=float),
        y_scale=np.asarray(y_scaler.scale_, dtype=float),
        ridge_coef=np.asarray(ridge.coef_, dtype=float),
        ridge_intercept=np.asarray(ridge.intercept_, dtype=float),
        training_residual_center=center,
        training_residual_scale=scale,
        model_sha256=_digest(payload),
    )


def _component_adjacency(catalog: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for edge in catalog.get("connections", []):
        left = str(edge.get("c1") or "")
        right = str(edge.get("c2") or "")
        if left and right:
            result[left].add(right)
            result[right].add(left)
    return result


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


def _select_mode(component: Mapping[str, Any], recent_values: np.ndarray) -> str | None:
    modes = list(component.get("fault_modes") or [])
    if not modes:
        return None
    if component.get("kind") == "sensor":
        if recent_values.size >= 4 and float(np.max(recent_values) - np.min(recent_values)) <= 1e-9 and "Stuck" in modes:
            return "Stuck"
        if "Offset" in modes:
            return "Offset"
        if "Stuck" in modes:
            return "Stuck"
    for mode in ("FailedOff", "FailedOpen", "StuckOpen", "StuckClosed", "FlowBlocked", "UnderSpeed", "OverSpeed", "Degraded"):
        if mode in modes:
            return mode
    return sorted(modes)[0]


def predict_hybrid_model_based(
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    model: DXCHybridResidualModel,
    config: DXCHybridResidualConfig,
) -> DXCHybridPrediction:
    from .data_exposure import guard_prediction_generation
    guard_prediction_generation()
    sequence = extract_numeric_sequence(records, catalog)
    if sequence.sensor_ids != model.sensor_ids or sequence.command_ids != model.command_ids:
        raise ValueError("DXC_HYBRID_MODEL_INPUT_SCHEMA_MISMATCH")
    graph: SystemModelGraph = build_system_model_graph(dict(catalog))
    components = {str(item["component_id"]): item for item in catalog.get("components", [])}
    adjacency = _component_adjacency(catalog)
    distance_maps = {
        sensor: _distances(adjacency, sensor, config.support_topology_hops)
        for sensor in model.sensor_ids
    }

    raw_residuals: list[np.ndarray] = []
    for index in range(1, len(sequence.time_s)):
        features = np.r_[sequence.sensor_values[index - 1], sequence.command_states[index]]
        normalized_features = (features - model.x_mean) / model.x_scale
        observed = (sequence.sensor_values[index] - model.y_mean) / model.y_scale
        predicted = model.ridge_coef @ normalized_features + model.ridge_intercept
        raw_residuals.append(
            (observed - predicted - model.training_residual_center) / model.training_residual_scale
        )
    residual_matrix = np.asarray(raw_residuals, dtype=float)
    baseline_count = sum(1 for value in sequence.time_s[1:] if value < config.startup_calibration_s)
    if baseline_count < 4:
        raise ValueError("DXC_HYBRID_STARTUP_REFERENCE_INSUFFICIENT")
    startup = residual_matrix[:baseline_count]
    startup_center = np.median(startup, axis=0)
    startup_mad = np.median(np.abs(startup - startup_center), axis=0)
    startup_scale = np.maximum(1.4826 * startup_mad, 1.0)

    command_transitions: list[tuple[float, tuple[str, ...]]] = []
    for transition_index in range(1, len(sequence.time_s)):
        changed = np.where(sequence.command_states[transition_index] != sequence.command_states[transition_index - 1])[0]
        if changed.size:
            command_transitions.append((
                float(sequence.time_s[transition_index]),
                tuple(model.command_ids[int(value)] for value in changed),
            ))

    persistence = np.zeros(len(model.sensor_ids), dtype=int)
    first_alarm: float | None = None
    final_alarm: float | None = None
    final_faults: tuple[str, ...] = ()
    final_components: tuple[str, ...] = ()
    final_mbd: MinimalDiagnosisResult | None = None

    for index in range(1, len(sequence.time_s)):
        time_s = float(sequence.time_s[index])
        if time_s < config.startup_calibration_s:
            continue
        z = np.abs((residual_matrix[index - 1] - startup_center) / startup_scale)
        active = z >= config.residual_z_threshold
        persistence = np.where(active, persistence + 1, 0)
        persistent = np.where(persistence >= config.persistence_decisions)[0]
        if persistent.size == 0:
            continue
        ranked_sensor_indices = sorted(
            (int(value) for value in persistent),
            key=lambda value: (-float(z[value]), model.sensor_ids[value]),
        )[: config.top_residual_sensors]
        support: dict[str, float] = defaultdict(float)
        for sensor_index in ranked_sensor_indices:
            sensor_id = model.sensor_ids[sensor_index]
            for component_id, hops in distance_maps[sensor_id].items():
                if component_id in components and components[component_id].get("fault_modes"):
                    support[component_id] += float(z[sensor_index]) / (1.0 + float(hops))
        ranked_components = sorted(support, key=lambda value: (-support[value], value))[: config.top_components]
        faults: list[str] = []
        for component_id in ranked_components:
            if component_id in model.sensor_ids:
                sensor_index = model.sensor_ids.index(component_id)
                recent = sequence.sensor_values[max(0, index - 5) : index + 1, sensor_index]
            else:
                recent = np.asarray([], dtype=float)
            mode = _select_mode(components[component_id], recent)
            if mode:
                faults.append(f"{component_id}:{mode}")

        recent_commands: tuple[str, ...] = ()
        for transition_time, command_ids in reversed(command_transitions):
            if transition_time > time_s:
                continue
            if time_s - transition_time <= config.command_context_s:
                recent_commands = command_ids
            break
        recent_command_id = recent_commands[0] if len(recent_commands) == 1 else None
        command_id = recent_command_id if config.include_recent_command_in_structural_conflict else None
        mismatches = [
            ObservationMismatch(
                mismatch_id=f"RESIDUAL-{int(round(time_s * 1000))}-{model.sensor_ids[sensor_index]}",
                sensor_component_id=model.sensor_ids[sensor_index],
                command_id=command_id,
                source_ids=tuple(filter(None, (model.sensor_ids[sensor_index], recent_command_id))),
                explanation=(
                    "Command-aware one-step residual exceeded the frozen development threshold; "
                    + ("the single recent command transition is included in the structural conflict." if command_id else ("a recent command transition was retained as context but not expanded into the structural conflict." if recent_command_id else "no unique recent command transition was admitted for this conflict."))
                ),
            )
            for sensor_index in ranked_sensor_indices
        ]
        try:
            conflicts = structural_conflicts_from_mismatches(
                graph,
                mismatches,
                max_topology_hops=config.conflict_topology_hops,
            )
            mbd = enumerate_minimal_diagnoses(
                conflicts,
                max_cardinality=config.max_diagnosis_cardinality,
                max_candidates=config.max_diagnosis_candidates,
                max_search_nodes=config.max_search_nodes,
            )
        except ValueError:
            mbd = None

        if first_alarm is None:
            first_alarm = time_s
        final_alarm = time_s
        final_components = tuple(ranked_components)
        final_faults = tuple(sorted(set(faults)))
        final_mbd = mbd

    return DXCHybridPrediction(
        predicted_fault_ids=final_faults,
        first_alarm_time_s=first_alarm,
        final_alarm_time_s=final_alarm,
        final_component_ids=final_components,
        minimal_diagnosis=final_mbd,
        model_sha256=model.model_sha256,
        graph_sha256=graph.graph_sha256,
    )


__all__ = [
    "DXCHybridPrediction",
    "DXCHybridResidualConfig",
    "DXCHybridResidualModel",
    "DXCNumericSequence",
    "build_training_pairs",
    "extract_numeric_sequence",
    "fit_hybrid_residual_model",
    "hybrid_model_artifact",
    "hybrid_model_from_artifact",
    "predict_hybrid_model_based",
]
