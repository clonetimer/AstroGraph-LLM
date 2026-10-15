"""Governed simulation/counterfactual interface for diagnosis review.

R23 deliberately ships a zero-extra-dependency provider that consumes only
host-admitted, precomputed traces.  Live Basilisk/Simulink/Sat adapters can
implement the same narrow provider contract later without becoming AstroGraph
core dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.model_based import MinimalDiagnosisCandidate


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("simulation trace contains a non-finite value")
    return result


class SimulationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-diagnostic-simulation-trace.v1"
    trace_id: str = Field(min_length=1)
    time_s: tuple[float, ...] = Field(min_length=1)
    channels: dict[str, tuple[float, ...]] = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_kind: str = "HOST_ADMITTED_PRECOMPUTED_TRACE"

    @model_validator(mode="after")
    def trace_integrity(self) -> "SimulationTrace":
        if any(not math.isfinite(value) for value in self.time_s):
            raise ValueError("trace timestamps must be finite")
        if any(b <= a for a, b in zip(self.time_s, self.time_s[1:])):
            raise ValueError("trace timestamps must be strictly increasing")
        for channel, values in self.channels.items():
            if not channel:
                raise ValueError("trace channel name cannot be empty")
            if len(values) != len(self.time_s):
                raise ValueError("trace channel length must match time_s")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("trace channel values must be finite")
        payload = {
            "trace_id": self.trace_id,
            "time_s": self.time_s,
            "channels": self.channels,
            "source_artifact_sha256": self.source_artifact_sha256,
            "provenance_kind": self.provenance_kind,
        }
        if _digest(payload) != self.trace_sha256:
            raise ValueError("simulation trace_sha256 mismatch")
        return self


def build_simulation_trace(
    *,
    trace_id: str,
    time_s: Sequence[float],
    channels: Mapping[str, Sequence[float]],
    source_artifact_sha256: str,
    provenance_kind: str = "HOST_ADMITTED_PRECOMPUTED_TRACE",
) -> SimulationTrace:
    time_values = _finite_tuple(time_s)
    channel_values = {str(name): _finite_tuple(values) for name, values in sorted(channels.items())}
    payload = {
        "trace_id": trace_id,
        "time_s": time_values,
        "channels": channel_values,
        "source_artifact_sha256": source_artifact_sha256,
        "provenance_kind": provenance_kind,
    }
    return SimulationTrace(**payload, trace_sha256=_digest(payload))


class SimulationHypothesisCandidate(BaseModel):
    """Neutral host-bound candidate for counterfactual simulation.

    Unlike ``MinimalDiagnosisCandidate``, this contract does not assert conflict
    coverage, subset minimality, or minimum cardinality.  It is appropriate for
    physics-generated hypotheses that are only being compared by a simulation
    provider.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnosis_id: str = Field(min_length=1)
    fault_ids: tuple[str, ...]
    candidate_semantics: str = "HOST_BOUND_SIMULATION_HYPOTHESIS_NOT_DIAGNOSIS_PROOF"

    @model_validator(mode="after")
    def candidate_integrity(self) -> "SimulationHypothesisCandidate":
        if tuple(sorted(set(self.fault_ids))) != self.fault_ids:
            raise ValueError("simulation hypothesis fault_ids must be unique and sorted")
        return self


class CounterfactualSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-counterfactual-simulation-request.v1"
    diagnosis_id: str = Field(min_length=1)
    fault_ids: tuple[str, ...]
    required_channels: tuple[str, ...] = Field(min_length=1)
    expected_time_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def request_integrity(self) -> "CounterfactualSimulationRequest":
        if tuple(sorted(set(self.fault_ids))) != self.fault_ids:
            raise ValueError("fault_ids must be unique and sorted")
        if tuple(sorted(set(self.required_channels))) != self.required_channels:
            raise ValueError("required_channels must be unique and sorted")
        payload = {
            "diagnosis_id": self.diagnosis_id,
            "fault_ids": self.fault_ids,
            "required_channels": self.required_channels,
            "expected_time_sha256": self.expected_time_sha256,
        }
        if _digest(payload) != self.request_sha256:
            raise ValueError("counterfactual request_sha256 mismatch")
        return self


def build_counterfactual_request(
    candidate: MinimalDiagnosisCandidate | SimulationHypothesisCandidate,
    *,
    required_channels: Sequence[str],
    time_s: Sequence[float],
) -> CounterfactualSimulationRequest:
    channels = tuple(sorted(set(str(value) for value in required_channels)))
    if not channels:
        raise ValueError("required_channels cannot be empty")
    payload = {
        "diagnosis_id": candidate.diagnosis_id,
        "fault_ids": candidate.fault_ids,
        "required_channels": channels,
        "expected_time_sha256": _digest(_finite_tuple(time_s)),
    }
    return CounterfactualSimulationRequest(**payload, request_sha256=_digest(payload))


class DiagnosticSimulationProvider(ABC):
    """Narrow optional simulation provider used by deterministic diagnosis review."""

    provider_id: str

    @abstractmethod
    def predict(self, request: CounterfactualSimulationRequest) -> SimulationTrace:
        raise NotImplementedError


class PrecomputedTraceSimulationProvider(DiagnosticSimulationProvider):
    """Zero-dependency provider over already admitted simulation traces.

    The provider never opens arbitrary paths and never receives an LLM-authored
    simulator configuration.  The host constructs it from content-verified traces.
    """

    provider_id = "precomputed-trace-v1"

    def __init__(self, traces_by_diagnosis_id: Mapping[str, SimulationTrace]):
        if not traces_by_diagnosis_id:
            raise ValueError("precomputed simulation provider requires at least one trace")
        self._traces = dict(traces_by_diagnosis_id)

    def predict(self, request: CounterfactualSimulationRequest) -> SimulationTrace:
        try:
            trace = self._traces[request.diagnosis_id]
        except KeyError as exc:
            raise KeyError(f"no admitted simulation trace for {request.diagnosis_id}") from exc
        if _digest(trace.time_s) != request.expected_time_sha256:
            raise ValueError("simulation trace timeline does not match the admitted request")
        missing = sorted(set(request.required_channels) - set(trace.channels))
        if missing:
            raise ValueError(f"simulation trace missing required channels: {missing}")
        return trace


class CounterfactualComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    diagnosis_id: str
    fault_ids: tuple[str, ...]
    aggregate_normalized_rmse: float = Field(ge=0.0)
    aggregate_normalized_mae: float = Field(ge=0.0)
    per_channel: dict[str, dict[str, float]]
    sample_count: int = Field(ge=1)
    simulation_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_semantics: str = "DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY"


class CounterfactualDiagnosisReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-counterfactual-diagnosis-review.v1"
    provider_id: str
    observed_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    channel_scales_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparisons: tuple[CounterfactualComparison, ...]
    best_diagnosis_id: str | None = None
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formal_evidence_eligible: bool = False
    claim_boundary: str = (
        "Counterfactual trace-distance review over host-admitted traces. Lower distance indicates closer trace fit only; "
        "it is not a posterior fault probability, causal proof, formal EvidenceFact, or release authority."
    )

    @model_validator(mode="after")
    def shadow_only(self) -> "CounterfactualDiagnosisReview":
        if self.formal_evidence_eligible:
            raise ValueError("R23 counterfactual diagnosis review is shadow-only")
        return self


def compare_counterfactual_diagnoses(
    *,
    observed: SimulationTrace,
    candidates: Sequence[MinimalDiagnosisCandidate | SimulationHypothesisCandidate],
    provider: DiagnosticSimulationProvider,
    channel_scales: Mapping[str, float],
    channels: Sequence[str] | None = None,
) -> CounterfactualDiagnosisReview:
    selected_channels = tuple(sorted(set(channels or observed.channels.keys())))
    if not selected_channels:
        raise ValueError("at least one comparison channel is required")
    missing_observed = sorted(set(selected_channels) - set(observed.channels))
    if missing_observed:
        raise ValueError(f"observed trace missing comparison channels: {missing_observed}")
    scales: dict[str, float] = {}
    for channel in selected_channels:
        scale = float(channel_scales[channel])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"channel scale must be finite and positive: {channel}")
        scales[channel] = scale

    comparisons: list[CounterfactualComparison] = []
    for candidate in candidates:
        request = build_counterfactual_request(candidate, required_channels=selected_channels, time_s=observed.time_s)
        predicted = provider.predict(request)
        if predicted.time_s != observed.time_s:
            raise ValueError("counterfactual trace timeline mismatch")
        per_channel: dict[str, dict[str, float]] = {}
        channel_rmse: list[float] = []
        channel_mae: list[float] = []
        for channel in selected_channels:
            obs = observed.channels[channel]
            pred = predicted.channels[channel]
            residual = [o - p for o, p in zip(obs, pred)]
            scale = scales[channel]
            rmse = math.sqrt(sum(value * value for value in residual) / len(residual)) / scale
            mae = sum(abs(value) for value in residual) / len(residual) / scale
            per_channel[channel] = {"normalized_rmse": rmse, "normalized_mae": mae, "scale": scale}
            channel_rmse.append(rmse)
            channel_mae.append(mae)
        comparisons.append(CounterfactualComparison(
            diagnosis_id=candidate.diagnosis_id,
            fault_ids=candidate.fault_ids,
            aggregate_normalized_rmse=sum(channel_rmse) / len(channel_rmse),
            aggregate_normalized_mae=sum(channel_mae) / len(channel_mae),
            per_channel=per_channel,
            sample_count=len(observed.time_s),
            simulation_trace_sha256=predicted.trace_sha256,
        ))

    comparisons.sort(key=lambda item: (item.aggregate_normalized_rmse, item.aggregate_normalized_mae, item.diagnosis_id))
    scales_sha = _digest(scales)
    payload = {
        "provider_id": provider.provider_id,
        "observed_trace_sha256": observed.trace_sha256,
        "channel_scales_sha256": scales_sha,
        "comparisons": [item.model_dump(mode="json") for item in comparisons],
        "best_diagnosis_id": comparisons[0].diagnosis_id if comparisons else None,
    }
    return CounterfactualDiagnosisReview(
        provider_id=provider.provider_id,
        observed_trace_sha256=observed.trace_sha256,
        channel_scales_sha256=scales_sha,
        comparisons=tuple(comparisons),
        best_diagnosis_id=comparisons[0].diagnosis_id if comparisons else None,
        result_sha256=_digest(payload),
    )


__all__ = [
    "CounterfactualComparison",
    "CounterfactualDiagnosisReview",
    "CounterfactualSimulationRequest",
    "DiagnosticSimulationProvider",
    "PrecomputedTraceSimulationProvider",
    "SimulationHypothesisCandidate",
    "SimulationTrace",
    "build_counterfactual_request",
    "build_simulation_trace",
    "compare_counterfactual_diagnoses",
]
