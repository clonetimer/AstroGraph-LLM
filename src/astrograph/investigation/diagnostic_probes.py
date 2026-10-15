from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.model_based import enumerate_minimal_diagnoses, MinimalDiagnosisCandidate
from astrograph.diagnosis.dxc_expected_response import DXCExpectedResponseDecision
from astrograph.diagnosis.dxc_forward_response import DXCForwardResponseDecision
from astrograph.diagnosis.dxc_reduced_order_physics import DXCReducedOrderPhysicsDecision
from astrograph.diagnosis.dxc_pump_reduced_order import DXCWaterPumpPhysicsDecision
from astrograph.simulation.diagnostic_provider import (
    PrecomputedTraceSimulationProvider,
    SimulationTrace,
    compare_counterfactual_diagnoses,
)

from astrograph.evidence import (
    ComparisonOperator,
    EpisodeEvidenceBundle,
    EvidenceFactStatus,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


class DiagnosticProbeKind(str, Enum):
    EVIDENCE_THRESHOLD_MARGIN = "EVIDENCE_THRESHOLD_MARGIN"
    PRE_POST_CHANGE = "PRE_POST_CHANGE"
    COMMAND_RESPONSE_RESIDUAL = "COMMAND_RESPONSE_RESIDUAL"
    LAG_CORRELATION = "LAG_CORRELATION"
    OSCILLATION_SIGNATURE = "OSCILLATION_SIGNATURE"
    SATURATION_LIMIT_CHECK = "SATURATION_LIMIT_CHECK"
    MINIMAL_DIAGNOSIS_REVIEW = "MINIMAL_DIAGNOSIS_REVIEW"
    SIMULATION_COUNTERFACTUAL_REVIEW = "SIMULATION_COUNTERFACTUAL_REVIEW"
    DXC_EXPECTED_RESPONSE_REVIEW = "DXC_EXPECTED_RESPONSE_REVIEW"
    DXC_FORWARD_SIMULATION_REVIEW = "DXC_FORWARD_SIMULATION_REVIEW"
    DXC_REDUCED_ORDER_SIMULATION_REVIEW = "DXC_REDUCED_ORDER_SIMULATION_REVIEW"
    DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW = "DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW"


class DiagnosticProbeSpec(BaseModel):
    """A host-bound deterministic diagnostic operation.

    The LLM selects only ``probe_id``. It cannot alter channels, time windows,
    thresholds, algorithms, or source identities carried by this immutable spec.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-diagnostic-probe-spec.v1"
    probe_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    kind: DiagnosticProbeKind
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    target_fault_ids: tuple[str, ...] = ()
    evidence_definition_id: str | None = None
    feature_result_id: str | None = None
    unit: str | None = None
    bound_inputs: dict[str, Any] = Field(default_factory=dict)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formal_evidence_eligible: bool = False
    claim_boundary: str = (
        "Shadow deterministic diagnostic probe. The model selects only the preregistered probe_id; "
        "host-bound inputs are immutable and probe output does not create or modify formal EvidenceFact records."
    )

    @model_validator(mode="after")
    def shadow_only(self) -> "DiagnosticProbeSpec":
        if self.formal_evidence_eligible:
            raise ValueError("v1 diagnostic probes are shadow-only and cannot be formal evidence")
        return self


class DiagnosticProbeCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-diagnostic-probe-catalog.v1"
    episode_id: str
    probes: tuple[DiagnosticProbeSpec, ...] = ()
    telemetry_admission_status: str = "NOT_ADMITTED"
    claim_boundary: str = (
        "Only host-bound probes are exposed. Raw time-series probes remain unavailable until a separate "
        "telemetry admission contract binds channels, units, windows, and source hashes."
    )

    @model_validator(mode="after")
    def unique_probe_ids(self) -> "DiagnosticProbeCatalog":
        ids = [item.probe_id for item in self.probes]
        if len(ids) != len(set(ids)):
            raise ValueError("diagnostic probe IDs must be unique")
        if any(item.episode_id != self.episode_id for item in self.probes):
            raise ValueError("probe catalog contains a different episode ID")
        return self

    def get(self, probe_id: str) -> DiagnosticProbeSpec:
        matches = [item for item in self.probes if item.probe_id == probe_id]
        if len(matches) != 1:
            raise KeyError(f"unknown diagnostic probe_id: {probe_id}")
        return matches[0]


class DiagnosticProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-diagnostic-probe-result.v1"
    observation_id: str
    probe_id: str
    episode_id: str
    kind: DiagnosticProbeKind
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    creates_new_raw_observation: bool = False
    formal_evidence_generated: bool = False
    formal_evidence_eligible: bool = False
    evidence_definition_id: str | None = None
    feature_result_id: str | None = None
    target_fault_ids: tuple[str, ...] = ()
    claim_boundary: str = (
        "Shadow diagnostic observation only. It may guide LLM investigation but cannot alter formal evidence, "
        "diagnosis disposition, release status, or control authorization."
    )


class DiagnosticProbeExecutor:
    """Execute only immutable probe specs created by the trusted host."""

    algorithm_version = "diagnostic-probe-executor.v1"

    @staticmethod
    def _threshold_margin(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        x = float(spec.bound_inputs["observed_value"])
        threshold = float(spec.bound_inputs["threshold_value"])
        comparison = ComparisonOperator(str(spec.bound_inputs["comparison"]))
        if comparison in {ComparisonOperator.GREATER_THAN, ComparisonOperator.GREATER_OR_EQUAL}:
            signed = x - threshold
        elif comparison in {ComparisonOperator.LESS_THAN, ComparisonOperator.LESS_OR_EQUAL}:
            signed = threshold - x
        elif comparison == ComparisonOperator.EQUAL:
            signed = -abs(x - threshold)
        elif comparison == ComparisonOperator.NOT_EQUAL:
            signed = abs(x - threshold)
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported comparison: {comparison}")
        scale = max(abs(threshold), 1e-12)
        return {
            "observed_value": x,
            "threshold_value": threshold,
            "comparison": comparison.value,
            "signed_margin": signed,
            "normalized_signed_margin": signed / scale,
            "unit": spec.unit,
            "interpretation": "SATISFY_SIDE" if signed >= 0.0 else "NON_SATISFY_SIDE",
        }

    @staticmethod
    def _pre_post_change(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        before = [float(v) for v in spec.bound_inputs.get("before", [])]
        after = [float(v) for v in spec.bound_inputs.get("after", [])]
        if not before or not after:
            raise ValueError("PRE_POST_CHANGE requires nonempty host-bound before/after samples")
        b = sum(before) / len(before)
        a = sum(after) / len(after)
        return {"before_mean": b, "after_mean": a, "delta": a - b, "unit": spec.unit}

    @staticmethod
    def _command_response_residual(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        command = [float(v) for v in spec.bound_inputs.get("command", [])]
        response = [float(v) for v in spec.bound_inputs.get("response", [])]
        if len(command) != len(response) or not command:
            raise ValueError("COMMAND_RESPONSE_RESIDUAL requires equal nonempty host-bound arrays")
        residuals = [c - r for c, r in zip(command, response)]
        mae = sum(abs(v) for v in residuals) / len(residuals)
        rms = math.sqrt(sum(v * v for v in residuals) / len(residuals))
        return {"mean_abs_residual": mae, "rms_residual": rms, "sample_count": len(residuals), "unit": spec.unit}

    @staticmethod
    def _lag_correlation(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        x = [float(v) for v in spec.bound_inputs.get("x", [])]
        y = [float(v) for v in spec.bound_inputs.get("y", [])]
        max_lag = int(spec.bound_inputs.get("max_lag_samples", 0))
        if len(x) != len(y) or len(x) < 3 or max_lag < 0:
            raise ValueError("LAG_CORRELATION requires equal host-bound arrays and nonnegative max lag")
        def corr(a: list[float], b: list[float]) -> float:
            ma, mb = sum(a)/len(a), sum(b)/len(b)
            da=[v-ma for v in a]; db=[v-mb for v in b]
            denom=math.sqrt(sum(v*v for v in da)*sum(v*v for v in db))
            return 0.0 if denom <= 1e-15 else sum(u*v for u,v in zip(da,db))/denom
        best=(0, corr(x,y))
        for lag in range(-max_lag, max_lag+1):
            if lag < 0:
                a,b=x[-lag:],y[:len(y)+lag]
            elif lag > 0:
                a,b=x[:len(x)-lag],y[lag:]
            else:
                a,b=x,y
            if len(a) < 3: continue
            c=corr(a,b)
            if abs(c) > abs(best[1]): best=(lag,c)
        return {"best_lag_samples": best[0], "correlation": best[1], "sample_count": len(x)}

    @staticmethod
    def _oscillation_signature(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        values=[float(v) for v in spec.bound_inputs.get("values", [])]
        if len(values) < 4:
            raise ValueError("OSCILLATION_SIGNATURE requires at least four host-bound samples")
        diffs=[values[i]-values[i-1] for i in range(1,len(values))]
        sign_changes=sum(1 for a,b in zip(diffs,diffs[1:]) if a*b < 0)
        energy=sum(d*d for d in diffs)/len(diffs)
        return {"derivative_sign_changes": sign_changes, "difference_energy": energy, "sample_count": len(values), "unit": spec.unit}

    @staticmethod
    def _saturation_limit_check(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        values=[float(v) for v in spec.bound_inputs.get("values", [])]
        lo=float(spec.bound_inputs["lower_limit"]); hi=float(spec.bound_inputs["upper_limit"])
        if not values or not lo < hi:
            raise ValueError("SATURATION_LIMIT_CHECK requires host-bound values and ordered limits")
        eps=max(abs(hi-lo)*float(spec.bound_inputs.get("edge_fraction",0.02)),1e-12)
        saturated=sum(v <= lo+eps or v >= hi-eps for v in values)
        return {"saturation_fraction": saturated/len(values), "sample_count": len(values), "lower_limit": lo, "upper_limit": hi, "unit": spec.unit}

    @staticmethod
    def _minimal_diagnosis_review(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        conflicts = spec.bound_inputs.get("conflicts", [])
        if not isinstance(conflicts, list) or not conflicts:
            raise ValueError("MINIMAL_DIAGNOSIS_REVIEW requires host-bound conflicts")
        normalized: list[tuple[str, ...]] = []
        for conflict in conflicts:
            if not isinstance(conflict, (list, tuple)) or not conflict:
                raise ValueError("MINIMAL_DIAGNOSIS_REVIEW conflict must be a nonempty sequence")
            normalized.append(tuple(sorted({str(value) for value in conflict if str(value)})))
        result = enumerate_minimal_diagnoses(
            normalized,
            max_cardinality=int(spec.bound_inputs.get("max_cardinality", 4)),
            max_candidates=int(spec.bound_inputs.get("max_candidates", 128)),
            max_search_nodes=int(spec.bound_inputs.get("max_search_nodes", 100_000)),
        )
        return {
            "subset_minimal_diagnoses": [item.model_dump(mode="json") for item in result.subset_minimal_diagnoses],
            "minimum_cardinality_diagnoses": [item.model_dump(mode="json") for item in result.minimum_cardinality_diagnoses],
            "search_complete": result.search_complete,
            "search_nodes": result.search_nodes,
            "reason_codes": list(result.reason_codes),
            "diagnosis_result_sha256": result.result_sha256,
            "score_semantics": "SET_MINIMALITY_NOT_PROBABILITY",
        }

    @staticmethod
    def _dxc_expected_response_review(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        payload = spec.bound_inputs.get("expected_response_decision")
        if not isinstance(payload, dict):
            raise ValueError("DXC_EXPECTED_RESPONSE_REVIEW requires one host-bound decision")
        decision = DXCExpectedResponseDecision.model_validate(payload)
        return {
            "selected_fault_ids": list(decision.selected_fault_ids),
            "candidate_count": len(decision.candidates),
            "candidates": [item.model_dump(mode="json") for item in decision.candidates],
            "used_discrete_state_branch": decision.used_discrete_state_branch,
            "reclassified_position_sensor_count": decision.reclassified_position_sensor_count,
            "expected_response_result_sha256": decision.result_sha256,
            "score_semantics": decision.score_semantics,
            "review_semantics": "READ_ONLY_MIRROR_OF_DETERMINISTIC_CORE_DECISION",
        }

    @staticmethod
    def _dxc_forward_simulation_review(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        payload = spec.bound_inputs.get("forward_response_decision")
        if not isinstance(payload, dict):
            raise ValueError("DXC_FORWARD_SIMULATION_REVIEW requires one host-bound decision")
        decision = DXCForwardResponseDecision.model_validate(payload)
        return {
            "selected_fault_ids": list(decision.selected_fault_ids),
            "base_fault_ids": list(decision.base_fault_ids),
            "vetoed_fault_ids": list(decision.vetoed_fault_ids),
            "unresolved_reason_codes": list(decision.unresolved_reason_codes),
            "fits": [item.model_dump(mode="json") for item in decision.fits],
            "forward_response_result_sha256": decision.result_sha256,
            "simulation_semantics": decision.simulation_semantics,
            "score_semantics": decision.score_semantics,
            "review_semantics": "READ_ONLY_MIRROR_OF_DETERMINISTIC_FORWARD_MODEL_DECISION",
        }

    @staticmethod
    def _dxc_reduced_order_simulation_review(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        payload = spec.bound_inputs.get("reduced_order_physics_decision")
        if not isinstance(payload, dict):
            raise ValueError("DXC_REDUCED_ORDER_SIMULATION_REVIEW requires one host-bound decision")
        decision = DXCReducedOrderPhysicsDecision.model_validate(payload)
        return {
            "selected_fault_ids": list(decision.selected_fault_ids),
            "base_fault_ids": list(decision.base_fault_ids),
            "application_status": decision.application_status,
            "reason_codes": list(decision.reason_codes),
            "selected_diagnosis_id": decision.selected_diagnosis_id,
            "candidate_fits": [item.model_dump(mode="json") for item in decision.candidate_fits],
            "reduced_order_result_sha256": decision.result_sha256,
            "simulation_semantics": decision.simulation_semantics,
            "model_class": decision.model_class,
            "review_semantics": "READ_ONLY_MIRROR_OF_DETERMINISTIC_REDUCED_ORDER_PHYSICS_DECISION",
        }

    @staticmethod
    def _dxc_waterpump_reduced_order_simulation_review(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        payload = spec.bound_inputs.get("waterpump_physics_decision")
        if not isinstance(payload, dict):
            raise ValueError("DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW requires one host-bound decision")
        decision = DXCWaterPumpPhysicsDecision.model_validate(payload)
        return {
            "selected_fault_ids": list(decision.selected_fault_ids),
            "base_fault_ids": list(decision.base_fault_ids),
            "physics_preferred_fault_ids": list(decision.physics_preferred_fault_ids),
            "application_status": decision.application_status,
            "reason_codes": list(decision.reason_codes),
            "supports_base_mode": decision.supports_base_mode,
            "candidate_fits": [item.model_dump(mode="json") for item in decision.candidate_fits],
            "waterpump_result_sha256": decision.result_sha256,
            "simulation_semantics": decision.simulation_semantics,
            "model_class": decision.model_class,
            "review_semantics": "READ_ONLY_MIRROR_OF_DETERMINISTIC_WATERPUMP_REDUCED_ORDER_DECISION",
        }

    @staticmethod
    def _simulation_counterfactual_review(spec: DiagnosticProbeSpec) -> dict[str, Any]:
        observed_payload = spec.bound_inputs.get("observed_trace")
        candidates_payload = spec.bound_inputs.get("candidate_traces")
        scales = spec.bound_inputs.get("channel_scales")
        if not isinstance(observed_payload, dict) or not isinstance(candidates_payload, list) or not candidates_payload:
            raise ValueError("SIMULATION_COUNTERFACTUAL_REVIEW requires host-bound observed/candidate traces")
        if not isinstance(scales, dict) or not scales:
            raise ValueError("SIMULATION_COUNTERFACTUAL_REVIEW requires host-bound channel scales")
        observed = SimulationTrace.model_validate(observed_payload)
        candidates: list[MinimalDiagnosisCandidate] = []
        traces: dict[str, SimulationTrace] = {}
        for row in candidates_payload:
            if not isinstance(row, dict):
                raise ValueError("candidate trace row must be an object")
            fault_ids = tuple(sorted({str(value) for value in row.get("fault_ids", [])}))
            diagnosis_id = str(row.get("diagnosis_id") or "")
            if not diagnosis_id or not fault_ids:
                raise ValueError("candidate trace row requires diagnosis_id and fault_ids")
            candidates.append(MinimalDiagnosisCandidate(
                diagnosis_id=diagnosis_id,
                fault_ids=fault_ids,
                cardinality=len(fault_ids),
                hits_all_conflicts=True,
                subset_minimal=True,
                minimum_cardinality=bool(row.get("minimum_cardinality", False)),
            ))
            traces[diagnosis_id] = SimulationTrace.model_validate(row["trace"])
        review = compare_counterfactual_diagnoses(
            observed=observed,
            candidates=candidates,
            provider=PrecomputedTraceSimulationProvider(traces),
            channel_scales={str(key): float(value) for key, value in scales.items()},
            channels=spec.bound_inputs.get("channels"),
        )
        return {
            "best_diagnosis_id": review.best_diagnosis_id,
            "comparisons": [item.model_dump(mode="json") for item in review.comparisons],
            "counterfactual_result_sha256": review.result_sha256,
            "score_semantics": "DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY",
        }

    def execute(self, *, catalog: DiagnosticProbeCatalog, probe_id: str) -> DiagnosticProbeResult:
        spec = catalog.get(probe_id)
        executors = {
            DiagnosticProbeKind.EVIDENCE_THRESHOLD_MARGIN: self._threshold_margin,
            DiagnosticProbeKind.PRE_POST_CHANGE: self._pre_post_change,
            DiagnosticProbeKind.COMMAND_RESPONSE_RESIDUAL: self._command_response_residual,
            DiagnosticProbeKind.LAG_CORRELATION: self._lag_correlation,
            DiagnosticProbeKind.OSCILLATION_SIGNATURE: self._oscillation_signature,
            DiagnosticProbeKind.SATURATION_LIMIT_CHECK: self._saturation_limit_check,
            DiagnosticProbeKind.MINIMAL_DIAGNOSIS_REVIEW: self._minimal_diagnosis_review,
            DiagnosticProbeKind.SIMULATION_COUNTERFACTUAL_REVIEW: self._simulation_counterfactual_review,
            DiagnosticProbeKind.DXC_EXPECTED_RESPONSE_REVIEW: self._dxc_expected_response_review,
            DiagnosticProbeKind.DXC_FORWARD_SIMULATION_REVIEW: self._dxc_forward_simulation_review,
            DiagnosticProbeKind.DXC_REDUCED_ORDER_SIMULATION_REVIEW: self._dxc_reduced_order_simulation_review,
            DiagnosticProbeKind.DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW: self._dxc_waterpump_reduced_order_simulation_review,
        }
        metrics = executors[spec.kind](spec)
        payload = {
            "probe_id": spec.probe_id,
            "episode_id": spec.episode_id,
            "kind": spec.kind.value,
            "metrics": metrics,
            "source_artifact_sha256": spec.source_artifact_sha256,
            "algorithm_version": self.algorithm_version,
        }
        result_hash = _digest(payload)
        return DiagnosticProbeResult(
            observation_id=f"PROBEOBS-{result_hash[:16].upper()}",
            probe_id=spec.probe_id,
            episode_id=spec.episode_id,
            kind=spec.kind,
            status="SUCCEEDED",
            metrics=metrics,
            source_artifact_sha256=spec.source_artifact_sha256,
            result_sha256=result_hash,
            creates_new_raw_observation=False,
            formal_evidence_generated=False,
            formal_evidence_eligible=False,
            evidence_definition_id=spec.evidence_definition_id,
            feature_result_id=spec.feature_result_id,
            target_fault_ids=spec.target_fault_ids,
        )


def build_model_based_probe_catalog(
    *,
    episode_id: str,
    conflicts: list[list[str]] | tuple[tuple[str, ...], ...],
    source_artifact_sha256: str,
    max_cardinality: int = 4,
    max_candidates: int = 128,
    max_search_nodes: int = 100_000,
) -> DiagnosticProbeCatalog:
    """Build one host-bound shadow probe for conflict/minimal-diagnosis review.

    This helper is deliberately separate from ``build_evidence_margin_probe_catalog``:
    model-based review is exposed only when the host has independently admitted
    conflict sets and their source identity.
    """
    normalized = [sorted({str(value) for value in conflict if str(value)}) for conflict in conflicts]
    if not normalized or any(not conflict for conflict in normalized):
        raise ValueError("model-based probe catalog requires nonempty admitted conflicts")
    bound_inputs = {
        "conflicts": normalized,
        "max_cardinality": int(max_cardinality),
        "max_candidates": int(max_candidates),
        "max_search_nodes": int(max_search_nodes),
    }
    source_hash = _digest({
        "upstream_source_artifact_sha256": source_artifact_sha256,
        "bound_inputs": bound_inputs,
    })
    spec = DiagnosticProbeSpec(
        probe_id=f"PROBE-MBD-{source_hash[:12].upper()}",
        episode_id=episode_id,
        kind=DiagnosticProbeKind.MINIMAL_DIAGNOSIS_REVIEW,
        title="Review admitted model conflicts",
        objective="Enumerate bounded subset-minimal and minimum-cardinality diagnoses from frozen conflicts.",
        bound_inputs=bound_inputs,
        source_artifact_sha256=source_hash,
    )
    return DiagnosticProbeCatalog(
        episode_id=episode_id, probes=(spec,), telemetry_admission_status="MODEL_CONFLICTS_HOST_BOUND"
    )


def build_counterfactual_probe_catalog(
    *,
    episode_id: str,
    observed_trace: SimulationTrace,
    candidate_rows: list[dict[str, Any]],
    channel_scales: dict[str, float],
    source_artifact_sha256: str,
    channels: list[str] | tuple[str, ...] | None = None,
) -> DiagnosticProbeCatalog:
    """Build one host-bound shadow probe over admitted observed/simulation traces."""
    if not candidate_rows:
        raise ValueError("counterfactual probe catalog requires candidate traces")
    bound_inputs: dict[str, Any] = {
        "observed_trace": observed_trace.model_dump(mode="json"),
        "candidate_traces": candidate_rows,
        "channel_scales": {str(key): float(value) for key, value in channel_scales.items()},
    }
    if channels is not None:
        bound_inputs["channels"] = list(channels)
    source_hash = _digest({
        "upstream_source_artifact_sha256": source_artifact_sha256,
        "bound_inputs": bound_inputs,
    })
    spec = DiagnosticProbeSpec(
        probe_id=f"PROBE-SIMCF-{source_hash[:12].upper()}",
        episode_id=episode_id,
        kind=DiagnosticProbeKind.SIMULATION_COUNTERFACTUAL_REVIEW,
        title="Review admitted simulation counterfactuals",
        objective="Compare frozen counterfactual traces with the admitted observed trace using fixed channel scales.",
        bound_inputs=bound_inputs,
        source_artifact_sha256=source_hash,
    )
    return DiagnosticProbeCatalog(
        episode_id=episode_id, probes=(spec,), telemetry_admission_status="SIMULATION_TRACES_HOST_BOUND"
    )


def build_dxc_expected_response_probe_catalog(
    *,
    episode_id: str,
    decision: DXCExpectedResponseDecision,
    source_artifact_sha256: str,
) -> DiagnosticProbeCatalog:
    """Expose an already-computed R27 expected-response decision for shadow review.

    The core decision is host-produced before the Agent sees the probe catalog.
    The probe cannot modify telemetry, candidate generation, thresholds, or the
    selected diagnosis; execution only validates and mirrors the content-bound
    deterministic decision for investigation/explanation.
    """
    bound_inputs = {"expected_response_decision": decision.model_dump(mode="json")}
    source_hash = _digest({
        "upstream_source_artifact_sha256": source_artifact_sha256,
        "expected_response_result_sha256": decision.result_sha256,
        "bound_inputs": bound_inputs,
    })
    spec = DiagnosticProbeSpec(
        probe_id=f"PROBE-DXC-EXPRESP-{source_hash[:12].upper()}",
        episode_id=episode_id,
        kind=DiagnosticProbeKind.DXC_EXPECTED_RESPONSE_REVIEW,
        title="Review DXC expected-response ranking",
        objective=(
            "Inspect the frozen deterministic component-mode/expected-response ranking without rerunning or changing the diagnosis core."
        ),
        target_fault_ids=decision.selected_fault_ids,
        bound_inputs=bound_inputs,
        source_artifact_sha256=source_hash,
    )
    return DiagnosticProbeCatalog(
        episode_id=episode_id, probes=(spec,), telemetry_admission_status="DXC_EXPECTED_RESPONSE_DECISION_HOST_BOUND"
    )


def build_dxc_forward_simulation_probe_catalog(
    *,
    episode_id: str,
    decision: DXCForwardResponseDecision,
    source_artifact_sha256: str,
) -> DiagnosticProbeCatalog:
    """Expose an already-computed R28 forward-model decision for shadow review.

    The Agent cannot rerun the model, alter transition semantics, change the
    selected fault set, or convert the result into formal evidence.
    """
    bound_inputs = {"forward_response_decision": decision.model_dump(mode="json")}
    source_hash = _digest({
        "upstream_source_artifact_sha256": source_artifact_sha256,
        "forward_response_result_sha256": decision.result_sha256,
        "bound_inputs": bound_inputs,
    })
    spec = DiagnosticProbeSpec(
        probe_id=f"PROBE-DXC-FWDSIM-{source_hash[:12].upper()}",
        episode_id=episode_id,
        kind=DiagnosticProbeKind.DXC_FORWARD_SIMULATION_REVIEW,
        title="Review DXC forward-response simulation consistency",
        objective=(
            "Inspect the frozen qualitative component-mode/state-transition forward-model result without rerunning or changing the diagnosis core."
        ),
        target_fault_ids=decision.selected_fault_ids,
        bound_inputs=bound_inputs,
        source_artifact_sha256=source_hash,
    )
    return DiagnosticProbeCatalog(
        episode_id=episode_id,
        probes=(spec,),
        telemetry_admission_status="DXC_FORWARD_RESPONSE_DECISION_HOST_BOUND",
    )


def build_dxc_reduced_order_simulation_probe_catalog(
    *,
    episode_id: str,
    decision: DXCReducedOrderPhysicsDecision,
    source_artifact_sha256: str,
) -> DiagnosticProbeCatalog:
    """Expose one already-computed R29 reduced-order decision for shadow review.

    The Agent receives a content-bound mirror only; it cannot change candidate
    generation, branch boundary conditions, component ratings, thresholds, or
    the selected fault set.
    """
    bound_inputs = {"reduced_order_physics_decision": decision.model_dump(mode="json")}
    source_hash = _digest({
        "upstream_source_artifact_sha256": source_artifact_sha256,
        "reduced_order_result_sha256": decision.result_sha256,
        "bound_inputs": bound_inputs,
    })
    spec = DiagnosticProbeSpec(
        probe_id=f"PROBE-DXC-R29SIM-{source_hash[:12].upper()}",
        episode_id=episode_id,
        kind=DiagnosticProbeKind.DXC_REDUCED_ORDER_SIMULATION_REVIEW,
        title="Review DXC reduced-order physical simulation consistency",
        objective=(
            "Inspect the frozen candidate-specific reduced-order electrical-load simulation result without rerunning or changing the diagnosis core."
        ),
        target_fault_ids=decision.selected_fault_ids,
        bound_inputs=bound_inputs,
        source_artifact_sha256=source_hash,
    )
    return DiagnosticProbeCatalog(
        episode_id=episode_id,
        probes=(spec,),
        telemetry_admission_status="DXC_REDUCED_ORDER_PHYSICS_DECISION_HOST_BOUND",
    )


def build_dxc_waterpump_reduced_order_simulation_probe_catalog(
    *,
    episode_id: str,
    decision: DXCWaterPumpPhysicsDecision,
    source_artifact_sha256: str,
) -> DiagnosticProbeCatalog:
    """Expose one already-computed R31 pump-physics decision for shadow review."""
    bound_inputs = {"waterpump_physics_decision": decision.model_dump(mode="json")}
    source_hash = _digest({
        "upstream_source_artifact_sha256": source_artifact_sha256,
        "waterpump_result_sha256": decision.result_sha256,
        "bound_inputs": bound_inputs,
    })
    spec = DiagnosticProbeSpec(
        probe_id=f"PROBE-DXC-R31PUMP-{source_hash[:12].upper()}",
        episode_id=episode_id,
        kind=DiagnosticProbeKind.DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW,
        title="Review DXC WaterPump reduced-order physical consistency",
        objective=(
            "Inspect the frozen current/flow reduced-order WaterPump corroboration without rerunning or changing the diagnosis core."
        ),
        target_fault_ids=decision.selected_fault_ids,
        bound_inputs=bound_inputs,
        source_artifact_sha256=source_hash,
    )
    return DiagnosticProbeCatalog(
        episode_id=episode_id,
        probes=(spec,),
        telemetry_admission_status="DXC_WATERPUMP_REDUCED_ORDER_DECISION_HOST_BOUND",
    )


def build_evidence_margin_probe_catalog(bundle: EpisodeEvidenceBundle) -> DiagnosticProbeCatalog:
    """Expose only low-risk probes derivable from already-admitted formal evidence.

    Time-series probe executors exist, but this catalog intentionally does not expose
    them because ``EpisodeEvidenceBundle`` does not grant raw telemetry access.
    """

    facts = {item.evidence_definition_id: item for item in bundle.evidence_facts}
    features = {item.feature_result_id: item for item in bundle.feature_results}
    probes: list[DiagnosticProbeSpec] = []
    for definition in sorted(bundle.evidence_definitions, key=lambda item: item.evidence_definition_id):
        fact = facts.get(definition.evidence_definition_id)
        if fact is None or fact.status in {
            EvidenceFactStatus.MISSING_DATA,
            EvidenceFactStatus.INVALID_DATA,
            EvidenceFactStatus.NOT_APPLICABLE,
            EvidenceFactStatus.UNKNOWN,
        }:
            continue
        if fact.observed_value is None:
            continue
        feature = features.get(fact.feature_result_id)
        source_payload = {
            "episode_id": bundle.episode.episode_id,
            "evidence_definition_id": definition.evidence_definition_id,
            "evidence_fact_id": fact.evidence_fact_id,
            "feature_result_id": fact.feature_result_id,
            "feature_input_sha256": feature.input_data_sha256 if feature else None,
            "observed_value": fact.observed_value,
            "threshold_value": fact.threshold_value,
            "comparison": fact.comparison.value,
            "unit": fact.unit,
        }
        source_hash = _digest(source_payload)
        probe_id = f"PROBE-MARGIN-{hashlib.sha256(definition.evidence_definition_id.encode()).hexdigest()[:12].upper()}"
        probes.append(DiagnosticProbeSpec(
            probe_id=probe_id,
            episode_id=bundle.episode.episode_id,
            kind=DiagnosticProbeKind.EVIDENCE_THRESHOLD_MARGIN,
            title=f"Recheck threshold margin for {definition.name}",
            objective=(
                f"Quantify how far admitted evidence {definition.evidence_definition_id} lies from its frozen threshold "
                "without changing the underlying EvidenceFact."
            ),
            target_fault_ids=(definition.target_fault_id,),
            evidence_definition_id=definition.evidence_definition_id,
            feature_result_id=fact.feature_result_id,
            unit=fact.unit,
            bound_inputs={
                "observed_value": fact.observed_value,
                "threshold_value": fact.threshold_value,
                "comparison": fact.comparison.value,
                "evidence_fact_status": fact.status.value,
            },
            source_artifact_sha256=source_hash,
        ))
    return DiagnosticProbeCatalog(episode_id=bundle.episode.episode_id, probes=tuple(probes))


__all__ = [
    "DiagnosticProbeCatalog",
    "DiagnosticProbeExecutor",
    "DiagnosticProbeKind",
    "DiagnosticProbeResult",
    "DiagnosticProbeSpec",
    "build_counterfactual_probe_catalog",
    "build_dxc_expected_response_probe_catalog",
    "build_dxc_forward_simulation_probe_catalog",
    "build_dxc_reduced_order_simulation_probe_catalog",
    "build_evidence_margin_probe_catalog",
    "build_model_based_probe_catalog",
]
