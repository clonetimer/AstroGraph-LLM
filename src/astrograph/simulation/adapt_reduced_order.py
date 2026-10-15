"""Reduced-order ADAPT electrical-load forward simulation.

R29 adds the first built-in candidate-specific numerical simulation provider used
inside the DXC deterministic diagnosis core.  The initial scope is deliberately
narrow: resistive ``LightBulb*W:FailedOff`` candidates on an admitted AC branch.
The provider uses the public ADAPT component rating embedded in the component
 type name together with host-admitted branch voltage/current boundary
conditions.  It predicts the post-transition current trace by removing the
candidate load contribution and scaling the remaining branch current with the
observed bus voltage.

This is a reduced-order engineering model, not VirtualADAPT/Simulink and not a
full continuous electrical-network solver.  It nevertheless produces a
candidate-specific numerical trajectory through the same governed
``DiagnosticSimulationProvider`` interface used by counterfactual review.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.simulation.diagnostic_provider import (
    CounterfactualSimulationRequest,
    DiagnosticSimulationProvider,
    SimulationTrace,
    build_simulation_trace,
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("ADAPT_REDUCED_ORDER_NONFINITE_OR_EMPTY_SERIES")
    return result


class ADAPTResistiveLoadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    fault_id: str = Field(pattern=r"^[A-Za-z0-9_]+:FailedOff$")
    component_id: str = Field(min_length=1)
    component_type: str = Field(pattern=r"^LightBulb[0-9]+W$")
    rated_power_w: float = Field(gt=0.0)
    current_sensor_id: str = Field(min_length=1)
    voltage_sensor_id: str = Field(min_length=1)
    local_temperature_sensor_id: str | None = None
    local_position_sensor_id: str | None = None


class ADAPTBranchBoundaryCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    branch_id: str = Field(min_length=1)
    current_sensor_id: str = Field(min_length=1)
    voltage_sensor_id: str = Field(min_length=1)
    time_s: tuple[float, ...] = Field(min_length=2)
    observed_voltage_v: tuple[float, ...] = Field(min_length=2)
    pre_current_a: float = Field(ge=0.0)
    pre_voltage_v: float = Field(gt=0.0)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    boundary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_boundary(self) -> "ADAPTBranchBoundaryCondition":
        if len(self.time_s) != len(self.observed_voltage_v):
            raise ValueError("ADAPT_REDUCED_ORDER_BOUNDARY_LENGTH_MISMATCH")
        if any(b <= a for a, b in zip(self.time_s, self.time_s[1:])):
            raise ValueError("ADAPT_REDUCED_ORDER_BOUNDARY_TIME_NOT_STRICT")
        payload = {
            "branch_id": self.branch_id,
            "current_sensor_id": self.current_sensor_id,
            "voltage_sensor_id": self.voltage_sensor_id,
            "time_s": self.time_s,
            "observed_voltage_v": self.observed_voltage_v,
            "pre_current_a": self.pre_current_a,
            "pre_voltage_v": self.pre_voltage_v,
            "source_artifact_sha256": self.source_artifact_sha256,
        }
        if _digest(payload) != self.boundary_sha256:
            raise ValueError("ADAPT_REDUCED_ORDER_BOUNDARY_SHA256_MISMATCH")
        return self


def build_branch_boundary_condition(
    *,
    branch_id: str,
    current_sensor_id: str,
    voltage_sensor_id: str,
    time_s: Sequence[float],
    observed_voltage_v: Sequence[float],
    pre_current_a: float,
    pre_voltage_v: float,
    source_artifact_sha256: str,
) -> ADAPTBranchBoundaryCondition:
    times = _finite(time_s)
    voltages = _finite(observed_voltage_v)
    payload = {
        "branch_id": str(branch_id),
        "current_sensor_id": str(current_sensor_id),
        "voltage_sensor_id": str(voltage_sensor_id),
        "time_s": times,
        "observed_voltage_v": voltages,
        "pre_current_a": float(pre_current_a),
        "pre_voltage_v": float(pre_voltage_v),
        "source_artifact_sha256": str(source_artifact_sha256),
    }
    return ADAPTBranchBoundaryCondition(**payload, boundary_sha256=_digest(payload))


class ADAPTLightBulbReducedOrderProvider(DiagnosticSimulationProvider):
    """Numerical reduced-order provider for one admitted AC branch.

    The observed post-transition bus voltage is treated as an exogenous boundary
    condition.  For a candidate FailedOff set, the provider removes the rated
    lamp current ``sum(P_i / V_pre)`` from the pre-transition branch current and
    propagates the remaining load current under the observed voltage trajectory.
    This is intentionally conservative and does not claim a full network solve.
    """

    provider_id = "adapt-resistive-lightbulb-reduced-order-v1"

    def __init__(
        self,
        *,
        boundary: ADAPTBranchBoundaryCondition,
        load_specs: Sequence[ADAPTResistiveLoadSpec],
    ) -> None:
        if not load_specs:
            raise ValueError("ADAPT_REDUCED_ORDER_LOAD_SPECS_EMPTY")
        self.boundary = boundary
        self._loads = {item.fault_id: item for item in load_specs}
        if len(self._loads) != len(load_specs):
            raise ValueError("ADAPT_REDUCED_ORDER_DUPLICATE_FAULT_ID")
        for item in load_specs:
            if item.current_sensor_id != boundary.current_sensor_id or item.voltage_sensor_id != boundary.voltage_sensor_id:
                raise ValueError("ADAPT_REDUCED_ORDER_LOAD_BRANCH_MISMATCH")
        self.model_sha256 = _digest({
            "provider_id": self.provider_id,
            "boundary_sha256": boundary.boundary_sha256,
            "loads": [item.model_dump(mode="json") for item in sorted(load_specs, key=lambda row: row.fault_id)],
        })

    def predict(self, request: CounterfactualSimulationRequest) -> SimulationTrace:
        unknown = sorted(set(request.fault_ids) - set(self._loads))
        if unknown:
            raise ValueError(f"ADAPT_REDUCED_ORDER_UNKNOWN_FAULT_IDS:{unknown}")
        if request.required_channels != (self.boundary.current_sensor_id,):
            raise ValueError("ADAPT_REDUCED_ORDER_ONLY_BRANCH_CURRENT_CHANNEL_SUPPORTED")
        expected_time_sha256 = _digest(self.boundary.time_s)
        if request.expected_time_sha256 != expected_time_sha256:
            raise ValueError("ADAPT_REDUCED_ORDER_TIMELINE_MISMATCH")
        total_power_w = sum(self._loads[fault_id].rated_power_w for fault_id in request.fault_ids)
        candidate_nominal_current_a = total_power_w / self.boundary.pre_voltage_v
        other_current_pre_a = max(self.boundary.pre_current_a - candidate_nominal_current_a, 0.0)
        predicted_current = tuple(
            other_current_pre_a * (voltage / self.boundary.pre_voltage_v)
            for voltage in self.boundary.observed_voltage_v
        )
        source_sha = _digest({
            "model_sha256": self.model_sha256,
            "request_sha256": request.request_sha256,
            "boundary_sha256": self.boundary.boundary_sha256,
        })
        return build_simulation_trace(
            trace_id=f"ADAPT-R29-{request.diagnosis_id}",
            time_s=self.boundary.time_s,
            channels={self.boundary.current_sensor_id: predicted_current},
            source_artifact_sha256=source_sha,
            provenance_kind="ASTROGRAPH_ADAPT_RESISTIVE_LOAD_REDUCED_ORDER_V1",
        )


__all__ = [
    "ADAPTBranchBoundaryCondition",
    "ADAPTLightBulbReducedOrderProvider",
    "ADAPTResistiveLoadSpec",
    "build_branch_boundary_condition",
]
