"""Reduced-order ADAPT water-pump forward simulation.

R31 adds the second built-in numerical mechanism family after R29's resistive
lamp model.  The provider models ADAPT ``WaterPump`` faults using two coupled
observables:

* electrical branch current under the admitted bus-voltage boundary condition;
* pump-flow decay from the pre-transition operating point.

``FailedOff`` removes the pump's rated electrical load and drives flow toward
zero. ``FlowBlocked`` keeps the pump electrically energized while flow decays
toward a host-fitted blocked-flow asymptote.  The asymptote and flow time
constant are nuisance parameters fitted from admitted telemetry by a separate
truth-blind deterministic routine; they are not probabilities or fault labels.

This remains a reduced-order component model.  It is not a full hydraulic or
electrical network solver and does not claim VirtualADAPT/Simulink fidelity.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
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
        raise ValueError("ADAPT_PUMP_NONFINITE_OR_EMPTY_SERIES")
    return result




class ADAPTWaterPumpTypeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    component_type: str = "WaterPump"
    rated_power_w: float = Field(gt=0.0)
    nominal_flow_gph: float = Field(gt=0.0)
    source_xml_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_semantics: str = "EXACT_PUBLIC_ADAPT_XML_COMPONENT_TYPE_AND_NOMINAL_MODE_DESCRIPTION"


def extract_water_pump_type_parameters(xml_bytes: bytes) -> ADAPTWaterPumpTypeParameters:
    """Extract the public ADAPT WaterPump rating without name/topology inference."""
    source_sha = hashlib.sha256(xml_bytes).hexdigest()
    root = ET.fromstring(xml_bytes)
    ns = {"d": "urn:org:dx-competition:system"}
    xsi = "{http://www.w3.org/2001/XMLSchema-instance}type"
    rated_power = None
    modes_ref = None
    for node in root.findall("d:componentTypeCatalog/d:componentType", ns):
        name = (node.findtext("d:name", default="", namespaces=ns) or "").strip()
        if name != "WaterPump":
            continue
        raw_power = (node.findtext("d:nominalWattage", default="", namespaces=ns) or "").strip()
        if not raw_power:
            raise ValueError("ADAPT_PUMP_XML_NOMINAL_WATTAGE_MISSING")
        rated_power = float(raw_power)
        modes_ref = (node.findtext("d:modesRef", default="", namespaces=ns) or "").strip()
        break
    if rated_power is None or not modes_ref:
        raise ValueError("ADAPT_PUMP_XML_COMPONENT_TYPE_MISSING")

    nominal_description = None
    for group in root.findall("d:modeCatalog/d:modeGroup", ns):
        group_name = (group.findtext("d:name", default="", namespaces=ns) or "").strip()
        if group_name != modes_ref:
            continue
        for mode in group.findall("d:mode", ns):
            if mode.attrib.get(xsi, "") != "mode":
                continue
            mode_name = (mode.findtext("d:name", default="", namespaces=ns) or "").strip()
            if mode_name != "Nominal":
                continue
            nominal_description = (mode.findtext("d:description", default="", namespaces=ns) or "").strip()
            break
    if not nominal_description:
        raise ValueError("ADAPT_PUMP_XML_NOMINAL_DESCRIPTION_MISSING")
    match = re.search(r"approximately\s+([0-9]+(?:\.[0-9]+)?)\s*GPH", nominal_description, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("ADAPT_PUMP_XML_NOMINAL_FLOW_UNPARSED")
    return ADAPTWaterPumpTypeParameters(
        rated_power_w=rated_power,
        nominal_flow_gph=float(match.group(1)),
        source_xml_sha256=source_sha,
    )


class ADAPTWaterPumpFaultSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    fault_id: str = Field(pattern=r"^[A-Za-z0-9_]+:(FailedOff|FlowBlocked)$")
    component_id: str = Field(min_length=1)
    fault_mode: str = Field(pattern=r"^(FailedOff|FlowBlocked)$")
    rated_power_w: float = Field(gt=0.0)
    nominal_flow_gph: float = Field(gt=0.0)
    current_sensor_id: str = Field(min_length=1)
    voltage_sensor_id: str = Field(min_length=1)
    flow_sensor_id: str = Field(min_length=1)
    flow_asymptote_ratio: float = Field(ge=0.0, le=0.95)
    flow_tau_s: float = Field(gt=0.0)
    parameter_semantics: str = "TRUTH_BLIND_NUISANCE_FIT_NOT_CALIBRATED_PARAMETER"

    @model_validator(mode="after")
    def validate_mode(self) -> "ADAPTWaterPumpFaultSpec":
        expected = f"{self.component_id}:{self.fault_mode}"
        if self.fault_id != expected:
            raise ValueError("ADAPT_PUMP_FAULT_ID_MODE_MISMATCH")
        if self.fault_mode == "FailedOff" and self.flow_asymptote_ratio != 0.0:
            raise ValueError("ADAPT_PUMP_FAILEDOFF_FLOW_ASYMPTOTE_MUST_BE_ZERO")
        return self


class ADAPTWaterPumpBoundaryCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    branch_id: str = Field(min_length=1)
    current_sensor_id: str = Field(min_length=1)
    voltage_sensor_id: str = Field(min_length=1)
    flow_sensor_id: str = Field(min_length=1)
    transition_time_s: float
    time_s: tuple[float, ...] = Field(min_length=2)
    observed_voltage_v: tuple[float, ...] = Field(min_length=2)
    pre_current_a: float = Field(ge=0.0)
    pre_voltage_v: float = Field(gt=0.0)
    pre_flow_gph: float = Field(ge=0.0)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    boundary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_boundary(self) -> "ADAPTWaterPumpBoundaryCondition":
        if len(self.time_s) != len(self.observed_voltage_v):
            raise ValueError("ADAPT_PUMP_BOUNDARY_LENGTH_MISMATCH")
        if any(b <= a for a, b in zip(self.time_s, self.time_s[1:])):
            raise ValueError("ADAPT_PUMP_BOUNDARY_TIME_NOT_STRICT")
        if self.time_s[0] < self.transition_time_s - 1e-9:
            raise ValueError("ADAPT_PUMP_BOUNDARY_STARTS_BEFORE_TRANSITION")
        payload = {
            "branch_id": self.branch_id,
            "current_sensor_id": self.current_sensor_id,
            "voltage_sensor_id": self.voltage_sensor_id,
            "flow_sensor_id": self.flow_sensor_id,
            "transition_time_s": self.transition_time_s,
            "time_s": self.time_s,
            "observed_voltage_v": self.observed_voltage_v,
            "pre_current_a": self.pre_current_a,
            "pre_voltage_v": self.pre_voltage_v,
            "pre_flow_gph": self.pre_flow_gph,
            "source_artifact_sha256": self.source_artifact_sha256,
        }
        if _digest(payload) != self.boundary_sha256:
            raise ValueError("ADAPT_PUMP_BOUNDARY_SHA256_MISMATCH")
        return self


def build_pump_boundary_condition(
    *,
    branch_id: str,
    current_sensor_id: str,
    voltage_sensor_id: str,
    flow_sensor_id: str,
    transition_time_s: float,
    time_s: Sequence[float],
    observed_voltage_v: Sequence[float],
    pre_current_a: float,
    pre_voltage_v: float,
    pre_flow_gph: float,
    source_artifact_sha256: str,
) -> ADAPTWaterPumpBoundaryCondition:
    times = _finite(time_s)
    voltages = _finite(observed_voltage_v)
    payload = {
        "branch_id": str(branch_id),
        "current_sensor_id": str(current_sensor_id),
        "voltage_sensor_id": str(voltage_sensor_id),
        "flow_sensor_id": str(flow_sensor_id),
        "transition_time_s": float(transition_time_s),
        "time_s": times,
        "observed_voltage_v": voltages,
        "pre_current_a": float(pre_current_a),
        "pre_voltage_v": float(pre_voltage_v),
        "pre_flow_gph": float(pre_flow_gph),
        "source_artifact_sha256": str(source_artifact_sha256),
    }
    return ADAPTWaterPumpBoundaryCondition(**payload, boundary_sha256=_digest(payload))


class ADAPTWaterPumpReducedOrderProvider(DiagnosticSimulationProvider):
    """Candidate-specific reduced-order pump simulator.

    The electrical branch is decomposed into a pump rated-power contribution and
    a residual background load.  ``FailedOff`` removes the pump contribution;
    ``FlowBlocked`` preserves it.  Flow follows a first-order transition toward
    the hypothesis-specific asymptote.
    """

    provider_id = "adapt-waterpump-reduced-order-v1"

    def __init__(
        self,
        *,
        boundary: ADAPTWaterPumpBoundaryCondition,
        fault_specs: Sequence[ADAPTWaterPumpFaultSpec],
    ) -> None:
        if not fault_specs:
            raise ValueError("ADAPT_PUMP_FAULT_SPECS_EMPTY")
        self.boundary = boundary
        self._faults = {item.fault_id: item for item in fault_specs}
        if len(self._faults) != len(fault_specs):
            raise ValueError("ADAPT_PUMP_DUPLICATE_FAULT_ID")
        component_ids = {item.component_id for item in fault_specs}
        if len(component_ids) != 1:
            raise ValueError("ADAPT_PUMP_PROVIDER_ONE_COMPONENT_PER_INSTANCE")
        for item in fault_specs:
            if (
                item.current_sensor_id != boundary.current_sensor_id
                or item.voltage_sensor_id != boundary.voltage_sensor_id
                or item.flow_sensor_id != boundary.flow_sensor_id
            ):
                raise ValueError("ADAPT_PUMP_FAULT_BRANCH_MISMATCH")
        self.model_sha256 = _digest({
            "provider_id": self.provider_id,
            "boundary_sha256": boundary.boundary_sha256,
            "fault_specs": [item.model_dump(mode="json") for item in sorted(fault_specs, key=lambda row: row.fault_id)],
        })

    def predict(self, request: CounterfactualSimulationRequest) -> SimulationTrace:
        if len(request.fault_ids) != 1:
            raise ValueError("ADAPT_PUMP_ONLY_SINGLE_COMPONENT_MODE_HYPOTHESIS_SUPPORTED")
        fault_id = request.fault_ids[0]
        try:
            spec = self._faults[fault_id]
        except KeyError as exc:
            raise ValueError(f"ADAPT_PUMP_UNKNOWN_FAULT_ID:{fault_id}") from exc
        required = tuple(sorted((self.boundary.current_sensor_id, self.boundary.flow_sensor_id)))
        if request.required_channels != required:
            raise ValueError("ADAPT_PUMP_REQUIRED_CHANNEL_SET_MISMATCH")
        if request.expected_time_sha256 != _digest(self.boundary.time_s):
            raise ValueError("ADAPT_PUMP_TIMELINE_MISMATCH")

        rated_pre_current = spec.rated_power_w / self.boundary.pre_voltage_v
        background_current_pre = max(self.boundary.pre_current_a - rated_pre_current, 0.0)
        predicted_current: list[float] = []
        predicted_flow: list[float] = []
        for timestamp, voltage in zip(self.boundary.time_s, self.boundary.observed_voltage_v):
            background = background_current_pre * (voltage / self.boundary.pre_voltage_v)
            if spec.fault_mode == "FailedOff":
                pump_current = 0.0
            else:
                pump_current = spec.rated_power_w / max(abs(voltage), 1e-9)
            predicted_current.append(background + pump_current)

            elapsed = max(float(timestamp) - self.boundary.transition_time_s, 0.0)
            relaxation = math.exp(-elapsed / spec.flow_tau_s)
            ratio = spec.flow_asymptote_ratio + (1.0 - spec.flow_asymptote_ratio) * relaxation
            predicted_flow.append(self.boundary.pre_flow_gph * ratio)

        source_sha = _digest({
            "model_sha256": self.model_sha256,
            "request_sha256": request.request_sha256,
            "boundary_sha256": self.boundary.boundary_sha256,
        })
        return build_simulation_trace(
            trace_id=f"ADAPT-R31-{request.diagnosis_id}",
            time_s=self.boundary.time_s,
            channels={
                self.boundary.current_sensor_id: predicted_current,
                self.boundary.flow_sensor_id: predicted_flow,
            },
            source_artifact_sha256=source_sha,
            provenance_kind="ASTROGRAPH_ADAPT_WATERPUMP_REDUCED_ORDER_V1",
        )


__all__ = [
    "ADAPTWaterPumpBoundaryCondition",
    "ADAPTWaterPumpTypeParameters",
    "ADAPTWaterPumpFaultSpec",
    "ADAPTWaterPumpReducedOrderProvider",
    "build_pump_boundary_condition",
    "extract_water_pump_type_parameters",
]
