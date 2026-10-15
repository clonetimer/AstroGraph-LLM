"""R31 blind synthetic campaign for ADAPT WaterPump mechanism classification.

The campaign freezes mode predictions before hidden synthetic mode labels are
joined.  It evaluates only *mode classification given a host-admitted event
transition*, not end-to-end event detection.  The generator deliberately
introduces parameter and dynamics mismatch relative to the R31 diagnosis model.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.dxc_pump_reduced_order import evaluate_dxc_water_pump_physics
from astrograph.simulation.adapt_pump_reduced_order import ADAPTWaterPumpTypeParameters


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_SYNTH_XML_SHA = "3" * 64


def _catalog() -> dict[str, Any]:
    return {
        "xml_sha256": _SYNTH_XML_SHA,
        "components": [
            {"component_id": "PUMP", "component_type": "WaterPump", "kind": "load", "unit": None, "fault_modes": ["FailedOff", "FlowBlocked"]},
            {"component_id": "FT", "component_type": "FlowTransmitter", "kind": "sensor", "unit": "GPH", "fault_modes": ["Offset", "Stuck"]},
            {"component_id": "W1", "component_type": "Wire", "kind": "", "unit": None, "fault_modes": []},
            {"component_id": "R", "component_type": "Relay", "kind": "", "unit": None, "fault_modes": ["StuckOpen", "StuckClosed"]},
            {"component_id": "W2", "component_type": "Wire", "kind": "", "unit": None, "fault_modes": []},
            {"component_id": "I", "component_type": "ACCurrentTransmitter12", "kind": "sensor", "unit": "Amps", "fault_modes": ["Offset", "Stuck"]},
            {"component_id": "V", "component_type": "ACVoltageSensor", "kind": "sensor", "unit": "VAC", "fault_modes": ["Offset", "Stuck"]},
        ],
        "connections": [
            {"c1": "PUMP", "c2": "FT"}, {"c1": "PUMP", "c2": "W1"}, {"c1": "W1", "c2": "R"},
            {"c1": "R", "c2": "W2"}, {"c1": "W2", "c2": "I"}, {"c1": "W2", "c2": "V"},
        ],
        "command_bindings": [{"command_id": "R_CL", "component_id": "R"}],
    }


class R31PumpBlindCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    case_id: str
    transition_time_s: float
    records: tuple[dict[str, Any], ...]
    hidden_fault_id: str = Field(pattern=r"^PUMP:(FailedOff|FlowBlocked)$")
    generator_variant: str
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify(self) -> "R31PumpBlindCase":
        payload = {key: value for key, value in self.model_dump(mode="json").items() if key != "case_sha256"}
        if _digest(payload) != self.case_sha256:
            raise ValueError("R31_PUMP_BLIND_CASE_SHA_MISMATCH")
        return self


class R31PumpBlindPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    case_id: str
    predicted_fault_id: str = Field(pattern=r"^PUMP:(FailedOff|FlowBlocked)$")
    failed_off_rmse: float = Field(ge=0.0)
    flow_blocked_rmse: float = Field(ge=0.0)
    prediction_semantics: str = "TRUTH_FREE_EVENT_CONDITIONAL_PUMP_MODE_CLASSIFICATION"


class R31PumpBlindCampaignReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-r31-pump-blind-simulation-campaign.v1"
    seed: int
    case_count: int = Field(ge=1)
    prediction_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    truth_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    failed_off_recall: float = Field(ge=0.0, le=1.0)
    flow_blocked_recall: float = Field(ge=0.0, le=1.0)
    status: str = "SYNTHETIC_BLIND_EVENT_CONDITIONAL_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION"
    claim_boundary: str = (
        "Predictions are frozen before hidden synthetic pump-mode truth is joined. The transition time is host-admitted, so this evaluates "
        "fault-mode classification conditional on an event, not detection. Generator dynamics intentionally differ from the diagnosis model. "
        "This is not DXC external validation, calibrated probability, HIL evidence, SOTA evidence, or flight qualification."
    )


def _generate_case(*, case_id: str, mode: str, rng: random.Random) -> R31PumpBlindCase:
    dt = 0.5
    transition = 10.0
    times = [round(index * dt, 6) for index in range(41)]
    actual_power = 35.0 * rng.uniform(0.84, 1.16)
    actual_nominal_flow = 250.0 * rng.uniform(0.90, 1.10)
    background_g = rng.uniform(0.012, 0.025)
    background_const = rng.uniform(0.08, 0.22)
    flow_tau = rng.uniform(0.45, 3.5)
    electrical_tau = rng.uniform(0.08, 0.65)
    blocked_ratio = rng.uniform(0.12, 0.78)
    blocked_power_factor = rng.uniform(0.90, 1.10)
    failed_off_leak_ratio = rng.uniform(0.0, 0.025)
    voltage_phase = rng.uniform(-math.pi, math.pi)
    records: list[dict[str, Any]] = []
    for timestamp in times:
        voltage = 120.0 + 2.0 * math.sin(0.23 * timestamp + voltage_phase) + rng.gauss(0.0, 0.22)
        elapsed = max(timestamp - transition, 0.0)
        if timestamp < transition:
            pump_power = actual_power
            flow_ratio = 1.0
        elif mode == "FailedOff":
            electrical_relax = math.exp(-elapsed / electrical_tau)
            pump_power = actual_power * electrical_relax
            flow_ratio = failed_off_leak_ratio + (1.0 - failed_off_leak_ratio) * math.exp(-elapsed / flow_tau)
        else:
            pump_power = actual_power * blocked_power_factor
            flow_ratio = blocked_ratio + (1.0 - blocked_ratio) * math.exp(-elapsed / flow_tau)
        current = background_const + background_g * voltage + pump_power / max(voltage, 80.0) + rng.gauss(0.0, 0.012)
        flow = actual_nominal_flow * flow_ratio + rng.gauss(0.0, 2.5)
        records.append({
            "kind": "sensors",
            "time_tick": int(round(timestamp * 1000)),
            "values": {"I": current, "V": voltage, "FT": max(flow, 0.0)},
        })
    payload = {
        "case_id": case_id,
        "transition_time_s": transition,
        "records": tuple(records),
        "hidden_fault_id": f"PUMP:{mode}",
        "generator_variant": "R31_PERTURBED_PUMP_ELECTRICAL_HYDRAULIC_DYNAMICS_V1",
    }
    return R31PumpBlindCase(**payload, case_sha256=_digest(payload))


def generate_blind_cases(seed: int = 3101, repetitions_per_mode: int = 40) -> tuple[R31PumpBlindCase, ...]:
    rng = random.Random(seed)
    cases: list[R31PumpBlindCase] = []
    modes = ["FailedOff", "FlowBlocked"]
    for rep in range(repetitions_per_mode):
        for mode in modes:
            cases.append(_generate_case(case_id=f"R31-{rep:02d}-{mode}", mode=mode, rng=rng))
    return tuple(cases)


def predict_cases_truth_free(cases: Sequence[R31PumpBlindCase]) -> tuple[R31PumpBlindPrediction, ...]:
    catalog = _catalog()
    type_parameters = ADAPTWaterPumpTypeParameters(
        rated_power_w=35.0,
        nominal_flow_gph=250.0,
        source_xml_sha256=_SYNTH_XML_SHA,
        extraction_semantics="R31_SYNTHETIC_PUBLIC_ADAPT_PARAMETER_IDENTITY",
    )
    predictions: list[R31PumpBlindPrediction] = []
    for case in cases:
        # Hidden mode truth is intentionally never consumed below. A fixed base
        # mode merely activates the non-authoritative physical corroborator; the
        # physics_preferred_fault_ids field is the blind classification output.
        decision = evaluate_dxc_water_pump_physics(
            records=case.records,
            catalog=catalog,
            type_parameters=type_parameters,
            base_fault_ids=("PUMP:FailedOff",),
            transition_time_s=case.transition_time_s,
        )
        if len(decision.physics_preferred_fault_ids) != 1:
            raise ValueError("R31_PUMP_BLIND_PHYSICS_DID_NOT_RETURN_ONE_MODE")
        by_mode = {item.fault_mode: item.aggregate_normalized_rmse for item in decision.candidate_fits}
        predictions.append(R31PumpBlindPrediction(
            case_id=case.case_id,
            predicted_fault_id=decision.physics_preferred_fault_ids[0],
            failed_off_rmse=by_mode["FailedOff"],
            flow_blocked_rmse=by_mode["FlowBlocked"],
        ))
    return tuple(predictions)


def evaluate_frozen_predictions(
    *,
    cases: Sequence[R31PumpBlindCase],
    predictions: Sequence[R31PumpBlindPrediction],
    seed: int,
) -> R31PumpBlindCampaignReport:
    if [case.case_id for case in cases] != [pred.case_id for pred in predictions]:
        raise ValueError("R31_PUMP_CASE_PREDICTION_IDENTITY_MISMATCH")
    pred_payload = [item.model_dump(mode="json") for item in predictions]
    truth_payload = [
        {"case_id": item.case_id, "hidden_fault_id": item.hidden_fault_id, "case_sha256": item.case_sha256}
        for item in cases
    ]
    labels = ("PUMP:FailedOff", "PUMP:FlowBlocked")
    exact = 0
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for case, pred in zip(cases, predictions):
            actual = case.hidden_fault_id == label
            predicted = pred.predicted_fault_id == label
            tp += int(actual and predicted)
            fp += int((not actual) and predicted)
            fn += int(actual and (not predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[label] = recall
        f1s.append(f1)
    exact = sum(case.hidden_fault_id == pred.predicted_fault_id for case, pred in zip(cases, predictions))
    return R31PumpBlindCampaignReport(
        seed=seed,
        case_count=len(cases),
        prediction_set_sha256=_digest(pred_payload),
        truth_set_sha256=_digest(truth_payload),
        accuracy=exact / len(cases),
        macro_f1=sum(f1s) / len(f1s),
        failed_off_recall=recalls["PUMP:FailedOff"],
        flow_blocked_recall=recalls["PUMP:FlowBlocked"],
    )


def run_blind_campaign(seed: int = 3101, repetitions_per_mode: int = 40):
    cases = generate_blind_cases(seed, repetitions_per_mode)
    predictions = predict_cases_truth_free(cases)
    report = evaluate_frozen_predictions(cases=cases, predictions=predictions, seed=seed)
    return report, predictions, cases


__all__ = [
    "R31PumpBlindCampaignReport",
    "R31PumpBlindCase",
    "R31PumpBlindPrediction",
    "evaluate_frozen_predictions",
    "generate_blind_cases",
    "predict_cases_truth_free",
    "run_blind_campaign",
]
