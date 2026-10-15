from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from astrograph.semantics import is_prohibited_field


class PairedDiagnosisStatus(str, Enum):
    DIAGNOSED = "DIAGNOSED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    DATA_ERROR = "DATA_ERROR"


class PairedRuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    field: str | None = None
    aliases: list[str]
    statistic: str
    nominal_value: float | None = None
    effect_value: float | None = None
    delta: float | None = None
    passed: bool
    required: bool = True
    reason_code: str
    claim_boundary: str = "Paired laboratory response trend; not calibrated probability."


class PairedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effect_id: str
    display_name: str
    subsystem: str
    backend: str
    score: float = Field(ge=0.0, le=1.0)
    observable_rule_count: int = Field(ge=0)
    required_rule_count: int = Field(ge=0)
    required_pass_count: int = Field(ge=0)
    eligible: bool
    evidence: list[PairedRuleResult]
    component_fault_ids: list[str] = Field(default_factory=list)


class PairedDiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-sat-sim-paired-diagnosis.v1"
    case_id: str
    status: PairedDiagnosisStatus
    backend: str
    conclusion_effect_id: str | None = None
    conclusion_fault_ids: list[str] = Field(default_factory=list)
    top_score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_margin: float = 0.0
    resolution_basis: str = "NO_RESOLUTION"
    competing_effect_ids: list[str] = Field(default_factory=list)
    candidates: list[PairedCandidate] = Field(default_factory=list)
    input_columns: list[str] = Field(default_factory=list)
    prohibited_columns_used: list[str] = Field(default_factory=list)
    labels_used: bool = False
    truth_used: bool = False
    claim_boundary: str = (
        "Basilisk laboratory paired-run deterministic evidence only; scores are rule coverage, "
        "not calibrated diagnostic probabilities or flight qualification."
    )


@dataclass(frozen=True)
class Rule:
    rule_id: str
    aliases: tuple[str, ...]
    statistic: str
    direction: str
    threshold: float
    required: bool = True
    upper: float | None = None
    transform: str = "identity"


@dataclass(frozen=True)
class Signature:
    effect_id: str
    display_name: str
    subsystem: str
    backend: str
    rules: tuple[Rule, ...]
    component_fault_ids: tuple[str, ...] = ()


def _r(rule_id: str, aliases: Sequence[str], statistic: str, direction: str, threshold: float,
       *, required: bool = True, upper: float | None = None, transform: str = "identity") -> Rule:
    return Rule(rule_id, tuple(aliases), statistic, direction, threshold, required, upper, transform)


# These signatures intentionally exclude labels, truth, effect.active_* and object effect-evidence
# columns. Direct mechanism-state telemetry (for example effective RW torque limit) is allowed and
# is identified as laboratory telemetry, not ground truth.
SIGNATURES: tuple[Signature, ...] = (
    Signature("adcs_rw_jamming", "Reaction-wheel jamming", "ADCS", "unified", (
        _r("RW_JAM_TORQUE", ["adcs.control.applied_torque_nm_0"], "mean_abs", "increase", 0.02),
        _r("RW_JAM_SPEED", ["adcs.rw.speed_rad_s_0"], "last", "change", 10.0),
        _r("RW_JAM_POINTING", ["adcs.pointing_error_deg"], "max", "increase", 0.5, required=False),
    ), ("ADCS_RW_JAM",)),
    Signature("adcs_rw_motor_failure", "Reaction-wheel motor failure", "ADCS", "unified", (
        _r("RW_MOTOR_LIMIT", ["adcs.rw.effective_max_torque_nm_1"], "mean", "decrease", 0.04),
        _r("RW_MOTOR_TORQUE", ["adcs.control.applied_torque_nm_1"], "mean_abs", "decrease", 0.0001, required=False),
        _r("RW_MOTOR_SPEED", ["adcs.rw.speed_rad_s_1"], "last", "change", 0.01),
    ), ("ADCS_RW_MOTOR_FAILURE",)),
    Signature("gyro_bias_step", "Gyroscope bias step", "ADCS", "unified", (
        _r("GYRO_BIAS_STATE", ["adcs.sensor.gyro_bias_norm_rad_s"], "mean", "increase", 0.001),
        _r("GYRO_BIAS_MEASURED", ["adcs.sensor.gyro_measured_rad_s_y"], "mean", "change", 0.0003),
    ), ("ADCS_GYRO_BIAS_STEP",)),
    Signature("payload_instrument_off", "Payload instrument off", "PAYLOAD", "unified", (
        _r("PAYLOAD_OFF_RATE", ["payload.generated_bps"], "mean", "decrease", 1_000_000.0),
        _r("PAYLOAD_OFF_ACTIVE", ["payload.active"], "mean", "decrease", 0.3),
        _r("PAYLOAD_COMM_STAYS", ["comm.active"], "mean", "abs_max", 0.1, required=False, upper=0.1),
    ), ("PAYLOAD_INSTRUMENT_OFF",)),
    Signature("comm_data_downlink_link_loss", "Downlink link loss", "COMM", "unified", (
        _r("DOWNLINK_RATE", ["comm.downlink_bps"], "mean", "decrease", 500_000.0),
        _r("DOWNLINK_COMM_ACTIVE", ["comm.active"], "mean", "decrease", 0.3),
        _r("DOWNLINK_PAYLOAD_STAYS", ["payload.active"], "mean", "abs_max", 0.1, required=False, upper=0.1),
    ), ("COMM_DOWNLINK_SERVICE_LOSS",)),
    Signature("eps_battery_capacity_loss", "Battery capacity loss", "EPS", "unified", (
        _r("BATTERY_CAPACITY", ["eps.effective_battery_capacity_wh"], "mean", "decrease", 20.0),
        _r("BATTERY_ENERGY", ["eps.battery_energy_j"], "last", "decrease", 10_000.0),
    ), ("EPS_BATTERY_CAPACITY_LOSS",)),
    Signature("gyro_noise_increase", "Gyroscope noise increase", "ADCS", "unified", (
        _r("GYRO_NOISE_STATE", ["adcs.sensor.gyro_noise_norm_rad_s"], "rms", "increase", 0.0002),
        _r("GYRO_NOISE_X", ["adcs.sensor.gyro_measured_rad_s_x"], "std", "increase", 0.000005),
    ), ("ADCS_GYRO_NOISE_INCREASE",)),
    Signature("rw_friction_degradation", "Reaction-wheel friction degradation", "ADCS", "unified", (
        _r("RW_FRICTION_DRAG", ["adcs.rw.effective_drag_nms_2"], "mean", "increase", 0.002),
        _r("RW_FRICTION_TORQUE", ["adcs.rw.effective_friction_torque_nm_2"], "rms", "increase", 0.01, transform="abs"),
        _r("RW_FRICTION_SPEED", ["adcs.rw.speed_rad_s_2"], "last", "change", 0.5, required=False),
    ), ("ADCS_RW_FRICTION_INCREASE",)),
    Signature("solar_panel_efficiency_loss", "Solar-panel efficiency loss", "EPS", "unified", (
        _r("SOLAR_POWER", ["eps.solar_array_power_w"], "mean", "decrease", 100.0),
        _r("SOLAR_BATTERY", ["eps.battery_energy_j"], "last", "decrease", 1_000.0),
    ), ("EPS_SOLAR_ARRAY_EFFICIENCY_LOSS",)),
    Signature("thermal_radiator_rejection_loss", "Radiator rejection loss", "THERMAL", "unified", (
        _r("RADIATOR_RATIO", ["thermal.radiator_rejection_ratio"], "mean", "decrease", 0.2),
        _r("RADIATOR_TEMP", ["thermal.payload_temp_k", "thermal.payload_temp_c"], "last", "increase", 0.5),
    ), ("THERMAL_RADIATOR_REJECTION_LOSS",)),
    Signature("adcs_reaction_wheel_speed_limit", "Reaction-wheel speed constraint", "ADCS", "unified", (
        _r("RW_SPEED_LIMIT", ["adcs.rw.effective_max_speed_rad_s_1"], "mean", "increase", 0.3),
        _r("RW_SPEED_RESPONSE", ["adcs.rw.speed_rad_s_1"], "last", "change", 0.025),
    ), ("ADCS_RW_SPEED_LIMIT",)),
    Signature("power_safe_mode_threshold", "Power safe-mode threshold", "EPS", "unified", (
        _r("SAFE_PAYLOAD", ["payload.active"], "mean", "decrease", 0.3),
        _r("SAFE_COMM", ["comm.active"], "mean", "decrease", 0.3),
        _r("SAFE_DOWNLINK", ["comm.downlink_bps"], "mean", "decrease", 700_000.0, required=False),
    ), ("EPS_POWER_SAFE_MODE",)),

    Signature("eps_battery_capacity_loss", "Battery capacity loss", "EPS", "composite", (
        _r("BATTERY_SOC_RESPONSE", ["battery_soc"], "last", "increase", 0.02),
        _r("BATTERY_BUS_STAYS", ["power_bus_available"], "mean", "abs_max", 0.1, required=True, upper=0.1),
    ), ("EPS_BATTERY_CAPACITY_LOSS",)),
    Signature("eps_battery_open_circuit", "Battery open circuit", "EPS", "composite", (
        _r("BATTERY_OPEN_BUS", ["power_bus_available"], "mean", "decrease", 0.5),
        _r("BATTERY_OPEN_ENERGY", ["battery_storage_j"], "last", "decrease", 200_000.0),
    ), ("EPS_BATTERY_OPEN_CIRCUIT",)),
    Signature("adcs_rw_jamming", "Reaction-wheel jamming", "ADCS", "composite", (
        _r("RW_JAM_RATE", ["rate_error_norm_rad_s"], "max", "increase", 0.002),
        _r("RW_JAM_POINTING_DIRECTION", ["attitude_error_deg"], "max", "decrease", 10.0),
    ), ("ADCS_RW_JAM",)),
    Signature("adcs_rw_bearing_seizure", "Reaction-wheel bearing seizure", "ADCS", "composite", (
        _r("RW_SEIZURE_SPEED", ["rw_speed_rad_s_0"], "mean", "decrease", 0.5),
        _r("RW_SEIZURE_COMMAND", ["rw_motor_torque_cmd_nm_0"], "mean_abs", "increase", 0.002),
        _r("RW_SEIZURE_POINTING", ["attitude_error_deg"], "mean", "increase", 0.05),
        _r("RW_SEIZURE_RATE", ["rate_error_norm_rad_s"], "max", "increase", 0.0001, required=False),
    ), ("ADCS_RW_BEARING_SEIZURE",)),
    Signature("propulsion_thruster_ignition_failure", "Thruster ignition failure", "PROPULSION", "composite", (
        _r("IGNITION_FLOW", ["fuel_mass_flow_kg_s"], "mean_abs", "decrease", 0.0002),
        _r("IGNITION_PROPELLANT", ["propellant_remaining_kg"], "last", "increase", 0.01),
    ), ("PROPULSION_THRUSTER_IGNITION_FAILURE",)),
    Signature("payload_instrument_off", "Payload instrument off", "PAYLOAD", "composite", (
        _r("PAYLOAD_OFF_DROP", ["payload_generated_bps"], "min", "decrease", 500_000.0),
        _r("PAYLOAD_OFF_ZERO_WINDOW", ["payload_generated_bps"], "min", "effect_max", 50_000.0),
        _r("PAYLOAD_DOWNLINK_INVARIANT", ["downlink_requested_rate_bps"], "last", "abs_max", 1.0, required=False, upper=5_000.0),
    ), ("PAYLOAD_INSTRUMENT_OFF",)),
    Signature("payload_instrument_degraded_rate", "Payload instrument degraded rate", "PAYLOAD", "composite", (
        _r("PAYLOAD_DEGRADED_DROP", ["payload_generated_bps"], "min", "decrease", 250_000.0),
        _r("PAYLOAD_DEGRADED_REMAINS_ACTIVE", ["payload_generated_bps"], "min", "effect_min", 100_000.0),
        _r("PAYLOAD_DEGRADED_BELOW_NOMINAL", ["payload_generated_bps"], "min", "effect_max", 2_250_000.0),
        _r("PAYLOAD_DEGRADED_DOWNLINK_INVARIANT", ["downlink_requested_rate_bps"], "last", "abs_max", 1.0, required=False, upper=5_000.0),
    ), ("PAYLOAD_INSTRUMENT_DEGRADED_RATE",)),
    Signature("comm_data_downlink_link_loss", "Downlink link loss", "COMM", "composite", (
        _r("DOWNLINK_SERVICE", ["downlink_requested_rate_bps"], "last", "decrease", 500_000.0),
        _r("DOWNLINK_EFFECT_FINAL", ["downlink_requested_rate_bps"], "last", "effect_max", 50_000.0),
        _r("DOWNLINK_PAYLOAD_BOUND", ["payload_generated_bps"], "last", "abs_max", 1.0, required=True, upper=5_000.0),
    ), ("COMM_DOWNLINK_SERVICE_LOSS",)),
    Signature("comm_data_storage_capacity_loss", "Data-storage capacity loss", "DATA", "composite", (
        _r("STORAGE_CAPACITY", ["data_storage_bits"], "max", "decrease", 10_000_000.0),
        _r("STORAGE_PAYLOAD_BOUND", ["payload_generated_bps"], "mean", "abs_max", 1.0, required=True, upper=5_000.0),
    ), ("DATA_STORAGE_CAPACITY_LOSS",)),
    Signature("thermal_heater_stuck_off", "Heater stuck off", "THERMAL", "composite", (
        _r("HEATER_OFF_POWER", ["heater_eps_load_w"], "mean", "decrease", 5.0),
        _r("HEATER_OFF_COUNT", ["heater_active_count"], "mean", "decrease", 0.4),
    ), ("THERMAL_HEATER_STUCK_OFF",)),
    Signature("thermal_heater_stuck_on", "Heater stuck on", "THERMAL", "composite", (
        _r("HEATER_ON_POWER", ["heater_eps_load_w"], "last", "increase", 5.0),
        _r("HEATER_ON_COUNT", ["heater_active_count"], "last", "increase", 0.4),
    ), ("THERMAL_HEATER_STUCK_ON",)),
    Signature("thermal_radiator_rejection_loss", "Radiator rejection loss", "THERMAL", "composite", (
        _r("RADIATOR_TEMP", ["thermal_temp_c"], "last", "increase", 3.0),
        _r("RADIATOR_MARGIN", ["thermal_margin_c"], "last", "decrease", 3.0),
    ), ("THERMAL_RADIATOR_REJECTION_LOSS",)),
    Signature("battery_capacity_loss_30pct", "Battery capacity degradation", "EPS", "composite", (
        _r("BATTERY_MODERATE_DROP", ["battery_storage_j"], "last", "decrease", 50_000.0, upper=200_000.0),
    ), ("EPS_BATTERY_CAPACITY_DEGRADATION",)),
    Signature("solar_panel_efficiency_loss_20pct", "Solar-panel efficiency degradation", "EPS", "composite", (
        _r("SOLAR_NET_POWER", ["net_power_w"], "mean", "decrease", 10.0),
        _r("SOLAR_ENERGY_SMALL", ["battery_storage_j"], "last", "decrease", 500.0, upper=20_000.0),
    ), ("EPS_SOLAR_ARRAY_EFFICIENCY_LOSS",)),
    Signature("thruster_thrust_loss_15pct", "Thruster thrust degradation", "PROPULSION", "composite", (
        _r("THRUST_FLOW", ["fuel_mass_flow_kg_s"], "mean_abs", "decrease", 0.00002, upper=0.00015),
        _r("THRUST_PROPELLANT", ["propellant_remaining_kg"], "last", "increase", 0.001, upper=0.02),
    ), ("PROPULSION_THRUST_LOSS",)),
    Signature("combined_degradation", "Combined EPS and propulsion degradation", "MULTI", "composite", (
        _r("COMBINED_BATTERY", ["battery_storage_j"], "last", "decrease", 50_000.0),
        _r("COMBINED_POWER", ["net_power_w"], "mean", "decrease", 5.0),
        _r("COMBINED_FLOW", ["fuel_mass_flow_kg_s"], "mean_abs", "decrease", 0.00001),
    ), ("EPS_BATTERY_CAPACITY_DEGRADATION", "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS", "PROPULSION_THRUST_LOSS")),
    Signature("adcs_rw_friction_and_sensor_noise", "RW friction plus sensor noise", "ADCS", "composite", (
        _r("ADCS_COMBINED_POINTING", ["attitude_error_deg"], "max", "increase", 5.0),
        _r("ADCS_COMBINED_RATE", ["rate_error_norm_rad_s"], "max", "increase", 0.002),
    ), ("ADCS_RW_FRICTION_INCREASE", "ADCS_GYRO_NOISE_INCREASE")),
    Signature("thermal_heater_and_radiator_degradation", "Heater and radiator degradation", "THERMAL", "composite", (
        _r("THERMAL_COMBINED_TEMP", ["thermal_temp_c"], "last", "increase", 0.5, upper=3.0),
        _r("THERMAL_COMBINED_MARGIN", ["thermal_margin_c"], "last", "decrease", 0.5, upper=3.0),
    ), ("THERMAL_HEATER_DEGRADATION", "THERMAL_RADIATOR_REJECTION_LOSS")),
    Signature("fuel_leak_and_pressure_loss", "Fuel leak and pressure loss", "PROPULSION", "composite", (
        _r("FUEL_LEAK_MASS", ["propellant_remaining_kg"], "last", "decrease", 0.05),
        _r("FUEL_PRESSURE_FLOW", ["fuel_mass_flow_kg_s"], "mean_abs", "decrease", 0.000005),
    ), ("PROPULSION_FUEL_LEAK", "PROPULSION_PRESSURE_LOSS")),
    Signature("multi_subsystem_end_of_life", "Multi-subsystem end of life", "MULTI", "composite", (
        _r("EOL_BATTERY", ["battery_storage_j"], "last", "decrease", 50_000.0),
        _r("EOL_PAYLOAD", ["payload_generated_bps"], "mean", "decrease", 500_000.0),
        _r("EOL_THERMAL", ["thermal_temp_c"], "last", "increase", 0.5),
        _r("EOL_PROPELLANT", ["propellant_remaining_kg"], "last", "decrease", 0.02),
    ), ("EPS_END_OF_LIFE", "PAYLOAD_END_OF_LIFE", "THERMAL_END_OF_LIFE", "PROPULSION_END_OF_LIFE")),
)

PROHIBITED_PREFIXES = ("label.", "truth.")
PROHIBITED_CONTAINS = ("effect.active", "effect_evidence")


def _series(frame: pd.DataFrame, aliases: Iterable[str]) -> tuple[str | None, pd.Series | None]:
    for field in aliases:
        if field in frame.columns:
            return field, pd.to_numeric(frame[field], errors="coerce")
    return None, None


def _stat(series: pd.Series, statistic: str, transform: str) -> float | None:
    data = series.dropna()
    if transform == "abs":
        data = data.abs()
    if data.empty:
        return None
    if statistic == "mean":
        return float(data.mean())
    if statistic == "mean_abs":
        return float(data.abs().mean())
    if statistic == "last":
        return float(data.iloc[-1])
    if statistic == "max":
        return float(data.max())
    if statistic == "min":
        return float(data.min())
    if statistic == "rms":
        return float(math.sqrt(float((data * data).mean())))
    if statistic == "std":
        return float(data.std(ddof=0))
    raise ValueError(f"unsupported statistic: {statistic}")


def _evaluate(rule: Rule, nominal: pd.DataFrame, effect: pd.DataFrame) -> PairedRuleResult:
    field, nominal_series = _series(nominal, rule.aliases)
    effect_field, effect_series = _series(effect, rule.aliases)
    if field is None or effect_field is None or nominal_series is None or effect_series is None:
        return PairedRuleResult(
            rule_id=rule.rule_id, aliases=list(rule.aliases), statistic=rule.statistic,
            passed=False, required=rule.required, reason_code="FIELD_NOT_OBSERVED",
        )
    nominal_value = _stat(nominal_series, rule.statistic, rule.transform)
    effect_value = _stat(effect_series, rule.statistic, rule.transform)
    if nominal_value is None or effect_value is None:
        return PairedRuleResult(
            rule_id=rule.rule_id, field=field, aliases=list(rule.aliases), statistic=rule.statistic,
            nominal_value=nominal_value, effect_value=effect_value, passed=False,
            required=rule.required, reason_code="NO_FINITE_VALUES",
        )
    delta = effect_value - nominal_value
    value = abs(delta) if rule.direction in {"change", "abs_max"} else delta
    if rule.direction == "increase":
        passed = delta >= rule.threshold
    elif rule.direction == "decrease":
        passed = delta <= -rule.threshold
    elif rule.direction == "change":
        passed = abs(delta) >= rule.threshold
    elif rule.direction == "abs_max":
        passed = abs(delta) <= (rule.upper if rule.upper is not None else rule.threshold)
    elif rule.direction == "effect_min":
        passed = effect_value >= rule.threshold
    elif rule.direction == "effect_max":
        passed = effect_value <= rule.threshold
    else:
        raise ValueError(f"unsupported direction: {rule.direction}")
    if passed and rule.upper is not None and rule.direction not in {"abs_max", "effect_min", "effect_max"}:
        passed = abs(value) <= rule.upper
    return PairedRuleResult(
        rule_id=rule.rule_id, field=field, aliases=list(rule.aliases), statistic=rule.statistic,
        nominal_value=nominal_value, effect_value=effect_value, delta=delta, passed=passed,
        required=rule.required, reason_code="TREND_MATCH" if passed else "TREND_MISMATCH",
    )


class SatSimPairedDiagnosisEngine:
    """Deterministic paired-run diagnosis for validated Sat-Agent Basilisk outputs.

    The engine never reads labels, truth manifests, effect identifiers, case directory
    names or Sat-Agent result.json files.  It consumes only paired telemetry and backend
    identity.  Scores represent signature evidence coverage.
    """

    engine_id = "astrograph-sat-sim-paired-signature-v1"

    def __init__(self, signatures: Sequence[Signature] = SIGNATURES) -> None:
        self.signatures = tuple(signatures)

    def diagnose_frames(
        self,
        *,
        case_id: str,
        backend: str,
        nominal: pd.DataFrame,
        effect: pd.DataFrame,
    ) -> PairedDiagnosisResult:
        if list(nominal.columns) != list(effect.columns) or len(nominal) != len(effect):
            return PairedDiagnosisResult(case_id=case_id, backend=backend, status=PairedDiagnosisStatus.DATA_ERROR)
        prohibited = [
            c for c in effect.columns
            if is_prohibited_field(c)
        ]
        # The builder should already have removed them; fail closed rather than silently ignore leakage.
        if prohibited:
            return PairedDiagnosisResult(
                case_id=case_id, backend=backend, status=PairedDiagnosisStatus.DATA_ERROR,
                prohibited_columns_used=prohibited, input_columns=list(effect.columns),
            )
        candidates: list[PairedCandidate] = []
        for signature in self.signatures:
            if signature.backend != backend:
                continue
            evidence = [_evaluate(rule, nominal, effect) for rule in signature.rules]
            observable = [item for item in evidence if item.reason_code != "FIELD_NOT_OBSERVED"]
            required = [item for item in evidence if item.required]
            required_pass = sum(item.passed for item in required)
            optional = [item for item in evidence if not item.required]
            optional_pass = sum(item.passed for item in optional)
            required_ratio = required_pass / len(required) if required else 0.0
            optional_ratio = optional_pass / len(optional) if optional else 1.0
            score = 0.85 * required_ratio + 0.15 * optional_ratio
            eligible = bool(required) and required_pass == len(required)
            candidates.append(PairedCandidate(
                effect_id=signature.effect_id,
                display_name=signature.display_name,
                subsystem=signature.subsystem,
                backend=signature.backend,
                score=max(0.0, min(1.0, score)),
                observable_rule_count=len(observable),
                required_rule_count=len(required),
                required_pass_count=required_pass,
                eligible=eligible,
                evidence=evidence,
                component_fault_ids=list(signature.component_fault_ids),
            ))
        candidates.sort(
            key=lambda c: (c.eligible, c.score, c.required_pass_count, c.required_rule_count, c.observable_rule_count),
            reverse=True,
        )
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        coverage_margin = 0.0 if top is None else top.score - (second.score if second else 0.0)
        competing = [
            item for item in candidates
            if top is not None and item.eligible and item.score >= max(0.85, top.score - 0.02)
        ]
        resolution_basis = "NO_ELIGIBLE_SIGNATURE"
        if top is None or top.score < 0.55:
            status = PairedDiagnosisStatus.UNKNOWN
            conclusion = None
            fault_ids: list[str] = []
        elif top.eligible and top.score >= 0.85 and coverage_margin >= 0.08:
            status = PairedDiagnosisStatus.DIAGNOSED
            conclusion = top.effect_id
            fault_ids = top.component_fault_ids
            resolution_basis = "EVIDENCE_COVERAGE_MARGIN"
        elif top.eligible and top.score >= 0.85 and len(competing) > 1:
            # A compound signature is allowed to dominate nested component signatures only
            # when it is the unique eligible candidate with the largest required evidence set.
            max_required = max(item.required_rule_count for item in competing)
            most_specific = [item for item in competing if item.required_rule_count == max_required]
            if len(most_specific) == 1 and most_specific[0].effect_id == top.effect_id:
                status = PairedDiagnosisStatus.DIAGNOSED
                conclusion = top.effect_id
                fault_ids = top.component_fault_ids
                resolution_basis = "UNIQUE_COMPOUND_SIGNATURE"
            else:
                status = PairedDiagnosisStatus.AMBIGUOUS
                conclusion = None
                fault_ids = []
                resolution_basis = "COMPETING_SIGNATURES"
        elif top.eligible and top.score >= 0.85 and len(competing) == 1:
            status = PairedDiagnosisStatus.DIAGNOSED
            conclusion = top.effect_id
            fault_ids = top.component_fault_ids
            resolution_basis = "SOLE_ELIGIBLE_SIGNATURE"
        else:
            status = PairedDiagnosisStatus.AMBIGUOUS
            conclusion = None
            fault_ids = []
            resolution_basis = "PARTIAL_SIGNATURE"
        # Margin is evidence coverage only.  Specificity resolution is reported separately.
        return PairedDiagnosisResult(
            case_id=case_id,
            backend=backend,
            status=status,
            conclusion_effect_id=conclusion,
            conclusion_fault_ids=fault_ids,
            top_score=0.0 if top is None else top.score,
            score_margin=coverage_margin,
            resolution_basis=resolution_basis,
            competing_effect_ids=[item.effect_id for item in competing],
            candidates=candidates[:5],
            input_columns=list(effect.columns),
            labels_used=False,
            truth_used=False,
        )

    def diagnose_paths(self, *, case_id: str, backend: str, nominal_path: str | Path, effect_path: str | Path) -> PairedDiagnosisResult:
        nominal = pd.read_csv(nominal_path, low_memory=False)
        effect = pd.read_csv(effect_path, low_memory=False)
        return self.diagnose_frames(case_id=case_id, backend=backend, nominal=nominal, effect=effect)


__all__ = [
    "PairedDiagnosisStatus", "PairedRuleResult", "PairedCandidate", "PairedDiagnosisResult",
    "SatSimPairedDiagnosisEngine", "SIGNATURES",
]
