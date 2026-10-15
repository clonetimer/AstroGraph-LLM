from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from astrograph.knowledge.sat_sim_fault_catalog import load_sat_sim_fault_catalog
from astrograph.semantics import is_prohibited_field

PROHIBITED_PREFIXES = ("label.", "truth.", "effect.", "object.")
PROHIBITED_EXACT = {"task_id", "case_id", "mode"}


class SingleTraceMultiFaultStatus(str, Enum):
    NORMAL = "NORMAL"
    KNOWN_FAULT = "KNOWN_FAULT"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DATA_ERROR = "DATA_ERROR"


class SingleTraceMultiFaultEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effect_id: str
    fields: list[str]
    statistic: str
    observed_value: float | None = None
    threshold: str
    passed: bool
    interpretation: str


class SingleTraceObservedImpact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path_id: str
    root_effect_id: str
    source_subsystem: str
    target_subsystem: str
    fields: list[str]
    observed_values: dict[str, float]
    interpretation: str
    claim_semantics: str = "OBSERVED_ASSOCIATIVE_COUPLING_NOT_CAUSAL_EFFECT_ESTIMATE"


class SingleTraceMultiFaultResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-sat-sim-multifault-single.v2"
    case_id: str
    status: SingleTraceMultiFaultStatus
    detected_effect_ids: list[str] = Field(default_factory=list)
    primary_effect_id: str | None = None
    secondary_effect_ids: list[str] = Field(default_factory=list)
    effect_scores: dict[str, float] = Field(default_factory=dict)
    primary_selection_semantics: str = "STRONGEST_NORMALIZED_PHYSICAL_SIGNATURE_NOT_CAUSAL_ROOT"
    affected_subsystems: list[str] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    propagation_paths: list[str] = Field(default_factory=list)
    observed_secondary_impacts: list[SingleTraceObservedImpact] = Field(default_factory=list)
    catalog_evidence_paths: list[str] = Field(default_factory=list)
    supporting_evidence: list[SingleTraceMultiFaultEvidence] = Field(default_factory=list)
    observed_field_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    prohibited_columns_used: list[str] = Field(default_factory=list)
    labels_used: bool = False
    truth_used: bool = False
    claim_boundary: str = (
        "Single-trace laboratory Basilisk multifault physical-signature diagnosis. "
        "Thresholds are calibrated from development and holdout campaigns; outputs are "
        "known-catalog experimental effects, not calibrated probabilities, flight qualification, "
        "or causal effects."
    )


EFFECT_META: dict[str, tuple[str, str]] = {
    "adcs_rw_jamming": ("ADCS", "ReactionWheel"),
    "eps_battery_capacity_loss": ("EPS", "Battery"),
    "eps_battery_open_circuit": ("EPS", "Battery"),
    "solar_panel_efficiency_loss_20pct": ("EPS", "SolarArray"),
    "payload_instrument_off": ("PAYLOAD", "Instrument"),
    "thruster_thrust_loss_15pct": ("PROPULSION", "Thruster"),
    "propulsion_thruster_ignition_failure": ("PROPULSION", "Thruster"),
    "thermal_radiator_rejection_loss": ("THERMAL", "Radiator"),
    "thermal_heater_stuck_off": ("THERMAL", "Heater"),
    "thermal_heater_stuck_on": ("THERMAL", "Heater"),
    "comm_data_downlink_link_loss": ("COMM_DATA", "Transmitter"),
    "comm_data_storage_capacity_loss": ("COMM_DATA", "Storage"),
}

# These thresholds are laboratory experiment boundaries derived only from the
# development and holdout campaigns. They are not hardware limits.
THRESHOLDS: dict[str, float] = {
    "rw_relative_torque_max": 0.76,
    "rw_speed_peak_min_rad_s": 380.0,
    "battery_soc_mean_min": 0.75,
    "power_bus_mean_open_max": 0.60,
    "solar_net_power_mean_max_w": -175.0,
    "solar_thermal_mean_max_c": 25.0,
    "solar_heater_load_mean_max_w": 3.0,
    "payload_rate_mean_max_bps": 2_200_000.0,
    "ignition_flow_mean_abs_max": 1.0e-6,
    "partial_flow_mean_abs_max": 3.7e-4,
    "radiator_temp_mean_min_c": 26.7,
    "radiator_temp_last_min_c": 25.0,
    "heater_off_active_mean_min": 0.05,
    "heater_off_active_mean_max": 0.40,
    "heater_on_active_mean_min": 0.60,
    "heater_on_active_mean_max": 0.95,
    "downlink_requested_mean_max_bps": 300_000.0,
    "storage_bits_max_max": 80_000_000.0,
}

REQUIRED_SIGNAL_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rw_motor_torque_nm_0", "rw_motor_torque_nm_1", "rw_motor_torque_nm_2", "rw_speed_rad_s_0"),
    ("battery_soc",),
    ("power_bus_available",),
    ("net_power_w", "thermal_temp_c", "heater_eps_load_w"),
    ("payload_generated_bps",),
    ("fuel_mass_flow_kg_s",),
    ("thermal_temp_c",),
    ("heater_active_count",),
    ("downlink_requested_rate_bps",),
    ("data_storage_bits",),
)


def _series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").dropna().astype(float)


def _mean(frame: pd.DataFrame, field: str, *, absolute: bool = False) -> float | None:
    values = _series(frame, field)
    if values.empty:
        return None
    if absolute:
        values = values.abs()
    return float(values.mean())


def _max(frame: pd.DataFrame, field: str) -> float | None:
    values = _series(frame, field)
    return None if values.empty else float(values.max())


def _last(frame: pd.DataFrame, field: str) -> float | None:
    values = _series(frame, field)
    return None if values.empty else float(values.iloc[-1])


def _ev(effect_id: str, fields: list[str], statistic: str, observed: float | None,
        threshold: str, passed: bool, interpretation: str) -> SingleTraceMultiFaultEvidence:
    return SingleTraceMultiFaultEvidence(
        effect_id=effect_id,
        fields=fields,
        statistic=statistic,
        observed_value=observed,
        threshold=threshold,
        passed=passed,
        interpretation=interpretation,
    )


def _effect_strengths(frame: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, float] = {}
    torques = [_mean(frame, f"rw_motor_torque_nm_{i}", absolute=True) for i in range(3)]
    speed_peak = _max(frame, "rw_speed_rad_s_0")
    if all(value is not None for value in torques):
        denominator = (float(torques[1]) + float(torques[2])) / 2.0
        ratio = float(torques[0]) / max(denominator, 1.0e-12)
        scores["adcs_rw_jamming"] = max(0.0, (THRESHOLDS["rw_relative_torque_max"] - ratio) / 0.10) + max(0.0, ((speed_peak or 0.0) - THRESHOLDS["rw_speed_peak_min_rad_s"]) / 100.0)
    soc = _mean(frame, "battery_soc")
    scores["eps_battery_capacity_loss"] = max(0.0, ((soc or 0.0) - THRESHOLDS["battery_soc_mean_min"]) / 0.10)
    power_bus = _mean(frame, "power_bus_available")
    scores["eps_battery_open_circuit"] = max(0.0, (THRESHOLDS["power_bus_mean_open_max"] - (power_bus or 1.0)) / 0.20)
    net = _mean(frame, "net_power_w")
    scores["solar_panel_efficiency_loss_20pct"] = max(0.0, (THRESHOLDS["solar_net_power_mean_max_w"] - (net or 0.0)) / 10.0)
    payload = _mean(frame, "payload_generated_bps")
    scores["payload_instrument_off"] = max(0.0, (THRESHOLDS["payload_rate_mean_max_bps"] - (payload or 0.0)) / 500_000.0)
    flow = _mean(frame, "fuel_mass_flow_kg_s", absolute=True)
    if flow is not None:
        scores["propulsion_thruster_ignition_failure"] = max(0.0, (THRESHOLDS["ignition_flow_mean_abs_max"] - flow) / THRESHOLDS["ignition_flow_mean_abs_max"]) * 3.0
        scores["thruster_thrust_loss_15pct"] = max(0.0, (THRESHOLDS["partial_flow_mean_abs_max"] - flow) / 5.0e-5)
    thermal_last = _last(frame, "thermal_temp_c")
    scores["thermal_radiator_rejection_loss"] = max(0.0, ((thermal_last or 0.0) - THRESHOLDS["radiator_temp_last_min_c"]) / 1.0)
    heater = _mean(frame, "heater_active_count")
    if heater is not None:
        scores["thermal_heater_stuck_off"] = max(0.0, (THRESHOLDS["heater_off_active_mean_max"] - heater) / 0.35)
        scores["thermal_heater_stuck_on"] = max(0.0, (heater - THRESHOLDS["heater_on_active_mean_min"]) / 0.35)
    downlink = _mean(frame, "downlink_requested_rate_bps")
    scores["comm_data_downlink_link_loss"] = max(0.0, (THRESHOLDS["downlink_requested_mean_max_bps"] - (downlink or 0.0)) / THRESHOLDS["downlink_requested_mean_max_bps"])
    storage = _max(frame, "data_storage_bits")
    scores["comm_data_storage_capacity_loss"] = max(0.0, (THRESHOLDS["storage_bits_max_max"] - (storage or 0.0)) / 30_000_000.0)
    return scores


def _catalog_paths(effect_ids: list[str], catalog_path: Path | None) -> list[str]:
    if catalog_path is None:
        catalog_path = Path(__file__).resolve().parents[3] / "reference" / "knowledge" / "sat_sim_fault_catalog.json"
    if not catalog_path.exists():
        return []
    catalog = load_sat_sim_fault_catalog(catalog_path).by_effect_id()
    paths: list[str] = []
    for effect_id in effect_ids:
        entry = catalog.get(effect_id)
        if entry is None:
            continue
        for target in entry.propagation_targets:
            paths.append(f"{effect_id}->{entry.subsystem}->{target}")
    return sorted(set(paths))


def _detect(frame: pd.DataFrame) -> list[SingleTraceMultiFaultEvidence]:
    evidence: list[SingleTraceMultiFaultEvidence] = []

    torques = [_mean(frame, f"rw_motor_torque_nm_{i}", absolute=True) for i in range(3)]
    rw_ratio: float | None = None
    if all(value is not None for value in torques):
        denominator = (float(torques[1]) + float(torques[2])) / 2.0
        if denominator > 1.0e-12:
            rw_ratio = float(torques[0]) / denominator
    speed_peak = _max(frame, "rw_speed_rad_s_0")
    rw_pass = (
        rw_ratio is not None
        and speed_peak is not None
        and rw_ratio < THRESHOLDS["rw_relative_torque_max"]
        and speed_peak > THRESHOLDS["rw_speed_peak_min_rad_s"]
    )
    evidence.append(_ev(
        "adcs_rw_jamming",
        ["rw_motor_torque_nm_0", "rw_motor_torque_nm_1", "rw_motor_torque_nm_2", "rw_speed_rad_s_0"],
        "rw0_mean_abs_torque_ratio_with_speed_peak",
        rw_ratio,
        f"ratio<{THRESHOLDS['rw_relative_torque_max']} and speed_peak>{THRESHOLDS['rw_speed_peak_min_rad_s']}",
        rw_pass,
        "target RW torque is selectively suppressed while its native speed response changes",
    ))

    soc = _mean(frame, "battery_soc")
    evidence.append(_ev(
        "eps_battery_capacity_loss", ["battery_soc"], "mean", soc,
        f">{THRESHOLDS['battery_soc_mean_min']}",
        soc is not None and soc > THRESHOLDS["battery_soc_mean_min"],
        "the same stored energy occupies a larger fraction of reduced battery capacity",
    ))

    power_bus = _mean(frame, "power_bus_available")
    evidence.append(_ev(
        "eps_battery_open_circuit", ["power_bus_available"], "mean", power_bus,
        f"<{THRESHOLDS['power_bus_mean_open_max']}",
        power_bus is not None and power_bus < THRESHOLDS["power_bus_mean_open_max"],
        "whole-spacecraft power-bus availability collapses after the battery open-circuit proxy",
    ))

    net_power = _mean(frame, "net_power_w")
    thermal_mean = _mean(frame, "thermal_temp_c")
    heater_load = _mean(frame, "heater_eps_load_w")
    solar_pass = (
        net_power is not None and thermal_mean is not None and heater_load is not None
        and net_power < THRESHOLDS["solar_net_power_mean_max_w"]
        and thermal_mean < THRESHOLDS["solar_thermal_mean_max_c"]
        and heater_load < THRESHOLDS["solar_heater_load_mean_max_w"]
    )
    evidence.append(_ev(
        "solar_panel_efficiency_loss_20pct", ["net_power_w", "thermal_temp_c", "heater_eps_load_w"],
        "mean_power_with_thermal_context", net_power,
        f"net<{THRESHOLDS['solar_net_power_mean_max_w']}, thermal<{THRESHOLDS['solar_thermal_mean_max_c']}, heater<{THRESHOLDS['solar_heater_load_mean_max_w']}",
        solar_pass,
        "net power loss is present outside an active heater/thermal operating regime",
    ))

    payload_rate = _mean(frame, "payload_generated_bps")
    evidence.append(_ev(
        "payload_instrument_off", ["payload_generated_bps"], "mean", payload_rate,
        f"<{THRESHOLDS['payload_rate_mean_max_bps']}",
        payload_rate is not None and payload_rate < THRESHOLDS["payload_rate_mean_max_bps"],
        "payload data production is below the active mission regime",
    ))

    flow = _mean(frame, "fuel_mass_flow_kg_s", absolute=True)
    ignition_pass = flow is not None and flow < THRESHOLDS["ignition_flow_mean_abs_max"]
    partial_pass = (
        flow is not None
        and THRESHOLDS["ignition_flow_mean_abs_max"] <= flow < THRESHOLDS["partial_flow_mean_abs_max"]
    )
    evidence.append(_ev(
        "propulsion_thruster_ignition_failure", ["fuel_mass_flow_kg_s"], "mean_abs", flow,
        f"<{THRESHOLDS['ignition_flow_mean_abs_max']}", ignition_pass,
        "native fuel flow is effectively absent during the commanded propulsion window",
    ))
    evidence.append(_ev(
        "thruster_thrust_loss_15pct", ["fuel_mass_flow_kg_s"], "mean_abs", flow,
        f">={THRESHOLDS['ignition_flow_mean_abs_max']} and <{THRESHOLDS['partial_flow_mean_abs_max']}", partial_pass,
        "native fuel flow is reduced but not eliminated",
    ))

    thermal_last = _last(frame, "thermal_temp_c")
    radiator_pass = (
        thermal_mean is not None and thermal_last is not None
        and thermal_mean > THRESHOLDS["radiator_temp_mean_min_c"]
        and thermal_last > THRESHOLDS["radiator_temp_last_min_c"]
    )
    evidence.append(_ev(
        "thermal_radiator_rejection_loss", ["thermal_temp_c"], "mean_and_last",
        thermal_last,
        f"mean>{THRESHOLDS['radiator_temp_mean_min_c']} and last>{THRESHOLDS['radiator_temp_last_min_c']}",
        radiator_pass,
        "the electronics thermal trajectory remains above the deployed-radiator nominal envelope",
    ))

    heater_active = _mean(frame, "heater_active_count")
    off_pass = (
        heater_active is not None
        and THRESHOLDS["heater_off_active_mean_min"] < heater_active < THRESHOLDS["heater_off_active_mean_max"]
    )
    on_pass = (
        heater_active is not None
        and THRESHOLDS["heater_on_active_mean_min"] < heater_active < THRESHOLDS["heater_on_active_mean_max"]
    )
    evidence.append(_ev(
        "thermal_heater_stuck_off", ["heater_active_count"], "mean", heater_active,
        f">{THRESHOLDS['heater_off_active_mean_min']} and <{THRESHOLDS['heater_off_active_mean_max']}",
        off_pass,
        "a cold-start heater is active for only a small residual fraction of the run",
    ))
    evidence.append(_ev(
        "thermal_heater_stuck_on", ["heater_active_count"], "mean", heater_active,
        f">{THRESHOLDS['heater_on_active_mean_min']} and <{THRESHOLDS['heater_on_active_mean_max']}",
        on_pass,
        "a normally inactive/warm-start heater remains active for most of the run",
    ))

    downlink_rate = _mean(frame, "downlink_requested_rate_bps")
    evidence.append(_ev(
        "comm_data_downlink_link_loss", ["downlink_requested_rate_bps"], "mean", downlink_rate,
        f"<{THRESHOLDS['downlink_requested_mean_max_bps']}",
        downlink_rate is not None and downlink_rate < THRESHOLDS["downlink_requested_mean_max_bps"],
        "requested downlink service is strongly suppressed during an active access window",
    ))

    storage_max = _max(frame, "data_storage_bits")
    evidence.append(_ev(
        "comm_data_storage_capacity_loss", ["data_storage_bits"], "max", storage_max,
        f"<{THRESHOLDS['storage_bits_max_max']}",
        storage_max is not None and storage_max < THRESHOLDS["storage_bits_max_max"],
        "stored data saturates below the normal storage-capacity envelope",
    ))
    return evidence


def _value(frame: pd.DataFrame, field: str, stat: str) -> float | None:
    if stat == "mean": return _mean(frame, field)
    if stat == "last": return _last(frame, field)
    if stat == "max": return _max(frame, field)
    raise ValueError(stat)


def _single_trace_impacts(detected: list[str], frame: pd.DataFrame) -> list[SingleTraceObservedImpact]:
    impacts: list[SingleTraceObservedImpact] = []
    present=set(detected)
    def add(path_id: str, root: str, source: str, target: str, checks: list[tuple[str,str,callable]], text: str) -> None:
        values: dict[str,float]={}
        for field,stat,predicate in checks:
            value=_value(frame,field,stat)
            if value is None or not predicate(value):
                return
            values[f"{field}:{stat}"]=float(value)
        impacts.append(SingleTraceObservedImpact(path_id=path_id,root_effect_id=root,
            source_subsystem=source,target_subsystem=target,fields=[x[0] for x in checks],
            observed_values=values,interpretation=text))
    if "eps_battery_open_circuit" in present:
        add("eps_open->power_bus->payload","eps_battery_open_circuit","EPS","PAYLOAD",
            [("power_bus_available","mean",lambda x:x<0.60),("payload_generated_bps","mean",lambda x:x<1_800_000.0)],
            "low bus availability coincides with suppressed payload production")
        add("eps_open->power_bus->downlink","eps_battery_open_circuit","EPS","COMM_DATA",
            [("power_bus_available","mean",lambda x:x<0.60),("downlink_requested_rate_bps","mean",lambda x:x<1_000_000.0)],
            "low bus availability coincides with suppressed requested downlink activity")
    if "thermal_heater_stuck_on" in present:
        add("heater_on->eps_load->net_power","thermal_heater_stuck_on","THERMAL","EPS",
            [("heater_eps_load_w","mean",lambda x:x>5.0),("net_power_w","mean",lambda x:x<-150.0)],
            "persistent heater load is visible in the integrated EPS power state")
    if "thermal_heater_stuck_off" in present:
        add("heater_off->eps_load->net_power","thermal_heater_stuck_off","THERMAL","EPS",
            [("heater_active_count","mean",lambda x:0.05<x<0.40),("heater_eps_load_w","mean",lambda x:x<5.0)],
            "suppressed heater activity removes expected EPS heater load")
    if "payload_instrument_off" in present:
        add("payload_off->data_generation->storage","payload_instrument_off","PAYLOAD","COMM_DATA",
            [("payload_generated_bps","mean",lambda x:x<2_200_000.0),("data_storage_bits","max",lambda x:x<100_000_000.0)],
            "suppressed payload production coincides with lower onboard data accumulation")
    if "comm_data_downlink_link_loss" in present:
        add("downlink_loss->comm_load->payload","comm_data_downlink_link_loss","COMM_DATA","PAYLOAD",
            [("downlink_requested_rate_bps","mean",lambda x:x<300_000.0),("payload_generated_bps","mean",lambda x:x>2_000_000.0)],
            "downlink suppression leaves the payload active under the reduced communication load")
    if "propulsion_thruster_ignition_failure" in present:
        add("ignition_failure->propulsion_activity->eps","propulsion_thruster_ignition_failure","PROPULSION","EPS",
            [("fuel_mass_flow_kg_s","mean",lambda x:abs(x)<1.0e-6),("battery_soc","last",lambda x:x>0.50)],
            "absent fuel flow coincides with preserved integrated EPS energy")
    return impacts


def diagnose_single_trace_multifault(*, case_id: str, telemetry_csv: Path,
                                     catalog_path: Path | None = None) -> SingleTraceMultiFaultResult:
    frame = pd.read_csv(telemetry_csv)
    prohibited = sorted(
        column for column in frame.columns
        if is_prohibited_field(column)
    )
    if prohibited:
        return SingleTraceMultiFaultResult(
            case_id=case_id,
            status=SingleTraceMultiFaultStatus.DATA_ERROR,
            prohibited_columns_used=prohibited,
        )

    observed_groups = sum(all(field in frame.columns for field in group) for group in REQUIRED_SIGNAL_GROUPS)
    observed_ratio = observed_groups / max(len(REQUIRED_SIGNAL_GROUPS), 1)
    if observed_ratio < 0.60:
        return SingleTraceMultiFaultResult(
            case_id=case_id,
            status=SingleTraceMultiFaultStatus.INSUFFICIENT_EVIDENCE,
            observed_field_ratio=observed_ratio,
        )

    all_evidence = _detect(frame)
    supporting = [item for item in all_evidence if item.passed]
    detected = sorted({item.effect_id for item in supporting})
    if not detected:
        return SingleTraceMultiFaultResult(
            case_id=case_id,
            status=SingleTraceMultiFaultStatus.NORMAL,
            observed_field_ratio=observed_ratio,
        )

    # Ignition failure and partial thrust loss are mutually exclusive by the flow rule,
    # but retain the explicit safeguard at the decision boundary.
    if "propulsion_thruster_ignition_failure" in detected:
        detected = [effect_id for effect_id in detected if effect_id != "thruster_thrust_loss_15pct"]

    # A collapsed power bus explains payload and downlink inactivity.  Those
    # downstream symptoms are retained as observed impacts rather than being
    # promoted to independent injected faults.
    if "eps_battery_open_circuit" in detected:
        detected = [effect_id for effect_id in detected if effect_id not in {
            "payload_instrument_off", "comm_data_downlink_link_loss",
            "comm_data_storage_capacity_loss"
        }]
    # Low storage occupancy can be a downstream consequence of payload
    # shutdown.  Do not promote it to an independent storage-capacity fault
    # when the payload production signature is already present.
    if "payload_instrument_off" in detected:
        detected = [effect_id for effect_id in detected if effect_id != "comm_data_storage_capacity_loss"]

    scores = _effect_strengths(frame)
    primary = max(detected, key=lambda effect_id: (scores.get(effect_id, 0.0), effect_id))
    impacts = _single_trace_impacts(detected,frame)
    subsystems = sorted({EFFECT_META[effect_id][0] for effect_id in detected}|{x.target_subsystem for x in impacts})
    components = sorted({EFFECT_META[effect_id][1] for effect_id in detected})
    return SingleTraceMultiFaultResult(
        case_id=case_id,
        status=SingleTraceMultiFaultStatus.KNOWN_FAULT,
        detected_effect_ids=detected,
        primary_effect_id=primary,
        secondary_effect_ids=[effect_id for effect_id in detected if effect_id != primary],
        effect_scores={effect_id: float(scores.get(effect_id, 0.0)) for effect_id in detected},
        affected_subsystems=subsystems,
        affected_components=components,
        propagation_paths=sorted({x.path_id for x in impacts}),
        observed_secondary_impacts=impacts,
        catalog_evidence_paths=_catalog_paths(detected, catalog_path),
        supporting_evidence=supporting,
        observed_field_ratio=observed_ratio,
    )
