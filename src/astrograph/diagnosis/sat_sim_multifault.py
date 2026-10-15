from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable

from astrograph.knowledge.sat_sim_fault_catalog import load_sat_sim_fault_catalog
from astrograph.semantics import is_prohibited_field

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

PROHIBITED_PREFIXES = ("label.", "truth.", "effect.", "object.")
PROHIBITED_EXACT = {"task_id", "case_id", "mode"}


class MultiFaultStatus(str, Enum):
    KNOWN_FAULT = "KNOWN_FAULT"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DATA_ERROR = "DATA_ERROR"


class MultiFaultEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effect_id: str
    field: str
    statistic: str
    nominal_value: float | None = None
    effect_value: float | None = None
    observed_delta: float | None = None
    strength: float = Field(ge=0.0)
    passed: bool
    interpretation: str


class ObservedSecondaryImpact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path_id: str
    root_effect_id: str
    source_subsystem: str
    target_subsystem: str
    field: str
    statistic: str
    nominal_value: float
    effect_value: float
    observed_delta: float
    interpretation: str
    claim_semantics: str = "OBSERVED_ASSOCIATIVE_COUPLING_NOT_CAUSAL_EFFECT_ESTIMATE"


class MultiFaultDiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-sat-sim-multifault.v2"
    case_id: str
    status: MultiFaultStatus
    detected_effect_ids: list[str] = Field(default_factory=list)
    primary_effect_id: str | None = None
    secondary_effect_ids: list[str] = Field(default_factory=list)
    affected_subsystems: list[str] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    propagation_paths: list[str] = Field(default_factory=list)
    observed_secondary_impacts: list[ObservedSecondaryImpact] = Field(default_factory=list)
    catalog_evidence_paths: list[str] = Field(default_factory=list)
    supporting_evidence: list[MultiFaultEvidence] = Field(default_factory=list)
    contradicting_evidence: list[MultiFaultEvidence] = Field(default_factory=list)
    observed_field_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    prohibited_columns_used: list[str] = Field(default_factory=list)
    labels_used: bool = False
    truth_used: bool = False
    claim_boundary: str = (
        "Paired laboratory Basilisk multifault trend diagnosis. Detected sets are known-catalog "
        "experimental effects, not calibrated probabilities, flight qualification, or causal effects."
    )


EFFECT_META: dict[str, tuple[str, str]] = {
    "adcs_rw_jamming": ("ADCS", "ReactionWheel"),
    "adcs_rw_bearing_seizure": ("ADCS", "ReactionWheel"),
    "eps_battery_capacity_loss": ("EPS", "Battery"),
    "eps_battery_open_circuit": ("EPS", "Battery"),
    "solar_panel_efficiency_loss_20pct": ("EPS", "SolarArray"),
    "payload_instrument_off": ("PAYLOAD", "Instrument"),
    "payload_instrument_degraded_rate": ("PAYLOAD", "Instrument"),
    "thruster_thrust_loss_15pct": ("PROPULSION", "Thruster"),
    "thermal_radiator_rejection_loss": ("THERMAL", "Radiator"),
    "thermal_heater_stuck_off": ("THERMAL", "Heater"),
    "thermal_heater_stuck_on": ("THERMAL", "Heater"),
    "comm_data_downlink_link_loss": ("COMM_DATA", "Transmitter"),
    "comm_data_storage_capacity_loss": ("COMM_DATA", "Storage"),
    "propulsion_thruster_ignition_failure": ("PROPULSION", "Thruster"),
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "adcs_rw_jamming": ("rw_motor_torque_nm_0", "rw_speed_rad_s_0"),
    "adcs_rw_bearing_seizure": ("rw_speed_rad_s_0", "rw_motor_torque_cmd_nm_0", "attitude_error_deg"),
    "eps_battery_capacity_loss": ("battery_soc",),
    "eps_battery_open_circuit": ("power_bus_available",),
    "solar_panel_efficiency_loss_20pct": ("net_power_w",),
    "payload_instrument_off": ("payload_generated_bps",),
    "payload_instrument_degraded_rate": ("payload_generated_bps",),
    "thruster_thrust_loss_15pct": ("fuel_mass_flow_kg_s",),
    "thermal_radiator_rejection_loss": ("thermal_temp_c", "thermal_margin_c"),
    "thermal_heater_stuck_off": ("heater_active_count", "heater_eps_load_w"),
    "thermal_heater_stuck_on": ("heater_active_count", "heater_eps_load_w"),
    "comm_data_downlink_link_loss": ("cumulative_delivered_bits",),
    "comm_data_storage_capacity_loss": ("data_storage_bits",),
    "propulsion_thruster_ignition_failure": ("fuel_mass_flow_kg_s",),
}


def _series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").dropna().astype(float)


def _stat(frame: pd.DataFrame, field: str, stat: str) -> float | None:
    s = _series(frame, field)
    if s.empty:
        return None
    if stat == "mean": return float(s.mean())
    if stat == "mean_abs": return float(s.abs().mean())
    if stat == "last": return float(s.iloc[-1])
    if stat == "max": return float(s.max())
    if stat == "min": return float(s.min())
    raise ValueError(stat)


def _evidence(effect_id: str, field: str, stat: str, nominal: pd.DataFrame, effect: pd.DataFrame,
              predicate: Callable[[float, float], tuple[bool, float]], interpretation: str) -> MultiFaultEvidence:
    nv, ev = _stat(nominal, field, stat), _stat(effect, field, stat)
    if nv is None or ev is None:
        return MultiFaultEvidence(effect_id=effect_id, field=field, statistic=stat, passed=False,
                                  strength=0.0, interpretation="required telemetry unavailable")
    passed, strength = predicate(nv, ev)
    return MultiFaultEvidence(effect_id=effect_id, field=field, statistic=stat,
                              nominal_value=nv, effect_value=ev, observed_delta=ev-nv,
                              passed=passed, strength=max(0.0, float(strength)),
                              interpretation=interpretation)


def _decrease(threshold: float) -> Callable[[float, float], tuple[bool, float]]:
    return lambda n, e: (n-e >= threshold, (n-e)/max(threshold, 1e-12))


def _increase(threshold: float) -> Callable[[float, float], tuple[bool, float]]:
    return lambda n, e: (e-n >= threshold, (e-n)/max(threshold, 1e-12))


def _change(threshold: float) -> Callable[[float, float], tuple[bool, float]]:
    return lambda n, e: (abs(e-n) >= threshold, abs(e-n)/max(threshold, 1e-12))


def _bounded_change(max_abs: float) -> Callable[[float, float], tuple[bool, float]]:
    return lambda n, e: (abs(e-n) <= max_abs, max(0.0, 1.0-abs(e-n)/max(max_abs,1e-12)))


def _effect_at_most(max_value: float) -> Callable[[float, float], tuple[bool, float]]:
    return lambda n, e: (e <= max_value, max(0.0, (max_value-e)/max(max_value, 1e-12)+1.0))


def _effect_between(min_value: float, max_value: float) -> Callable[[float, float], tuple[bool, float]]:
    def check(n: float, e: float) -> tuple[bool, float]:
        passed = min_value <= e <= max_value
        if not passed:
            return False, 0.0
        midpoint = (min_value + max_value) / 2.0
        half = max((max_value - min_value) / 2.0, 1e-12)
        return True, max(0.0, 1.0 - abs(e-midpoint)/half) + 1.0
    return check


def _flow_ratio(max_ratio: float, min_drop: float = 1e-8) -> Callable[[float, float], tuple[bool, float]]:
    def check(n: float, e: float) -> tuple[bool, float]:
        nabs, eabs = abs(n), abs(e)
        ratio = eabs/max(nabs, 1e-12)
        drop = nabs-eabs
        passed = drop >= min_drop and ratio <= max_ratio
        return passed, max(0.0, (max_ratio-ratio)/max(max_ratio, 1e-12)+drop/max(min_drop,1e-12))
    return check


def _rw_relative_torque_evidence(nominal: pd.DataFrame, effect: pd.DataFrame) -> MultiFaultEvidence:
    def ratio(frame: pd.DataFrame) -> float | None:
        values=[_stat(frame,f"rw_motor_torque_nm_{i}","mean_abs") for i in range(3)]
        if any(v is None for v in values):
            return None
        denominator=(float(values[1])+float(values[2]))/2.0
        if denominator <= 1.0e-12:
            return None
        return float(values[0])/denominator
    nv,ev=ratio(nominal),ratio(effect)
    if nv is None or ev is None:
        return MultiFaultEvidence(effect_id="adcs_rw_jamming",field="rw0_relative_motor_torque_ratio",
                                  statistic="mean_abs_ratio",passed=False,strength=0.0,
                                  interpretation="required three-wheel torque telemetry unavailable")
    delta=ev-nv; threshold=0.05
    return MultiFaultEvidence(effect_id="adcs_rw_jamming",field="rw0_relative_motor_torque_ratio",
                              statistic="mean_abs_ratio",nominal_value=nv,effect_value=ev,observed_delta=delta,
                              passed=delta <= -threshold,strength=max(0.0,-delta/threshold),
                              interpretation="target RW0 torque falls relative to the non-target wheels")


def _detectors(nominal: pd.DataFrame, effect: pd.DataFrame) -> dict[str, list[MultiFaultEvidence]]:
    out: dict[str, list[MultiFaultEvidence]] = {}
    out["adcs_rw_jamming"] = [
        _rw_relative_torque_evidence(nominal, effect),
        _evidence("adcs_rw_jamming", "rw_speed_rad_s_0", "max", nominal, effect, _change(10.0), "native RW speed response changes"),
    ]
    out["adcs_rw_bearing_seizure"] = [
        _evidence("adcs_rw_bearing_seizure", "rw_speed_rad_s_0", "mean", nominal, effect, _decrease(0.5), "target wheel speed falls under bearing drag"),
        _evidence("adcs_rw_bearing_seizure", "rw_motor_torque_cmd_nm_0", "mean_abs", nominal, effect, _increase(0.002), "controller increases target-wheel motor command"),
        _evidence("adcs_rw_bearing_seizure", "attitude_error_deg", "mean", nominal, effect, _increase(0.05), "pointing performance degrades"),
    ]
    out["eps_battery_capacity_loss"] = [
        _evidence("eps_battery_capacity_loss", "battery_soc", "mean", nominal, effect, _increase(0.04), "same stored energy maps to higher SOC after capacity loss"),
    ]
    out["eps_battery_open_circuit"] = [
        _evidence("eps_battery_open_circuit", "power_bus_available", "mean", nominal, effect, _decrease(0.50), "battery open-circuit proxy collapses whole-spacecraft power-bus availability"),
    ]
    out["solar_panel_efficiency_loss_20pct"] = [
        _evidence("solar_panel_efficiency_loss_20pct", "net_power_w", "mean", nominal, effect, _decrease(4.0), "solar generation loss reduces net bus power"),
        _evidence("solar_panel_efficiency_loss_20pct", "heater_eps_load_w", "mean", nominal, effect, _bounded_change(1.0), "power drop is not explained by heater load switching"),
    ]
    out["payload_instrument_off"] = [
        _evidence("payload_instrument_off", "payload_generated_bps", "min", nominal, effect, _decrease(500_000.0), "payload data generation decreases"),
        _evidence("payload_instrument_off", "payload_generated_bps", "min", nominal, effect, _effect_at_most(50_000.0), "payload final rate is effectively zero"),
    ]
    out["payload_instrument_degraded_rate"] = [
        _evidence("payload_instrument_degraded_rate", "payload_generated_bps", "min", nominal, effect, _decrease(250_000.0), "payload data generation decreases"),
        _evidence("payload_instrument_degraded_rate", "payload_generated_bps", "min", nominal, effect, _effect_between(100_000.0, 2_250_000.0), "payload remains active at a reduced non-zero rate"),
    ]
    out["thermal_radiator_rejection_loss"] = [
        _evidence("thermal_radiator_rejection_loss", "thermal_temp_c", "last", nominal, effect, _increase(0.15), "reduced rejection increases final temperature"),
        _evidence("thermal_radiator_rejection_loss", "thermal_margin_c", "last", nominal, effect, _decrease(0.15), "reduced rejection decreases thermal margin"),
    ]
    out["thermal_heater_stuck_off"] = [
        _evidence("thermal_heater_stuck_off", "heater_active_count", "mean", nominal, effect, _decrease(0.15), "heater active fraction decreases"),
        _evidence("thermal_heater_stuck_off", "heater_eps_load_w", "mean", nominal, effect, _decrease(1.0), "heater electrical load decreases"),
    ]
    out["thermal_heater_stuck_on"] = [
        _evidence("thermal_heater_stuck_on", "heater_active_count", "last", nominal, effect, _increase(0.5), "heater remains active at the end of the window"),
        _evidence("thermal_heater_stuck_on", "heater_eps_load_w", "last", nominal, effect, _increase(5.0), "heater electrical load remains elevated"),
    ]
    out["comm_data_downlink_link_loss"] = [
        _evidence("comm_data_downlink_link_loss", "downlink_requested_rate_bps", "mean", nominal, effect, _decrease(100_000.0), "native mission gate suppresses requested downlink service"),
    ]
    out["comm_data_storage_capacity_loss"] = [
        _evidence("comm_data_storage_capacity_loss", "data_storage_bits", "max", nominal, effect, _decrease(5_000_000.0), "maximum stored data decreases"),
        _evidence("comm_data_storage_capacity_loss", "payload_generated_bps", "mean", nominal, effect, _bounded_change(500_000.0), "storage reduction is not caused by payload shutdown"),
    ]
    out["propulsion_thruster_ignition_failure"] = [
        _evidence("propulsion_thruster_ignition_failure", "fuel_mass_flow_kg_s", "mean_abs", nominal, effect, _flow_ratio(0.10), "ignition failure suppresses nearly all fuel flow"),
    ]
    # Partial thrust loss uses the same physical field but excludes near-zero ignition failures.
    def partial(n: float, e: float) -> tuple[bool, float]:
        nabs,eabs=abs(n),abs(e); ratio=eabs/max(nabs,1e-12); drop=nabs-eabs
        return (drop>=1e-8 and 0.10 < ratio < 0.95, max(0.0, drop/1e-8))
    out["thruster_thrust_loss_15pct"] = [
        _evidence("thruster_thrust_loss_15pct", "fuel_mass_flow_kg_s", "mean_abs", nominal, effect, partial, "partial thrust loss reduces but does not eliminate fuel flow"),
    ]
    return out


def _observed_impact(
    *, root_effect_id: str, path_id: str, source_subsystem: str, target_subsystem: str,
    field: str, statistic: str, nominal: pd.DataFrame, effect: pd.DataFrame,
    predicate: Callable[[float, float], tuple[bool, float]], interpretation: str,
) -> ObservedSecondaryImpact | None:
    nv, ev = _stat(nominal, field, statistic), _stat(effect, field, statistic)
    if nv is None or ev is None:
        return None
    passed, _ = predicate(nv, ev)
    if not passed:
        return None
    return ObservedSecondaryImpact(
        path_id=path_id,
        root_effect_id=root_effect_id,
        source_subsystem=source_subsystem,
        target_subsystem=target_subsystem,
        field=field,
        statistic=statistic,
        nominal_value=nv,
        effect_value=ev,
        observed_delta=ev-nv,
        interpretation=interpretation,
    )


def _observed_secondary_impacts(
    detected: list[str], nominal: pd.DataFrame, effect: pd.DataFrame,
) -> list[ObservedSecondaryImpact]:
    candidates: list[ObservedSecondaryImpact | None] = []
    present=set(detected)
    if "eps_battery_open_circuit" in present:
        candidates += [
            _observed_impact(root_effect_id="eps_battery_open_circuit",path_id="eps_open->power_bus->payload",source_subsystem="EPS",target_subsystem="PAYLOAD",field="payload_generated_bps",statistic="mean",nominal=nominal,effect=effect,predicate=_decrease(250_000.0),interpretation="power-bus loss reduces payload production"),
            _observed_impact(root_effect_id="eps_battery_open_circuit",path_id="eps_open->power_bus->downlink",source_subsystem="EPS",target_subsystem="COMM_DATA",field="downlink_requested_rate_bps",statistic="mean",nominal=nominal,effect=effect,predicate=_decrease(150_000.0),interpretation="power-bus loss reduces requested downlink activity"),
        ]
    if "thermal_heater_stuck_on" in present:
        candidates += [
            _observed_impact(root_effect_id="thermal_heater_stuck_on",path_id="heater_on->eps_load->battery",source_subsystem="THERMAL",target_subsystem="EPS",field="battery_storage_j",statistic="last",nominal=nominal,effect=effect,predicate=_decrease(150.0),interpretation="stuck-on heater load depletes stored battery energy"),
            _observed_impact(root_effect_id="thermal_heater_stuck_on",path_id="heater_on->eps_load->net_power",source_subsystem="THERMAL",target_subsystem="EPS",field="net_power_w",statistic="mean",nominal=nominal,effect=effect,predicate=_decrease(0.8),interpretation="stuck-on heater makes integrated net power more negative"),
        ]
    if "thermal_heater_stuck_off" in present:
        candidates += [
            _observed_impact(root_effect_id="thermal_heater_stuck_off",path_id="heater_off->eps_load->battery",source_subsystem="THERMAL",target_subsystem="EPS",field="battery_storage_j",statistic="last",nominal=nominal,effect=effect,predicate=_increase(150.0),interpretation="stuck-off heater removes load and preserves stored battery energy"),
            _observed_impact(root_effect_id="thermal_heater_stuck_off",path_id="heater_off->eps_load->net_power",source_subsystem="THERMAL",target_subsystem="EPS",field="net_power_w",statistic="mean",nominal=nominal,effect=effect,predicate=_increase(0.8),interpretation="stuck-off heater makes integrated net power less negative"),
        ]
    if "payload_instrument_off" in present:
        candidates += [
            _observed_impact(root_effect_id="payload_instrument_off",path_id="payload_off->eps_load->battery",source_subsystem="PAYLOAD",target_subsystem="EPS",field="battery_storage_j",statistic="last",nominal=nominal,effect=effect,predicate=_increase(300.0),interpretation="payload shutdown removes load and preserves stored battery energy"),
            _observed_impact(root_effect_id="payload_instrument_off",path_id="payload_off->data_generation->storage",source_subsystem="PAYLOAD",target_subsystem="COMM_DATA",field="data_storage_bits",statistic="max",nominal=nominal,effect=effect,predicate=_decrease(250_000.0),interpretation="payload shutdown reduces data accumulated in storage"),
        ]
    if "comm_data_downlink_link_loss" in present:
        candidates += [
            _observed_impact(root_effect_id="comm_data_downlink_link_loss",path_id="downlink_loss->comm_load->battery",source_subsystem="COMM_DATA",target_subsystem="EPS",field="battery_storage_j",statistic="last",nominal=nominal,effect=effect,predicate=_increase(300.0),interpretation="downlink loss removes communication load and preserves stored battery energy"),
            _observed_impact(root_effect_id="comm_data_downlink_link_loss",path_id="downlink_loss->comm_load->payload",source_subsystem="COMM_DATA",target_subsystem="PAYLOAD",field="payload_generated_bps",statistic="mean",nominal=nominal,effect=effect,predicate=_increase(100_000.0),interpretation="reduced communication load preserves EPS margin and payload availability"),
        ]
    if "propulsion_thruster_ignition_failure" in present:
        candidates += [
            _observed_impact(root_effect_id="propulsion_thruster_ignition_failure",path_id="ignition_failure->propulsion_activity->battery",source_subsystem="PROPULSION",target_subsystem="EPS",field="battery_storage_j",statistic="last",nominal=nominal,effect=effect,predicate=_increase(300.0),interpretation="failed ignition reduces integrated propulsion energy demand"),
            _observed_impact(root_effect_id="propulsion_thruster_ignition_failure",path_id="ignition_failure->propulsion_activity->net_power",source_subsystem="PROPULSION",target_subsystem="EPS",field="net_power_w",statistic="mean",nominal=nominal,effect=effect,predicate=_increase(0.8),interpretation="failed ignition makes integrated net power less negative"),
        ]
    return [item for item in candidates if item is not None]


def _catalog_propagation_paths(effect_ids: list[str], catalog_path: Path | None = None) -> list[str]:
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


def diagnose_paired_multifault(*, case_id: str, nominal_csv: Path, effect_csv: Path,
                               catalog_path: Path | None = None) -> MultiFaultDiagnosisResult:
    nominal, effect = pd.read_csv(nominal_csv), pd.read_csv(effect_csv)
    columns = sorted(set(nominal.columns) | set(effect.columns))
    prohibited = [c for c in columns if is_prohibited_field(c)]
    if prohibited:
        return MultiFaultDiagnosisResult(case_id=case_id, status=MultiFaultStatus.DATA_ERROR,
                                         prohibited_columns_used=sorted(prohibited))
    detector_results = _detectors(nominal, effect)
    observed_required = sum(all(field in columns for field in fields) for fields in REQUIRED_FIELDS.values())
    ratio = observed_required/max(len(REQUIRED_FIELDS),1)
    detected: list[str] = []
    supporting: list[MultiFaultEvidence] = []
    contradicting: list[MultiFaultEvidence] = []
    scores: dict[str,float] = {}
    for effect_id, evidence in detector_results.items():
        passed = [item for item in evidence if item.passed]
        failed = [item for item in evidence if not item.passed]
        # Two-channel mechanisms require both channels; single-channel mechanisms require one.
        eligible = len(passed) == len(evidence)
        if eligible:
            detected.append(effect_id)
            supporting.extend(passed)
            scores[effect_id] = sum(item.strength for item in passed)/len(passed)
        else:
            contradicting.extend(failed)
    # Ignition failure and partial thrust loss are mutually exclusive.
    if "propulsion_thruster_ignition_failure" in detected:
        detected = [x for x in detected if x != "thruster_thrust_loss_15pct"]
    # A collapsed power bus directly suppresses payload and downlink activity.
    # Preserve those telemetry changes as secondary impacts instead of
    # misclassifying them as independently injected faults.
    if "eps_battery_open_circuit" in detected:
        detected = [x for x in detected if x not in {
            "payload_instrument_off", "payload_instrument_degraded_rate", "comm_data_downlink_link_loss"
        }]
    # Direct wheel-speed and motor-command evidence identifies the root ADCS fault.
    # Payload/downlink changes are mission-gate consequences and must not be promoted
    # to independently injected faults in this paired single-fault workflow.
    if "adcs_rw_bearing_seizure" in detected:
        detected = [x for x in detected if x not in {
            "payload_instrument_off", "payload_instrument_degraded_rate", "comm_data_downlink_link_loss"
        }]
    if not detected:
        status = MultiFaultStatus.INSUFFICIENT_EVIDENCE if ratio < 0.6 else MultiFaultStatus.AMBIGUOUS
        return MultiFaultDiagnosisResult(case_id=case_id, status=status, observed_field_ratio=ratio,
                                         supporting_evidence=supporting, contradicting_evidence=contradicting,
                                         catalog_evidence_paths=_catalog_propagation_paths([], catalog_path))
    detected=sorted(set(detected))
    primary=max(detected,key=lambda x:(scores.get(x,0.0),x))
    impacts=_observed_secondary_impacts(detected,nominal,effect)
    subsystems=sorted({EFFECT_META[x][0] for x in detected}|{x.target_subsystem for x in impacts})
    components=sorted({EFFECT_META[x][1] for x in detected})
    return MultiFaultDiagnosisResult(
        case_id=case_id,status=MultiFaultStatus.KNOWN_FAULT,detected_effect_ids=detected,
        primary_effect_id=primary,secondary_effect_ids=[x for x in detected if x!=primary],
        affected_subsystems=subsystems,affected_components=components,
        propagation_paths=sorted({x.path_id for x in impacts}),observed_secondary_impacts=impacts,
        catalog_evidence_paths=_catalog_propagation_paths(detected, catalog_path),supporting_evidence=supporting,
        contradicting_evidence=contradicting,observed_field_ratio=ratio,
    )
