from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .unified_models import DiagnosticEvidence


@dataclass(frozen=True)
class SingleFaultProfile:
    subsystem: str
    component: str
    onset_field: str
    onset_threshold: float
    severity_kind: str


PROFILES: dict[str, SingleFaultProfile] = {
    "eps_battery_capacity_loss": SingleFaultProfile("EPS", "Battery", "battery_soc", 0.002, "capacity_loss"),
    "adcs_rw_bearing_seizure": SingleFaultProfile("ADCS", "ReactionWheel", "rw_speed_rad_s_0", 0.1, "rw_response"),
    "thermal_heater_stuck_on": SingleFaultProfile("THERMAL", "Heater", "heater_eps_load_w", 1.0, "complete"),
    "payload_instrument_off": SingleFaultProfile("PAYLOAD", "Instrument", "payload_generated_bps", 100_000.0, "complete"),
    "payload_instrument_degraded_rate": SingleFaultProfile(
        "PAYLOAD", "Instrument", "payload_generated_bps", 100_000.0, "fractional_loss"
    ),
    "comm_data_downlink_link_loss": SingleFaultProfile(
        "COMM_DATA", "Transmitter", "downlink_requested_rate_bps", 100_000.0, "complete"
    ),
}


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").astype(float)


def _time(frame: pd.DataFrame) -> pd.Series:
    for field in ("time_s", "sim_time_s", "timestamp_s"):
        values = _numeric(frame, field)
        if not values.dropna().empty:
            return values
    return pd.Series(range(len(frame)), dtype=float)


def _severity_band(score: float) -> str:
    if score < 0.25:
        return "MINOR"
    if score < 0.55:
        return "MODERATE"
    if score < 0.85:
        return "SEVERE"
    return "CRITICAL"


def _estimate_onset(
    nominal: pd.DataFrame,
    effect: pd.DataFrame,
    *,
    field: str,
    threshold: float,
) -> tuple[float | None, str, DiagnosticEvidence | None]:
    n = _numeric(nominal, field)
    e = _numeric(effect, field)
    if n.empty or e.empty or len(n) != len(e):
        return None, "NOT_ESTIMABLE", None
    delta = (e - n).abs()
    valid = delta.ge(threshold).fillna(False)
    indexes = [i for i, passed in enumerate(valid.tolist()) if passed]
    if not indexes:
        return None, "NOT_ESTIMABLE", DiagnosticEvidence(
            evidence_id="ONSET_SIGNAL_NOT_OBSERVED",
            field=field,
            statistic="first_sustained_absolute_residual",
            nominal_value=None,
            effect_value=None,
            observed_delta=None,
            interpretation=f"No residual reached the frozen threshold {threshold:g}.",
        )
    first = indexes[0]
    times = _time(effect)
    # Sat-Agent applies runtime faults between recorder samples.  The first changed
    # sample therefore bounds the onset to the preceding sample interval.
    onset_index = max(0, first - 1)
    onset = float(times.iloc[onset_index])
    response_time = float(times.iloc[first])
    return onset, ("AT_OR_BEFORE_START" if onset_index == 0 else "ESTIMATED"), DiagnosticEvidence(
        evidence_id="ONSET_FROM_PAIRED_RESIDUAL",
        field=field,
        statistic="first_sustained_absolute_residual",
        nominal_value=float(n.iloc[first]),
        effect_value=float(e.iloc[first]),
        observed_delta=float(e.iloc[first] - n.iloc[first]),
        threshold=threshold,
        interpretation=(
            f"First response at {response_time:g}s; onset bounded to preceding sample at {onset:g}s."
        ),
    )


def _severity(
    effect_id: str,
    nominal: pd.DataFrame,
    effect: pd.DataFrame,
    profile: SingleFaultProfile,
) -> tuple[float | None, str | None, list[DiagnosticEvidence]]:
    evidence: list[DiagnosticEvidence] = []
    score: float | None = None
    if profile.severity_kind == "capacity_loss":
        n = _numeric(nominal, "battery_soc")
        e = _numeric(effect, "battery_soc")
        if not n.empty and not e.empty and float(e.iloc[-1]) > 1e-12:
            score = max(0.0, min(1.0, 1.0 - float(n.iloc[-1]) / float(e.iloc[-1])))
            evidence.append(DiagnosticEvidence(
                evidence_id="CAPACITY_LOSS_FROM_SOC_RATIO",
                field="battery_soc",
                statistic="last_ratio",
                nominal_value=float(n.iloc[-1]),
                effect_value=float(e.iloc[-1]),
                observed_delta=float(e.iloc[-1] - n.iloc[-1]),
                interpretation="Same stored-energy state maps to a higher SOC after usable-capacity loss.",
            ))
    elif profile.severity_kind == "fractional_loss":
        n = _numeric(nominal, "payload_generated_bps")
        e = _numeric(effect, "payload_generated_bps")
        if not n.empty and not e.empty and float(n.iloc[-1]) > 1e-12:
            score = max(0.0, min(1.0, 1.0 - float(e.iloc[-1]) / float(n.iloc[-1])))
            evidence.append(DiagnosticEvidence(
                evidence_id="PAYLOAD_RATE_RETENTION",
                field="payload_generated_bps",
                statistic="last_ratio",
                nominal_value=float(n.iloc[-1]),
                effect_value=float(e.iloc[-1]),
                observed_delta=float(e.iloc[-1] - n.iloc[-1]),
                interpretation="Fractional loss is estimated from the final retained payload rate.",
            ))
    elif profile.severity_kind == "rw_response":
        n = _numeric(nominal, "rw_speed_rad_s_0")
        e = _numeric(effect, "rw_speed_rad_s_0")
        cmd_n = _numeric(nominal, "rw_motor_torque_cmd_nm_0")
        cmd_e = _numeric(effect, "rw_motor_torque_cmd_nm_0")
        if not n.empty and not e.empty:
            speed_loss = max(0.0, float(n.mean() - e.mean()))
            # Laboratory response score, not a recovered hardware friction coefficient.
            score = max(0.0, min(1.0, speed_loss / 3.5))
            evidence.append(DiagnosticEvidence(
                evidence_id="RW_SPEED_RESPONSE_LOSS",
                field="rw_speed_rad_s_0",
                statistic="mean",
                nominal_value=float(n.mean()),
                effect_value=float(e.mean()),
                observed_delta=float(e.mean() - n.mean()),
                interpretation="Target wheel mean speed falls under the injected bearing drag.",
            ))
        if not cmd_n.empty and not cmd_e.empty:
            evidence.append(DiagnosticEvidence(
                evidence_id="RW_CONTROL_EFFORT_INCREASE",
                field="rw_motor_torque_cmd_nm_0",
                statistic="mean_abs",
                nominal_value=float(cmd_n.abs().mean()),
                effect_value=float(cmd_e.abs().mean()),
                observed_delta=float(cmd_e.abs().mean() - cmd_n.abs().mean()),
                interpretation="Controller effort rises while target-wheel speed falls.",
            ))
    elif profile.severity_kind == "complete":
        score = 1.0
        field = profile.onset_field
        n = _numeric(nominal, field)
        e = _numeric(effect, field)
        if not n.empty and not e.empty:
            evidence.append(DiagnosticEvidence(
                evidence_id="COMPLETE_STATE_CHANGE",
                field=field,
                statistic="last",
                nominal_value=float(n.iloc[-1]),
                effect_value=float(e.iloc[-1]),
                observed_delta=float(e.iloc[-1] - n.iloc[-1]),
                interpretation="Final telemetry is consistent with a complete stuck/off/service-loss state.",
            ))
    return score, (_severity_band(score) if score is not None else None), evidence


def assess_trusted_single_fault(
    *,
    effect_id: str,
    nominal: pd.DataFrame,
    effect: pd.DataFrame,
) -> dict[str, object]:
    profile = PROFILES.get(effect_id)
    if profile is None:
        return {}
    onset, onset_status, onset_evidence = _estimate_onset(
        nominal, effect, field=profile.onset_field, threshold=profile.onset_threshold
    )
    severity_score, severity, evidence = _severity(effect_id, nominal, effect, profile)
    if onset_evidence is not None:
        evidence.insert(0, onset_evidence)
    return {
        "primary_subsystem": profile.subsystem,
        "primary_component": profile.component,
        "estimated_onset_time_s": onset,
        "onset_status": onset_status,
        "estimated_severity": severity,
        "severity_score": severity_score,
        "diagnostic_evidence": evidence,
    }


__all__ = ["PROFILES", "SingleFaultProfile", "assess_trusted_single_fault"]
