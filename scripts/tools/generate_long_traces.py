#!/usr/bin/env python3
"""Generate long component traces from the Sat-Agent platform (work item: 小规模验证管线第一步).

This script drives the Sat-Agent capability adapters (battery / IMU / reaction wheel) to produce
LONG labeled traces (nominal / fault / degradation), which feed AstroGraph's parameter-identification
feasibility experiments and label adapter.

IMPORTANT (venv separation):
  * This script MUST be run with the Sat-Agent venv (Python 3.12.12):
        /d/Project/test_sat/.venv/Scripts/python.exe scripts/generate_long_traces.py
  * It must NOT be run with the AstroGraph venv (3.14.2), because it imports ``sat_sim`` / ``components``
    from the Sat-Agent platform (installed editable in the Sat-Agent venv).
  * No Sat-Agent source is modified. The script only builds canonical TaskSpecs, compiles them through
    the platform's own pipeline (``compile_task_spec`` -> ``to_runtime_task_spec`` -> adapter.run), and
    materializes the trace rows to AstroGraph's ``data/generated_traces/``.

Output:
  * One CSV per scenario under ``data/generated_traces/``. Columns are the full adapter trace rows,
    INCLUDING ``label.*`` columns (kept for the label adapter). Public (label-stripped) materialization is
    a separate later step.
  * ``data/generated_traces/manifest.json`` with per-file statistics, scenario descriptors and generation params.
  * Physical sanity checks (time monotonic, finite values, numeric range guards) are run on every trace.

Note on "seed": the three component adapters here are deterministic pure-Python models (no RNG in the
adapter execution path). "Seed" variation is therefore expressed as parameter variation (onset time,
magnitude/severity, physical config), which is documented in the manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "generated_traces"

BATT_CAP = 160.0
BATT_SOC0 = 0.62


def _canonical(
    *,
    capability_id: str,
    target_name: str,
    mode: str,
    duration_s: float,
    sample_s: float,
    params: dict[str, Any],
    case_id: str,
    level: str = "component",
    faults: list[dict[str, Any]] | None = None,
    degradations: list[dict[str, Any]] | None = None,
    legacy_degradations: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical TaskSpec dict in the same shape the platform synthesizes."""
    task_id = case_id
    return {
        "schema_version": "1.0.0",
        "task": {"id": task_id, "name": case_id, "description": f"long trace {case_id}", "tags": []},
        "simulation": {
            "level": level,
            "subsystem": target_name if level == "subsystem" else None,
            "duration_s": duration_s,
            "step_s": sample_s,
            "sample_s": sample_s,
            "backend": "selective_unified_basilisk_assembly",
            "time_base": "simulation_seconds",
            "time_system": "UTC",
        },
        "mission": {},
        "parameters": {"profile": "demo", "values": params, "overrides": []},
        "events": {
            "faults": faults or [],
            "degradations": degradations or [],
        },
        "outputs": {
            "output_root": "runs",
            "trace_format": "csv",
            "qoi": ["time_s"],
            "files": [],
            "plots": [],
            "include_summary": True,
            "include_trace": True,
            "include_labels": True,
            "include_manifest": True,
        },
        "assurance": {
            "fidelity_level": "declared_by_capability",
            "claim_level": "analysis_only",
            "validation_profile": "default",
            "parameter_profile": "demo",
            "allow_proxy": False,
        },
        "model": {
            "capability_id": capability_id,
            "target": {"level": level, "name": target_name, "mode": mode},
            "spacecraft": {},
            "orbit_environment": {},
            "config": config or {},
            "legacy_degradations": legacy_degradations or {},
            "campaign": {},
            "simulation_extra": {},
            "validation": {},
        },
        "provenance": {"fields": {}, "assumptions": [], "pending_confirmations": []},
        "metadata": {"template_source": "astrograph_long_trace_generator", "case_id": case_id},
    }


def _legacy_fault(fid: str, effect: str, start_s: float, magnitude: float) -> dict[str, Any]:
    return {
        "id": fid,
        "target": "battery",
        "target_type": "battery",
        "effect": effect,
        "start_s": start_s,
        "magnitude": magnitude,
        "implementation": "auto",
        "delivery": "legacy_fault",
        "parameters": {},
        "event_type": "fault",
    }


def _battery_cycle_profile(n: int) -> list[float]:
    """Square-wave charge/discharge cycling between +35 W and -35 W.

    Gap-1 closure (2026-08-04): the storage v2 efficiency branch identifies
    ``discharge_efficiency`` only when BOTH charge (P >= 0) and discharge
    (P < 0) excitation are present.  Half-period of 300 samples at 1.0 s gives
    a 300 s dwell on each sign, so the charge and discharge slopes are both
    seen repeatedly over the 6000 s trace (net energy stays negative because
    discharge uses the 1/eff factor).
    """
    return [35.0 if (i // 300) % 2 == 0 else -35.0 for i in range(n)]


def _rw_cycle_profile(n: int, dt: float) -> list[float]:
    """Slow positive sine torque command within +/- max_motor_torque.

    The offset keeps the command positive so the wheel spin rate stays inside
    the ``actuator.spinning_rotor`` physical range [0, 1e5] rad/s while still
    varying the input enough to identify J and b.
    """
    return [round(0.01 + 0.009 * math.sin(0.05 * index * dt), 12) for index in range(n)]


def _thermal_internal_power_profile(n: int, period_samples: int = 10) -> list[float]:
    """Internal dissipation profile cycling between 20 W and 40 W."""
    return [40.0 if (i // period_samples) % 2 == 0 else 20.0 for i in range(n)]


def _thermal_shadow_profile(n: int, period_samples: int = 60, eclipse_fraction: float = 0.4) -> list[float]:
    """Orbit-style shadow profile: sunlit then eclipse, cycling.

    ``period_samples`` is the orbit length in samples; ``eclipse_fraction`` is
    the fraction of each period spent fully shadowed (shadow_factor = 0).
    """
    out: list[float] = []
    eclipse_samples = max(1, int(round(period_samples * eclipse_fraction)))
    for i in range(n):
        phase = i % period_samples
        out.append(0.0 if phase < eclipse_samples else 1.0)
    return out


def _square_wave(n: int, low: float, high: float, half_period_samples: int) -> list[float]:
    """Square wave cycling between ``low`` and ``high`` every ``half_period_samples`` samples."""
    return [high if (i // half_period_samples) % 2 == 0 else low for i in range(n)]


def _mtb_command_profile(n: int) -> list[list[float]]:
    """Time-varying MTB dipole command: x cycles 0/0.10 A*m^2, y steady 0.05, z 0.

    The square-wave x command (50 s dwell) excites the first-order dipole lag
    (``d(actual)/dt = (command - actual)/tau``) while staying inside the
    nominal 0.2 A*m^2 dipole limit.
    """
    return [[0.10 if (i // 50) % 2 == 0 else 0.0, 0.05, 0.0] for i in range(n)]


def scenarios() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # ---- Battery (storage.energy_buffer), sample_s min 1.0s per capability YAML ----
    batt_params = {
        "capacity_wh": BATT_CAP,
        "initial_soc": BATT_SOC0,
        "net_power_w": -20.0,
        "charge_efficiency": 0.98,
        "discharge_efficiency": 0.97,
        "min_soc": 0.0,
        "max_soc": 1.0,
    }
    batt_model = {
        "capability_id": "component.battery.v1",
        "target_name": "battery",
    }
    out.append({
        "name": "battery-nominal-long",
        "case_id": "BATT-NOM-6000s",
        "capability": "component.battery.v1",
        "scenario": "nominal",
        "duration_s": 6000.0, "sample_s": 1.0,
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="nominal",
            duration_s=6000.0, sample_s=1.0, params=batt_params, case_id="BATT-NOM-6000s",
            config=batt_params,
        ),
    })
    out.append({
        "name": "battery-sudden_capacity_loss-onset2000-mag0.35-long",
        "case_id": "BATT-SCL-2000-0.35",
        "capability": "component.battery.v1",
        "scenario": "fault:sudden_capacity_loss",
        "duration_s": 6000.0, "sample_s": 1.0,
        "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": 2000.0, "magnitude": 0.35},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="fault",
            duration_s=6000.0, sample_s=1.0, params=batt_params, case_id="BATT-SCL-2000-0.35",
            config=batt_params,
            faults=[_legacy_fault("f_scl_a", "sudden_capacity_loss", 2000.0, 0.35)],
        ),
    })
    out.append({
        "name": "battery-sudden_capacity_loss-onset4000-mag0.20-long",
        "case_id": "BATT-SCL-4000-0.20",
        "capability": "component.battery.v1",
        "scenario": "fault:sudden_capacity_loss",
        "duration_s": 6000.0, "sample_s": 1.0,
        "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": 4000.0, "magnitude": 0.20},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="fault",
            duration_s=6000.0, sample_s=1.0, params=batt_params, case_id="BATT-SCL-4000-0.20",
            config=batt_params,
            faults=[_legacy_fault("f_scl_b", "sudden_capacity_loss", 4000.0, 0.20)],
        ),
    })
    out.append({
        "name": "battery-open_circuit-onset1500-long",
        "case_id": "BATT-OC-1500",
        "capability": "component.battery.v1",
        "scenario": "fault:open_circuit",
        "duration_s": 6000.0, "sample_s": 1.0,
        "fault": {"fault_type": "open_circuit", "onset_time_s": 1500.0, "magnitude": 1.0},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="fault",
            duration_s=6000.0, sample_s=1.0, params=batt_params, case_id="BATT-OC-1500",
            config=batt_params,
            faults=[_legacy_fault("f_oc", "open_circuit", 1500.0, 1.0)],
        ),
    })
    out.append({
        "name": "battery-capacity_loss_pct-30-degradation-long",
        "case_id": "BATT-DEG-CL30",
        "capability": "component.battery.v1",
        "scenario": "degradation:capacity_loss_pct",
        "duration_s": 6000.0, "sample_s": 1.0,
        "degradation": {"capacity_loss_pct": 30.0},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="degradation",
            duration_s=6000.0, sample_s=1.0, params=batt_params, case_id="BATT-DEG-CL30",
            config=batt_params,
            legacy_degradations={"eps": {"battery": {"capacity_loss_pct": 30.0}}},
        ),
    })

    # ---- IMU (sensor.measurement), sample_s min 0.001s ----
    imu_model = {"capability_id": "component.imu.v1", "target_name": "imu"}
    imu_params = {"true_gyro_rad_s": [0.1, 0.2, 0.3], "true_accel_m_s2": [0.0, 0.0, 0.0]}
    out.append({
        "name": "imu-nominal-long",
        "case_id": "IMU-NOM-600s",
        "capability": "component.imu.v1",
        "scenario": "nominal",
        "duration_s": 600.0, "sample_s": 0.1,
        "canonical": _canonical(
            capability_id="component.imu.v1", target_name="imu", mode="nominal",
            duration_s=600.0, sample_s=0.1, params=imu_params, case_id="IMU-NOM-600s",
        ),
    })
    out.append({
        "name": "imu-bias_step-onset150-sev0.6-long",
        "case_id": "IMU-BS-150-0.6",
        "capability": "component.imu.v1",
        "scenario": "fault:bias_step",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "bias_step", "onset_time_s": 150.0, "severity": 0.6},
        "canonical": _canonical(
            capability_id="component.imu.v1", target_name="imu", mode="fault",
            duration_s=600.0, sample_s=0.1, params=imu_params, case_id="IMU-BS-150-0.6",
            faults=[{
                "id": "f_bs_a", "target": "imu", "target_type": "imu", "effect": "bias_step",
                "start_s": 150.0, "magnitude": 0.6, "implementation": "auto",
                "delivery": "modifier", "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "imu-bias_step-onset300-sev0.3-long",
        "case_id": "IMU-BS-300-0.3",
        "capability": "component.imu.v1",
        "scenario": "fault:bias_step",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "bias_step", "onset_time_s": 300.0, "severity": 0.3},
        "canonical": _canonical(
            capability_id="component.imu.v1", target_name="imu", mode="fault",
            duration_s=600.0, sample_s=0.1, params=imu_params, case_id="IMU-BS-300-0.3",
            faults=[{
                "id": "f_bs_b", "target": "imu", "target_type": "imu", "effect": "bias_step",
                "start_s": 300.0, "magnitude": 0.3, "implementation": "auto",
                "delivery": "modifier", "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "imu-bias_drift-rate0.01-max0.5-degradation-long",
        "case_id": "IMU-DEG-BDRIFT",
        "capability": "component.imu.v1",
        "scenario": "degradation:bias_drift",
        "duration_s": 600.0, "sample_s": 0.1,
        "degradation": {"severity": 0.5, "adapter_defaults": "rate_per_s=0.01 max_fraction=severity"},
        "canonical": _canonical(
            capability_id="component.imu.v1", target_name="imu", mode="degradation",
            duration_s=600.0, sample_s=0.1, params=imu_params, case_id="IMU-DEG-BDRIFT",
            degradations=[{
                "id": "d_bd", "target": "imu", "target_type": "imu", "effect": "bias_drift",
                "start_s": 0.0, "magnitude": 0.5, "implementation": "auto",
                "parameters": {}, "event_type": "degradation",
            }],
        ),
    })

    # ---- Reaction wheel (actuator.controlled_effort), sample_s min 0.001s ----
    rw_params = {
        "num_wheels": 1,
        "wheel_inertia_kg_m2": 0.1,
        "max_motor_torque_nm": 0.02,
        "max_speed_rad_s": 1000.0,
        "damping_nms": 1.0e-5,
        "initial_wheel_speeds_rad_s": 100.0,
        "command_torque_nm": 0.01,
    }
    rw_model = {"capability_id": "component.reaction_wheel.v1", "target_name": "reaction_wheel"}
    rw_config = dict(rw_params)
    out.append({
        "name": "reaction_wheel-nominal-long",
        "case_id": "RW-NOM-600s",
        "capability": "component.reaction_wheel.v1",
        "scenario": "nominal",
        "duration_s": 600.0, "sample_s": 0.1,
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="nominal",
            duration_s=600.0, sample_s=0.1, params=rw_params, case_id="RW-NOM-600s", config=rw_config,
        ),
    })
    out.append({
        "name": "reaction_wheel-rw_jamming-onset200-long",
        "case_id": "RW-JAM-200",
        "capability": "component.reaction_wheel.v1",
        "scenario": "fault:rw_jamming",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "rw_jamming", "onset_time_s": 200.0, "magnitude": 1.0},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="fault",
            duration_s=600.0, sample_s=0.1, params=rw_params, case_id="RW-JAM-200", config=rw_config,
            faults=[{
                "id": "f_jam", "target": "rw_0", "target_type": "reaction_wheel", "effect": "rw_jamming",
                "start_s": 200.0, "magnitude": 1.0, "implementation": "auto",
                "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "reaction_wheel-rw_motor_failure-onset300-mag0.8-long",
        "case_id": "RW-MOT-300-0.8",
        "capability": "component.reaction_wheel.v1",
        "scenario": "fault:rw_motor_failure",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "rw_motor_failure", "onset_time_s": 300.0, "magnitude": 0.8},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="fault",
            duration_s=600.0, sample_s=0.1, params=rw_params, case_id="RW-MOT-300-0.8", config=rw_config,
            faults=[{
                "id": "f_mot", "target": "rw_0", "target_type": "reaction_wheel", "effect": "rw_motor_failure",
                "start_s": 300.0, "magnitude": 0.8, "implementation": "auto",
                "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "reaction_wheel-friction_increase-50-degradation-long",
        "case_id": "RW-DEG-FRICT50",
        "capability": "component.reaction_wheel.v1",
        "scenario": "degradation:friction_increase_pct",
        "duration_s": 600.0, "sample_s": 0.1,
        "degradation": {"friction_increase_pct": 50.0},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="degradation",
            duration_s=600.0, sample_s=0.1, params=rw_params, case_id="RW-DEG-FRICT50", config=rw_config,
            legacy_degradations={"adcs": {"reaction_wheel": {"friction_increase_pct": 50.0}}},
        ),
    })

    # ---- Excitation-injected battery: time-varying load profile (work item ①) ----
    # The adapter consumes ``power_profile_w`` (list[float], per-sample power)
    # natively; no Sat-Agent source is modified.
    batt_cycled_power = _battery_cycle_profile(6000)
    batt_cycled_params = dict(batt_params)
    batt_cycled_params["power_profile_w"] = batt_cycled_power
    batt_cycled_config = dict(batt_cycled_params)
    out.append({
        "name": "battery-cycled-nominal-long",
        "case_id": "BATT-CYC-NOM-6000s",
        "capability": "component.battery.v1",
        "scenario": "nominal",
        "duration_s": 6000.0, "sample_s": 1.0,
        "excitation": {"profile": "square_wave_power", "power_levels_w": [-35.0, -10.0], "half_period_samples": 300, "sample_s": 1.0},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="nominal",
            duration_s=6000.0, sample_s=1.0, params=batt_cycled_params, case_id="BATT-CYC-NOM-6000s",
            config=batt_cycled_config,
        ),
    })
    out.append({
        "name": "battery-cycled-sudden_capacity_loss-onset2000-mag0.35-long",
        "case_id": "BATT-CYC-SCL-2000-0.35",
        "capability": "component.battery.v1",
        "scenario": "fault:sudden_capacity_loss",
        "duration_s": 6000.0, "sample_s": 1.0,
        "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": 2000.0, "magnitude": 0.35},
        "excitation": {"profile": "square_wave_power", "power_levels_w": [-35.0, -10.0], "half_period_samples": 300, "sample_s": 1.0},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="fault",
            duration_s=6000.0, sample_s=1.0, params=batt_cycled_params, case_id="BATT-CYC-SCL-2000-0.35",
            config=batt_cycled_config,
            faults=[_legacy_fault("f_scl_cyc", "sudden_capacity_loss", 2000.0, 0.35)],
        ),
    })
    out.append({
        "name": "battery-cycled-capacity_loss_pct-30-degradation-long",
        "case_id": "BATT-CYC-DEG-CL30",
        "capability": "component.battery.v1",
        "scenario": "degradation:capacity_loss_pct",
        "duration_s": 6000.0, "sample_s": 1.0,
        "degradation": {"capacity_loss_pct": 30.0},
        "excitation": {"profile": "square_wave_power", "power_levels_w": [-35.0, -10.0], "half_period_samples": 300, "sample_s": 1.0},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="degradation",
            duration_s=6000.0, sample_s=1.0, params=batt_cycled_params, case_id="BATT-CYC-DEG-CL30",
            config=batt_cycled_config,
            legacy_degradations={"eps": {"battery": {"capacity_loss_pct": 30.0}}},
        ),
    })
    out.append({
        "name": "battery-cycled-open_circuit-onset1500-long",
        "case_id": "BATT-CYC-OC-1500",
        "capability": "component.battery.v1",
        "scenario": "fault:open_circuit",
        "duration_s": 6000.0, "sample_s": 1.0,
        "fault": {"fault_type": "open_circuit", "onset_time_s": 1500.0, "magnitude": 1.0},
        "excitation": {"profile": "square_wave_power", "power_levels_w": [-35.0, -10.0], "half_period_samples": 300, "sample_s": 1.0},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="fault",
            duration_s=6000.0, sample_s=1.0, params=batt_cycled_params, case_id="BATT-CYC-OC-1500",
            config=batt_cycled_config,
            faults=[_legacy_fault("f_oc_cyc", "open_circuit", 1500.0, 1.0)],
        ),
    })
    out.append({
        "name": "battery-cycled-sudden_capacity_loss-onset4000-mag0.20-long",
        "case_id": "BATT-CYC-SCL-4000-0.20",
        "capability": "component.battery.v1",
        "scenario": "fault:sudden_capacity_loss",
        "duration_s": 6000.0, "sample_s": 1.0,
        "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": 4000.0, "magnitude": 0.20},
        "excitation": {"profile": "square_wave_power", "power_levels_w": [-35.0, -10.0], "half_period_samples": 300, "sample_s": 1.0},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="fault",
            duration_s=6000.0, sample_s=1.0, params=batt_cycled_params, case_id="BATT-CYC-SCL-4000-0.20",
            config=batt_cycled_config,
            faults=[_legacy_fault("f_scl_cyc_b", "sudden_capacity_loss", 4000.0, 0.20)],
        ),
    })

    # ---- Excitation-injected reaction wheel: time-varying torque profile (work item ①) ----
    # ``torque_profile_nm`` is a list (or list[list] per wheel) of per-sample
    # torque commands; the adapter expands it with ``_vector(item, count, 0.0)``.
    rw_cycled_torque = _rw_cycle_profile(6000, 0.1)
    rw_cycled_params = dict(rw_params)
    rw_cycled_params["torque_profile_nm"] = rw_cycled_torque
    rw_cycled_config = dict(rw_cycled_params)
    out.append({
        "name": "reaction_wheel-cycled-nominal-long",
        "case_id": "RW-CYC-NOM-600s",
        "capability": "component.reaction_wheel.v1",
        "scenario": "nominal",
        "duration_s": 600.0, "sample_s": 0.1,
        "excitation": {"profile": "positive_sine_torque", "amplitude_nm": 0.009, "offset_nm": 0.01, "angular_freq_rad_s": 0.05, "sample_s": 0.1},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="nominal",
            duration_s=600.0, sample_s=0.1, params=rw_cycled_params, case_id="RW-CYC-NOM-600s", config=rw_cycled_config,
        ),
    })
    out.append({
        "name": "reaction_wheel-cycled-rw_jamming-onset200-long",
        "case_id": "RW-CYC-JAM-200",
        "capability": "component.reaction_wheel.v1",
        "scenario": "fault:rw_jamming",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "rw_jamming", "onset_time_s": 200.0, "magnitude": 1.0},
        "excitation": {"profile": "positive_sine_torque", "amplitude_nm": 0.009, "offset_nm": 0.01, "angular_freq_rad_s": 0.05, "sample_s": 0.1},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="fault",
            duration_s=600.0, sample_s=0.1, params=rw_cycled_params, case_id="RW-CYC-JAM-200", config=rw_cycled_config,
            faults=[{
                "id": "f_jam_cyc", "target": "rw_0", "target_type": "reaction_wheel", "effect": "rw_jamming",
                "start_s": 200.0, "magnitude": 1.0, "implementation": "auto",
                "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "reaction_wheel-cycled-friction_increase-50-degradation-long",
        "case_id": "RW-CYC-DEG-FRICT50",
        "capability": "component.reaction_wheel.v1",
        "scenario": "degradation:friction_increase_pct",
        "duration_s": 600.0, "sample_s": 0.1,
        "degradation": {"friction_increase_pct": 50.0},
        "excitation": {"profile": "positive_sine_torque", "amplitude_nm": 0.009, "offset_nm": 0.01, "angular_freq_rad_s": 0.05, "sample_s": 0.1},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="degradation",
            duration_s=600.0, sample_s=0.1, params=rw_cycled_params, case_id="RW-CYC-DEG-FRICT50", config=rw_cycled_config,
            legacy_degradations={"adcs": {"reaction_wheel": {"friction_increase_pct": 50.0}}},
        ),
    })
    out.append({
        "name": "reaction_wheel-cycled-rw_motor_failure-onset300-mag0.8-long",
        "case_id": "RW-CYC-MOT-300-0.8",
        "capability": "component.reaction_wheel.v1",
        "scenario": "fault:rw_motor_failure",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "rw_motor_failure", "onset_time_s": 300.0, "magnitude": 0.8},
        "excitation": {"profile": "positive_sine_torque", "amplitude_nm": 0.009, "offset_nm": 0.01, "angular_freq_rad_s": 0.05, "sample_s": 0.1},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="fault",
            duration_s=600.0, sample_s=0.1, params=rw_cycled_params, case_id="RW-CYC-MOT-300-0.8", config=rw_cycled_config,
            faults=[{
                "id": "f_mot_cyc", "target": "rw_0", "target_type": "reaction_wheel", "effect": "rw_motor_failure",
                "start_s": 300.0, "magnitude": 0.8, "implementation": "auto",
                "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
            }],
        ),
    })

    # ---- Thermal (subsystem.thermal.basic_lumped.v1): long excitation-injected traces (work item ①) ----
    # Dual-node (bus + battery) lumped model with radiator + heater hysteresis.
    # Excitation comes from ``internal_power_profile_w`` and ``shadow_profile``.
    thermal_params = {
        "initial_bus_temp_c": 20.0,
        "initial_battery_temp_c": 18.0,
        "bus_thermal_capacity_j_k": 18000.0,
        "battery_thermal_capacity_j_k": 9000.0,
        "internal_power_w": 28.0,
        "battery_internal_power_w": 3.0,
        "solar_heat_w": 45.0,
        "shadow_factor": 1.0,
        "sink_temp_c": -35.0,
        "radiator_area_m2": 0.35,
        "radiator_emissivity": 0.82,
        "radiator_degradation_factor": 1.0,
        "thermal_conductance_bus_battery_w_k": 0.45,
        "heater_power_w": 20.0,
        "heater_setpoint_c": 5.0,
        "heater_deadband_c": 2.0,
        "bus_min_temp_c": -10.0,
        "bus_max_temp_c": 45.0,
        "battery_min_temp_c": 0.0,
        "battery_max_temp_c": 40.0,
    }
    thermal_capability = "subsystem.thermal.basic_lumped.v1"
    thermal_duration_s = 30000.0
    thermal_sample_s = 60.0
    thermal_n = int(math.floor(thermal_duration_s / thermal_sample_s)) + 1
    thermal_profiles = {
        "internal_power_profile_w": _thermal_internal_power_profile(thermal_n, period_samples=10),
        "shadow_profile": _thermal_shadow_profile(thermal_n, period_samples=60, eclipse_fraction=0.4),
    }
    thermal_cycled_params = dict(thermal_params)
    thermal_cycled_params.update(thermal_profiles)
    thermal_config = dict(thermal_cycled_params)
    thermal_excitation = {
        "profile": "internal_power_square_20_40W + orbit_shadow_eclipse",
        "internal_power_period_samples": 10,
        "shadow_period_samples": 60,
        "shadow_eclipse_fraction": 0.4,
        "sample_s": 60.0,
        "duration_s": 30000.0,
    }
    out.append({
        "name": "thermal-cycled-nominal-long",
        "case_id": "TH-CYC-NOM-30000s",
        "capability": thermal_capability,
        "scenario": "nominal",
        "level": "subsystem",
        "duration_s": thermal_duration_s, "sample_s": thermal_sample_s,
        "excitation": thermal_excitation,
        "canonical": _canonical(
            capability_id=thermal_capability, target_name="thermal", mode="nominal",
            duration_s=thermal_duration_s, sample_s=thermal_sample_s, params=thermal_cycled_params,
            case_id="TH-CYC-NOM-30000s", level="subsystem", config=thermal_config,
        ),
    })
    out.append({
        "name": "thermal-cycled-heating_or_rejection_failure-onset12000-long",
        "case_id": "TH-CYC-HRF-12000",
        "capability": thermal_capability,
        "scenario": "fault:heating_or_rejection_failure",
        "level": "subsystem",
        "duration_s": thermal_duration_s, "sample_s": thermal_sample_s,
        "fault": {"fault_type": "heating_or_rejection_failure", "onset_time_s": 12000.0, "magnitude": 1.0},
        "excitation": thermal_excitation,
        "canonical": _canonical(
            capability_id=thermal_capability, target_name="thermal", mode="fault",
            duration_s=thermal_duration_s, sample_s=thermal_sample_s, params=thermal_cycled_params,
            case_id="TH-CYC-HRF-12000", level="subsystem", config=thermal_config,
            faults=[{
                "id": "f_hrf", "target": "thermal", "target_type": "thermal",
                "effect": "heating_or_rejection_failure", "start_s": 12000.0,
                "magnitude": 1.0, "implementation": "auto", "delivery": "modifier",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "thermal-cycled-radiator_degradation-factor0.7-long",
        "case_id": "TH-CYC-DEG-RAD0.7",
        "capability": thermal_capability,
        "scenario": "degradation:radiator_degradation_factor",
        "level": "subsystem",
        "duration_s": thermal_duration_s, "sample_s": thermal_sample_s,
        "degradation": {"radiator_degradation_factor": 0.7},
        "excitation": thermal_excitation,
        "canonical": _canonical(
            capability_id=thermal_capability, target_name="thermal", mode="degradation",
            duration_s=thermal_duration_s, sample_s=thermal_sample_s, params=thermal_cycled_params,
            case_id="TH-CYC-DEG-RAD0.7", level="subsystem", config=thermal_config,
            degradations=[{
                "id": "d_rad_deg", "target": "thermal", "target_type": "thermal",
                "effect": "radiator_degradation_factor", "start_s": 0.0,
                "magnitude": 0.7, "implementation": "auto", "delivery": "modifier",
                "parameters": {"factor": 0.7}, "event_type": "degradation",
            }],
        ),
    })

    # ---- Loop 13: all-object long traces (MTB / data_queue / onboard_storage /
    #      solar_panel / EPS subsystem).  Each object gets nominal + 1-2 key
    #      fault/degradation with native excitation profiles where supported. ----

    # -- MTB (component.mtb.v1): first-order dipole lag + m x B torque.
    #    Excitation via ``command_profile_am2`` (square-wave x command).
    mtb_params = {
        "dipole_limit_am2": [0.2, 0.2, 0.2],
        "lag_tau_s": 0.2,
        "magnetic_field_t": [2.0e-5, -1.0e-5, 3.0e-5],
        "command_mode": "direct_dipole",
        "command_profile_am2": _mtb_command_profile(1201),
    }
    mtb_config = dict(mtb_params)
    out.append({
        "name": "mtb-nominal-long",
        "case_id": "MTB-NOM-1200s",
        "capability": "component.mtb.v1",
        "scenario": "nominal",
        "duration_s": 1200.0, "sample_s": 1.0,
        "excitation": {"profile": "square_wave_dipole_x_0_0.1_Am2", "half_period_samples": 50, "y_cmd_am2": 0.05},
        "canonical": _canonical(
            capability_id="component.mtb.v1", target_name="mtb", mode="nominal",
            duration_s=1200.0, sample_s=1.0, params=mtb_params, case_id="MTB-NOM-1200s", config=mtb_config,
        ),
    })
    out.append({
        "name": "mtb-mtb_coil_open-onset300-long",
        "case_id": "MTB-COIL-300",
        "capability": "component.mtb.v1",
        "scenario": "fault:mtb_coil_open",
        "duration_s": 1200.0, "sample_s": 1.0,
        "fault": {"fault_type": "mtb_coil_open", "onset_time_s": 300.0, "magnitude": 1.0, "axis_index": 0},
        "excitation": {"profile": "square_wave_dipole_x_0_0.1_Am2", "half_period_samples": 50, "y_cmd_am2": 0.05},
        "canonical": _canonical(
            capability_id="component.mtb.v1", target_name="mtb", mode="fault",
            duration_s=1200.0, sample_s=1.0, params=mtb_params, case_id="MTB-COIL-300", config=mtb_config,
            faults=[{
                "id": "f_mtb_coil", "target": "mtb", "target_type": "mtb", "effect": "mtb_coil_open",
                "start_s": 300.0, "magnitude": 1.0, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {"axis_index": 0}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "mtb-mtb_dipole_capacity_loss-degradation-long",
        "case_id": "MTB-DEG-CAPLOSS",
        "capability": "component.mtb.v1",
        "scenario": "degradation:mtb_dipole_capacity_loss",
        "duration_s": 1200.0, "sample_s": 1.0,
        "degradation": {"remaining_capacity_ratio": 0.7},
        "excitation": {"profile": "square_wave_dipole_x_0_0.1_Am2", "half_period_samples": 50, "y_cmd_am2": 0.05},
        "canonical": _canonical(
            capability_id="component.mtb.v1", target_name="mtb", mode="degradation",
            duration_s=1200.0, sample_s=1.0, params=mtb_params, case_id="MTB-DEG-CAPLOSS", config=mtb_config,
            degradations=[{
                "id": "d_mtb_cap", "target": "mtb", "target_type": "mtb", "effect": "mtb_dipole_capacity_loss",
                "start_s": 0.0, "magnitude": 0.3, "implementation": "auto", "delivery": "modifier",
                "parameters": {"remaining_capacity_ratio": 0.7}, "event_type": "degradation",
            }],
        ),
    })

    # -- Data queue (component.data_queue.v1): bounded-inventory queue.
    #    Excitation via ``generated_profile_bps`` (square wave 20k/80k bit/s).
    dq_params = {
        "capacity_bits": 1000000.0,
        "initial_queue_bits": 500000.0,
        "downlink_bps": 30000.0,
        "generated_profile_bps": _square_wave(601, 20000.0, 80000.0, 50),
    }
    dq_config = dict(dq_params)
    out.append({
        "name": "data_queue-nominal-long",
        "case_id": "DQ-NOM-600s",
        "capability": "component.data_queue.v1",
        "scenario": "nominal",
        "duration_s": 600.0, "sample_s": 1.0,
        "excitation": {"profile": "square_wave_generated_20_80kbps", "half_period_samples": 50, "downlink_bps": 30000.0},
        "canonical": _canonical(
            capability_id="component.data_queue.v1", target_name="data_queue", mode="nominal",
            duration_s=600.0, sample_s=1.0, params=dq_params, case_id="DQ-NOM-600s", config=dq_config,
        ),
    })
    out.append({
        "name": "data_queue-overflow-onset300-long",
        "case_id": "DQ-OVF-300",
        "capability": "component.data_queue.v1",
        "scenario": "fault:overflow",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "overflow", "onset_time_s": 300.0, "magnitude": 0.8},
        "excitation": {"profile": "square_wave_generated_20_80kbps", "half_period_samples": 50, "downlink_bps": 30000.0},
        "canonical": _canonical(
            capability_id="component.data_queue.v1", target_name="data_queue", mode="fault",
            duration_s=600.0, sample_s=1.0, params=dq_params, case_id="DQ-OVF-300", config=dq_config,
            faults=[{
                "id": "f_dq_ovf", "target": "data_queue", "target_type": "data_queue", "effect": "overflow",
                "start_s": 300.0, "magnitude": 0.8, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "data_queue-throughput_decay-degradation-long",
        "case_id": "DQ-DEG-TDECAY",
        "capability": "component.data_queue.v1",
        "scenario": "degradation:throughput_decay",
        "duration_s": 600.0, "sample_s": 1.0,
        "degradation": {"severity": 0.5},
        "excitation": {"profile": "square_wave_generated_20_80kbps", "half_period_samples": 50, "downlink_bps": 30000.0},
        "canonical": _canonical(
            capability_id="component.data_queue.v1", target_name="data_queue", mode="degradation",
            duration_s=600.0, sample_s=1.0, params=dq_params, case_id="DQ-DEG-TDECAY", config=dq_config,
            degradations=[{
                "id": "d_dq_td", "target": "data_queue", "target_type": "data_queue", "effect": "throughput_decay",
                "start_s": 0.0, "magnitude": 0.5, "implementation": "auto", "delivery": "modifier",
                "parameters": {}, "event_type": "degradation",
            }],
        ),
    })

    # -- Onboard storage (component.onboard_storage.v1): bounded inventory.
    #    Excitation via ``generated_profile_bps`` + ``downlink_profile_bps``.
    os_params = {
        "capacity_bits": 20000000.0,
        "initial_stored_bits": 5000000.0,
        "high_watermark": 0.9,
        "generated_profile_bps": _square_wave(601, 30000.0, 80000.0, 100),
        "downlink_profile_bps": _square_wave(601, 10000.0, 40000.0, 50),
    }
    os_config = dict(os_params)
    out.append({
        "name": "onboard_storage-nominal-long",
        "case_id": "OS-NOM-600s",
        "capability": "component.onboard_storage.v1",
        "scenario": "nominal",
        "duration_s": 600.0, "sample_s": 1.0,
        "excitation": {"profile": "square_wave_generated_30_80kbps + downlink_10_40kbps"},
        "canonical": _canonical(
            capability_id="component.onboard_storage.v1", target_name="onboard_storage", mode="nominal",
            duration_s=600.0, sample_s=1.0, params=os_params, case_id="OS-NOM-600s", config=os_config,
        ),
    })
    out.append({
        "name": "onboard_storage-read_only-onset300-long",
        "case_id": "OS-RO-300",
        "capability": "component.onboard_storage.v1",
        "scenario": "fault:read_only",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "read_only", "onset_time_s": 300.0, "magnitude": 1.0},
        "excitation": {"profile": "square_wave_generated_30_80kbps + downlink_10_40kbps"},
        "canonical": _canonical(
            capability_id="component.onboard_storage.v1", target_name="onboard_storage", mode="fault",
            duration_s=600.0, sample_s=1.0, params=os_params, case_id="OS-RO-300", config=os_config,
            faults=[{
                "id": "f_os_ro", "target": "onboard_storage", "target_type": "onboard_storage", "effect": "read_only",
                "start_s": 300.0, "magnitude": 1.0, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "onboard_storage-capacity_loss-degradation-long",
        "case_id": "OS-DEG-CAPLOSS",
        "capability": "component.onboard_storage.v1",
        "scenario": "degradation:capacity_loss",
        "duration_s": 600.0, "sample_s": 1.0,
        "degradation": {"severity": 0.3, "adapter_defaults": "rate_per_s=0.01 max_fraction=severity"},
        "excitation": {"profile": "square_wave_generated_30_80kbps + downlink_10_40kbps"},
        "canonical": _canonical(
            capability_id="component.onboard_storage.v1", target_name="onboard_storage", mode="degradation",
            duration_s=600.0, sample_s=1.0, params=os_params, case_id="OS-DEG-CAPLOSS", config=os_config,
            degradations=[{
                "id": "d_os_cl", "target": "onboard_storage", "target_type": "onboard_storage", "effect": "capacity_loss",
                "start_s": 0.0, "magnitude": 0.3, "implementation": "auto", "delivery": "modifier",
                "parameters": {}, "event_type": "degradation",
            }],
        ),
    })

    # -- Solar panel (component.solar_panel.v1): converter.
    #    Excitation via ``shadow_profile`` (eclipse cycling).
    sp_params = {
        "max_power_w": 120.0,
        "panel_count": 1,
        "efficiency": 1.0,
        "deployment_fraction": 1.0,
        "shadow_profile": _thermal_shadow_profile(601, period_samples=60, eclipse_fraction=0.4),
    }
    sp_config = dict(sp_params)
    out.append({
        "name": "solar_panel-nominal-long",
        "case_id": "SP-NOM-600s",
        "capability": "component.solar_panel.v1",
        "scenario": "nominal",
        "duration_s": 600.0, "sample_s": 1.0,
        "excitation": {"profile": "orbit_shadow_eclipse", "period_samples": 60, "eclipse_fraction": 0.4},
        "canonical": _canonical(
            capability_id="component.solar_panel.v1", target_name="solar_panel", mode="nominal",
            duration_s=600.0, sample_s=1.0, params=sp_params, case_id="SP-NOM-600s", config=sp_config,
        ),
    })
    out.append({
        "name": "solar_panel-solar_panel_failure-onset300-long",
        "case_id": "SP-FAIL-300",
        "capability": "component.solar_panel.v1",
        "scenario": "fault:solar_panel_failure",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "solar_panel_failure", "onset_time_s": 300.0, "magnitude": 0.8},
        "excitation": {"profile": "orbit_shadow_eclipse", "period_samples": 60, "eclipse_fraction": 0.4},
        "canonical": _canonical(
            capability_id="component.solar_panel.v1", target_name="solar_panel", mode="fault",
            duration_s=600.0, sample_s=1.0, params=sp_params, case_id="SP-FAIL-300", config=sp_config,
            faults=[{
                "id": "f_sp_fail", "target": "solar_panel", "target_type": "solar_panel", "effect": "solar_panel_failure",
                "start_s": 300.0, "magnitude": 0.8, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "solar_panel-efficiency_loss-degradation-long",
        "case_id": "SP-DEG-EFF",
        "capability": "component.solar_panel.v1",
        "scenario": "degradation:efficiency_loss_pct",
        "duration_s": 600.0, "sample_s": 1.0,
        "degradation": {"efficiency_loss_pct": 20.0},
        "excitation": {"profile": "orbit_shadow_eclipse", "period_samples": 60, "eclipse_fraction": 0.4},
        "canonical": _canonical(
            capability_id="component.solar_panel.v1", target_name="solar_panel", mode="degradation",
            duration_s=600.0, sample_s=1.0, params=sp_params, case_id="SP-DEG-EFF", config=sp_config,
            legacy_degradations={"eps": {"solar_panel": {"efficiency_loss_pct": 20.0}}},
        ),
    })

    # -- EPS subsystem (subsystem.eps.basic.v1): battery + solar + PDU energy
    #    balance.  Excitation via ``load_profile_w`` (40/60 W square wave) +
    #    ``shadow_profile`` (eclipse cycling).
    eps_params = {
        "battery_capacity_wh": 160.0,
        "initial_soc": 0.7,
        "solar_array_max_power_w": 120.0,
        "solar_panel_count": 1,
        "solar_array_efficiency": 1.0,
        "pdu_efficiency": 0.97,
        "pdu_bus_max_w": 90.0,
        "bus_load_power_w": 25.0,
        "payload_load_power_w": 30.0,
        "adcs_load_power_w": 8.0,
        "comm_load_power_w": 6.0,
        "load_profile_w": _square_wave(601, 40.0, 60.0, 120),
        "shadow_profile": _thermal_shadow_profile(601, period_samples=100, eclipse_fraction=0.4),
    }
    eps_config = dict(eps_params)
    eps_excitation = {
        "profile": "square_wave_load_40_60W + orbit_shadow_eclipse",
        "load_half_period_samples": 120,
        "shadow_period_samples": 100,
        "shadow_eclipse_fraction": 0.4,
    }
    out.append({
        "name": "eps-nominal-long",
        "case_id": "EPS-NOM-600s",
        "capability": "subsystem.eps.basic.v1",
        "scenario": "nominal",
        "level": "subsystem",
        "duration_s": 600.0, "sample_s": 1.0,
        "excitation": eps_excitation,
        "canonical": _canonical(
            capability_id="subsystem.eps.basic.v1", target_name="eps", mode="nominal",
            duration_s=600.0, sample_s=1.0, params=eps_params, case_id="EPS-NOM-600s",
            level="subsystem", config=eps_config,
        ),
    })
    out.append({
        "name": "eps-sudden_capacity_loss-onset300-long",
        "case_id": "EPS-SCL-300",
        "capability": "subsystem.eps.basic.v1",
        "scenario": "fault:sudden_capacity_loss",
        "level": "subsystem",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": 300.0, "magnitude": 0.35},
        "excitation": eps_excitation,
        "canonical": _canonical(
            capability_id="subsystem.eps.basic.v1", target_name="eps", mode="fault",
            duration_s=600.0, sample_s=1.0, params=eps_params, case_id="EPS-SCL-300",
            level="subsystem", config=eps_config,
            faults=[_legacy_fault("f_eps_scl", "sudden_capacity_loss", 300.0, 0.35)],
        ),
    })
    out.append({
        "name": "eps-solar_panel_failure-onset300-long",
        "case_id": "EPS-SOLAR-FAIL-300",
        "capability": "subsystem.eps.basic.v1",
        "scenario": "fault:solar_panel_failure",
        "level": "subsystem",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "solar_panel_failure", "onset_time_s": 300.0, "magnitude": 0.8},
        "excitation": eps_excitation,
        "canonical": _canonical(
            capability_id="subsystem.eps.basic.v1", target_name="eps", mode="fault",
            duration_s=600.0, sample_s=1.0, params=eps_params, case_id="EPS-SOLAR-FAIL-300",
            level="subsystem", config=eps_config,
            faults=[{
                "id": "f_eps_sp", "target": "solar_panel", "target_type": "solar_panel", "effect": "solar_panel_failure",
                "start_s": 300.0, "magnitude": 0.8, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })

    # ==================== Loop 17: data expansion (severity x onset variants) ===
    # Increases per-mechanism samples and cross-object representatives for the
    # TEST mechanism classes (CAPACITY_LOSS / FROZEN / GAIN_LOSS).  All variants
    # follow the same physical-gate discipline as the loop-13 base scenarios.
    # ---------------------------------------------------------------------------

    # -- battery: more CAPACITY_LOSS / FROZEN severity x onset samples -----------
    for mag, onset in ((0.15, 2000.0), (0.30, 1000.0), (0.40, 5000.0)):
        cid = f"BATT-SCL-{int(onset)}-{mag}"
        out.append({
            "name": f"battery-sudden_capacity_loss-onset{int(onset)}-mag{mag}-long",
            "case_id": cid, "capability": "component.battery.v1",
            "scenario": "fault:sudden_capacity_loss",
            "duration_s": 6000.0, "sample_s": 1.0,
            "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": onset, "magnitude": mag},
            "canonical": _canonical(
                capability_id="component.battery.v1", target_name="battery", mode="fault",
                duration_s=6000.0, sample_s=1.0, params=batt_params, case_id=cid,
                config=batt_params,
                faults=[_legacy_fault(f"f_scl_{int(onset)}", "sudden_capacity_loss", onset, mag)],
            ),
        })
    out.append({
        "name": "battery-cycled-sudden_capacity_loss-onset1500-mag0.45-long",
        "case_id": "BATT-CYC-SCL-1500-0.45",
        "capability": "component.battery.v1",
        "scenario": "fault:sudden_capacity_loss",
        "duration_s": 6000.0, "sample_s": 1.0,
        "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": 1500.0, "magnitude": 0.45},
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="fault",
            duration_s=6000.0, sample_s=1.0, params=batt_cycled_params, case_id="BATT-CYC-SCL-1500-0.45",
            config=batt_cycled_config,
            faults=[_legacy_fault("f_scl_cyc_1500", "sudden_capacity_loss", 1500.0, 0.45)],
        ),
    })
    for onset in (3000.0, 5000.0):
        cid = f"BATT-OC-{int(onset)}"
        out.append({
            "name": f"battery-open_circuit-onset{int(onset)}-long",
            "case_id": cid, "capability": "component.battery.v1",
            "scenario": "fault:open_circuit",
            "duration_s": 6000.0, "sample_s": 1.0,
            "fault": {"fault_type": "open_circuit", "onset_time_s": onset, "magnitude": 1.0},
            "canonical": _canonical(
                capability_id="component.battery.v1", target_name="battery", mode="fault",
                duration_s=6000.0, sample_s=1.0, params=batt_params, case_id=cid,
                config=batt_params,
                faults=[_legacy_fault(f"f_oc_{int(onset)}", "open_circuit", onset, 1.0)],
            ),
        })
    for pct in (20.0, 40.0):
        cid = f"BATT-DEG-CL{int(pct)}"
        out.append({
            "name": f"battery-capacity_loss_pct-{int(pct)}-degradation-long",
            "case_id": cid, "capability": "component.battery.v1",
            "scenario": "degradation:capacity_loss_pct",
            "duration_s": 6000.0, "sample_s": 1.0,
            "degradation": {"capacity_loss_pct": pct},
            "canonical": _canonical(
                capability_id="component.battery.v1", target_name="battery", mode="degradation",
                duration_s=6000.0, sample_s=1.0, params=batt_params, case_id=cid,
                config=batt_params,
                legacy_degradations={"eps": {"battery": {"capacity_loss_pct": pct}}},
            ),
        })
    batt_nom2_params = dict(batt_params)
    batt_nom2_params["initial_soc"] = 0.70
    out.append({
        "name": "battery-nominal-soc0.70-long",
        "case_id": "BATT-NOM2-6000s",
        "capability": "component.battery.v1",
        "scenario": "nominal",
        "duration_s": 6000.0, "sample_s": 1.0,
        "canonical": _canonical(
            capability_id="component.battery.v1", target_name="battery", mode="nominal",
            duration_s=6000.0, sample_s=1.0, params=batt_nom2_params, case_id="BATT-NOM2-6000s",
            config=batt_nom2_params,
        ),
    })

    # -- reaction_wheel: more FROZEN / MOTOR_FAILURE / FRICTION samples ---------
    out.append({
        "name": "reaction_wheel-rw_jamming-onset400-long",
        "case_id": "RW-JAM-400",
        "capability": "component.reaction_wheel.v1",
        "scenario": "fault:rw_jamming",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "rw_jamming", "onset_time_s": 400.0, "magnitude": 1.0},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="fault",
            duration_s=600.0, sample_s=0.1, params=rw_params, case_id="RW-JAM-400", config=rw_config,
            faults=[{
                "id": "f_jam_400", "target": "rw_0", "target_type": "reaction_wheel", "effect": "rw_jamming",
                "start_s": 400.0, "magnitude": 1.0, "implementation": "auto",
                "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "reaction_wheel-cycled-rw_jamming-onset500-long",
        "case_id": "RW-CYC-JAM-500",
        "capability": "component.reaction_wheel.v1",
        "scenario": "fault:rw_jamming",
        "duration_s": 600.0, "sample_s": 0.1,
        "fault": {"fault_type": "rw_jamming", "onset_time_s": 500.0, "magnitude": 1.0},
        "canonical": _canonical(
            capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="fault",
            duration_s=600.0, sample_s=0.1, params=rw_cycled_params, case_id="RW-CYC-JAM-500", config=rw_cycled_config,
            faults=[{
                "id": "f_jam_cyc_500", "target": "rw_0", "target_type": "reaction_wheel", "effect": "rw_jamming",
                "start_s": 500.0, "magnitude": 1.0, "implementation": "auto",
                "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    for mag, onset in ((0.5, 200.0), (0.95, 450.0)):
        cid = f"RW-MOT-{int(onset)}-{mag}"
        out.append({
            "name": f"reaction_wheel-rw_motor_failure-onset{int(onset)}-mag{mag}-long",
            "case_id": cid, "capability": "component.reaction_wheel.v1",
            "scenario": "fault:rw_motor_failure",
            "duration_s": 600.0, "sample_s": 0.1,
            "fault": {"fault_type": "rw_motor_failure", "onset_time_s": onset, "magnitude": mag},
            "canonical": _canonical(
                capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="fault",
                duration_s=600.0, sample_s=0.1, params=rw_params, case_id=cid, config=rw_config,
                faults=[{
                    "id": f"f_mot_{int(onset)}", "target": "rw_0", "target_type": "reaction_wheel", "effect": "rw_motor_failure",
                    "start_s": onset, "magnitude": mag, "implementation": "auto",
                    "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
                }],
            ),
        })
    for pct in (25.0, 75.0):
        cid = f"RW-DEG-FRICT{int(pct)}"
        out.append({
            "name": f"reaction_wheel-friction_increase-{int(pct)}-degradation-long",
            "case_id": cid, "capability": "component.reaction_wheel.v1",
            "scenario": "degradation:friction_increase_pct",
            "duration_s": 600.0, "sample_s": 0.1,
            "degradation": {"friction_increase_pct": pct},
            "canonical": _canonical(
                capability_id="component.reaction_wheel.v1", target_name="reaction_wheel", mode="degradation",
                duration_s=600.0, sample_s=0.1, params=rw_params, case_id=cid, config=rw_config,
                legacy_degradations={"adcs": {"reaction_wheel": {"friction_increase_pct": pct}}},
            ),
        })

    # -- imu: more BIAS / DRIFT samples -----------------------------------------
    for sev, onset in ((0.15, 200.0), (0.45, 450.0)):
        cid = f"IMU-BS-{int(onset)}-{sev}"
        out.append({
            "name": f"imu-bias_step-onset{int(onset)}-sev{sev}-long",
            "case_id": cid, "capability": "component.imu.v1",
            "scenario": "fault:bias_step",
            "duration_s": 600.0, "sample_s": 0.1,
            "fault": {"fault_type": "bias_step", "onset_time_s": onset, "severity": sev},
            "canonical": _canonical(
                capability_id="component.imu.v1", target_name="imu", mode="fault",
                duration_s=600.0, sample_s=0.1, params=imu_params, case_id=cid,
                faults=[{
                    "id": f"f_bs_{int(onset)}", "target": "imu", "target_type": "imu", "effect": "bias_step",
                    "start_s": onset, "magnitude": sev, "implementation": "auto",
                    "delivery": "legacy_fault", "parameters": {}, "event_type": "fault",
                }],
            ),
        })
    for sev in (0.3, 0.7):
        cid = f"IMU-DEG-BDRIFT-{sev}"
        out.append({
            "name": f"imu-bias_drift-sev{sev}-degradation-long",
            "case_id": cid, "capability": "component.imu.v1",
            "scenario": "degradation:bias_drift",
            "duration_s": 600.0, "sample_s": 0.1,
            "degradation": {"severity": sev, "adapter_defaults": "rate_per_s=0.01 max_fraction=severity"},
            "canonical": _canonical(
                capability_id="component.imu.v1", target_name="imu", mode="degradation",
                duration_s=600.0, sample_s=0.1, params=imu_params, case_id=cid,
                degradations=[{
                    "id": f"d_bd_{sev}", "target": "imu", "target_type": "imu", "effect": "bias_drift",
                    "start_s": 0.0, "magnitude": sev, "implementation": "auto",
                    "parameters": {}, "event_type": "degradation",
                }],
            ),
        })

    # -- thermal: more THERMAL_FAILURE / THERMAL_RADIATOR samples ---------------
    for onset in (8000.0, 20000.0):
        cid = f"TH-CYC-HRF-{int(onset)}"
        out.append({
            "name": f"thermal-cycled-heating_or_rejection_failure-onset{int(onset)}-long",
            "case_id": cid, "capability": thermal_capability,
            "scenario": "fault:heating_or_rejection_failure",
            "level": "subsystem",
            "duration_s": thermal_duration_s, "sample_s": thermal_sample_s,
            "fault": {"fault_type": "heating_or_rejection_failure", "onset_time_s": onset, "magnitude": 1.0},
            "excitation": thermal_excitation,
            "canonical": _canonical(
                capability_id=thermal_capability, target_name="thermal", mode="fault",
                duration_s=thermal_duration_s, sample_s=thermal_sample_s, params=thermal_cycled_params,
                case_id=cid, level="subsystem", config=thermal_config,
                faults=[{
                    "id": f"f_hrf_{int(onset)}", "target": "thermal", "target_type": "thermal",
                    "effect": "heating_or_rejection_failure", "start_s": onset,
                    "magnitude": 1.0, "implementation": "auto", "delivery": "modifier",
                    "parameters": {}, "event_type": "fault",
                }],
            ),
        })
    out.append({
        "name": "thermal-cycled-radiator_degradation-factor0.5-long",
        "case_id": "TH-CYC-DEG-RAD0.5",
        "capability": thermal_capability,
        "scenario": "degradation:radiator_degradation_factor",
        "level": "subsystem",
        "duration_s": thermal_duration_s, "sample_s": thermal_sample_s,
        "degradation": {"radiator_degradation_factor": 0.5},
        "excitation": thermal_excitation,
        "canonical": _canonical(
            capability_id=thermal_capability, target_name="thermal", mode="degradation",
            duration_s=thermal_duration_s, sample_s=thermal_sample_s, params=thermal_cycled_params,
            case_id="TH-CYC-DEG-RAD0.5", level="subsystem", config=thermal_config,
            degradations=[{
                "id": "d_rad_deg_05", "target": "thermal", "target_type": "thermal",
                "effect": "radiator_degradation_factor", "start_s": 0.0,
                "magnitude": 0.5, "implementation": "auto", "delivery": "modifier",
                "parameters": {"factor": 0.5}, "event_type": "degradation",
            }],
        ),
    })

    # -- eps (TRAIN cross-object reps for CAPACITY_LOSS / GAIN_LOSS) ------------
    out.append({
        "name": "eps-sudden_capacity_loss-onset200-mag0.5-long",
        "case_id": "EPS-SCL-200-0.5",
        "capability": "subsystem.eps.basic.v1",
        "scenario": "fault:sudden_capacity_loss",
        "level": "subsystem",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "sudden_capacity_loss", "onset_time_s": 200.0, "magnitude": 0.5},
        "excitation": eps_excitation,
        "canonical": _canonical(
            capability_id="subsystem.eps.basic.v1", target_name="eps", mode="fault",
            duration_s=600.0, sample_s=1.0, params=eps_params, case_id="EPS-SCL-200-0.5",
            level="subsystem", config=eps_config,
            faults=[_legacy_fault("f_eps_scl_05", "sudden_capacity_loss", 200.0, 0.5)],
        ),
    })
    out.append({
        "name": "eps-solar_panel_failure-onset150-mag0.5-long",
        "case_id": "EPS-SOLAR-FAIL-150-0.5",
        "capability": "subsystem.eps.basic.v1",
        "scenario": "fault:solar_panel_failure",
        "level": "subsystem",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "solar_panel_failure", "onset_time_s": 150.0, "magnitude": 0.5},
        "excitation": eps_excitation,
        "canonical": _canonical(
            capability_id="subsystem.eps.basic.v1", target_name="eps", mode="fault",
            duration_s=600.0, sample_s=1.0, params=eps_params, case_id="EPS-SOLAR-FAIL-150-0.5",
            level="subsystem", config=eps_config,
            faults=[{
                "id": "f_eps_sp_05", "target": "solar_panel", "target_type": "solar_panel", "effect": "solar_panel_failure",
                "start_s": 150.0, "magnitude": 0.5, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })

    # -- data_queue / onboard_storage (VALIDATION cross-object reps) ------------
    out.append({
        "name": "data_queue-overflow-onset150-mag0.25-long",
        "case_id": "DQ-OVF-150-0.25",
        "capability": "component.data_queue.v1",
        "scenario": "fault:overflow",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "overflow", "onset_time_s": 150.0, "magnitude": 0.25},
        "excitation": {"profile": "square_wave_generated_20_80kbps", "half_period_samples": 50, "downlink_bps": 30000.0},
        "canonical": _canonical(
            capability_id="component.data_queue.v1", target_name="data_queue", mode="fault",
            duration_s=600.0, sample_s=1.0, params=dq_params, case_id="DQ-OVF-150-0.25", config=dq_config,
            faults=[{
                "id": "f_dq_ovf_025", "target": "data_queue", "target_type": "data_queue", "effect": "overflow",
                "start_s": 150.0, "magnitude": 0.25, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "data_queue-throughput_decay-sev0.3-degradation-long",
        "case_id": "DQ-DEG-TDECAY-0.3",
        "capability": "component.data_queue.v1",
        "scenario": "degradation:throughput_decay",
        "duration_s": 600.0, "sample_s": 1.0,
        "degradation": {"severity": 0.3},
        "excitation": {"profile": "square_wave_generated_20_80kbps", "half_period_samples": 50, "downlink_bps": 30000.0},
        "canonical": _canonical(
            capability_id="component.data_queue.v1", target_name="data_queue", mode="degradation",
            duration_s=600.0, sample_s=1.0, params=dq_params, case_id="DQ-DEG-TDECAY-0.3", config=dq_config,
            degradations=[{
                "id": "d_dq_td_03", "target": "data_queue", "target_type": "data_queue", "effect": "throughput_decay",
                "start_s": 0.0, "magnitude": 0.3, "implementation": "auto", "delivery": "modifier",
                "parameters": {}, "event_type": "degradation",
            }],
        ),
    })
    out.append({
        "name": "onboard_storage-capacity_loss-sev0.15-degradation-long",
        "case_id": "OS-DEG-CAPLOSS-0.15",
        "capability": "component.onboard_storage.v1",
        "scenario": "degradation:capacity_loss",
        "duration_s": 600.0, "sample_s": 1.0,
        "degradation": {"severity": 0.15, "adapter_defaults": "rate_per_s=0.01 max_fraction=severity"},
        "excitation": {"profile": "square_wave_generated_30_80kbps + downlink_10_40kbps"},
        "canonical": _canonical(
            capability_id="component.onboard_storage.v1", target_name="onboard_storage", mode="degradation",
            duration_s=600.0, sample_s=1.0, params=os_params, case_id="OS-DEG-CAPLOSS-0.15", config=os_config,
            degradations=[{
                "id": "d_os_cl_015", "target": "onboard_storage", "target_type": "onboard_storage", "effect": "capacity_loss",
                "start_s": 0.0, "magnitude": 0.15, "implementation": "auto", "delivery": "modifier",
                "parameters": {}, "event_type": "degradation",
            }],
        ),
    })
    out.append({
        "name": "onboard_storage-read_only-onset200-long",
        "case_id": "OS-RO-200",
        "capability": "component.onboard_storage.v1",
        "scenario": "fault:read_only",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "read_only", "onset_time_s": 200.0, "magnitude": 1.0},
        "excitation": {"profile": "square_wave_generated_30_80kbps + downlink_10_40kbps"},
        "canonical": _canonical(
            capability_id="component.onboard_storage.v1", target_name="onboard_storage", mode="fault",
            duration_s=600.0, sample_s=1.0, params=os_params, case_id="OS-RO-200", config=os_config,
            faults=[{
                "id": "f_os_ro_200", "target": "onboard_storage", "target_type": "onboard_storage", "effect": "read_only",
                "start_s": 200.0, "magnitude": 1.0, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })

    # -- mtb / solar_panel (whole-object TEST probes, more samples) -------------
    out.append({
        "name": "mtb-mtb_dipole_capacity_loss-ratio0.8-degradation-long",
        "case_id": "MTB-DEG-CAPLOSS-0.8",
        "capability": "component.mtb.v1",
        "scenario": "degradation:mtb_dipole_capacity_loss",
        "duration_s": 1200.0, "sample_s": 1.0,
        "degradation": {"remaining_capacity_ratio": 0.8},
        "excitation": {"profile": "square_wave_dipole_x_0_0.1_Am2", "half_period_samples": 50, "y_cmd_am2": 0.05},
        "canonical": _canonical(
            capability_id="component.mtb.v1", target_name="mtb", mode="degradation",
            duration_s=1200.0, sample_s=1.0, params=mtb_params, case_id="MTB-DEG-CAPLOSS-0.8", config=mtb_config,
            degradations=[{
                "id": "d_mtb_cap_08", "target": "mtb", "target_type": "mtb", "effect": "mtb_dipole_capacity_loss",
                "start_s": 0.0, "magnitude": 0.2, "implementation": "auto", "delivery": "modifier",
                "parameters": {"remaining_capacity_ratio": 0.8}, "event_type": "degradation",
            }],
        ),
    })
    out.append({
        "name": "mtb-mtb_coil_open-onset500-long",
        "case_id": "MTB-COIL-500",
        "capability": "component.mtb.v1",
        "scenario": "fault:mtb_coil_open",
        "duration_s": 1200.0, "sample_s": 1.0,
        "fault": {"fault_type": "mtb_coil_open", "onset_time_s": 500.0, "magnitude": 1.0, "axis_index": 0},
        "excitation": {"profile": "square_wave_dipole_x_0_0.1_Am2", "half_period_samples": 50, "y_cmd_am2": 0.05},
        "canonical": _canonical(
            capability_id="component.mtb.v1", target_name="mtb", mode="fault",
            duration_s=1200.0, sample_s=1.0, params=mtb_params, case_id="MTB-COIL-500", config=mtb_config,
            faults=[{
                "id": "f_mtb_coil_500", "target": "mtb", "target_type": "mtb", "effect": "mtb_coil_open",
                "start_s": 500.0, "magnitude": 1.0, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {"axis_index": 0}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "solar_panel-solar_panel_failure-onset150-mag0.6-long",
        "case_id": "SP-FAIL-150-0.6",
        "capability": "component.solar_panel.v1",
        "scenario": "fault:solar_panel_failure",
        "duration_s": 600.0, "sample_s": 1.0,
        "fault": {"fault_type": "solar_panel_failure", "onset_time_s": 150.0, "magnitude": 0.6},
        "excitation": {"profile": "orbit_shadow_eclipse", "period_samples": 60, "eclipse_fraction": 0.4},
        "canonical": _canonical(
            capability_id="component.solar_panel.v1", target_name="solar_panel", mode="fault",
            duration_s=600.0, sample_s=1.0, params=sp_params, case_id="SP-FAIL-150-0.6", config=sp_config,
            faults=[{
                "id": "f_sp_fail_06", "target": "solar_panel", "target_type": "solar_panel", "effect": "solar_panel_failure",
                "start_s": 150.0, "magnitude": 0.6, "implementation": "auto", "delivery": "legacy_fault",
                "parameters": {}, "event_type": "fault",
            }],
        ),
    })
    out.append({
        "name": "solar_panel-efficiency_loss-15-degradation-long",
        "case_id": "SP-DEG-EFF-15",
        "capability": "component.solar_panel.v1",
        "scenario": "degradation:efficiency_loss_pct",
        "duration_s": 600.0, "sample_s": 1.0,
        "degradation": {"efficiency_loss_pct": 15.0},
        "excitation": {"profile": "orbit_shadow_eclipse", "period_samples": 60, "eclipse_fraction": 0.4},
        "canonical": _canonical(
            capability_id="component.solar_panel.v1", target_name="solar_panel", mode="degradation",
            duration_s=600.0, sample_s=1.0, params=sp_params, case_id="SP-DEG-EFF-15", config=sp_config,
            legacy_degradations={"eps": {"solar_panel": {"efficiency_loss_pct": 15.0}}},
        ),
    })

    return out


def _adapter_for(capability_id: str) -> Any:
    if capability_id == "component.battery.v1":
        from sat_sim.adapters.component_battery import BatteryAdapter
        return BatteryAdapter()
    if capability_id == "component.imu.v1":
        from sat_sim.adapters.component_source_native_misc import ImuAdapter
        return ImuAdapter()
    if capability_id == "component.reaction_wheel.v1":
        from sat_sim.adapters.component_reaction_wheel import ReactionWheelAdapter
        return ReactionWheelAdapter()
    if capability_id == "subsystem.thermal.basic_lumped.v1":
        from sat_sim.adapters.subsystem_thermal_basic_lumped import ThermalBasicLumpedAdapter
        return ThermalBasicLumpedAdapter()
    if capability_id == "component.mtb.v1":
        from sat_sim.adapters.component_mtb import MtbAdapter
        return MtbAdapter()
    if capability_id == "component.data_queue.v1":
        from sat_sim.adapters.component_data_queue import DataQueueAdapter
        return DataQueueAdapter()
    if capability_id == "component.onboard_storage.v1":
        from sat_sim.adapters.component_onboard_storage import OnboardStorageAdapter
        return OnboardStorageAdapter()
    if capability_id == "component.solar_panel.v1":
        from sat_sim.adapters.component_solar_panel import SolarPanelAdapter
        return SolarPanelAdapter()
    if capability_id == "subsystem.eps.basic.v1":
        from sat_sim.adapters.subsystem_eps_basic import EpsBasicAdapter
        return EpsBasicAdapter()
    raise ValueError(f"no adapter for {capability_id}")


def _sanity_check(rows: list[dict[str, Any]], scenario: dict[str, Any]) -> list[str]:
    """Physical sanity checks. Returns list of violations (empty = PASS)."""
    violations: list[str] = []
    times = [float(r["time_s"]) for r in rows]
    # time monotonic (non-decreasing) and no duplicate stamps
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            violations.append(f"time non-monotonic at index {i}: {times[i-1]} -> {times[i]}")
    if times[0] != 0.0:
        violations.append(f"time does not start at 0: {times[0]}")
    # finite numeric values
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                violations.append(f"non-finite {k} at t={r.get('time_s')}: {v}")
    cap = scenario["capability"]
    if cap == "component.battery.v1":
        for r in rows:
            soc, storage, effcap = r["eps.battery.soc"], r["eps.battery.storage_wh"], r["eps.battery.effective_capacity_wh"]
            if not (-1e-9 <= soc <= 1.0 + 1e-9):
                violations.append(f"battery soc out of [0,1] at t={r['time_s']}: {soc}")
            if storage < -1e-9 or storage > effcap + 1e-6:
                violations.append(f"battery storage_wh out of [0,eff_cap] at t={r['time_s']}: {storage}/{effcap}")
            if effcap > BATT_CAP + 1e-6:
                violations.append(f"battery effective capacity exceeds nominal: {effcap} > {BATT_CAP}")
    if cap == "component.reaction_wheel.v1":
        for r in rows:
            speed = abs(float(r.get("adcs.reaction_wheel.speed_rad_s_0", 0.0)))
            max_sp = float(r.get("adcs.reaction_wheel.effective_max_speed_rad_s_0", 0.0)) or 1e-15
            if speed > max_sp + 1e-6:
                violations.append(f"rw speed {speed} exceeds effective max {max_sp} at t={r['time_s']}")
            applied = abs(float(r.get("adcs.reaction_wheel.applied_torque_nm_0", 0.0)))
            max_tq = float(r.get("adcs.reaction_wheel.effective_max_torque_nm_0", 0.0)) or 1e-15
            if applied > max_tq + 1e-9:
                violations.append(f"rw applied torque {applied} exceeds effective max {max_tq} at t={r['time_s']}")
    if cap == "subsystem.thermal.basic_lumped.v1":
        for r in rows:
            for key in ("thermal.node.bus_temp_c", "thermal.node.battery_temp_c"):
                value = float(r[key])
                if not math.isfinite(value):
                    violations.append(f"non-finite {key} at t={r['time_s']}: {value}")
                if value < -100.0 or value > 500.0:
                    violations.append(f"{key} out of physical range [-100,500] C at t={r['time_s']}: {value}")
            shadow = float(r.get("environment.shadow_factor", 1.0))
            if not (-1e-9 <= shadow <= 1.0 + 1e-9):
                violations.append(f"shadow_factor out of [0,1] at t={r['time_s']}: {shadow}")
            for key in ("thermal.heat.internal_power_w", "thermal.heat.solar_input_w", "thermal.heater.power_w", "thermal.radiator.heat_reject_w"):
                value = float(r[key])
                if not math.isfinite(value) or value < -1e-9:
                    violations.append(f"{key} non-finite or negative at t={r['time_s']}: {value}")
    if cap == "component.mtb.v1":
        field_norm = math.sqrt(sum(float(r.get(f"adcs.mtb.magnetic_field_t_{i}", 0.0)) ** 2 for i in range(3)))
        for r in rows:
            for axis in range(3):
                actual = abs(float(r.get(f"adcs.mtb.actual_dipole_am2_{axis}", 0.0)))
                limit = float(r.get(f"adcs.mtb.effective_dipole_limit_am2_{axis}", 0.0))
                if actual > limit + 1e-9:
                    violations.append(f"mtb actual dipole axis {axis} {actual} exceeds effective limit {limit} at t={r['time_s']}")
                command = abs(float(r.get(f"adcs.mtb.command_dipole_am2_{axis}", 0.0)))
                if command > 0.2 + 1e-9:
                    violations.append(f"mtb command dipole axis {axis} {command} exceeds nominal limit 0.2 at t={r['time_s']}")
            torque_norm = float(r["adcs.mtb.torque_norm_nm"])
            # Physical bound: |m x B| <= sqrt(3) * max_dipole * |B|.
            max_limit = max(float(r.get(f"adcs.mtb.effective_dipole_limit_am2_{axis}", 0.0)) for axis in range(3))
            bound = math.sqrt(3.0) * max_limit * field_norm + 1e-12
            if torque_norm > bound + 1e-9:
                violations.append(f"mtb torque norm {torque_norm} exceeds physical bound {bound} at t={r['time_s']}")
            saturated = bool(r["adcs.mtb.saturation_flag"])
            if saturated and not any(
                abs(float(r.get(f"adcs.mtb.actual_dipole_am2_{axis}", 0.0))) >= float(r.get(f"adcs.mtb.effective_dipole_limit_am2_{axis}", 0.0)) - 1e-9
                for axis in range(3)
            ):
                violations.append(f"mtb saturation_flag True with no dipole at limit at t={r['time_s']}")
    if cap in ("component.data_queue.v1", "component.onboard_storage.v1"):
        prefix = "comm.data_queue" if cap == "component.data_queue.v1" else "comm.storage"
        stored_key = f"{prefix}.queue_bits" if cap == "component.data_queue.v1" else f"{prefix}.stored_bits"
        gen_key = f"{prefix}.generated_bps"
        dlnk_key = f"{prefix}.downlink_bps"
        eff_key = f"{prefix}.effective_capacity_bits"
        for index, r in enumerate(rows):
            stored = float(r[stored_key])
            eff_cap = float(r[eff_key])
            fill = float(r[f"{prefix}.fill_ratio"])
            if stored < -1e-9 or stored > eff_cap + 1e-6:
                violations.append(f"queue stored {stored} out of [0, eff_cap {eff_cap}] at t={r['time_s']}")
            if not (-1e-9 <= fill <= 1.0 + 1e-9):
                violations.append(f"queue fill_ratio out of [0,1] at t={r['time_s']}: {fill}")
            for key in (gen_key, dlnk_key):
                value = float(r[key])
                if not math.isfinite(value) or value < -1e-9:
                    violations.append(f"{key} non-finite or negative at t={r['time_s']}: {value}")
            # Conservation: the queue is a bounded integrator
            #   next = min(max(0, stored + (generated - downlink)*dt), capacity)
            # with the excess dropped/overflowed.  Downlink is clipped to the
            # available data when the queue runs dry, so the commanded downlink
            # is only an upper bound.
            if index > 0:
                prev = rows[index - 1]
                dt = float(r["time_s"]) - float(prev["time_s"])
                if dt > 0:
                    gen = float(r[gen_key])
                    dlnk = float(r[dlnk_key])
                    eff_cap = float(r[eff_key])
                    prev_stored = float(prev[stored_key])
                    expected = min(max(0.0, prev_stored + (gen - dlnk) * dt), eff_cap)
                    if abs(stored - expected) > 1.0 + 1e-6:
                        violations.append(
                            f"queue conservation violated at t={r['time_s']}: stored={stored} expected={expected}"
                        )
    if cap == "component.solar_panel.v1":
        for r in rows:
            array_power = float(r["eps.solar.array_power_w"])
            nominal_max = float(r["eps.solar.nominal_max_power_w"])
            if array_power < -1e-9 or array_power > nominal_max + 1e-6:
                violations.append(f"solar array_power {array_power} out of [0, nominal_max {nominal_max}] at t={r['time_s']}")
            panel_power = float(r["eps.solar.panel_power_w"])
            panel_count = int(r["eps.solar.panel_count"])
            if abs(array_power - panel_count * panel_power) > 1e-6:
                violations.append(f"solar array_power != panel_count*panel_power at t={r['time_s']}")
            for key in ("eps.solar.efficiency", "eps.solar.incidence_cos", "eps.solar.shadow_factor", "eps.solar.fault_multiplier"):
                value = float(r[key])
                if not (-1e-9 <= value <= 1.0 + 1e-9):
                    violations.append(f"solar {key} out of [0,1] at t={r['time_s']}: {value}")
    if cap == "subsystem.eps.basic.v1":
        for r in rows:
            soc = float(r["eps.battery.soc"])
            if not (-1e-9 <= soc <= 1.0 + 1e-9):
                violations.append(f"eps soc out of [0,1] at t={r['time_s']}: {soc}")
            storage = float(r["eps.battery.storage_wh"])
            eff_cap = float(r["eps.battery.effective_capacity_wh"])
            if storage < -1e-9 or storage > eff_cap + 1e-6:
                violations.append(f"eps storage_wh out of [0, eff_cap] at t={r['time_s']}: {storage}/{eff_cap}")
            array_power = float(r["eps.solar.array_power_w"])
            if array_power < -1e-9 or array_power > 120.0 + 1e-6:
                violations.append(f"eps array_power {array_power} out of [0, 120] at t={r['time_s']}")
            for key in ("environment.shadow_factor", "eps.solar.incidence_cos", "eps.solar.fault_multiplier"):
                value = float(r[key])
                if not (-1e-9 <= value <= 1.0 + 1e-9):
                    violations.append(f"eps {key} out of [0,1] at t={r['time_s']}: {value}")
            served = float(r["eps.loads.served_power_w"])
            requested = float(r["eps.loads.requested_power_w"])
            if served < -1e-9 or served > requested + 1e-6:
                violations.append(f"eps served_power {served} > requested {requested} at t={r['time_s']}")
            unserved = float(r["eps.loads.unserved_power_w"])
            if unserved < -1e-9:
                violations.append(f"eps unserved_power negative at t={r['time_s']}: {unserved}")
    return violations


def _row_to_csv_flat(rows: list[dict[str, Any]]) -> None:
    # placeholder replaced below; each scenario writes its own file
    _ = rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    ok_count = 0
    for sc in scenarios():
        canonical = sc["canonical"]
        from sat_sim.task_compiler import compile_task_spec
        from sat_sim.task_models import to_runtime_task_spec

        # NOTE: compile-level validation (validate=True) is skipped because the
        # platform's global KNOWN_FAULT_TYPES catalog lags the adapter-supported
        # effects for the loop-13 objects (e.g. component.mtb.v1 supports
        # mtb_coil_open but the catalog predates it).  The authoritative contract
        # check is adapter.validate(rt) below, which validates per-capability
        # fault/degradation effects and parameter ranges.  The compiled dataclass
        # is discarded; only the runtime task spec drives the adapter.
        try:
            compiled = compile_task_spec(canonical, validate=False)
            _ = compiled
            rt = to_runtime_task_spec(canonical)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"ERROR compile {sc['name']}: {exc}")
            continue

        adapter = _adapter_for(sc["capability"])
        issues = [i for i in adapter.validate(rt) if i.severity == "error"]
        if issues:
            print(f"ERROR validate {sc['name']}: {[i.message for i in issues]}")
            continue
        result = adapter.run(rt)
        rows = list(result.trace_rows)
        violations = _sanity_check(rows, sc)
        status = "PASS" if not violations else "FAIL"
        if status == "PASS":
            ok_count += 1

        filename = f"{sc['name']}.csv"
        out_path = OUT_DIR / filename
        columns = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        label_cols = [c for c in columns if c.startswith("label.")]
        rec = {
            "file": filename,
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "case_id": sc["case_id"],
            "capability": sc["capability"],
            "scenario": sc["scenario"],
            "mode": rt["target"].get("mode"),
            "rows": len(rows),
            "duration_s": sc["duration_s"],
            "sample_s": sc["sample_s"],
            "columns": columns,
            "label_columns": label_cols,
            "physical_check": status,
            "violations": violations,
            "generation_params": {
                "fault": sc.get("fault"),
                "degradation": sc.get("degradation"),
                "excitation": sc.get("excitation"),
            },
            "labels": result.labels,
        }
        records.append(rec)
        print(f"{status}  {filename:60s} rows={len(rows):5d} dur={sc['duration_s']:6.0f}s sample={sc['sample_s']}")

    manifest = {
        "schema_version": "astrograph-long-traces.v1",
        "generator": "scripts/generate_long_traces.py",
        "generator_notes": "Runs Sat-Agent component/subsystem adapters via Sat-Agent venv (3.12.12); deterministic models, no RNG in adapter path. Loop 13 added MTB / data_queue / onboard_storage / solar_panel / EPS subsystem (15 traces). Compile-level fault-catalog validation is deferred to adapter.validate() because the platform's global KNOWN_FAULT_TYPES catalog predates loop-13 adapter-supported effects; per-capability adapter validation is authoritative.",
        "source": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "trace_count": len(records),
        "passed_count": ok_count,
        "traces": records,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest: {manifest_path}")
    print(f"traces: {ok_count}/{len(records)} passed physical checks")
    return 0 if ok_count == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
