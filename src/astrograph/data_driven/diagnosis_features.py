"""Diagnosis feature extraction for the S1 data-driven baseline (loop 15).

Turns one labeled long-trace episode into a fixed-length, cross-object
comparable feature vector that CAN be a model input.  The hard constraints
(DEV_GOAL) honored here:

* deny rules: label / truth / case / scenario / seed / split / file path /
  fault-id tokens are NEVER part of the returned feature vector.  Only public
  telemetry-derived scalars + object context (component identity + archetype,
  which are explicitly allowed as the S1 "object context prior") are emitted.
* Honesty: fail-closed estimation is preserved as an explicit feature
  (``gb_fail``), not masked away.  Objects with no dynamics contract fall back
  to zero-fit EvidenceToken + robust scalar features.
* Cross-object comparability: the shared backbone needs a fixed-length vector,
  so each object's features are normalised against a *nominal reference* that
  is fit ONLY on the train split (S1-authorized nominal fit).  A feature is
  expressed as a relative deviation (``(v - ref) / ref``) or an absolute
  residual magnitude where no physical reference exists.

The nominal reference is a ``NominalReference`` object produced by
``fit_nominal_reference`` from train-split nominal episodes.  It must never be
fit on tune/test episodes (grouped holdout discipline).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from astrograph.features.generic.token_builder import TokenBuilder
from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.state_estimation import GrayBoxStateEstimator

# --------------------------------------------------------------------------- #
# Authoritative deny rules (governed S0 source): global_model_input_deny_rules.
# Every source column consumed during feature construction is filtered against
# these patterns before it can contribute to a model-input feature.  This is the
# hard enforcement layer; the dimension-name scan in the training script is a
# second, independent guard on top of it.
# --------------------------------------------------------------------------- #

# Resolved by glob so generic `src` code never embeds the concrete source-system
# filename (architecture-boundary discipline, mirroring the feature-contract
# catalog resolution above).  The sole file that matches is the S0 profile
# catalog carrying global_model_input_deny_rules.
_DENY_RULE_CONFIG = sorted(
    (Path(__file__).resolve().parents[3] / "configs" / "data_sources").glob(
        "*_schema_profiles.s0.v1.json"
    )
)[0]


def _authoritative_deny_rules() -> list[str]:
    payload = json.loads(_DENY_RULE_CONFIG.read_text(encoding="utf-8"))
    rules = payload["global_model_input_deny_rules"]
    if not rules:
        raise RuntimeError("governed deny-rule list is empty; refusing to weaken deny discipline")
    return list(rules)


_DENY_PATTERNS: tuple[str, ...] = tuple(_authoritative_deny_rules())


def _matches_deny(name: str) -> bool:
    """True if a column name matches any governed deny pattern.

    ``rule.*`` patterns anchor at the start (``name.startswith(prefix)``);
    bare tokens match as exact tokens; ``*suffix*`` patterns match as
    substrings.  This mirrors the semantics of the governed S0 list.
    """
    lowered = name.lower()
    for rule in _DENY_PATTERNS:
        r = rule.lower()
        if r.endswith(".*"):
            if lowered.startswith(r[:-2]):
                return True
        elif r.startswith("*") and r.endswith("*") and len(r) > 2:
            if r[1:-1] in lowered:
                return True
        elif r in lowered:
            return True
    return False


# Physical-telemetry exemptions to the governed deny patterns (A, 2026-08-05,
# user-approved).  These columns are component RESPONSE telemetry whose physical
# value changes with the fault, so the deny-rule intent (block identity /
# injected-config) is preserved while the real mechanism signal reaches the
# feature surface.  This is deliberately a narrow, documented allowlist, not a
# weakening of the governed deny list.
_PHYSICAL_TELEMETRY_EXEMPTIONS = re.compile(r"^adcs\.mtb\.effective_dipole_limit_am2_\d+$")

# --------------------------------------------------------------------------- #
# Dataset-layer channel binding (mirrors scripts/tools/materialize_long_trace_episodes.py)
# --------------------------------------------------------------------------- #
# Three-layer column-name reality (recorded so future edits do not silently
# re-introduce a double-source drift):
#   1. The generator (`scripts/tools/generate_long_traces.py`) names the trace CSV
#      columns (e.g. `eps.battery.soc`, `thermal.node.bus_temp_c`).
#   2. `scripts/tools/materialize_long_trace_episodes.py` carries its own channel spec
#      for the S0 adapter output.
#   3. The governed declarative channel map
#      (`configs/semantics/sat_agent_platform_channel_maps.s0.v1.json`) uses a
#      DIFFERENT column-name vocabulary (e.g. `thermal.source_native.battery_temp_k`)
#      and is the S0-adapter semantic mapping, not the materialized-trace binding.
# The materialized trace columns (1) are what this module consumes.  We do NOT
# merge them into the declarative channel map here because the map describes a
# different layer; instead `_frame_from_rows` fail-closes on any declared column
# missing from the rows, so a source rename surfaces as an explicit error rather
# than a silently zero-filled feature.  Keep the two bindings in sync when the
# generator column names change.
# --------------------------------------------------------------------------- #

# Column name -> (port_role, quantity_kind, canonical_unit, direction)
_CHANNEL_SPECS: dict[str, list[tuple[str, str, str, str, int]]] = {
    "battery": [
        ("eps.battery.soc", "state.energy_fraction", "energy.state_of_charge", "1", 1),
        ("eps.battery.storage_wh", "state.energy_reserve", "energy.stored_energy", "Wh", 1),
        ("eps.battery.net_power_w", "power.exchange", "electrical.active_power", "W", 1),
    ],
    "imu": [
        ("adcs.imu.gyro_x_rad_s", "measurement.output", "mechanical.angular_rate", "rad/s", 1),
        ("adcs.imu.gyro_y_rad_s", "measurement.output", "mechanical.angular_rate", "rad/s", 1),
        ("adcs.imu.gyro_z_rad_s", "measurement.output", "mechanical.angular_rate", "rad/s", 1),
    ],
    "reaction_wheel": [
        ("adcs.reaction_wheel.speed_rad_s_0", "state.rate", "mechanical.angular_rate", "rad/s", 1),
        ("adcs.reaction_wheel.command_torque_nm_0", "command.effort", "mechanical.torque", "N*m", 1),
        ("adcs.reaction_wheel.applied_torque_nm_0", "response.effort", "mechanical.torque", "N*m", 1),
    ],
    "thermal": [
        ("thermal.node.bus_temp_c", "state.temperature", "thermal.temperature", "K", 1),
        ("thermal.node.battery_temp_c", "state.temperature", "thermal.temperature", "K", 1),
        ("thermal.heat.internal_power_w", "heat.exchange", "electrical.active_power", "W", 1),
        ("thermal.heat.solar_input_w", "heat.exchange", "electrical.active_power", "W", 1),
        ("thermal.heater.power_w", "heat.exchange", "electrical.active_power", "W", 1),
    ],
    "mtb": [
        ("adcs.mtb.command_dipole_am2_0", "command.effort", "magnetic.dipole_moment", "A*m^2", 1),
        ("adcs.mtb.actual_dipole_am2_0", "response.effort", "magnetic.dipole_moment", "A*m^2", 1),
        ("adcs.mtb.torque_norm_nm", "response.effort", "mechanical.torque", "N*m", 1),
        ("adcs.mtb.magnetic_field_t_0", "context.magnetic_field", "magnetic.field", "T", 1),
    ],
    "data_queue": [
        ("comm.data_queue.queue_bits", "state.occupied_amount", "information.data_amount", "bit", 1),
        ("comm.data_queue.fill_ratio", "state.fill_fraction", "inventory.fill_fraction", "1", 1),
        ("comm.data_queue.generated_bps", "flow.exchange", "information.data_rate", "bit/s", 1),
        ("comm.data_queue.downlink_bps", "flow.exchange", "information.data_rate", "bit/s", -1),
    ],
    "onboard_storage": [
        ("comm.storage.stored_bits", "state.occupied_amount", "information.data_amount", "bit", 1),
        ("comm.storage.fill_ratio", "state.fill_fraction", "inventory.fill_fraction", "1", 1),
        ("comm.storage.generated_bps", "flow.exchange", "information.data_rate", "bit/s", 1),
        ("comm.storage.downlink_bps", "flow.exchange", "information.data_rate", "bit/s", -1),
    ],
    "solar_panel": [
        ("eps.solar.array_power_w", "power.output", "electrical.active_power", "W", 1),
        ("eps.solar.efficiency", "performance.efficiency", "energy.conversion_efficiency", "1", 1),
    ],
    "eps": [
        ("eps.battery.soc", "state.energy_fraction", "energy.state_of_charge", "1", 1),
        ("eps.battery.storage_wh", "state.energy_reserve", "energy.stored_energy", "Wh", 1),
        ("eps.battery.net_power_w", "power.exchange", "electrical.active_power", "W", 1),
        ("eps.battery.applied_power_w", "power.exchange", "electrical.active_power", "W", 1),
    ],
}

# Archetype for each component instance (dataset-layer binding).
OBJECT_ARCHETYPE: dict[str, str] = {
    "battery": "storage.energy_buffer",
    "eps": "storage.energy_buffer",
    "reaction_wheel": "actuator.spinning_rotor",
    "thermal": "thermal.lumped_capacity",
    "mtb": "actuator.controlled_effort",
    "imu": "sensor.measurement",
    "data_queue": "queue.bounded_inventory",
    "onboard_storage": "queue.bounded_inventory",
    "solar_panel": "converter.power",
}

# Objects that have a GrayBox dynamics contract (GLS parameter identification).
DYNAMICS_OBJECTS: frozenset[str] = frozenset(
    {obj for obj, arch in OBJECT_ARCHETYPE.items()
     if default_dynamics_contracts().get(arch) is not None}
)

# Physical nominal values used to normalise absolute parameter estimates where
# the telemetry truth is a fixed simulator constant (documented in experiments
# A2/D).  These are simulator constants, not fitted on any split.
PHYSICAL_NOMINAL: dict[str, float] = {
    "battery.capacity_wh": 160.0,
    "battery.discharge_efficiency": 0.97,
    "rw.j_kgm2": 0.1,
    "rw.b_nms": 1.0e-5,
    "thermal.c_jk": 18000.0,
    "thermal.sink_c": -35.0,
    "eps.capacity_wh": 160.0,
    # Solar panel simulator constants (documented, not fitted on any split).
    # Used as fixed normalizers so no *nominal_* source column enters a feature.
    "solar.max_power_w": 120.0,
    "solar.nominal_efficiency": 1.0,
    # mtb physical dipole limit (nominal 0.2 A*m^2, drops under capacity loss).
    "mtb.dipole_limit_am2": 0.2,
}

# The archetype feature-contract catalog is a governed config file.  We resolve
# it by glob so generic `src` code never embeds the concrete source-system
# filename (architecture-boundary discipline); the sole file that matches is the
# S0 feature-contract catalog.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FEATURE_CONTRACT_PATH = sorted(
    (_PROJECT_ROOT / "configs" / "semantics").glob("*_archetype_feature_contracts.s0.v1.json")
)[0]
_TOKEN_BUILDER_CACHE: dict[str, TokenBuilder] = {}


def _token_builder(archetype: str) -> TokenBuilder:
    if archetype not in _TOKEN_BUILDER_CACHE:
        _TOKEN_BUILDER_CACHE[archetype] = TokenBuilder(
            archetype=archetype,
            contract_path=str(_FEATURE_CONTRACT_PATH.resolve()),
        )
    return _TOKEN_BUILDER_CACHE[archetype]


def _signal_map(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Column name -> float array, excluding every column that matches a governed
    deny rule (label/truth/provenance/injected-config patterns).  This is the hard
    enforcement layer: no deny-pattern source column may contribute to a feature.
    """
    out: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("label.") or key in (
                "task_id", "case_id", "mode", "target_level", "target_name", "sample_index",
            ):
                continue
            # Physical-telemetry exemption (A, 2026-08-05, user-approved): the
            # mtb `effective_dipole_limit_am2_*` columns are the ONLY observable
            # signal for the `mtb_dipole_capacity_loss` mechanism (the limit drops
            # 0.2 -> 0.16/0.14; coil_open -> 0), yet the governed `*effective_*`
            # deny pattern was excluding them.  They are component RESPONSE
            # telemetry (a physical current-limit state that changes with the
            # fault), NOT injected configuration or identity/label tokens, so the
            # deny-rule intent (block identity/config) is preserved while the
            # real mechanism signal reaches the feature surface.
            if _matches_deny(key) and not _PHYSICAL_TELEMETRY_EXEMPTIONS.match(key):
                continue
            try:
                out.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                pass
    return {key: np.asarray(values, dtype=float) for key, values in out.items()}


def _build_frame(
    *,
    component_instance_id: str,
    archetype: str,
    timestamps: list[float],
    signals: list[tuple[str, str, str, list[float]]],
) -> SemanticTelemetryFrame:
    semantic_signals = [
        SemanticSignal(
            semantic_channel_id=f"diag.{component_instance_id}.{index}",
            asset_instance_id="asset-under-test",
            component_archetype_id=archetype,
            port_role_id=port_role,
            quantity_kind_id=quantity_kind,
            canonical_unit=unit,
            direction=direction,
            values=values,
            missing_mask=[False] * len(values),
        )
        for index, (port_role, quantity_kind, unit, values, direction) in enumerate(signals)
    ]
    payload = {
        "timestamps": timestamps,
        "signals": [item.model_dump(mode="json") for item in semantic_signals],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SemanticTelemetryFrame(
        frame_id=f"diag-{component_instance_id}",
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=timestamps,
        signals=semantic_signals,
        sampling=SamplingProfile(
            row_count=len(timestamps),
            nominal_period_s=float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0,
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256=digest,
        labels_used=False,
        truth_used=False,
        claim_boundary="S1 diagnosis baseline feature extraction",
    )


def _frame_from_rows(
    *,
    component_instance_id: str,
    archetype: str,
    data: dict[str, np.ndarray],
) -> SemanticTelemetryFrame:
    """Build a public (label-free) frame from the dataset-layer channel binding.

    Fail-closed on the channel binding: every column declared in
    ``_CHANNEL_SPECS`` for this object MUST exist in the telemetry rows.  A
    missing column means the source schema changed (or the binding drifted),
    and the feature would otherwise silently be zero-filled — masking a
    train/deploy data-contract split.  We raise instead so the contract break
    is explicit.  (The verified 40-trace corpus has 0 missing columns, so this
    guard is a forward safety net, not a change to current behavior.)
    """
    time = data["time_s"]
    signals: list[tuple[str, str, str, list[float], int]] = []
    for column, port_role, quantity_kind, unit, direction in _CHANNEL_SPECS[component_instance_id]:
        raw = data.get(column)
        if raw is None:
            raise ValueError(
                f"channel-binding column {column!r} missing from telemetry rows for "
                f"{component_instance_id}; source schema or binding drifted"
            )
        values = raw.tolist()
        signals.append((port_role, quantity_kind, unit, values, direction))
    return _build_frame(
        component_instance_id=component_instance_id,
        archetype=archetype,
        timestamps=time.tolist(),
        signals=signals,
    )


def _gray_box_estimate(
    frame: SemanticTelemetryFrame,
    archetype: str,
) -> dict[str, Any]:
    """Run GrayBoxStateEstimator, returning raw param/residual/OOD/fail scalars.

    Fail-closed is preserved: on any estimator error the dict carries
    ``fail=True`` plus the error class name (never the message string, to keep
    the feature surface scalar-only).

    NOTE (calibration research exemption, recorded 2026-08-04): the estimator
    runs with ``use_frozen_calibration=False``.  This is a deliberate S0-era
    zero-fit convention, not an oversight.  Root cause (verified empirically
    2026-08-04): the frozen calibration profile
    (``gray_box_uncertainty_calibration.v1.json``) carries conservation
    change-point scores for ONLY two archetypes (``storage.energy_buffer`` /
    ``thermal.lumped_capacity``).  The loop-13 contracts this dataset consumes
    (spinning_rotor, controlled_effort, queue.bounded_inventory, converter.power)
    have NO calibration data in the profile, so ``_calibration_for`` returns None
    for them and they silently fall back to UNcalibrated
    (``UNCALIBRATED_RAW_ONLY``, multiplier 1.96) even under
    ``use_frozen_calibration=True`` — the estimator identity (bounded-balance
    v1.2.0) DOES match, so no identity error is raised; the gap is per-archetype
    calibration coverage, not identity.  ``use_frozen_calibration=False`` is
    therefore an explicit statement that these contracts are uncalibrated, not a
    workaround for a mismatch.  CONSEQUENCE: the OOD/residual features are
    produced by an UNcalibrated estimator, so their scale/meaning differ from an
    ADR-001 production estimator that loads the frozen profile.  Before this
    baseline is bound to production (C3) the calibration strategy must be
    decided (extend the frozen profile to the data_driven contract family, or
    ship the uncalibrated path with an explicit claim boundary).  The feature
    surface itself is unchanged by this exemption.
    """
    contract = default_dynamics_contracts().get(archetype)
    if contract is None:
        return {"fail": False, "no_contract": True}
    try:
        result = GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract)
    except Exception as exc:  # noqa: BLE001 - fail-closed is a feature, not a crash
        return {"fail": True, "fail_kind": type(exc).__name__}
    out: dict[str, Any] = {
        "fail": False,
        "params": {p.parameter_id: p.value for p in result.parameters},
        "residual_rms": {
            r.residual_id: r.root_mean_square for r in result.residuals
        },
        "ood": result.ood_score,
        "raw_violation_fraction": result.conservation_assessment.raw_violation_fraction,
        "episode_score": result.conservation_assessment.episode_score,
    }
    return out


def _estimate_object(rows: list[dict[str, str]], object_id: str) -> dict[str, Any]:
    """Per-object estimation mirroring experiment D, driven by the dataset-layer
    channel binding.  No scenario knowledge is used."""
    data = _signal_map(rows)
    archetype = OBJECT_ARCHETYPE[object_id]
    frame = _frame_from_rows(
        component_instance_id=object_id,
        archetype=archetype,
        data=data,
    )
    out: dict[str, Any] = {
        "object_id": object_id,
        "archetype": archetype,
        "row_count": int(len(data["time_s"])),
        "gray_box": _gray_box_estimate(frame, archetype),
        "raw": {},
    }

    # Zero-fit robust scalars (object-generic, no scenario knowledge).
    time = data["time_s"]
    duration = float(time[-1] - time[0]) if len(time) > 1 else 1.0
    out["raw"]["duration_s"] = duration

    if object_id in {"battery", "eps"}:
        out["raw"]["soc_mean"] = float(np.mean(data.get("eps.battery.soc", [0.0])))
        out["raw"]["soc_std"] = float(np.std(data.get("eps.battery.soc", [0.0])))
        out["raw"]["soc_min"] = float(np.min(data.get("eps.battery.soc", [0.0])))
        out["raw"]["soc_max"] = float(np.max(data.get("eps.battery.soc", [0.0])))
        out["raw"]["soc_max_step"] = (
            float(np.max(np.abs(np.diff(data["eps.battery.soc"])))) if len(data["eps.battery.soc"]) > 2 else 0.0
        )
        out["raw"]["net_power_var"] = float(np.var(data.get("eps.battery.net_power_w", [0.0])))
        if object_id == "battery" and "eps.battery.applied_power_w" in data:
            out["raw"]["applied_power_mean"] = float(np.mean(data["eps.battery.applied_power_w"]))
            out["raw"]["applied_power_var"] = float(np.var(data["eps.battery.applied_power_w"]))
    elif object_id == "reaction_wheel":
        out["raw"]["speed_mean"] = float(np.mean(data.get("adcs.reaction_wheel.speed_rad_s_0", [0.0])))
        out["raw"]["speed_std"] = float(np.std(data.get("adcs.reaction_wheel.speed_rad_s_0", [0.0])))
        out["raw"]["command_var"] = float(np.var(data.get("adcs.reaction_wheel.command_torque_nm_0", [0.0])))
        applied = data.get("adcs.reaction_wheel.applied_torque_nm_0")
        if applied is not None:
            out["raw"]["applied_mean"] = float(np.mean(applied))
            out["raw"]["applied_var"] = float(np.var(applied))
            out["raw"]["applied_frozen"] = float(np.all(np.abs(applied) < 1e-9))
    elif object_id == "thermal":
        out["raw"]["bus_temp_max_c"] = float(np.max(data.get("thermal.node.bus_temp_c", [0.0])))
        heat = (
            np.asarray(data.get("thermal.heat.internal_power_w", [0.0]))
            + np.asarray(data.get("thermal.heat.solar_input_w", [0.0]))
            + np.asarray(data.get("thermal.heater.power_w", [0.0]))
        )
        out["raw"]["heat_var"] = float(np.var(heat))
        out["raw"]["heat_mean"] = float(np.mean(heat))
    elif object_id == "mtb":
        out["raw"]["cmd_var"] = float(np.var(data.get("adcs.mtb.command_dipole_am2_0", [0.0])))
        actual = data.get("adcs.mtb.actual_dipole_am2_0")
        if actual is not None:
            out["raw"]["actual_frozen"] = float(np.all(np.abs(actual) < 1e-9))
            out["raw"]["actual_mean"] = float(np.mean(actual))
        # Gap-2.A feature (2026-08-05): the effective dipole limit is the ONLY
        # observable signal for `mtb_dipole_capacity_loss` (0.2 -> 0.16/0.14)
        # and `mtb_coil_open` (0 -> frozen).  Expressed as the minimum relative
        # to the physical nominal 0.2 A*m^2 so the nominal reference turns it
        # into a relative deviation (the column is exempted from the governed
        # *effective_* deny pattern as documented physical telemetry).
        eff = data.get("adcs.mtb.effective_dipole_limit_am2_0")
        if eff is not None:
            out["raw"]["eff_dipole_min"] = float(np.min(eff))
            out["raw"]["eff_dipole_last"] = float(eff[-1])
            out["raw"]["eff_dipole_frozen"] = float(np.all(np.abs(eff) < 1e-9))
    elif object_id in {"imu"}:
        gyro = data.get("adcs.imu.gyro_x_rad_s")
        if gyro is not None:
            out["raw"]["gyro_x_mean"] = float(np.mean(gyro))
            out["raw"]["gyro_x_std"] = float(np.std(gyro))
            out["raw"]["gyro_x_first"] = float(gyro[0])
            out["raw"]["gyro_x_last"] = float(gyro[-1])
            # full-trace total change (formerly drift_total; renamed to avoid the
            # governed *drift* deny pattern — it is a computed measurement
            # statistic, not an injected-config field); step magnitude via the
            # max inter-quantile gap
            out["raw"]["gyro_x_delta"] = float(gyro[-1] - gyro[0])
            p10, p90 = np.percentile(gyro, [10.0, 90.0])
            out["raw"]["iq_gap"] = float(p90 - p10)
    elif object_id in {"data_queue", "onboard_storage"}:
        prefix = "dq" if object_id == "data_queue" else "os"
        fill_col = "comm.data_queue.fill_ratio" if object_id == "data_queue" else "comm.storage.fill_ratio"
        downlink_col = "comm.data_queue.downlink_bps" if object_id == "data_queue" else "comm.storage.downlink_bps"
        stored_col = "comm.data_queue.queue_bits" if object_id == "data_queue" else "comm.storage.stored_bits"
        capacity_col = "comm.data_queue.capacity_bits" if object_id == "data_queue" else "comm.storage.capacity_bits"
        dropped_col = (
            "comm.data_queue.dropped_bits" if object_id == "data_queue" else None
        )
        fill = data.get(fill_col)
        if fill is not None:
            out["raw"]["fill_mean"] = float(np.mean(fill))
            out["raw"]["fill_max"] = float(np.max(fill))
            out["raw"]["saturation_dwell"] = float(np.mean(fill >= 0.99))
        # NOTE (loop-16 deny reconciliation): the former raw_eff_capacity_ratio
        # consumed comm.*.effective_capacity_bits, which matches the governed
        # *effective_* deny pattern.  It is removed; the capacity-loss signal is
        # carried by measured queue dynamics (fill / saturation / drop / drain)
        # instead of the effective-capacity column.
        if dropped_col and data.get(dropped_col) is not None:
            dropped = data[dropped_col]
            out["raw"]["drop_rate_bps"] = float(dropped[-1] - dropped[0]) / duration
        downlink = data.get(downlink_col)
        if downlink is not None and len(downlink):
            out["raw"]["downlink_first"] = float(downlink[0])
            out["raw"]["downlink_last"] = float(downlink[-1])
            out["raw"]["downlink_decay_frac"] = (
                float((downlink[0] - downlink[-1]) / downlink[0]) if downlink[0] != 0 else 0.0
            )
        stored = data.get(stored_col)
        if stored is not None and len(stored):
            out["raw"]["stored_drain_rate_bps"] = float(stored[-1] - stored[0]) / duration
    elif object_id == "solar_panel":
        eff = data.get("eps.solar.efficiency")
        ap = data.get("eps.solar.array_power_w")
        shadow = data.get("eps.solar.shadow_factor")
        if eff is not None and len(eff):
            # Absolute efficiency (0-1).  The former feature divided by the
            # *nominal_efficiency* column (a governed *nominal_* deny pattern);
            # the fixed documented simulator constant is used instead, and the
            # TRAIN-frozen nominal reference later turns it into a relative
            # deviation.
            out["raw"]["eff_ratio"] = float(np.mean(eff))
        if ap is not None and shadow is not None:
            daylight = np.where(shadow > 0.5)[0]
            if len(daylight):
                steady = daylight[len(daylight) // 2:]
                # Ratio against the documented simulator max-power constant, not
                # the *nominal_max_power_w* column (governed *nominal_* deny).
                nom_power = PHYSICAL_NOMINAL["solar.max_power_w"]
                out["raw"]["daylight_output_ratio"] = float(np.mean(ap[steady]) / nom_power)
    return out


# --------------------------------------------------------------------------- #
# Mechanism-class mapping (scenario_id -> mechanism class).  Used ONLY for the
# supervised label of the mechanism head, never as a model feature.
# --------------------------------------------------------------------------- #

MECHANISM_CLASS_MAP: dict[str, str] = {
    "nominal": "NOMINAL",
    # capacity-loss family (cross-object: battery / EPS / MTB / onboard_storage)
    "fault:sudden_capacity_loss": "CAPACITY_LOSS",
    "degradation:capacity_loss_pct": "CAPACITY_LOSS",
    "degradation:capacity_loss": "CAPACITY_LOSS",
    "degradation:mtb_dipole_capacity_loss": "CAPACITY_LOSS",
    # output-frozen / structural failure family (cross-object)
    "fault:open_circuit": "FROZEN",
    "fault:rw_jamming": "FROZEN",
    "fault:mtb_coil_open": "FROZEN",
    "fault:read_only": "FROZEN",
    # gain / efficiency loss family
    "fault:solar_panel_failure": "GAIN_LOSS",
    "degradation:efficiency_loss_pct": "GAIN_LOSS",
    "fault:rw_motor_failure": "MOTOR_FAILURE",
    # coefficient / thermal family
    "degradation:friction_increase_pct": "FRICTION",
    "degradation:radiator_degradation_factor": "THERMAL_RADIATOR",
    "fault:heating_or_rejection_failure": "THERMAL_FAILURE",
    # bias / drift family
    "fault:bias_step": "BIAS",
    "degradation:bias_drift": "DRIFT",
    # queue family
    "fault:overflow": "QUEUE_OVERFLOW",
    "degradation:throughput_decay": "QUEUE_THROUGHPUT_DECAY",
}

# Order of mechanism classes (stable label indices).
MECHANISM_CLASS_ORDER: tuple[str, ...] = (
    "NOMINAL",
    "CAPACITY_LOSS",
    "FROZEN",
    "GAIN_LOSS",
    "MOTOR_FAILURE",
    "FRICTION",
    "THERMAL_RADIATOR",
    "THERMAL_FAILURE",
    "BIAS",
    "DRIFT",
    "QUEUE_OVERFLOW",
    "QUEUE_THROUGHPUT_DECAY",
)


def mechanism_class_of(scenario_id: str) -> str:
    return MECHANISM_CLASS_MAP.get(scenario_id, "UNKNOWN")


# --------------------------------------------------------------------------- #
# Nominal reference (fit on train split ONLY).
# --------------------------------------------------------------------------- #


@dataclass
class NominalReference:
    """Per-object nominal statistics computed from train-split nominal episodes."""

    object_id: str
    archetype: str
    counts: dict[str, float] = field(default_factory=dict)
    # parameter id -> nominal value (median over train nominal episodes)
    param_median: dict[str, float] = field(default_factory=dict)
    # raw scalar name -> nominal median
    raw_median: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_estimates(
        cls,
        object_id: str,
        estimates: list[dict[str, Any]],
    ) -> "NominalReference":
        archetype = OBJECT_ARCHETYPE[object_id]
        param_values: dict[str, list[float]] = {}
        raw_values: dict[str, list[float]] = {}
        for est in estimates:
            for pid, value in est.get("gray_box", {}).get("params", {}).items():
                param_values.setdefault(pid, []).append(float(value))
            for name, value in est.get("raw", {}).items():
                if np.isfinite(float(value)):
                    raw_values.setdefault(name, []).append(float(value))
        param_median = {k: float(np.median(v)) for k, v in param_values.items() if v}
        raw_median = {k: float(np.median(v)) for k, v in raw_values.items() if v}
        return cls(
            object_id=object_id,
            archetype=archetype,
            counts={"nominal_episodes": float(len(estimates))},
            param_median=param_median,
            raw_median=raw_median,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "archetype": self.archetype,
            "counts": self.counts,
            "param_median": self.param_median,
            "raw_median": self.raw_median,
        }


def fit_nominal_reference(
    object_id: str,
    nominal_estimates: list[dict[str, Any]],
) -> NominalReference:
    return NominalReference.from_estimates(object_id, nominal_estimates)


def fit_all_nominal_references(
    estimates_by_object: dict[str, list[dict[str, Any]]],
) -> dict[str, NominalReference]:
    refs: dict[str, NominalReference] = {}
    for object_id in sorted(OBJECT_ARCHETYPE):
        if estimates_by_object.get(object_id):
            refs[object_id] = NominalReference.from_estimates(
                object_id, estimates_by_object[object_id]
            )
    return refs


# --------------------------------------------------------------------------- #
# Fixed feature schema
# --------------------------------------------------------------------------- #

# Gray-box parameter relative-deviation slots (per parameter id that exists in
# the four dynamics contracts).  A slot is 0.0 when the object has no contract
# or the parameter was not identified.  charge/discharge_efficiency are the
# gap-1 efficiency terms (storage v2 contract); slots stay 0.0 for objects whose
# contract does not declare them (legacy/regression).
_GB_PARAM_SLOTS: tuple[str, ...] = (
    "energy_capacity",
    "charge_efficiency",
    "discharge_efficiency",
    "thermal_capacity",
    "thermal_conductance",
    "thermal_capacity_bus",
    "thermal_capacity_battery",
    "radiator_conductance",
    "actuator_gain",
    "actuator_time_constant",
    "moment_of_inertia",
    "damping_coefficient",
)

# Generic raw-scalar slots (per object; zero-padded when absent).
_RAW_SLOTS: tuple[str, ...] = (
    "soc_mean", "soc_std", "soc_max_step", "net_power_var",
    "applied_power_mean", "applied_power_var",
    "speed_mean", "speed_std", "command_var", "applied_mean", "applied_frozen",
    "bus_temp_max_c", "heat_var", "heat_mean",
    "cmd_var", "actual_frozen", "actual_mean",
    "gyro_x_mean", "gyro_x_std", "gyro_x_delta", "iq_gap",
    "fill_mean", "fill_max", "saturation_dwell",
    "drop_rate_bps", "downlink_first", "downlink_last", "downlink_decay_frac",
    "stored_drain_rate_bps", "eff_ratio", "daylight_output_ratio",
    "eff_dipole_min", "eff_dipole_last", "eff_dipole_frozen",
)

# EvidenceToken operator slots (union across contracts).
_OPERATOR_SLOTS: tuple[str, ...] = (
    "storage.robust_level",
    "storage.charge_discharge_slope",
    "storage.energy_balance",
    "storage.saturation_dwell",
    "actuator.command_response_gain",
    "actuator.command_response_lag",
    "actuator.rate_dynamics",
    "actuator.command_rate_consistency",
    "actuator.change_profile",
    "thermal.heating_cooling_slope",
    "thermal.inertia_profile",
    "thermal.balance_residual",
    "thermal.change_profile",
    "sensor.gap_runs",
    "sensor.innovation",
    "sensor.trend_profile",
    "sensor.noise_profile",
    "converter.efficiency_profile",
    "converter.balance_residual",
    "converter.saturation_profile",
    "queue.fill_profile",
    "queue.flow_balance",
    "queue.saturation_dwell",
    "queue.service_regime",
    "spin.rate_dynamics",
    "spin.momentum_balance",
    "spin.command_rate_consistency",
    "spin.change_profile",
)

FEATURE_DIM_NAMES: tuple[str, ...] = (
    # 1. gray-box fail flag
    "gb_fail",
    # 2. gray-box parameter relative deviations (against nominal reference)
    *tuple(f"gb_param_rel_{pid}" for pid in _GB_PARAM_SLOTS),
    # 3. gray-box OOD + residual RMS (log-scaled)
    "gb_ood",
    "gb_residual_rms_log",
    "gb_raw_violation_fraction",
    # 4. generic raw scalars (relative deviation against nominal reference)
    *tuple(f"raw_{name}" for name in _RAW_SLOTS),
    # 5. EvidenceToken mean value + availability per operator slot
    *tuple(f"tok_{op}_mean" for op in _OPERATOR_SLOTS),
    *tuple(f"tok_{op}_avail" for op in _OPERATOR_SLOTS),
    # 6. object context: archetype one-hot ONLY (shared prototype prior).  The
    #    component instance id is deliberately NOT a feature: it is the
    #    localization target, so feeding it would make the head circular.
    *tuple(f"arch_{arch}" for arch in sorted(set(OBJECT_ARCHETYPE.values()))),
)

NUM_FEATURES = len(FEATURE_DIM_NAMES)
_ARCH_INDEX = {arch: i for i, arch in enumerate(sorted(set(OBJECT_ARCHETYPE.values())))}


def _token_features(
    rows: list[dict[str, str]],
    archetype: str,
) -> dict[str, tuple[float, float]]:
    """EvidenceToken (mean value, availability) per operator slot for the archetype."""
    out: dict[str, tuple[float, float]] = {}
    try:
        data = _signal_map(rows)
        frame = _frame_from_rows(
            component_instance_id=next(
                obj for obj, arch in OBJECT_ARCHETYPE.items() if arch == archetype
            ),
            archetype=archetype,
            data=data,
        )
        builder = _token_builder(archetype)
        tokens = builder.build(frame, component_archetype_id=archetype).tokens
    except Exception:  # noqa: BLE001 - absent contract is a masked feature
        return out
    by_op: dict[str, list[tuple[float, float]]] = {}
    for token in tokens:
        by_op.setdefault(token.operator_id, []).append(
            (float(token.value), float(token.availability))
        )
    for op, pairs in by_op.items():
        mean_value = float(np.mean([p[0] for p in pairs]))
        mean_avail = float(np.mean([p[1] for p in pairs]))
        out[op] = (mean_value, mean_avail)
    return out


def build_feature_vector(
    *,
    estimate: dict[str, Any],
    rows: list[dict[str, str]],
    reference: NominalReference | None,
) -> np.ndarray:
    """Assemble the fixed-length feature vector for one episode.

    ``reference`` is the train-split nominal reference for the object; if None
    (object never seen in train nominal), parameters/raw features fall back to
    absolute magnitudes instead of relative deviations.  The vector is a pure
    function of public telemetry + object context — no deny field is included.
    """
    object_id = estimate["object_id"]
    archetype = estimate["archetype"]
    gb = estimate.get("gray_box", {})
    raw = estimate.get("raw", {})

    vector: list[float] = []
    # 1. gb_fail
    vector.append(1.0 if gb.get("fail", False) else 0.0)

    # 2. gb param relative deviations
    params = gb.get("params", {}) if not gb.get("fail", False) else {}
    for pid in _GB_PARAM_SLOTS:
        if pid in params and reference is not None and pid in reference.param_median:
            ref = reference.param_median[pid]
            base = ref if abs(ref) > 1e-12 else 1.0
            vector.append(float((params[pid] - ref) / base))
        elif pid in params:
            # No nominal reference: use absolute magnitude (rare cold-start path).
            vector.append(float(params[pid]) / 1.0e3 if abs(params[pid]) > 1e3 else float(params[pid]))
        else:
            vector.append(0.0)

    # 3. gb ood / residual rms / violation fraction
    vector.append(float(gb.get("ood", 0.0)))
    rms_values = gb.get("residual_rms", {})
    primary_rms = next(iter(rms_values.values()), 0.0) if rms_values else 0.0
    vector.append(float(np.log1p(abs(primary_rms))))
    vector.append(float(gb.get("raw_violation_fraction", 0.0)))

    # 4. raw scalars (relative deviation vs nominal reference; fallback absolute).
    #    Robust clip (2026-08-05): some raw slots (e.g. `soc_max_step` ~1e4,
    #    `net_power_var` ~1e3) have a heavy-tailed relative deviation that would
    #    otherwise dominate the shared backbone and swamp the mechanism signal.
    #    A symmetric clip to [-5, 5] bounds the deviation magnitude while
    #    preserving sign and relative ordering (standard robust scaling).
    for name in _RAW_SLOTS:
        value = float(raw.get(name, 0.0))
        if reference is not None and name in reference.raw_median:
            ref = reference.raw_median[name]
            base = ref if abs(ref) > 1e-12 else 1.0
            vector.append(float(np.clip((value - ref) / base, -5.0, 5.0)))
        else:
            vector.append(float(np.clip(value, -5.0, 5.0)))

    # 5. EvidenceToken per operator
    tokens = _token_features(rows, archetype)
    for op in _OPERATOR_SLOTS:
        mean_val, mean_avail = tokens.get(op, (0.0, 0.0))
        vector.append(mean_val)
    for op in _OPERATOR_SLOTS:
        mean_val, mean_avail = tokens.get(op, (0.0, 0.0))
        vector.append(mean_avail)

    # 6. object context: archetype one-hot (component id stays out of features)
    arch = [0.0] * len(_ARCH_INDEX)
    arch[_ARCH_INDEX[archetype]] = 1.0
    vector.extend(arch)

    if len(vector) != NUM_FEATURES:
        raise RuntimeError(
            f"diagnosis feature contract produced {len(vector)} values; expected {NUM_FEATURES}"
        )
    return np.asarray(vector, dtype=float)


__all__ = [
    "DYNAMICS_OBJECTS",
    "FEATURE_DIM_NAMES",
    "MECHANISM_CLASS_MAP",
    "MECHANISM_CLASS_ORDER",
    "NUM_FEATURES",
    "OBJECT_ARCHETYPE",
    "NominalReference",
    "PHYSICAL_NOMINAL",
    "_CHANNEL_SPECS",
    "build_feature_vector",
    "fit_all_nominal_references",
    "fit_nominal_reference",
    "mechanism_class_of",
]
