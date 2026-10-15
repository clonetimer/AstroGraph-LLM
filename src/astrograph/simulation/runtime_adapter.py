from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astrograph.evidence import (
    ComparisonOperator,
    EpisodeEvidenceBundle,
    EpisodeWindow,
    EvidenceDefinition,
    EvidenceRole,
    FeatureAggregation,
    FeatureDefinition,
    FeatureResult,
    evaluate_evidence,
)
from astrograph.evidence.executable import stable_id
from astrograph.semantics import TelemetrySemanticsRegistry
from astrograph.simulation.models import OnlineDataGateReport, SimulationRunCompleted


def _sha256_pair(*paths: Path) -> str:
    h = hashlib.sha256()
    for path in paths:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def validate_completed_simulation_run(
    event: SimulationRunCompleted,
    semantics: TelemetrySemanticsRegistry,
) -> OnlineDataGateReport:
    run_dir = Path(event.run_dir)
    nominal_dir = Path(event.nominal_run_dir)
    telemetry = run_dir / "telemetry.csv"
    nominal = nominal_dir / "telemetry.csv"
    manifest = run_dir / "run_manifest.json"
    missing_files = [str(p) for p in (telemetry, nominal, manifest) if not p.is_file()]
    if missing_files:
        return OnlineDataGateReport(
            passed=False,
            run_id=event.run_id,
            missing_required_fields=missing_files,
            protocol_version=semantics.registry_version,
        )
    frame = pd.read_csv(telemetry, low_memory=False)
    baseline = pd.read_csv(nominal, low_memory=False)
    required = semantics.model_input_fields()
    missing_fields = sorted(set(required) - set(frame.columns))
    prohibited = sorted(c for c in required if c.startswith("label.") or c.startswith("truth."))
    times = pd.to_numeric(frame.get("time_s"), errors="coerce")
    nominal_times = pd.to_numeric(baseline.get("time_s"), errors="coerce")
    monotonic = bool(times.notna().all() and times.is_monotonic_increasing and times.is_unique)
    aligned = bool(len(times) == len(nominal_times) and np.allclose(times, nominal_times, rtol=0, atol=1e-9))
    missing_values = int(frame[[c for c in required if c in frame]].isna().sum().sum())
    row_count_ok = int(json.loads(manifest.read_text(encoding="utf-8")).get("row_count", -1)) == len(frame)
    warnings = [] if row_count_ok else ["run_manifest row_count does not match telemetry rows"]
    passed = not missing_fields and not prohibited and monotonic and aligned and missing_values == 0 and row_count_ok
    return OnlineDataGateReport(
        passed=passed,
        run_id=event.run_id,
        row_count=len(frame),
        field_count=len(frame.columns),
        required_field_count=len(required),
        time_monotonic=monotonic,
        time_aligned_with_nominal=aligned,
        missing_required_fields=missing_fields,
        missing_values=missing_values,
        prohibited_model_input_fields=prohibited,
        protocol_version=semantics.registry_version,
        warnings=warnings,
    )


class WholeSpacecraftRuntimeEvidenceBuilder:
    """Build label-free E0 evidence from one completed runtime run and its paired nominal run."""

    builder_version = "0.5.9-online-runtime-evidence-v4"

    def __init__(self, protocol_root: str | Path):
        self.semantics = TelemetrySemanticsRegistry(protocol_root)

    def build(self, event: SimulationRunCompleted) -> tuple[EpisodeEvidenceBundle, OnlineDataGateReport]:
        gate = validate_completed_simulation_run(event, self.semantics)
        if not gate.passed:
            raise ValueError(f"online data gate failed: {gate.model_dump(mode='json')}")
        run_dir = Path(event.run_dir)
        nominal_dir = Path(event.nominal_run_dir)
        frame = pd.read_csv(run_dir / "telemetry.csv", low_memory=False)
        baseline = pd.read_csv(nominal_dir / "telemetry.csv", low_memory=False)
        t = frame["time_s"].astype(float).to_numpy()
        input_hash = _sha256_pair(run_dir / "telemetry.csv", nominal_dir / "telemetry.csv")

        heater_mask = (frame["status.thermal.heater_command_on"].astype(float) < 0.5) & (frame["status.thermal.heater_on"].astype(float) > 0.5)
        star_mask = (frame["status.adcs.star_tracker_valid"].astype(float) < 0.5) & (frame["status.adcs.star_tracker_locked"].astype(float) < 0.5)
        drag_fields = [f"adcs.rw.effective_drag_nms_{i}" for i in range(3)]
        torque_fields = [f"adcs.rw.effective_max_torque_nm_{i}" for i in range(3)]
        drag_scores = [float(frame[c].abs().max()) for c in drag_fields]
        torque_losses = [float(((baseline[c] - frame[c]).clip(lower=0) / baseline[c].replace(0, np.nan)).max()) for c in torque_fields]
        rw_index = int(np.argmax([max(drag_scores[i] / 0.001, torque_losses[i] / 0.1) for i in range(3)]))
        friction_mask = frame[drag_fields[rw_index]].abs() >= 0.001
        torque_mask = ((baseline[torque_fields[rw_index]] - frame[torque_fields[rw_index]]) / baseline[torque_fields[rw_index]].replace(0, np.nan)) >= 0.1

        jam_scores = [
            float(((frame[f"status.adcs.rw.available_{i}"].astype(float) < 0.5) &
                   (frame[f"status.adcs.rw.driver_code_{i}"].astype(float) >= 2.0)).mean())
            for i in range(3)
        ]
        jam_index = int(np.argmax(jam_scores))
        jam_mask = (
            (frame[f"status.adcs.rw.available_{jam_index}"].astype(float) < 0.5)
            & (frame[f"status.adcs.rw.driver_code_{jam_index}"].astype(float) >= 2.0)
        )
        battery_ir_mask = frame["eps.battery_internal_resistance_scale"].astype(float) >= 1.25
        solar_string_mask = (
            frame["status.eps.solar_array_available_strings"].astype(float)
            < baseline["status.eps.solar_array_available_strings"].astype(float)
        )
        gyro_bias_mask = (
            frame["adcs.sensor.gyro_bias_rad_s_x"].astype(float)
            - baseline["adcs.sensor.gyro_bias_rad_s_x"].astype(float)
        ).abs() >= 3e-4
        transmitter_mask = (
            baseline["comm.transmitter_rf_power_scale"].astype(float)
            - frame["comm.transmitter_rf_power_scale"].astype(float)
        ) >= 0.1
        storage_mask = (
            frame["status.data.storage_write_available"].astype(float)
            < baseline["status.data.storage_write_available"].astype(float)
        ) | (
            (baseline["data.accepted_write_bps"].astype(float) > 0.0)
            & (frame["data.accepted_write_bps"].astype(float) <= 0.8 * baseline["data.accepted_write_bps"].astype(float))
        )

        # Remaining six E0 mechanisms.  These direct runtime states are used only to
        # identify the closed mechanism Episode; formal diagnosis still depends on
        # versioned FeatureResult/EvidenceFact evaluation below.
        capacity_ratio_series = (
            frame["eps.effective_battery_capacity_wh"].astype(float)
            / baseline["eps.effective_battery_capacity_wh"].replace(0, np.nan).astype(float)
        )
        battery_capacity_mask = capacity_ratio_series <= 0.90

        solar_power_ratio_series = (
            frame["eps.solar_array_power_w"].astype(float)
            / baseline["eps.solar_array_power_w"].replace(0, np.nan).astype(float)
        )
        nominal_strings_series = baseline["status.eps.solar_array_available_strings"].astype(float)
        solar_efficiency_mask = (solar_power_ratio_series <= 0.80) & (
            frame["status.eps.solar_array_available_strings"].astype(float)
            >= nominal_strings_series - 0.1
        )

        downlink_service_mask = (
            frame["status.comm.transmitter_command_on"].astype(float) > 0.5
        ) & (
            frame["comm.active"].astype(float) < 0.5
        ) & (
            frame["status.comm.transmitter_on"].astype(float) < 0.5
        ) & (
            frame["comm.downlink_rate_bps"].astype(float)
            <= 0.10 * baseline["comm.downlink_rate_bps"].replace(0, np.nan).astype(float)
        )

        temperature_bias_series = frame["thermal.temperature_sensor_bias_c"].astype(float)
        sensor_fault_mask = temperature_bias_series.abs() >= 0.05
        nonzero_sensor = sensor_fault_mask.to_numpy()
        if bool(nonzero_sensor.any()):
            sensor_times = t[nonzero_sensor]
            sensor_bias_values = temperature_bias_series.to_numpy()[nonzero_sensor]
            sensor_bias_slope = (
                float(np.polyfit(sensor_times, sensor_bias_values, 1)[0])
                if len(sensor_times) >= 2
                else 0.0
            )
        else:
            sensor_bias_slope = 0.0
        temperature_bias_mask = sensor_fault_mask if abs(sensor_bias_slope) <= 0.005 else pd.Series(False, index=frame.index)
        temperature_drift_mask = sensor_fault_mask if abs(sensor_bias_slope) >= 0.02 else pd.Series(False, index=frame.index)

        masks = [
            heater_mask, star_mask, friction_mask.fillna(False), torque_mask.fillna(False),
            jam_mask.fillna(False), battery_ir_mask.fillna(False), solar_string_mask.fillna(False),
            gyro_bias_mask.fillna(False), transmitter_mask.fillna(False), storage_mask.fillna(False),
            battery_capacity_mask.fillna(False), solar_efficiency_mask.fillna(False),
            downlink_service_mask.fillna(False), temperature_bias_mask.fillna(False),
            temperature_drift_mask.fillna(False),
        ]
        mechanism_mask = masks[0].copy()
        for registered_mask in masks[1:]:
            mechanism_mask = mechanism_mask | registered_mask
        model_fields = [c for c in self.semantics.model_input_fields() if c in frame.columns]
        residual_max: dict[str, float] = {}
        generic_mask = np.zeros(len(frame), dtype=bool)
        for field in model_fields:
            a = pd.to_numeric(frame[field], errors="coerce").astype(float)
            b = pd.to_numeric(baseline[field], errors="coerce").astype(float)
            if a.notna().all() and b.notna().all():
                diff = (a - b).abs().to_numpy()
                residual_max[field] = float(np.max(diff))
                generic_mask |= diff > 1e-8
        active_mask = mechanism_mask.to_numpy() if bool(mechanism_mask.any()) else generic_mask
        if bool(active_mask.any()):
            indices = np.flatnonzero(active_mask)
            start_s, end_s = float(t[indices[0]]), float(t[indices[-1]])
            source = "label-free runtime mechanism/paired-residual detector"
        else:
            start_s, end_s = float(t[0]), float(t[-1])
            source = "label-free full-run nominal analysis"
        episode_id = stable_id("EP", {"run_id": event.run_id, "start_s": start_s, "end_s": end_s})
        episode = EpisodeWindow(
            episode_id=episode_id,
            run_id=event.run_id,
            paired_nominal_run_id=Path(event.nominal_run_dir).name,
            start_s=start_s,
            end_s=end_s,
            source=source,
            source_is_label_free=True,
            operating_condition={"runtime_id": event.runtime_id},
        )
        window = (frame["time_s"] >= start_s) & (frame["time_s"] <= end_s)
        base_window = (baseline["time_s"] >= start_s) & (baseline["time_s"] <= end_s)

        defs: list[FeatureDefinition] = []
        results: list[FeatureResult] = []
        edefs: list[EvidenceDefinition] = []

        def add_feature(fid: str, name: str, field: str, agg: FeatureAggregation, unit: str, value: float, component: str, algorithm: str, feature_mask=None) -> None:
            local_mask = window if feature_mask is None or not bool(feature_mask.any()) else feature_mask
            # The numerical sample selector may be narrower than the mechanism Episode,
            # but the FeatureDefinition remains scoped to the closed Episode window.
            # Record the Episode boundaries deterministically and preserve the effective
            # sample count separately.
            local_start = start_s
            local_end = end_s
            fd = FeatureDefinition(feature_definition_id=fid, name=name, source_field_template=field, aggregation=agg, unit=unit, algorithm_version=algorithm, source_refs=["PHM whole-spacecraft minimal runtime v1.1", "ProjectA PHM FMEA"], claim_boundary="E0 runtime feature only; not flight calibrated.")
            fr = FeatureResult(
                feature_result_id=stable_id("FR", {"run": event.run_id, "fid": fid, "episode": episode_id}),
                feature_definition_id=fid, run_id=event.run_id, paired_nominal_run_id=Path(event.nominal_run_dir).name,
                episode_id=episode_id, resolved_field=field, component_instance=component, value=float(value), unit=unit,
                window_start_s=local_start, window_end_s=local_end, sample_count=int(local_mask.sum()), algorithm_version=algorithm,
                input_data_sha256=input_hash,
            )
            defs.append(fd); results.append(fr)

        def add_evidence(eid: str, fault: str, name: str, fid: str, op: ComparisonOperator, threshold: float, unit: str, critical: bool = True) -> None:
            edefs.append(EvidenceDefinition(evidence_definition_id=eid, target_fault_id=fault, name=name, feature_definition_id=fid, role=EvidenceRole.REQUIRED if critical else EvidenceRole.SUPPORTING, comparison=op, threshold=threshold, threshold_unit=unit, critical=critical, source_refs=["PHM whole-spacecraft minimal runtime v1.1", "AstroGraph V0.5.3 validated effect matrix", "AstroGraph V0.5.9 compact-reasoning executable evidence review"], claim_boundary="E0 online integration threshold; not a flight-certified diagnostic limit."))

        heater_feature_mask = heater_mask if bool(heater_mask.any()) else window
        mismatch_fraction = float(heater_mask[heater_feature_mask].mean()) if int(heater_feature_mask.sum()) else 0.0
        heater_current = float(frame.loc[heater_feature_mask, "sensor.thermal.heater_current_a"].mean())
        add_feature("FD-ONLINE-HEATER-MISMATCH-V1", "指令关闭而加热器开启比例", "status.thermal.heater_on", FeatureAggregation.TRUE_FRACTION, "1", mismatch_fraction, "THERMAL_HEATER", "runtime-closed-window-v1", heater_feature_mask)
        add_feature("FD-ONLINE-HEATER-CURRENT-V1", "加热器电流均值", "sensor.thermal.heater_current_a", FeatureAggregation.MEAN_VALUE, "A", heater_current, "THERMAL_HEATER", "runtime-closed-window-v1", heater_feature_mask)
        add_evidence("ED-ONLINE-HEATER-MISMATCH-V1", "THERMAL_HEATER_STUCK_ON", "指令关闭而实际开启", "FD-ONLINE-HEATER-MISMATCH-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.9, "1")
        add_evidence("ED-ONLINE-HEATER-CURRENT-V1", "THERMAL_HEATER_STUCK_ON", "加热器电流超过最低效果门槛", "FD-ONLINE-HEATER-CURRENT-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.2, "A")

        star_feature_mask = star_mask if bool(star_mask.any()) else window
        invalid_fraction = float((frame.loc[star_feature_mask, "status.adcs.star_tracker_valid"].astype(float) < 0.5).mean())
        unlocked_fraction = float((frame.loc[star_feature_mask, "status.adcs.star_tracker_locked"].astype(float) < 0.5).mean())
        age_max = float(frame.loc[star_feature_mask, "sensor.adcs.star_tracker_data_age_s"].max())
        fusion_min = float(frame.loc[star_feature_mask, "status.adcs.sensor_fusion_mode_code"].min())
        for fid, name, field, agg, unit, value in [
            ("FD-ONLINE-STAR-INVALID-V1", "星敏无效比例", "status.adcs.star_tracker_valid", FeatureAggregation.FALSE_FRACTION, "1", invalid_fraction),
            ("FD-ONLINE-STAR-UNLOCKED-V1", "星敏失锁比例", "status.adcs.star_tracker_locked", FeatureAggregation.FALSE_FRACTION, "1", unlocked_fraction),
            ("FD-ONLINE-STAR-AGE-V1", "星敏数据龄期最大值", "sensor.adcs.star_tracker_data_age_s", FeatureAggregation.MAX_VALUE, "s", age_max),
            ("FD-ONLINE-STAR-FUSION-V1", "融合模式最小值", "status.adcs.sensor_fusion_mode_code", FeatureAggregation.MIN_VALUE, "code", fusion_min),
        ]: add_feature(fid, name, field, agg, unit, value, "STAR_TRACKER", "runtime-closed-window-v1")
        add_evidence("ED-ONLINE-STAR-INVALID-V1", "ADCS_STAR_TRACKER_DROPOUT", "星敏数据无效", "FD-ONLINE-STAR-INVALID-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.9, "1")
        add_evidence("ED-ONLINE-STAR-UNLOCKED-V1", "ADCS_STAR_TRACKER_DROPOUT", "星敏失锁", "FD-ONLINE-STAR-UNLOCKED-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.9, "1")
        add_evidence("ED-ONLINE-STAR-AGE-V1", "ADCS_STAR_TRACKER_DROPOUT", "星敏数据龄期超限", "FD-ONLINE-STAR-AGE-V1", ComparisonOperator.GREATER_OR_EQUAL, 30.0, "s")
        add_evidence("ED-ONLINE-STAR-FUSION-V1", "ADCS_STAR_TRACKER_DROPOUT", "融合切换到陀螺递推", "FD-ONLINE-STAR-FUSION-V1", ComparisonOperator.GREATER_OR_EQUAL, 2.0, "code")

        rw_feature_mask = (friction_mask | torque_mask) if bool((friction_mask | torque_mask).any()) else window
        drag_max = float(frame.loc[rw_feature_mask, drag_fields[rw_index]].abs().max())
        current_field = f"sensor.adcs.rw.motor_current_a_{rw_index}"
        current_residual = float((frame.loc[rw_feature_mask, current_field].astype(float) - baseline.loc[rw_feature_mask, current_field].astype(float)).abs().max()) if int(rw_feature_mask.sum()) else 0.0
        add_feature("FD-ONLINE-RW-DRAG-V1", "反作用轮有效拖曳系数峰值", drag_fields[rw_index], FeatureAggregation.MAX_VALUE, "N*m*s", drag_max, f"RW{rw_index}", "runtime-mechanism-state-v1", rw_feature_mask)
        add_feature("FD-ONLINE-RW-CURRENT-RESIDUAL-V1", "反作用轮电流配对残差峰值", current_field, FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "A", abs(current_residual), f"RW{rw_index}", "runtime-paired-residual-v1", rw_feature_mask)
        add_evidence("ED-ONLINE-RW-FRICTION-DIRECT-V1", "ADCS_RW_FRICTION_INCREASE", "运行时有效拖曳系数增大", "FD-ONLINE-RW-DRAG-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.001, "N*m*s")
        add_evidence("ED-ONLINE-RW-FRICTION-CURRENT-V1", "ADCS_RW_FRICTION_INCREASE", "电流相对配对正常运行发生变化", "FD-ONLINE-RW-CURRENT-RESIDUAL-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.005, "A", critical=False)

        loss_ratio = float(((baseline.loc[rw_feature_mask, torque_fields[rw_index]] - frame.loc[rw_feature_mask, torque_fields[rw_index]]).clip(lower=0) / baseline.loc[rw_feature_mask, torque_fields[rw_index]].replace(0, np.nan)).max()) if int(rw_feature_mask.sum()) else 0.0
        available_min = float(frame.loc[rw_feature_mask, f"status.adcs.rw.available_{rw_index}"].astype(float).min())
        add_feature("FD-ONLINE-RW-TORQUE-LOSS-RATIO-V1", "反作用轮最大力矩能力下降比例", torque_fields[rw_index], FeatureAggregation.MAX_VALUE, "1", loss_ratio, f"RW{rw_index}", "runtime-mechanism-state-v1", rw_feature_mask)
        add_feature("FD-ONLINE-RW-AVAILABLE-MIN-V1", "反作用轮可用状态最小值", f"status.adcs.rw.available_{rw_index}", FeatureAggregation.MIN_VALUE, "bool", available_min, f"RW{rw_index}", "runtime-closed-window-v1", rw_feature_mask)
        add_evidence("ED-ONLINE-RW-TORQUE-LOSS-V1", "ADCS_RW_TORQUE_AUTHORITY_LOSS", "最大力矩能力下降", "FD-ONLINE-RW-TORQUE-LOSS-RATIO-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.1, "1")
        add_evidence("ED-ONLINE-RW-TORQUE-AVAILABLE-V1", "ADCS_RW_TORQUE_AUTHORITY_LOSS", "轮仍保持可用以排除卡滞", "FD-ONLINE-RW-AVAILABLE-MIN-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.0, "bool")

        jam_feature_mask = jam_mask if bool(jam_mask.any()) else window
        jam_available = float(frame.loc[jam_feature_mask, f"status.adcs.rw.available_{jam_index}"].astype(float).min())
        jam_driver = float(frame.loc[jam_feature_mask, f"status.adcs.rw.driver_code_{jam_index}"].astype(float).min())
        add_feature("FD-ONLINE-RW-JAM-UNAVAILABLE-V1", "卡滞轮可用状态最小值", f"status.adcs.rw.available_{jam_index}", FeatureAggregation.MIN_VALUE, "bool", jam_available, f"RW{jam_index}", "runtime-closed-window-v1", jam_feature_mask)
        add_feature("FD-ONLINE-RW-JAM-DRIVER-V1", "卡滞轮驱动状态码最小值", f"status.adcs.rw.driver_code_{jam_index}", FeatureAggregation.MIN_VALUE, "code", jam_driver, f"RW{jam_index}", "runtime-closed-window-v1", jam_feature_mask)
        add_evidence("ED-ONLINE-RW-JAM-UNAVAILABLE-V1", "ADCS_RW_JAM", "反作用轮不可用", "FD-ONLINE-RW-JAM-UNAVAILABLE-V1", ComparisonOperator.LESS_OR_EQUAL, 0.0, "bool")
        add_evidence("ED-ONLINE-RW-JAM-DRIVER-V1", "ADCS_RW_JAM", "驱动状态进入卡滞码", "FD-ONLINE-RW-JAM-DRIVER-V1", ComparisonOperator.GREATER_OR_EQUAL, 2.0, "code")

        battery_feature_mask = battery_ir_mask if bool(battery_ir_mask.any()) else window
        battery_scale = float(frame.loc[battery_feature_mask, "eps.battery_internal_resistance_scale"].astype(float).min())
        battery_ohmic = float(frame.loc[battery_feature_mask, "eps.battery_ohmic_loss_w"].astype(float).min())
        battery_estimate_delta = float((frame.loc[battery_feature_mask, "estimate.eps.battery_internal_resistance_ohm"].astype(float) - baseline.loc[battery_feature_mask, "estimate.eps.battery_internal_resistance_ohm"].astype(float)).abs().max())
        add_feature("FD-ONLINE-BATTERY-IR-SCALE-V1", "电池有效内阻比例最小值", "eps.battery_internal_resistance_scale", FeatureAggregation.MIN_VALUE, "1", battery_scale, "BATTERY", "runtime-mechanism-state-v1", battery_feature_mask)
        add_feature("FD-ONLINE-BATTERY-OHMIC-LOSS-V1", "电池欧姆损耗最小值", "eps.battery_ohmic_loss_w", FeatureAggregation.MIN_VALUE, "W", battery_ohmic, "BATTERY", "runtime-closed-window-v1", battery_feature_mask)
        add_feature("FD-ONLINE-BATTERY-IR-ESTIMATE-DELTA-V1", "电池内阻估计配对残差峰值", "estimate.eps.battery_internal_resistance_ohm", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "ohm", battery_estimate_delta, "BATTERY", "runtime-paired-residual-v1", battery_feature_mask)
        add_evidence("ED-ONLINE-BATTERY-IR-SCALE-V1", "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE", "有效内阻比例增大", "FD-ONLINE-BATTERY-IR-SCALE-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.25, "1")
        add_evidence("ED-ONLINE-BATTERY-OHMIC-LOSS-V1", "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE", "欧姆损耗超过工程门槛", "FD-ONLINE-BATTERY-OHMIC-LOSS-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.1, "W")
        add_evidence("ED-ONLINE-BATTERY-IR-ESTIMATE-V1", "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE", "内阻估计相对正常基线变化", "FD-ONLINE-BATTERY-IR-ESTIMATE-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.02, "ohm", critical=False)

        solar_feature_mask = solar_string_mask if bool(solar_string_mask.any()) else window
        solar_strings = float(frame.loc[solar_feature_mask, "status.eps.solar_array_available_strings"].astype(float).max())
        solar_current_delta = float((frame.loc[solar_feature_mask, "sensor.eps.solar_array_current_a"].astype(float) - baseline.loc[solar_feature_mask, "sensor.eps.solar_array_current_a"].astype(float)).abs().max())
        solar_power_delta = float((frame.loc[solar_feature_mask, "estimate.eps.solar_array_power_w"].astype(float) - baseline.loc[solar_feature_mask, "estimate.eps.solar_array_power_w"].astype(float)).abs().max())
        add_feature("FD-ONLINE-SOLAR-AVAILABLE-STRINGS-V1", "太阳阵可用串数最大值", "status.eps.solar_array_available_strings", FeatureAggregation.MAX_VALUE, "count", solar_strings, "SOLAR_ARRAY", "runtime-closed-window-v1", solar_feature_mask)
        add_feature("FD-ONLINE-SOLAR-CURRENT-DELTA-V1", "太阳阵电流配对残差峰值", "sensor.eps.solar_array_current_a", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "A", solar_current_delta, "SOLAR_ARRAY", "runtime-paired-residual-v1", solar_feature_mask)
        add_feature("FD-ONLINE-SOLAR-POWER-DELTA-V1", "太阳阵功率配对残差峰值", "estimate.eps.solar_array_power_w", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "W", solar_power_delta, "SOLAR_ARRAY", "runtime-paired-residual-v1", solar_feature_mask)
        nominal_string_count = float(baseline["status.eps.solar_array_available_strings"].astype(float).max())
        add_evidence("ED-ONLINE-SOLAR-STRING-DROP-V1", "EPS_SOLAR_ARRAY_STRING_FAILURE", "太阳阵可用串数下降", "FD-ONLINE-SOLAR-AVAILABLE-STRINGS-V1", ComparisonOperator.LESS_OR_EQUAL, nominal_string_count - 1.0, "count")
        add_evidence("ED-ONLINE-SOLAR-CURRENT-DROP-V1", "EPS_SOLAR_ARRAY_STRING_FAILURE", "太阳阵电流变化超过门槛", "FD-ONLINE-SOLAR-CURRENT-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.2, "A")
        add_evidence("ED-ONLINE-SOLAR-POWER-DROP-V1", "EPS_SOLAR_ARRAY_STRING_FAILURE", "太阳阵功率变化超过门槛", "FD-ONLINE-SOLAR-POWER-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 5.0, "W", critical=False)

        gyro_feature_mask = gyro_bias_mask if bool(gyro_bias_mask.any()) else window
        gyro_bias_max = float((frame.loc[gyro_feature_mask, "adcs.sensor.gyro_bias_rad_s_x"].astype(float) - baseline.loc[gyro_feature_mask, "adcs.sensor.gyro_bias_rad_s_x"].astype(float)).abs().max())
        gyro_rate_delta = float((frame.loc[gyro_feature_mask, "sensor.adcs.gyro_rate_rad_s_x"].astype(float) - baseline.loc[gyro_feature_mask, "sensor.adcs.gyro_rate_rad_s_x"].astype(float)).abs().max())
        add_feature("FD-ONLINE-GYRO-BIAS-DELTA-V1", "陀螺X轴零偏累积峰值", "adcs.sensor.gyro_bias_rad_s_x", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "rad/s", gyro_bias_max, "GYROSCOPE", "runtime-mechanism-state-v1", gyro_feature_mask)
        add_feature("FD-ONLINE-GYRO-RATE-DELTA-V1", "陀螺X轴测量配对残差峰值", "sensor.adcs.gyro_rate_rad_s_x", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "rad/s", gyro_rate_delta, "GYROSCOPE", "runtime-paired-residual-v1", gyro_feature_mask)
        add_evidence("ED-ONLINE-GYRO-BIAS-V1", "ADCS_GYRO_BIAS_DRIFT", "陀螺零偏随时间累积", "FD-ONLINE-GYRO-BIAS-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.0003, "rad/s")
        add_evidence("ED-ONLINE-GYRO-RATE-V1", "ADCS_GYRO_BIAS_DRIFT", "陀螺原始测量相对正常基线偏移", "FD-ONLINE-GYRO-RATE-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.0002, "rad/s")

        transmitter_feature_mask = transmitter_mask if bool(transmitter_mask.any()) else window
        rf_scale = float(frame.loc[transmitter_feature_mask, "comm.transmitter_rf_power_scale"].astype(float).max())
        rf_power_drop = float((baseline.loc[transmitter_feature_mask, "sensor.comm.rf_output_power_w"].astype(float) - frame.loc[transmitter_feature_mask, "sensor.comm.rf_output_power_w"].astype(float)).clip(lower=0).max())
        downlink_drop = float((baseline.loc[transmitter_feature_mask, "comm.downlink_rate_bps"].astype(float) - frame.loc[transmitter_feature_mask, "comm.downlink_rate_bps"].astype(float)).clip(lower=0).max())
        add_feature("FD-ONLINE-TX-RF-SCALE-V1", "发射机射频功率比例最大值", "comm.transmitter_rf_power_scale", FeatureAggregation.MAX_VALUE, "1", rf_scale, "TRANSMITTER", "runtime-mechanism-state-v1", transmitter_feature_mask)
        add_feature("FD-ONLINE-TX-RF-POWER-DROP-V1", "射频输出功率下降峰值", "sensor.comm.rf_output_power_w", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "W", rf_power_drop, "TRANSMITTER", "runtime-paired-residual-v1", transmitter_feature_mask)
        add_feature("FD-ONLINE-TX-DOWNLINK-DROP-V1", "下行速率下降峰值", "comm.downlink_rate_bps", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "bps", downlink_drop, "TRANSMITTER", "runtime-paired-residual-v1", transmitter_feature_mask)
        add_evidence("ED-ONLINE-TX-RF-SCALE-V1", "COMM_TRANSMITTER_OUTPUT_POWER_LOSS", "射频输出能力比例下降", "FD-ONLINE-TX-RF-SCALE-V1", ComparisonOperator.LESS_OR_EQUAL, 0.8, "1")
        add_evidence("ED-ONLINE-TX-RF-POWER-DROP-V1", "COMM_TRANSMITTER_OUTPUT_POWER_LOSS", "射频输出功率下降", "FD-ONLINE-TX-RF-POWER-DROP-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.0, "W")
        add_evidence("ED-ONLINE-TX-DOWNLINK-DROP-V1", "COMM_TRANSMITTER_OUTPUT_POWER_LOSS", "下行速率下降", "FD-ONLINE-TX-DOWNLINK-DROP-V1", ComparisonOperator.GREATER_OR_EQUAL, 100000.0, "bps", critical=False)

        storage_feature_mask = storage_mask if bool(storage_mask.any()) else window
        storage_available = float(frame.loc[storage_feature_mask, "status.data.storage_write_available"].astype(float).max())
        accepted_ratio = float((frame.loc[storage_feature_mask, "data.accepted_write_bps"].astype(float) / baseline.loc[storage_feature_mask, "data.accepted_write_bps"].replace(0, np.nan).astype(float)).fillna(0.0).max())
        storage_delta = float((baseline.loc[storage_feature_mask, "data.storage_bits"].astype(float) - frame.loc[storage_feature_mask, "data.storage_bits"].astype(float)).clip(lower=0).max())
        add_feature("FD-ONLINE-STORAGE-AVAILABLE-V1", "存储写入可用状态最大值", "status.data.storage_write_available", FeatureAggregation.MAX_VALUE, "bool", storage_available, "ONBOARD_STORAGE", "runtime-closed-window-v1", storage_feature_mask)
        add_feature("FD-ONLINE-STORAGE-WRITE-RATIO-V1", "接受写入速率剩余比例", "data.accepted_write_bps", FeatureAggregation.MAX_VALUE, "1", accepted_ratio, "ONBOARD_STORAGE", "runtime-paired-ratio-v1", storage_feature_mask)
        add_feature("FD-ONLINE-STORAGE-INTEGRAL-DELTA-V1", "存储积分相对正常基线下降峰值", "data.storage_bits", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "bit", storage_delta, "ONBOARD_STORAGE", "runtime-paired-residual-v1", storage_feature_mask)
        add_evidence("ED-ONLINE-STORAGE-AVAILABLE-V1", "DATA_STORAGE_WRITE_FAILURE", "存储写入服务不可用", "FD-ONLINE-STORAGE-AVAILABLE-V1", ComparisonOperator.LESS_OR_EQUAL, 0.0, "bool")
        add_evidence("ED-ONLINE-STORAGE-WRITE-RATIO-V1", "DATA_STORAGE_WRITE_FAILURE", "接受写入速率下降", "FD-ONLINE-STORAGE-WRITE-RATIO-V1", ComparisonOperator.LESS_OR_EQUAL, 0.8, "1")
        add_evidence("ED-ONLINE-STORAGE-INTEGRAL-V1", "DATA_STORAGE_WRITE_FAILURE", "存储积分相对正常运行下降", "FD-ONLINE-STORAGE-INTEGRAL-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.0, "bit", critical=False)

        # Battery effective-capacity loss.
        capacity_feature_mask = battery_capacity_mask if bool(battery_capacity_mask.any()) else window
        capacity_ratio = float(capacity_ratio_series.loc[capacity_feature_mask].min())
        capacity_soh = float(frame.loc[capacity_feature_mask, "estimate.eps.battery_soh"].astype(float).min())
        capacity_estimate_ratio = float((
            frame.loc[capacity_feature_mask, "estimate.eps.battery_capacity_wh"].astype(float)
            / baseline.loc[capacity_feature_mask, "estimate.eps.battery_capacity_wh"].replace(0, np.nan).astype(float)
        ).min())
        add_feature("FD-ONLINE-BATTERY-CAPACITY-RATIO-V1", "电池有效容量剩余比例", "eps.effective_battery_capacity_wh", FeatureAggregation.MIN_VALUE, "1", capacity_ratio, "BATTERY", "runtime-paired-ratio-v1", capacity_feature_mask)
        add_feature("FD-ONLINE-BATTERY-SOH-MIN-V1", "电池SOH估计最小值", "estimate.eps.battery_soh", FeatureAggregation.MIN_VALUE, "1", capacity_soh, "BATTERY", "runtime-closed-window-v1", capacity_feature_mask)
        add_feature("FD-ONLINE-BATTERY-CAPACITY-ESTIMATE-RATIO-V1", "容量估计相对配对正常比例", "estimate.eps.battery_capacity_wh", FeatureAggregation.MIN_VALUE, "1", capacity_estimate_ratio, "BATTERY", "runtime-paired-ratio-v1", capacity_feature_mask)
        add_evidence("ED-ONLINE-BATTERY-CAPACITY-RATIO-V1", "EPS_BATTERY_CAPACITY_LOSS", "有效容量剩余比例下降", "FD-ONLINE-BATTERY-CAPACITY-RATIO-V1", ComparisonOperator.LESS_OR_EQUAL, 0.80, "1")
        add_evidence("ED-ONLINE-BATTERY-SOH-V1", "EPS_BATTERY_CAPACITY_LOSS", "SOH估计下降", "FD-ONLINE-BATTERY-SOH-MIN-V1", ComparisonOperator.LESS_OR_EQUAL, 0.80, "1")
        add_evidence("ED-ONLINE-BATTERY-CAPACITY-ESTIMATE-V1", "EPS_BATTERY_CAPACITY_LOSS", "容量估计相对正常基线下降", "FD-ONLINE-BATTERY-CAPACITY-ESTIMATE-RATIO-V1", ComparisonOperator.LESS_OR_EQUAL, 0.80, "1", critical=False)

        # Solar-array continuous efficiency loss; available string count remains nominal,
        # which separates this degradation from discrete string failure.
        solar_eff_feature_mask = solar_efficiency_mask if bool(solar_efficiency_mask.any()) else window
        solar_power_ratio = float(solar_power_ratio_series.loc[solar_eff_feature_mask].min())
        solar_current_ratio = float((
            frame.loc[solar_eff_feature_mask, "sensor.eps.solar_array_current_a"].astype(float)
            / baseline.loc[solar_eff_feature_mask, "sensor.eps.solar_array_current_a"].replace(0, np.nan).astype(float)
        ).min())
        solar_string_ratio = float((
            frame.loc[solar_eff_feature_mask, "status.eps.solar_array_available_strings"].astype(float)
            / baseline.loc[solar_eff_feature_mask, "status.eps.solar_array_available_strings"].replace(0, np.nan).astype(float)
        ).min())
        add_feature("FD-ONLINE-SOLAR-EFFICIENCY-POWER-RATIO-V1", "太阳阵功率剩余比例", "eps.solar_array_power_w", FeatureAggregation.MIN_VALUE, "1", solar_power_ratio, "SOLAR_ARRAY", "runtime-paired-ratio-v1", solar_eff_feature_mask)
        add_feature("FD-ONLINE-SOLAR-EFFICIENCY-CURRENT-RATIO-V1", "太阳阵电流剩余比例", "sensor.eps.solar_array_current_a", FeatureAggregation.MIN_VALUE, "1", solar_current_ratio, "SOLAR_ARRAY", "runtime-paired-ratio-v1", solar_eff_feature_mask)
        add_feature("FD-ONLINE-SOLAR-EFFICIENCY-STRING-RATIO-V1", "太阳阵可用串数保持比例", "status.eps.solar_array_available_strings", FeatureAggregation.MIN_VALUE, "1", solar_string_ratio, "SOLAR_ARRAY", "runtime-paired-ratio-v1", solar_eff_feature_mask)
        add_evidence("ED-ONLINE-SOLAR-EFFICIENCY-POWER-V1", "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS", "太阳阵功率连续下降", "FD-ONLINE-SOLAR-EFFICIENCY-POWER-RATIO-V1", ComparisonOperator.LESS_OR_EQUAL, 0.80, "1")
        add_evidence("ED-ONLINE-SOLAR-EFFICIENCY-CURRENT-V1", "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS", "太阳阵电流连续下降", "FD-ONLINE-SOLAR-EFFICIENCY-CURRENT-RATIO-V1", ComparisonOperator.LESS_OR_EQUAL, 0.80, "1")
        add_evidence("ED-ONLINE-SOLAR-EFFICIENCY-STRINGS-V1", "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS", "可用串数保持正常以排除字符串失效", "FD-ONLINE-SOLAR-EFFICIENCY-STRING-RATIO-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.99, "1")

        # Radiator rejection loss has no direct coefficient telemetry in the E0 contract.
        # Use paired mean temperature changes from two independent observed channels.
        radiator_feature_mask = window
        payload_temp_mean_delta = float((
            frame.loc[radiator_feature_mask, "sensor.thermal.payload_temp_c"].astype(float)
            - baseline.loc[base_window, "sensor.thermal.payload_temp_c"].astype(float).to_numpy()
        ).mean())
        radiator_temp_mean_delta = float((
            frame.loc[radiator_feature_mask, "sensor.thermal.radiator_temp_c"].astype(float)
            - baseline.loc[base_window, "sensor.thermal.radiator_temp_c"].astype(float).to_numpy()
        ).mean())
        add_feature("FD-ONLINE-RADIATOR-PAYLOAD-TEMP-MEAN-DELTA-V1", "载荷温度配对均值变化", "sensor.thermal.payload_temp_c", FeatureAggregation.MEAN_PAIRED_RESIDUAL, "degC", payload_temp_mean_delta, "THERMAL_RADIATOR", "runtime-paired-mean-residual-v1", radiator_feature_mask)
        add_feature("FD-ONLINE-RADIATOR-TEMP-MEAN-DELTA-V1", "散热器温度配对均值变化", "sensor.thermal.radiator_temp_c", FeatureAggregation.MEAN_PAIRED_RESIDUAL, "degC", radiator_temp_mean_delta, "THERMAL_RADIATOR", "runtime-paired-mean-residual-v1", radiator_feature_mask)
        add_evidence("ED-ONLINE-RADIATOR-PAYLOAD-TEMP-V1", "THERMAL_RADIATOR_REJECTION_LOSS", "载荷温度相对正常基线升高", "FD-ONLINE-RADIATOR-PAYLOAD-TEMP-MEAN-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.04, "degC")
        add_evidence("ED-ONLINE-RADIATOR-TEMP-V1", "THERMAL_RADIATOR_REJECTION_LOSS", "散热器观测温度相对正常基线升高", "FD-ONLINE-RADIATOR-TEMP-MEAN-DELTA-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.05, "degC")

        # Complete loss of downlink service is separated from transmitter power
        # degradation by command-on/actual-off status and near-zero service rate.
        downlink_feature_mask = downlink_service_mask if bool(downlink_service_mask.any()) else window
        tx_command_fraction = float((frame.loc[downlink_feature_mask, "status.comm.transmitter_command_on"].astype(float) > 0.5).mean())
        tx_on_fraction = float((frame.loc[downlink_feature_mask, "status.comm.transmitter_on"].astype(float) > 0.5).mean())
        service_active_fraction = float((frame.loc[downlink_feature_mask, "comm.active"].astype(float) > 0.5).mean())
        downlink_ratio = float((
            frame.loc[downlink_feature_mask, "comm.downlink_rate_bps"].astype(float)
            / baseline.loc[downlink_feature_mask, "comm.downlink_rate_bps"].replace(0, np.nan).astype(float)
        ).fillna(0.0).max())
        storage_backlog_delta = float((
            frame.loc[downlink_feature_mask, "data.storage_bits"].astype(float)
            - baseline.loc[downlink_feature_mask, "data.storage_bits"].astype(float)
        ).clip(lower=0).max())
        add_feature("FD-ONLINE-DOWNLINK-COMMAND-FRACTION-V1", "发射机指令开启比例", "status.comm.transmitter_command_on", FeatureAggregation.TRUE_FRACTION, "1", tx_command_fraction, "COMM_DOWNLINK", "runtime-closed-window-v1", downlink_feature_mask)
        add_feature("FD-ONLINE-DOWNLINK-ACTUAL-ON-FRACTION-V1", "发射机实际开启比例", "status.comm.transmitter_on", FeatureAggregation.TRUE_FRACTION, "1", tx_on_fraction, "COMM_DOWNLINK", "runtime-closed-window-v1", downlink_feature_mask)
        add_feature("FD-ONLINE-DOWNLINK-SERVICE-ACTIVE-FRACTION-V1", "下行服务激活比例", "comm.active", FeatureAggregation.TRUE_FRACTION, "1", service_active_fraction, "COMM_DOWNLINK", "runtime-closed-window-v1", downlink_feature_mask)
        add_feature("FD-ONLINE-DOWNLINK-RATE-RATIO-V1", "下行服务速率剩余比例", "comm.downlink_rate_bps", FeatureAggregation.MAX_VALUE, "1", downlink_ratio, "COMM_DOWNLINK", "runtime-paired-ratio-v1", downlink_feature_mask)
        add_feature("FD-ONLINE-DOWNLINK-STORAGE-BACKLOG-V1", "存储积压相对正常运行增量", "data.storage_bits", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "bit", storage_backlog_delta, "COMM_DOWNLINK", "runtime-paired-residual-v1", downlink_feature_mask)
        add_evidence("ED-ONLINE-DOWNLINK-COMMAND-V1", "COMM_DOWNLINK_SERVICE_LOSS", "计划要求发射机开启", "FD-ONLINE-DOWNLINK-COMMAND-FRACTION-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.90, "1")
        add_evidence("ED-ONLINE-DOWNLINK-ACTUAL-OFF-V1", "COMM_DOWNLINK_SERVICE_LOSS", "发射机实际保持关闭", "FD-ONLINE-DOWNLINK-ACTUAL-ON-FRACTION-V1", ComparisonOperator.LESS_OR_EQUAL, 0.10, "1")
        add_evidence("ED-ONLINE-DOWNLINK-SERVICE-INACTIVE-V1", "COMM_DOWNLINK_SERVICE_LOSS", "下行服务状态保持未激活", "FD-ONLINE-DOWNLINK-SERVICE-ACTIVE-FRACTION-V1", ComparisonOperator.LESS_OR_EQUAL, 0.10, "1")
        add_evidence("ED-ONLINE-DOWNLINK-RATE-LOSS-V1", "COMM_DOWNLINK_SERVICE_LOSS", "下行服务速率接近归零", "FD-ONLINE-DOWNLINK-RATE-RATIO-V1", ComparisonOperator.LESS_OR_EQUAL, 0.10, "1")
        add_evidence("ED-ONLINE-DOWNLINK-STORAGE-BACKLOG-V1", "COMM_DOWNLINK_SERVICE_LOSS", "存储积压增加", "FD-ONLINE-DOWNLINK-STORAGE-BACKLOG-V1", ComparisonOperator.GREATER_OR_EQUAL, 1000000.0, "bit", critical=False)

        # Temperature sensor bias and drift share the same measured channel but are
        # separated by the slope of the E0 injector bias state.  The bias state is
        # never described as flight truth.
        temp_sensor_feature_mask = sensor_fault_mask if bool(sensor_fault_mask.any()) else window
        bias_abs_max = float(temperature_bias_series.loc[temp_sensor_feature_mask].abs().max())
        measured_temp_residual = float((
            frame.loc[temp_sensor_feature_mask, "sensor.thermal.payload_temp_c"].astype(float)
            - baseline.loc[temp_sensor_feature_mask, "sensor.thermal.payload_temp_c"].astype(float)
        ).abs().max())
        bias_end_abs = float(abs(temperature_bias_series.loc[temp_sensor_feature_mask].iloc[-1]))
        slope_abs = float(abs(sensor_bias_slope))
        add_feature("FD-ONLINE-TEMP-SENSOR-BIAS-MAX-V1", "温度传感器偏置状态绝对峰值", "thermal.temperature_sensor_bias_c", FeatureAggregation.MAX_VALUE, "degC", bias_abs_max, "PAYLOAD_TEMPERATURE_SENSOR", "runtime-mechanism-state-v1", temp_sensor_feature_mask)
        add_feature("FD-ONLINE-TEMP-SENSOR-BIAS-SLOPE-V1", "温度传感器偏置变化斜率绝对值", "thermal.temperature_sensor_bias_c", FeatureAggregation.LINEAR_SLOPE, "degC/s", slope_abs, "PAYLOAD_TEMPERATURE_SENSOR", "runtime-linear-slope-v1", temp_sensor_feature_mask)
        add_feature("FD-ONLINE-TEMP-SENSOR-RESIDUAL-V1", "温度测量相对配对正常残差峰值", "sensor.thermal.payload_temp_c", FeatureAggregation.MAX_ABS_PAIRED_RESIDUAL, "degC", measured_temp_residual, "PAYLOAD_TEMPERATURE_SENSOR", "runtime-paired-residual-v1", temp_sensor_feature_mask)
        add_feature("FD-ONLINE-TEMP-SENSOR-BIAS-END-V1", "温度传感器偏置状态末值绝对值", "thermal.temperature_sensor_bias_c", FeatureAggregation.MAX_VALUE, "degC", bias_end_abs, "PAYLOAD_TEMPERATURE_SENSOR", "runtime-last-value-v1", temp_sensor_feature_mask)
        add_evidence("ED-ONLINE-TEMP-SENSOR-BIAS-MAGNITUDE-V1", "THERMAL_TEMPERATURE_SENSOR_BIAS", "温度传感器存在稳定偏置", "FD-ONLINE-TEMP-SENSOR-BIAS-MAX-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.0, "degC")
        add_evidence("ED-ONLINE-TEMP-SENSOR-BIAS-SLOPE-V1", "THERMAL_TEMPERATURE_SENSOR_BIAS", "偏置斜率接近零以区别漂移", "FD-ONLINE-TEMP-SENSOR-BIAS-SLOPE-V1", ComparisonOperator.LESS_OR_EQUAL, 0.005, "degC/s")
        add_evidence("ED-ONLINE-TEMP-SENSOR-BIAS-RESIDUAL-V1", "THERMAL_TEMPERATURE_SENSOR_BIAS", "测量温度相对正常基线偏移", "FD-ONLINE-TEMP-SENSOR-RESIDUAL-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.0, "degC", critical=False)
        add_evidence("ED-ONLINE-TEMP-SENSOR-DRIFT-END-V1", "THERMAL_TEMPERATURE_SENSOR_DRIFT", "温度传感器漂移末值达到工程门槛", "FD-ONLINE-TEMP-SENSOR-BIAS-END-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.0, "degC")
        add_evidence("ED-ONLINE-TEMP-SENSOR-DRIFT-SLOPE-V1", "THERMAL_TEMPERATURE_SENSOR_DRIFT", "偏置状态呈持续变化斜率", "FD-ONLINE-TEMP-SENSOR-BIAS-SLOPE-V1", ComparisonOperator.GREATER_OR_EQUAL, 0.02, "degC/s")
        add_evidence("ED-ONLINE-TEMP-SENSOR-DRIFT-RESIDUAL-V1", "THERMAL_TEMPERATURE_SENSOR_DRIFT", "测量温度相对正常基线产生递增偏移", "FD-ONLINE-TEMP-SENSOR-RESIDUAL-V1", ComparisonOperator.GREATER_OR_EQUAL, 1.0, "degC", critical=False)

        result_by_id = {r.feature_definition_id: r for r in results}
        facts = [evaluate_evidence(e, result_by_id.get(e.feature_definition_id)) for e in edefs]
        unexplained = sorted(field for field, value in residual_max.items() if value > 1e-8)
        detected_mechanisms = {
            "THERMAL_HEATER_STUCK_ON": bool(heater_mask.any()),
            "ADCS_STAR_TRACKER_DROPOUT": bool(star_mask.any()),
            "ADCS_RW_FRICTION_INCREASE": bool(friction_mask.any()),
            "ADCS_RW_TORQUE_AUTHORITY_LOSS": bool(torque_mask.any()),
            "ADCS_RW_JAM": bool(jam_mask.any()),
            "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE": bool(battery_ir_mask.any()),
            "EPS_SOLAR_ARRAY_STRING_FAILURE": bool(solar_string_mask.any()),
            "ADCS_GYRO_BIAS_DRIFT": bool(gyro_bias_mask.any()),
            "COMM_TRANSMITTER_OUTPUT_POWER_LOSS": bool(transmitter_mask.any()),
            "DATA_STORAGE_WRITE_FAILURE": bool(storage_mask.any()),
            "EPS_BATTERY_CAPACITY_LOSS": bool(battery_capacity_mask.any()),
            "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS": bool(solar_efficiency_mask.any()),
            "THERMAL_RADIATOR_REJECTION_LOSS": bool(
                payload_temp_mean_delta >= 0.04 and radiator_temp_mean_delta >= 0.05
            ),
            "COMM_DOWNLINK_SERVICE_LOSS": bool(downlink_service_mask.any()),
            "THERMAL_TEMPERATURE_SENSOR_BIAS": bool(temperature_bias_mask.any()),
            "THERMAL_TEMPERATURE_SENSOR_DRIFT": bool(temperature_drift_mask.any()),
        }
        supported_direct = any(detected_mechanisms.values())
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        bundle = EpisodeEvidenceBundle(
            episode=episode,
            domain_id="WHOLE_SPACECRAFT_ONLINE",
            analysis_wheel_index=rw_index,
            analysis_component_instance="WHOLE_SPACECRAFT",
            selected_wheel_index=(
                rw_index if bool(friction_mask.any() or torque_mask.any())
                else jam_index if bool(jam_mask.any())
                else None
            ),
            selected_component_instance=(
                f"RW{rw_index}" if bool(friction_mask.any() or torque_mask.any())
                else f"RW{jam_index}" if bool(jam_mask.any())
                else None
            ),
            wheel_selection_status=(
                "MECHANISM_DETECTED"
                if bool(friction_mask.any() or torque_mask.any() or jam_mask.any())
                else "NOT_APPLICABLE"
            ),
            component_selection_status="MECHANISM_DETECTED" if supported_direct else ("NO_MECHANISM_DETECTED" if not unexplained else "NOT_APPLICABLE"),
            analysis_metadata={
                "generic_anomaly_field_count": len(unexplained),
                "generic_anomaly_fields": unexplained,
                "supported_mechanism_detected": supported_direct,
                "labels_and_truth_excluded_from_features": True,
                "runtime_mechanism_fields_are_E0_only": True,
                "registered_mechanisms_detected": sorted(
                    fault_id for fault_id, detected in detected_mechanisms.items() if detected
                ),
            },
            feature_definitions=defs,
            feature_results=results,
            evidence_definitions=edefs,
            evidence_facts=facts,
            source_protocol_hashes=self.semantics.validation.protocol_hashes,
            evaluation_metadata={
                "scenario_for_evaluation_only": manifest.get("scenario", ""),
                "simulation_event_count_context": manifest.get("event_count", 0),
                "simulation_event_descriptors_context": [
                    {
                        "id": event_item.get("id"),
                        "category": event_item.get("category"),
                        "effect": event_item.get("effect"),
                        "target": event_item.get("target"),
                        "start_s": event_item.get("start_s"),
                        "duration_s": event_item.get("duration_s"),
                    }
                    for event_item in manifest.get("events", [])
                ],
            },
            claim_boundary="E0 online simulation-to-diagnosis integration evidence; runtime mechanism state is not a flight telemetry claim.",
        )
        return bundle, gate
