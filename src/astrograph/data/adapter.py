from __future__ import annotations

from pathlib import Path

import pandas as pd

from astrograph.schemas import AnomalyEvidence


# Canonical engineering units for the built-in semantic signal registry.
# Generic-series detection may run without units, but physical semantic mapping
# must fail closed when a registered signal unit is absent or incompatible.
SIGNAL_TO_UNIT = {
    "sensor.eps.battery_voltage_v": "V",
    "sensor.eps.battery_current_a": "A",
    "sensor.eps.battery_temp_c": "degC",
    "sensor.eps.solar_array_current_a": "A",
    "sensor.eps.solar_array_voltage_v": "V",
    "sensor.thermal.payload_temp_c": "degC",
    "sensor.thermal.radiator_temp_c": "degC",
    "sensor.adcs.gyro_rate_rad_s_x": "rad/s",
    "sensor.adcs.gyro_rate_rad_s_y": "rad/s",
    "sensor.adcs.gyro_rate_rad_s_z": "rad/s",
    "sensor.adcs.rw.speed_rad_s_0": "rad/s",
    "sensor.adcs.rw.speed_rad_s_1": "rad/s",
    "sensor.adcs.rw.speed_rad_s_2": "rad/s",
    "sensor.adcs.rw.motor_current_a_0": "A",
    "sensor.adcs.rw.winding_temp_c_0": "degC",
    "adcs.rw.command_torque_nm_0": "N*m",
    "comm.downlink_rate_bps": "bit/s",
    "sensor.comm.pa_current_a": "A",
    "sensor.comm.rf_output_power_w": "W",
    "counter.comm.packet_error_count": "1",
    "data.storage_bits": "bit",
}


def normalize_engineering_unit(value: str | None) -> str | None:
    if value is None:
        return None
    token=str(value).strip().lower().replace("°", "deg").replace(" ", "")
    aliases={
        "v":"v", "volt":"v", "volts":"v",
        "a":"a", "amp":"a", "amps":"a", "ampere":"a", "amperes":"a",
        "c":"degc", "degc":"degc", "celsius":"degc",
        "rad/s":"rad/s", "rad_s":"rad/s", "rads-1":"rad/s", "radsec-1":"rad/s",
        "n*m":"n*m", "nm":"n*m", "n-m":"n*m",
        "1":"1", "count":"1", "counts":"1",
        "bps":"bit/s", "bit/s":"bit/s", "bits/s":"bit/s",
        "w":"w", "watt":"w", "watts":"w",
        "bit":"bit", "bits":"bit",
    }
    return aliases.get(token, token)


def engineering_unit_matches(expected: str, actual: str | None) -> bool:
    normalized_actual=normalize_engineering_unit(actual)
    return normalized_actual is not None and normalized_actual==normalize_engineering_unit(expected)


SIGNAL_TO_ANOMALY = {
    "sensor.eps.battery_voltage_v": ("ANOM_BATTERY_VOLTAGE_DEVIATION", "电池端电压异常偏移"),
    "sensor.eps.battery_current_a": ("ANOM_BATTERY_CURRENT_DEVIATION", "电池电流异常偏移"),
    "sensor.eps.battery_temp_c": ("ANOM_BATTERY_TEMP_DEVIATION", "电池温度异常偏移"),
    "sensor.eps.solar_array_current_a": ("ANOM_SOLAR_CURRENT_DEVIATION", "太阳阵输出电流异常偏移"),
    "sensor.eps.solar_array_voltage_v": ("ANOM_SOLAR_VOLTAGE_DEVIATION", "太阳阵输出电压异常偏移"),
    "sensor.thermal.payload_temp_c": ("ANOM_PAYLOAD_TEMP_DEVIATION", "载荷温度异常偏移"),
    "sensor.thermal.radiator_temp_c": ("ANOM_RADIATOR_TEMP_DEVIATION", "散热器温度异常偏移"),
    "sensor.adcs.gyro_rate_rad_s_x": ("ANOM_GYRO_X_DEVIATION", "X轴角速度异常"),
    "sensor.adcs.gyro_rate_rad_s_y": ("ANOM_GYRO_Y_DEVIATION", "Y轴角速度异常"),
    "sensor.adcs.gyro_rate_rad_s_z": ("ANOM_GYRO_Z_DEVIATION", "Z轴角速度异常"),
    "sensor.adcs.rw.speed_rad_s_0": ("ANOM_RW0_SPEED_DEVIATION", "反作用轮0轮速异常"),
    "sensor.adcs.rw.speed_rad_s_1": ("ANOM_RW1_SPEED_DEVIATION", "反作用轮1轮速异常"),
    "sensor.adcs.rw.speed_rad_s_2": ("ANOM_RW2_SPEED_DEVIATION", "反作用轮2轮速异常"),
    "sensor.adcs.rw.motor_current_a_0": ("ANOM_RW0_CURRENT_DEVIATION", "反作用轮0电机电流异常"),
    "sensor.adcs.rw.winding_temp_c_0": ("ANOM_RW0_TEMP_DEVIATION", "反作用轮0绕组温度异常"),
    "adcs.rw.command_torque_nm_0": ("ANOM_RW0_TORQUE_DEVIATION", "反作用轮0指令力矩异常"),
    "comm.downlink_rate_bps": ("ANOM_DOWNLINK_RATE_DEVIATION", "下行速率异常"),
    "sensor.comm.pa_current_a": ("ANOM_PA_CURRENT_DEVIATION", "功放电流异常"),
    "sensor.comm.rf_output_power_w": ("ANOM_RF_POWER_DEVIATION", "射频输出功率异常"),
    "counter.comm.packet_error_count": ("ANOM_PACKET_ERROR_INCREASE", "通信误包计数异常"),
    "data.storage_bits": ("ANOM_STORAGE_ACCUMULATION", "星上存储积压异常"),
}


def residual_strength_confidence(peak: float, threshold: float) -> float:
    """Residual strength in [0, 1]. It is not a calibrated fault probability."""
    peak = max(0.0, float(peak))
    threshold = max(1e-12, float(threshold))
    return min(0.999, peak / (peak + threshold))


def load_case_anomalies(path: str | Path, case_id: str) -> list[AnomalyEvidence]:
    frame = pd.read_csv(path)
    detected = frame["detected"].astype(str).str.lower().isin({"true", "1"})
    rows = frame[(frame["case_id"].astype(str) == case_id) & detected].copy()
    best: dict[str, AnomalyEvidence] = {}
    for _, row in rows.iterrows():
        signal = str(row["signal"])
        mapped = SIGNAL_TO_ANOMALY.get(signal)
        if mapped is None:
            continue
        anomaly_id, name = mapped
        evidence = AnomalyEvidence(
            anomaly_id=anomaly_id,
            name=name,
            signal=signal,
            confidence=residual_strength_confidence(row["peak_residual_abs"], row["threshold_abs"]),
            observed_at_s=None if pd.isna(row.get("detection_time_s")) else float(row["detection_time_s"]),
            metadata={
                "case_id": case_id,
                "peak_residual_abs": float(row["peak_residual_abs"]),
                "threshold_abs": float(row["threshold_abs"]),
                "detection_delay_s": None if pd.isna(row.get("detection_delay_s")) else float(row["detection_delay_s"]),
                "source": "quick_fault_detection",
            },
        )
        previous = best.get(anomaly_id)
        if previous is None or evidence.confidence > previous.confidence:
            best[anomaly_id] = evidence
    return sorted(best.values(), key=lambda item: item.anomaly_id)
