from __future__ import annotations

import math

import numpy as np
import pandas as pd

from astrograph.conditions.catalog import ConditionCatalog
from astrograph.conditions.models import OperatingCondition


def _fraction_category(value: float, active_id: str, inactive_id: str, mixed_id: str, catalog: ConditionCatalog) -> str:
    if math.isnan(value):
        return "UNKNOWN"
    if value >= catalog.thresholds.active_fraction:
        return active_id
    if value <= catalog.thresholds.inactive_fraction:
        return inactive_id
    return mixed_id


class OperatingConditionResolver:
    """Deterministic E0 resolver for the current synthetic PHM streams.

    The resolver is deliberately configuration-driven.  A future simulation-side
    official condition stream can replace this resolver without changing the
    TelemetryWindow protocol.
    """

    resolver_id = "deterministic_condition_resolver_v2"

    def __init__(self, catalog: ConditionCatalog):
        self.catalog = catalog

    def resolve(self, legacy: pd.DataFrame, adcs: pd.DataFrame) -> OperatingCondition:
        thresholds = self.catalog.thresholds
        solar_current = pd.to_numeric(
            legacy.get("sensor.eps.solar_array_current_a", pd.Series(dtype=float)), errors="coerce"
        )
        sunlight_fraction = (
            float((solar_current > thresholds.solar_current_active_a).mean()) if not solar_current.empty else math.nan
        )
        orbit_phase = _fraction_category(
            sunlight_fraction, "SUNLIGHT", "ECLIPSE", "ORBIT_PHASE_TRANSITION", self.catalog
        )

        payload = pd.to_numeric(legacy.get("payload.active", pd.Series(dtype=float)), errors="coerce")
        payload_fraction = (
            float((payload > thresholds.payload_active_threshold).mean()) if not payload.empty else math.nan
        )
        payload_mode = _fraction_category(
            payload_fraction, "PAYLOAD_ACTIVE", "PAYLOAD_IDLE", "PAYLOAD_TRANSITION", self.catalog
        )

        transmitter = (
            legacy.get("status.comm.transmitter_on", pd.Series(dtype=bool))
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        )
        rate = pd.to_numeric(legacy.get("comm.downlink_rate_bps", pd.Series(dtype=float)), errors="coerce")
        downlink_fraction = (
            float((transmitter | (rate > thresholds.downlink_rate_active_bps)).mean()) if len(legacy) else math.nan
        )
        communication_mode = _fraction_category(
            downlink_fraction, "DOWNLINK_ACTIVE", "DOWNLINK_IDLE", "DOWNLINK_TRANSITION", self.catalog
        )

        torque_cols = [column for column in adcs.columns if column.startswith("adcs.rw.command_torque_nm_")]
        gyro_cols = [column for column in adcs.columns if column.startswith("sensor.adcs.gyro_rate_rad_s_")]
        max_torque = float(adcs[torque_cols].abs().max().max()) if torque_cols and not adcs.empty else 0.0
        gyro_norm = 0.0
        if gyro_cols and not adcs.empty:
            values = adcs[gyro_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            gyro_norm = float(np.linalg.norm(values, axis=1).max())
        attitude_mode = (
            "ATTITUDE_MANEUVER"
            if max_torque > thresholds.attitude_torque_maneuver_nm
            or gyro_norm > thresholds.gyro_rate_maneuver_rad_s
            else "STEADY_POINTING"
        )

        condition_ids = sorted({orbit_phase, payload_mode, communication_mode, attitude_mode} - {"UNKNOWN"})
        unknown_dimensions = [
            key
            for key, value in {
                "orbit_phase": orbit_phase,
                "payload_mode": payload_mode,
                "communication_mode": communication_mode,
                "attitude_mode": attitude_mode,
            }.items()
            if value == "UNKNOWN"
        ]
        return OperatingCondition(
            orbit_phase=orbit_phase,
            payload_mode=payload_mode,
            communication_mode=communication_mode,
            attitude_mode=attitude_mode,
            condition_ids=condition_ids,
            catalog_version=self.catalog.catalog_version,
            inference_method=self.resolver_id,
            confidence="E0_SYNTHETIC_HEURISTIC",
            unknown_dimensions=unknown_dimensions,
            metadata={
                "sunlight_fraction": None if math.isnan(sunlight_fraction) else sunlight_fraction,
                "payload_active_fraction": None if math.isnan(payload_fraction) else payload_fraction,
                "downlink_active_fraction": None if math.isnan(downlink_fraction) else downlink_fraction,
                "max_command_torque_nm": max_torque,
                "max_gyro_norm_rad_s": gyro_norm,
            },
        )
