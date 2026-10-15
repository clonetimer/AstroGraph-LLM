"""Operator-facing vocabulary for spacecraft maintenance UI.

Internal telemetry identifiers remain stable machine contracts.  This module only
maps them to concise Chinese labels for the maintenance presentation layer.
"""
from __future__ import annotations

import re


_EXACT_CHANNEL_ZH: dict[str, str] = {
    "adcs.pointing_error_deg": "姿态指向误差",
    "adcs.rw.command_execution_residual_norm_nm": "反作用轮执行总残差",
    "adcs.rw.speed_response_residual_norm_nm": "反作用轮转速响应总残差",
}

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^adcs\.rw\.measured_speed_rad_s_(\d+)$"), "反作用轮 #{wheel} 转速"),
    (re.compile(r"^adcs\.rw\.motor_current_a_(\d+)$"), "反作用轮 #{wheel} 电机电流"),
    (re.compile(r"^adcs\.rw\.command_execution_residual_nm_(\d+)$"), "反作用轮 #{wheel} 指令执行残差"),
    (re.compile(r"^adcs\.rw\.speed_response_residual_nm_(\d+)$"), "反作用轮 #{wheel} 转速响应残差"),
    (re.compile(r"^adcs\.rw\.command_torque_nm_(\d+)$"), "反作用轮 #{wheel} 指令扭矩"),
)


def channel_label_zh(channel: str, fallback: str | None = None) -> str:
    """Return a maintenance-operator label without changing the channel ID."""
    name = str(channel or "")
    if name in _EXACT_CHANNEL_ZH:
        return _EXACT_CHANNEL_ZH[name]
    for pattern, template in _PATTERNS:
        match = pattern.match(name)
        if match:
            # Dataset suffixes are zero-based; operator wheel numbers are one-based.
            return template.format(wheel=int(match.group(1)) + 1)
    return str(fallback or name)
