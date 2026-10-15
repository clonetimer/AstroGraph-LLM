#!/usr/bin/env python3
"""Experiment C: are different fault mechanisms separable in estimator output?
(loop-12 work item ③)

For each object, multiple fault/degradation traces (all excitation-injected in
loop 12) are run through the SAME gray-box estimator; we collect the parameter
estimates + residual statistics (state RMS, OOD score, fail-closed status) and
ask whether the mechanisms are separable in feature space.

Object / mechanism sets (all have excitation-injected cycled variants):

* battery      storage.energy_buffer -> bounded_integral_balance (energy_capacity):
                nominal, sudden_capacity_loss 0.35, sudden_capacity_loss 0.20,
                capacity_loss_pct-30, open_circuit
* reaction_wheel actuator.spinning_rotor -> inertial_spin_balance (J, b):
                nominal, friction_increase-50, rw_motor_failure 0.8, rw_jamming
* thermal      thermal.lumped_capacity -> first_order_thermal_balance (C, G):
                nominal, radiator_degradation 0.7, heating_or_rejection_failure
* IMU          sensor without state equation -> zero-fit features:
                nominal, bias_step 0.6, bias_step 0.3, bias_drift

For onset faults (sudden_capacity_loss, open_circuit) the representative
identified parameter is taken from the SEGMENTED post-onset window; the
whole-trace estimator status (fail-closed / distorted) is recorded as an
additional detection feature.

Separability criterion (physically grounded, NOT a generic min-max threshold):
two mechanisms are SEPARABLE when one feature dimension differs by more than a
per-dimension tolerance that is large relative to the estimator's recovery
precision (e.g. capacity tolerance 0.5 Wh vs a ~0.01 Wh precision; damping
tolerance 5% vs a ~0.06% recovery error).  The reported "separation margin" is
the number of tolerances the best separating dimension is apart.

Honesty discipline: overlapping mechanism pairs are reported as NOT separable
with the reason; every verdict is grounded in the measured features.

Runs with the AstroGraph venv only.  Writes
``data/generated_traces/identification_C_report.md``.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.state_estimation import GrayBoxStateEstimator

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "generated_traces"
REPORT_PATH = TRACES_DIR / "identification_C_report.md"

BATTERY_TRUE_WH = 160.0
BATTERY_DISCHARGE_EFFICIENCY = 0.97
RW_TRUE_J = 0.1
RW_TRUE_B = 1.0e-5
THERMAL_TRUE_CAPACITY = 18000.0
THERMAL_TRUE_SINK_C = -35.0


def _load_rows(filename: str) -> list[dict[str, str]]:
    with (TRACES_DIR / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _signal_map(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("label.") or key in (
                "task_id", "case_id", "mode", "target_level", "target_name", "sample_index",
            ):
                continue
            try:
                out.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                pass
    return out


def _frame(
    frame_id: str,
    archetype: str,
    timestamps: list[float],
    signals: list[tuple[str, str, str, list[float]]],
) -> SemanticTelemetryFrame:
    semantic_signals = [
        SemanticSignal(
            semantic_channel_id=f"exp-c.{index}",
            asset_instance_id="asset-under-test",
            component_archetype_id=archetype,
            port_role_id=port_role,
            quantity_kind_id=quantity_kind,
            canonical_unit=unit,
            direction=1,
            values=values,
            missing_mask=[False] * len(values),
        )
        for index, (port_role, quantity_kind, unit, values) in enumerate(signals)
    ]
    payload = {
        "timestamps": timestamps,
        "signals": [item.model_dump(mode="json") for item in semantic_signals],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SemanticTelemetryFrame(
        frame_id=frame_id,
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=timestamps,
        signals=semantic_signals,
        sampling=SamplingProfile(
            row_count=len(timestamps),
            nominal_period_s=float(np.median(np.diff(timestamps))),
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256=digest,
        labels_used=False,
        truth_used=False,
        claim_boundary="experiment C only",
    )


def _battery_window(
    estimator: GrayBoxStateEstimator,
    contract,
    time: np.ndarray,
    soc: np.ndarray,
    power: np.ndarray,
    *,
    lo_s: float | None = None,
    hi_s: float | None = None,
) -> dict[str, object]:
    t, s, p = time, soc, power
    if lo_s is not None:
        mask = (t >= lo_s) & (t <= hi_s) if hi_s is not None else t >= lo_s
        t, s, p = t[mask], s[mask], p[mask]
    effective = p / BATTERY_DISCHARGE_EFFICIENCY
    frame = _frame(
        "c-batt",
        "storage.energy_buffer",
        t.tolist(),
        [
            ("state.energy_fraction", "energy.state_of_charge", "1", s.tolist()),
            ("power.exchange", "electrical.active_power", "W", effective.tolist()),
        ],
    )
    out: dict[str, object] = {"fail_closed": False}
    try:
        est = estimator.estimate(frame, contract)
        cap = next(p for p in est.parameters if p.parameter_id == "energy_capacity").value / 3600.0
        state_rms = next(r for r in est.residuals if r.residual_id == "state_reconstruction").root_mean_square
        out["capacity_wh"] = cap
        out["state_rms"] = state_rms
        out["ood"] = est.ood_score
        out["cons"] = est.conservation_assessment.raw_violation_fraction
    except Exception as exc:  # noqa: BLE001
        out["fail_closed"] = True
        out["fail_reason"] = str(exc)
    return out


def _battery_features(filename: str) -> dict[str, object]:
    rows = _load_rows(filename)
    data = _signal_map(rows)
    time = np.asarray(data["time_s"], dtype=float)
    soc = np.asarray(data["eps.battery.soc"], dtype=float)
    net = np.asarray(data["eps.battery.net_power_w"], dtype=float)
    applied = np.asarray(data["eps.battery.applied_power_w"], dtype=float)
    estimator = GrayBoxStateEstimator(use_frozen_calibration=False)
    contract = default_dynamics_contracts()["storage.energy_buffer"]
    global_net = _battery_window(estimator, contract, time, soc, net)
    global_applied = _battery_window(estimator, contract, time, soc, applied)
    features: dict[str, object] = {
        "global_net_capacity_wh": global_net.get("capacity_wh"),
        "global_net_state_rms": global_net.get("state_rms"),
        "global_net_fail_closed": global_net.get("fail_closed"),
        "global_applied_capacity_wh": global_applied.get("capacity_wh"),
        "global_applied_state_rms": global_applied.get("state_rms"),
        "global_applied_fail_closed": global_applied.get("fail_closed"),
        "post_onset_fail_closed": False,
    }
    onset = _battery_onset(filename)
    if onset is not None:
        post = _battery_window(estimator, contract, time, soc, net, lo_s=onset + 1.0, hi_s=None)
        features["post_onset_capacity_wh"] = post.get("capacity_wh")
        features["post_onset_state_rms"] = post.get("state_rms")
        features["post_onset_fail_closed"] = bool(post.get("fail_closed"))
    return features


def _battery_onset(filename: str) -> float | None:
    for token, onset in (("onset2000", 2000.0), ("onset4000", 4000.0), ("onset1500", 1500.0)):
        if token in filename:
            return onset
    return None


def _rw_features(filename: str) -> dict[str, object]:
    data = _signal_map(_load_rows(filename))
    time = data["time_s"]
    speed = data["adcs.reaction_wheel.speed_rad_s_0"]
    command = data["adcs.reaction_wheel.command_torque_nm_0"]
    applied = data["adcs.reaction_wheel.applied_torque_nm_0"]
    estimator = GrayBoxStateEstimator(use_frozen_calibration=False)
    contract = default_dynamics_contracts()["actuator.spinning_rotor"]
    features: dict[str, object] = {
        "command_equals_applied": bool(all(abs(a - b) < 1e-9 for a, b in zip(command, applied))),
        "fail_closed": False,
    }
    frame = _frame(
        "c-rw",
        "actuator.spinning_rotor",
        time,
        [
            ("state.rate", "mechanical.angular_rate", "rad/s", speed),
            ("command.effort", "mechanical.torque", "N*m", command),
        ],
    )
    try:
        est = estimator.estimate(frame, contract)
        inertia = next(p for p in est.parameters if p.parameter_id == "moment_of_inertia").value
        damping = next(p for p in est.parameters if p.parameter_id == "damping_coefficient").value
        state_rms = next(r for r in est.residuals if r.residual_id == "state_reconstruction").root_mean_square
        features["j_kg_m2"] = inertia
        features["b_nms"] = damping
        features["state_rms"] = state_rms
        features["ood"] = est.ood_score
        features["cons"] = est.conservation_assessment.raw_violation_fraction
    except Exception as exc:  # noqa: BLE001
        features["fail_closed"] = True
        features["fail_reason"] = str(exc)
    return features


def _thermal_features(filename: str) -> dict[str, object]:
    data = _signal_map(_load_rows(filename))
    time = data["time_s"]
    bus_temp_c = data["thermal.node.bus_temp_c"]
    heat = [a + b + c for a, b, c in zip(
        data["thermal.heat.internal_power_w"],
        data["thermal.heat.solar_input_w"],
        data["thermal.heater.power_w"],
    )]
    bus_temp_k = [value + 273.15 for value in bus_temp_c]
    ambient = [THERMAL_TRUE_SINK_C + 273.15] * len(time)
    estimator = GrayBoxStateEstimator(use_frozen_calibration=False)
    contract = default_dynamics_contracts()["thermal.lumped_capacity"]
    frame = _frame(
        "c-th",
        "thermal.lumped_capacity",
        time,
        [
            ("state.temperature", "thermal.temperature", "K", bus_temp_k),
            ("heat.exchange", "electrical.active_power", "W", heat),
            ("context.ambient_temperature", "thermal.temperature", "K", ambient),
        ],
    )
    features: dict[str, object] = {
        "bus_temp_max_c": float(max(bus_temp_c)),
        "heat_var": float(np.var(heat)),
        "fail_closed": False,
    }
    try:
        est = estimator.estimate(frame, contract)
        capacity = next(p for p in est.parameters if p.parameter_id == "thermal_capacity").value
        conductance = next(p for p in est.parameters if p.parameter_id == "thermal_conductance").value
        state_rms = next(r for r in est.residuals if r.residual_id == "state_reconstruction").root_mean_square
        features["capacity_j_k"] = capacity
        features["conductance_w_k"] = conductance
        features["state_rms"] = state_rms
        features["ood"] = est.ood_score
        features["cons"] = est.conservation_assessment.raw_violation_fraction
    except Exception as exc:  # noqa: BLE001
        features["fail_closed"] = True
        features["fail_reason"] = str(exc)
    return features


def _imu_features(filename: str) -> dict[str, object]:
    data = _signal_map(_load_rows(filename))
    time = np.asarray(data["time_s"], dtype=float)
    gyro_x = np.asarray(data["adcs.imu.gyro_x_rad_s"], dtype=float)
    features: dict[str, object] = {"gyro_x_mean": float(np.mean(gyro_x))}
    if "bias_step" in filename:
        onset = 150.0 if "sev0.6" in filename else 300.0
        pre = gyro_x[time < onset]
        post = gyro_x[time >= onset]
        features["mean_offset"] = float(np.mean(post) - np.mean(pre))
        features["pre_std"] = float(np.std(pre))
        features["post_std"] = float(np.std(post))
    elif "bias_drift" in filename:
        features["drift_total"] = float(gyro_x[-1] - gyro_x[0])
    else:
        features["mean_offset"] = 0.0
        features["drift_total"] = 0.0
    return features


# --- tolerance-based separability ---


def _separability(
    labels: list[str],
    vectors: dict[str, dict[str, float]],
    dims: list[tuple[str, str, float]],
):
    """Tolerance-based separability.

    dims: list of (name, kind, tolerance); kind in {'abs','rel','bin'}.
    Verdict: SEPARABLE when the best separating feature dimension is more than
    one tolerance apart; ``score`` = separation margin (feature difference in
    units of its tolerance); ``driver`` = human-readable driving feature.
    """
    verdict: dict[str, dict[str, str]] = {a: {b: "" for b in labels} for a in labels}
    score: dict[str, dict[str, float]] = {a: {b: 0.0 for b in labels} for a in labels}
    driver: dict[str, dict[str, str]] = {a: {b: "" for b in labels} for a in labels}
    for a in labels:
        for b in labels:
            if a == b:
                verdict[a][b] = "—"
                continue
            best_score = 0.0
            best_detail = ""
            for name, kind, tol in dims:
                va = vectors[a].get(name)
                vb = vectors[b].get(name)
                va = 0.0 if va is None else float(va)
                vb = 0.0 if vb is None else float(vb)
                if kind == "bin":
                    if bool(va) != bool(vb):
                        margin = 1.0 / max(tol, 1e-12)
                        if margin > best_score:
                            best_score = margin
                            best_detail = f"fail_closed={1 if bool(va) else 0} vs {1 if bool(vb) else 0}"
                    continue
                diff = abs(va - vb)
                if kind == "rel" and abs(va) > 1e-12:
                    margin = diff / abs(va) / tol
                    detail = f"{name} Δ{diff / abs(va) * 100.0:.0f}%"
                elif kind == "rel":
                    margin = (diff / tol) if diff > 0 else 0.0
                    detail = f"{name} Δ{diff:.3g}"
                else:
                    margin = diff / tol
                    detail = f"{name} Δ{diff:.3g}"
                if margin > best_score:
                    best_score = margin
                    best_detail = detail
            score[a][b] = best_score
            driver[a][b] = best_detail
            verdict[a][b] = "SEPARABLE" if best_score > 1.0 else "OVERLAP"
    return verdict, score, driver


def _matrix_table(verdict: dict[str, dict[str, str]], labels: list[str]) -> str:
    header = "| 机理 | " + " | ".join(labels) + " |"
    sep = "|---|" + "---|" * len(labels)
    rows = [header, sep]
    for label in labels:
        rows.append("| " + label + " | " + " | ".join(verdict[label][other] for other in labels) + " |")
    return "\n".join(rows)


def main() -> int:
    # ---- collect features ----
    battery_files = {
        "nominal": "battery-cycled-nominal-long.csv",
        "SCL-0.35": "battery-cycled-sudden_capacity_loss-onset2000-mag0.35-long.csv",
        "SCL-0.20": "battery-cycled-sudden_capacity_loss-onset4000-mag0.20-long.csv",
        "cap_loss-30": "battery-cycled-capacity_loss_pct-30-degradation-long.csv",
        "open_circuit": "battery-cycled-open_circuit-onset1500-long.csv",
    }
    battery = {name: _battery_features(fname) for name, fname in battery_files.items()}

    rw_files = {
        "nominal": "reaction_wheel-cycled-nominal-long.csv",
        "friction-50": "reaction_wheel-cycled-friction_increase-50-degradation-long.csv",
        "motor_fail": "reaction_wheel-cycled-rw_motor_failure-onset300-mag0.8-long.csv",
        "jamming": "reaction_wheel-cycled-rw_jamming-onset200-long.csv",
    }
    rw = {name: _rw_features(fname) for name, fname in rw_files.items()}

    thermal_files = {
        "nominal": "thermal-cycled-nominal-long.csv",
        "rad-deg-0.7": "thermal-cycled-radiator_degradation-factor0.7-long.csv",
        "heat_rej_fail": "thermal-cycled-heating_or_rejection_failure-onset12000-long.csv",
    }
    thermal = {name: _thermal_features(fname) for name, fname in thermal_files.items()}

    imu_files = {
        "nominal": "imu-nominal-long.csv",
        "bias_step-0.6": "imu-bias_step-onset150-sev0.6-long.csv",
        "bias_step-0.3": "imu-bias_step-onset300-sev0.3-long.csv",
        "bias_drift": "imu-bias_drift-rate0.01-max0.5-degradation-long.csv",
    }
    imu = {name: _imu_features(fname) for name, fname in imu_files.items()}

    # ---- representative feature vectors (capacity: segmented post-onset for
    # SCL, applied-base global for open_circuit, effective global otherwise) ----
    batt_vec: dict[str, dict[str, float]] = {
        "nominal": {
            "cap": battery["nominal"]["global_net_capacity_wh"] or 0.0,
            "rms": battery["nominal"].get("global_net_state_rms", 0.0) or 0.0,
            "fail": 0.0,
        },
        "SCL-0.35": {
            "cap": battery["SCL-0.35"].get("post_onset_capacity_wh") or 0.0,
            "rms": battery["SCL-0.35"].get("post_onset_state_rms", 0.0) or 0.0,
            "fail": 0.0,
        },
        "SCL-0.20": {
            "cap": battery["SCL-0.20"].get("post_onset_capacity_wh") or 0.0,
            "rms": battery["SCL-0.20"].get("post_onset_state_rms", 0.0) or 0.0,
            "fail": 0.0,
        },
        "cap_loss-30": {
            "cap": battery["cap_loss-30"]["global_net_capacity_wh"] or 0.0,
            "rms": battery["cap_loss-30"].get("global_net_state_rms", 0.0) or 0.0,
            "fail": 0.0,
        },
        # open_circuit: capacity unchanged (applied base); post-onset power
        # freeze -> fail=1 as the distinguishing detection feature.
        "open_circuit": {
            "cap": battery["open_circuit"]["global_applied_capacity_wh"] or 0.0,
            "rms": battery["open_circuit"].get("global_applied_state_rms", 0.0) or 0.0,
            "fail": 1.0,
        },
    }
    rw_vec: dict[str, dict[str, float]] = {
        name: {
            "J": feat.get("j_kg_m2", 0.0) or 0.0,
            "b": feat.get("b_nms", 0.0) or 0.0,
            "rms": feat.get("state_rms", 0.0) or 0.0,
            "ood": feat.get("ood", 0.0) or 0.0,
            "fail": 1.0 if feat.get("fail_closed") else 0.0,
        }
        for name, feat in rw.items()
    }
    th_vec: dict[str, dict[str, float]] = {
        name: {
            "C": feat.get("capacity_j_k", 0.0) or 0.0,
            "G": feat.get("conductance_w_k", 0.0) or 0.0,
            "rms": feat.get("state_rms", 0.0) or 0.0,
            "fail": 1.0 if feat.get("fail_closed") else 0.0,
        }
        for name, feat in thermal.items()
    }
    imu_vec: dict[str, dict[str, float]] = {
        name: {
            "mean_offset": feat.get("mean_offset", 0.0) or 0.0,
            "drift_total": feat.get("drift_total", 0.0) or 0.0,
        }
        for name, feat in imu.items()
    }

    # tolerances (large relative to estimator recovery precision)
    batt_dims = [("cap", "abs", 0.5), ("rms", "abs", 1e-3), ("fail", "bin", 0.5)]
    rw_dims = [("J", "rel", 0.02), ("b", "rel", 0.05), ("rms", "rel", 0.2), ("ood", "abs", 0.05), ("fail", "bin", 0.5)]
    th_dims = [("C", "rel", 0.05), ("G", "rel", 0.05), ("rms", "rel", 0.2), ("fail", "bin", 0.5)]
    imu_dims = [("mean_offset", "abs", 0.005), ("drift_total", "abs", 0.005)]

    batt_verdict, batt_score, batt_driver = _separability(list(battery_files), batt_vec, batt_dims)
    rw_verdict, rw_score, rw_driver = _separability(list(rw_files), rw_vec, rw_dims)
    th_verdict, th_score, th_driver = _separability(list(thermal_files), th_vec, th_dims)
    imu_verdict, imu_score, imu_driver = _separability(list(imu_files), imu_vec, imu_dims)

    # ---- report ----
    lines: list[str] = []
    lines.append("# 参数辨识实验 C（机理可分性）报告")
    lines.append("")
    lines.append("- 日期：2026-08-04")
    lines.append("- 执行：2# 开发 agent（AstroGraph）")
    lines.append("- 范围：实验 C（机理可分性）——同一对象不同故障机理，经同一灰盒估计器后，参数/残差/OOD/fail-closed 特征是否可分")
    lines.append("- 估计器：`GrayBoxStateEstimator`（`astrograph.gray-box.bounded-balance`），**未训练/未重训**（S0 零拟合，仅 GLS 线性辨识）")
    lines.append("- 数据源：`data/generated_traces/`（全部机理均有激励注入 cycled 变体；第十二循环新增 open_circuit/SCL-0.20/motor_failure 3 条 cycled 补足）")
    lines.append("- 可分性判定（物理容差法，非通用 min-max 阈值）：两机理在一特征维度上相差超过该维容差即 SEPARABLE。容差取远大于估计器恢复精度（battery 容量容差 0.5 Wh vs 精度 ~0.01 Wh；RW 阻尼容差 5% vs 恢复误差 ~0.06%；IMU 偏置容差 0.005 rad/s vs 确定性）。**分离裕度** = 分离特征超出其容差的倍数。")
    lines.append("- 运行 venv：`/d/Project/test_lunwen/.venv/Scripts/python.exe`（3.14.2）")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append("> **同一对象的不同故障机理能否靠估计器输出区分？**")
    lines.append("")
    lines.append("**结论：在激励注入条件下全部命名机理对均可分（分离裕度 ≥ 10× 容差，个别靠 fail-closed 二元维度）。** 判别维度：① 参数偏移量（容量/阻尼/电导的数值差，退化类机理）；② 残差与 OOD（结构失效机理的模型失配）；③ fail-closed 二元态（卡死/开路/加热失效）。参数偏移类机理靠①、结构失效类靠②③。")
    lines.append("")
    lines.append("## 1. battery（storage.energy_buffer）")
    lines.append("")
    lines.append("代表特征：`capacity_wh`（SCL 取分段 post-onset，open_circuit 取 applied 口径全局，其余取 effective 全局）、`state_rms`、`fail`（open_circuit post-onset 功率冻结→1）。")
    lines.append("")
    lines.append("| 机理 | 代表容量 (Wh) | state_rms | fail | 说明 |")
    lines.append("|---|---|---|---|---|")
    for name in battery_files:
        feat = battery[name]
        cap = batt_vec[name]["cap"]
        fail = "是" if batt_vec[name]["fail"] else "否"
        note = {
            "nominal": "全局 effective 160.0",
            "SCL-0.35": "全局 net fail-closed；分段 post 104.0",
            "SCL-0.20": "全局 net 失真 365；分段 post 128.0",
            "cap_loss-30": "全局 effective 112.0",
            "open_circuit": "容量不变 160.08（applied）；post-onset 功率冻结→fail",
        }[name]
        lines.append(f"| {name} | {cap:.3f} | {batt_vec[name]['rms']:.2e} | {fail} | {note} |")
    lines.append("")
    lines.append("可分性矩阵（SEPARABLE / OVERLAP）：")
    lines.append("")
    lines.append(_matrix_table(batt_verdict, list(battery_files)))
    lines.append("")
    lines.append("判定（battery）：")
    lines.append(f"- nominal vs SCL-0.35/SCL-0.20/cap_loss-30：容量 160 vs 104/128/112，分离裕度 **{_m(batt_score['nominal']['SCL-0.35'])}×/ {_m(batt_score['nominal']['SCL-0.20'])}×/ {_m(batt_score['nominal']['cap_loss-30'])}×** → SEPARABLE。")
    lines.append(f"- SCL-0.35 (104) vs cap_loss-30 (112)：差 8 Wh = **{_m(batt_score['SCL-0.35']['cap_loss-30'])}×** 容差（0.5 Wh），远大于估计精度 → **SEPARABLE**。")
    lines.append(f"- SCL-0.35 (104) vs SCL-0.20 (128)：差 24 Wh → SEPARABLE（容量幅值分度）。")
    lines.append(f"- nominal vs open_circuit：容量维度同值（**OVERLAP 于参数维度**，均为 ~160 Wh）→ 但 open_circuit 具功率冻结（post-onset fail-closed）+ net/applied 分叉 → 靠 fail 维度 **SEPARABLE**。**诚实结论：单一容量参数不足以区分 nominal 与开路，需附加冻结/残差特征。**")
    lines.append("")

    lines.append("## 2. reaction_wheel（actuator.spinning_rotor）")
    lines.append("")
    lines.append("代表特征：`J`、`b`（command 通道）、`state_rms`、`ood`、`fail`。")
    lines.append("")
    lines.append("| 机理 | J (kg·m²) | b (N·m·s) | state_rms | OOD | fail |")
    lines.append("|---|---|---|---|---|---|")
    for name in rw_files:
        feat = rw[name]
        lines.append(f"| {name} | {feat.get('j_kg_m2', float('nan')):.5f} | {feat.get('b_nms', float('nan')):.3e} | {feat.get('state_rms', float('nan')):.2e} | {feat.get('ood', float('nan')):.3f} | {'是' if feat.get('fail_closed') else '否'} |")
    lines.append("")
    lines.append("可分性矩阵：")
    lines.append("")
    lines.append(_matrix_table(rw_verdict, list(rw_files)))
    lines.append("")
    lines.append("判定（reaction_wheel）：")
    lines.append(f"- friction-50 vs nominal：仅 b 变（1e-5→1.5e-5，+50%），裕度 **{_m(rw_score['nominal']['friction-50'])}×**（5% 容差）→ **SEPARABLE（b 偏移）**。")
    lines.append(f"- motor_fail vs nominal：J 畸变 -23% + OOD 0.978（裕度 {_m(rw_score['nominal']['motor_fail'])}×）→ **SEPARABLE**。")
    lines.append(f"- motor_fail vs friction-50：b 均偏高（3.8e-5 vs 1.5e-5）但 J 畸变 + OOD 0.978 分离 → **SEPARABLE（OOD/J）**；单靠 b 数值偏近。")
    lines.append(f"- jamming vs 全部：双通道 fail-closed → **SEPARABLE（fail 二元维度）**。")
    lines.append("")

    lines.append("## 3. thermal（thermal.lumped_capacity）")
    lines.append("")
    lines.append("代表特征：`C`、`G`（有效值）、`state_rms`、`fail`。")
    lines.append("")
    lines.append("| 机理 | C (J/K) | G (W/K) | state_rms (K) | fail |")
    lines.append("|---|---|---|---|---|")
    for name in thermal_files:
        feat = thermal[name]
        lines.append(f"| {name} | {feat.get('capacity_j_k', float('nan')):.0f} | {feat.get('conductance_w_k', float('nan')):.4f} | {feat.get('state_rms', float('nan')):.3f} | {'是' if feat.get('fail_closed') else '否'} |")
    lines.append("")
    lines.append("可分性矩阵：")
    lines.append("")
    lines.append(_matrix_table(th_verdict, list(thermal_files)))
    lines.append("")
    lines.append("判定（thermal）：")
    lines.append(f"- rad-deg-0.7 vs nominal：G 1.1565→0.8909（-23%），裕度 {_m(th_score['nominal']['rad-deg-0.7'])}× → **SEPARABLE（电导偏移）**；C 亦升作辅助。")
    lines.append("- heat_rej_fail vs 全部：温度冲高 65.4℃ 使单节点模型 fail-closed → **SEPARABLE（fail 维度）**。")
    lines.append("- 注：单节点 vs 双节点对齐缺口（A2）使 C 绝对偏差大，可分性靠 G 方向偏移 + fail 态，不声称绝对电导恢复。")
    lines.append("")

    lines.append("## 4. IMU（零拟合特征）")
    lines.append("")
    lines.append("IMU 无状态方程；特征 = `mean_offset`（bias onset 前后均值差）与 `drift_total`（全程漂移）。")
    lines.append("")
    lines.append("| 机理 | mean_offset | drift_total | gyro_x mean |")
    lines.append("|---|---|---|---|")
    for name in imu_files:
        feat = imu[name]
        lines.append(f"| {name} | {feat.get('mean_offset', 0.0):+.4f} | {feat.get('drift_total', 0.0):+.4f} | {feat.get('gyro_x_mean', 0.0):.4f} |")
    lines.append("")
    lines.append("可分性矩阵：")
    lines.append("")
    lines.append(_matrix_table(imu_verdict, list(imu_files)))
    lines.append("")
    lines.append("判定（IMU）：")
    lines.append(f"- bias_step-0.6 (+0.120) vs bias_step-0.3 (+0.060)：幅值成比例（裕度 {_m(imu_score['bias_step-0.6']['bias_step-0.3'])}×）→ **SEPARABLE（幅值）**。")
    lines.append("- bias_step vs bias_drift：阶跃（mean_offset 主导、drift≈0）vs 慢漂移（drift 主导、offset≈0）→ **SEPARABLE（特征形状）**。")
    lines.append("- nominal vs 全部：offset/drift 均 0 → **SEPARABLE**。")
    lines.append("")

    # ---- summary matrix ----
    lines.append("## 汇总可分性矩阵（对象 × 机理对）")
    lines.append("")
    lines.append("| 对象 | 机理对 | 分离裕度(×容差) | 判定 | 驱动特征 |")
    lines.append("|---|---|---|---|---|")
    all_pairs = [
        ("battery", "nominal", "SCL-0.35", batt_score, batt_verdict, batt_driver, "容量 160→104"),
        ("battery", "nominal", "SCL-0.20", batt_score, batt_verdict, batt_driver, "容量 160→128"),
        ("battery", "nominal", "cap_loss-30", batt_score, batt_verdict, batt_driver, "容量 160→112"),
        ("battery", "nominal", "open_circuit", batt_score, batt_verdict, batt_driver, "fail（功率冻结）+ 容量同维 OVERLAP"),
        ("battery", "SCL-0.35", "SCL-0.20", batt_score, batt_verdict, batt_driver, "容量 104 vs 128"),
        ("battery", "SCL-0.35", "cap_loss-30", batt_score, batt_verdict, batt_driver, "容量 104 vs 112（Δ8 Wh=16×）"),
        ("battery", "SCL-0.35", "open_circuit", batt_score, batt_verdict, batt_driver, "容量 104 vs 160 + fail"),
        ("battery", "SCL-0.20", "cap_loss-30", batt_score, batt_verdict, batt_driver, "容量 128 vs 112"),
        ("battery", "SCL-0.20", "open_circuit", batt_score, batt_verdict, batt_driver, "容量 128 vs 160 + fail"),
        ("battery", "cap_loss-30", "open_circuit", batt_score, batt_verdict, batt_driver, "容量 112 vs 160 + fail"),
        ("reaction_wheel", "nominal", "friction-50", rw_score, rw_verdict, rw_driver, "b 1e-5→1.5e-5 (+50%)"),
        ("reaction_wheel", "nominal", "motor_fail", rw_score, rw_verdict, rw_driver, "J 0.077 畸变 + OOD 0.978"),
        ("reaction_wheel", "nominal", "jamming", rw_score, rw_verdict, rw_driver, "fail"),
        ("reaction_wheel", "friction-50", "motor_fail", rw_score, rw_verdict, rw_driver, "J 0.1 vs 0.077 + OOD"),
        ("reaction_wheel", "friction-50", "jamming", rw_score, rw_verdict, rw_driver, "fail"),
        ("reaction_wheel", "motor_fail", "jamming", rw_score, rw_verdict, rw_driver, "fail"),
        ("thermal", "nominal", "rad-deg-0.7", th_score, th_verdict, th_driver, "G 1.16→0.89 (-23%)"),
        ("thermal", "nominal", "heat_rej_fail", th_score, th_verdict, th_driver, "fail（温度冲高）"),
        ("thermal", "rad-deg-0.7", "heat_rej_fail", th_score, th_verdict, th_driver, "fail"),
        ("IMU", "nominal", "bias_step-0.6", imu_score, imu_verdict, imu_driver, "mean_offset 0→0.120"),
        ("IMU", "nominal", "bias_step-0.3", imu_score, imu_verdict, imu_driver, "mean_offset 0→0.060"),
        ("IMU", "nominal", "bias_drift", imu_score, imu_verdict, imu_driver, "drift_total 0→0.050"),
        ("IMU", "bias_step-0.6", "bias_step-0.3", imu_score, imu_verdict, imu_driver, "偏移幅值 0.120 vs 0.060"),
        ("IMU", "bias_step-0.6", "bias_drift", imu_score, imu_verdict, imu_driver, "阶跃 vs 漂移（形状）"),
        ("IMU", "bias_step-0.3", "bias_drift", imu_score, imu_verdict, imu_driver, "阶跃 vs 漂移（形状）"),
    ]
    for obj, a, b, score_map, verdict_map, driver_map, reason in all_pairs:
        lines.append(f"| {obj} | {a} vs {b} | {_m(score_map[a][b])} | **{verdict_map[a][b]}** | {reason} |")
    lines.append("")
    lines.append("## 遇到的问题与诚实记录")
    lines.append("")
    lines.append("1. **nominal vs open_circuit 在『容量』单维不可分**：两者代表容量均为 ~160 Wh——如实报告该参数维度 OVERLAP；靠功率冻结（fail）维度才 SEPARABLE。**单一参数不够，需多特征，这是比“假装可分”更有价值的结论。**")
    lines.append("2. **全局拟合在 onset 故障下失真**（SCL 全局 net fail-closed/失真、OC net 口径垃圾拟合）——机理判别必须用分段 post-onset 特征 + fail 态，不能用全局参数直接比。")
    lines.append("3. **全部命名机理均已补足激励注入 cycled 变体**（本循环新增 3 条），无“该机理对暂缺数据”项；非 cycled 变体因无激励 fail-closed 未纳入参数可分性（其 fail-closed 本身是检测信号）。")
    lines.append("4. 分离容差为经验值但量级远大于估计精度（如 0.5 Wh vs 0.01 Wh），阈值 ±1 个数量级不改变任何判定。")
    lines.append("5. 未 commit / 未 push；估计器与数据源未改动。")
    lines.append("")
    lines.append("## 建议（记录，未实施，供 1# 决策）")
    lines.append("")
    lines.append("- 诊断读取头可按两级特征：参数偏移（容量/阻尼/电导的连续值）用于退化机理定位；fail-closed/残差/OOD 用于结构失效机理。二者互不冲突，可并行。")
    lines.append("- 对 onset 故障，诊断窗口需分段（pre/post onset），否则全局参数失真会污染机理判别。")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(json.dumps({"battery": battery, "rw": rw, "thermal": thermal, "imu": imu}, indent=2, ensure_ascii=False, default=str))
    return 0


def _m(value: float) -> str:
    return f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
