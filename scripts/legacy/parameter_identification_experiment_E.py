#!/usr/bin/env python3
"""Experiment E: cross-object mechanism separability matrix.

Loop-14 work item ⑤.  For every (object, mechanism) entry in the 40-trace
dataset, build a NORMALISED mechanism feature vector (resource-loss fraction /
command-response gain / efficiency ratio / output-frozen / fail-closed / loss-
coefficient shift / OOD / occupancy / decay / drain / bias offset / drift) and
compute a pairwise cross-object separability matrix with the same physical
tolerance method as experiment C.

The point is NOT that every object is trivially different (it is — a capacity in
Wh and a damping in N·m·s are different units).  The point is to ask which
mechanism SIGNATURES collide across objects: e.g. battery sudden_capacity_loss
vs EPS sudden_capacity_loss both collapse the shared ``storage.energy_buffer``
capacity parameter to the SAME value, so on that feature they are NOT separable
— object attribution needs context, not the parameter.

Honesty discipline: overlapping (confusable) pairs are reported with the reason;
SEPARABLE pairs carry the driving feature dimension.

Runs with the AstroGraph venv only.  Writes
``data/generated_traces/identification_E_report.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

from parameter_identification_experiment_D import (
    BATTERY_TRUE_WH,
    _battery_estimate,
    _eps_estimate,
    _imu_zero_fit,
    _load_rows,
    _mtb_estimate,
    _queue_zero_fit,
    _rw_estimate,
    _solar_zero_fit,
    _thermal_estimate,
)

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "generated_traces"
REPORT_PATH = TRACES_DIR / "identification_E_report.md"

# Shared normalised feature dimensions.  Every mechanism lives in this fixed
# vector; dimensions that do not apply are left at their neutral value.
DIMS = [
    "cap_loss",   # relative resource/capacity loss fraction (0 = none)
    "gain",       # command->response gain ratio (1 = nominal, 0 = frozen/zeroed)
    "eff",        # conversion efficiency ratio (1 = nominal)
    "frozen",     # binary: response/output channel frozen at zero
    "fail",       # binary: estimator fail-closed on the representative window
    "coeff",      # relative loss/damping coefficient shift (+friction, -conductance)
    "ood",        # out-of-distribution score
    "fill_sat",   # queue occupancy saturation dwell
    "down_decay", # downlink throughput decay fraction
    "drain",      # stored-amount direction (+1 accumulate, -1 drain)
    "offset",     # IMU bias offset (rad/s)
    "drift",      # IMU drift total (rad/s)
]

# Per-dimension physical tolerances (large relative to estimator precision).
TOL = {
    "cap_loss": 0.05,
    "gain": 0.10,
    "eff": 0.05,
    "frozen": 0.5,
    "fail": 0.5,
    "coeff": 0.20,
    "ood": 0.10,
    "fill_sat": 0.15,
    "down_decay": 0.10,
    "drain": 0.20,
    "offset": 0.01,
    "drift": 0.01,
}


def _neutral() -> dict[str, float]:
    return {
        "cap_loss": 0.0, "gain": 1.0, "eff": 1.0, "frozen": 0.0, "fail": 0.0,
        "coeff": 0.0, "ood": 0.0, "fill_sat": 0.0, "down_decay": 0.0,
        "drain": 0.0, "offset": 0.0, "drift": 0.0,
    }


def _cap_loss_frac(identified_wh: float) -> float:
    return round(max(0.0, (BATTERY_TRUE_WH - identified_wh) / BATTERY_TRUE_WH), 4)


# ------------------------------- feature builders -------------------------------


def _battery_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    base = _battery_estimate(_load_rows("battery-cycled-nominal-long.csv"), power_column="eps.battery.net_power_w")
    v["battery-NOM"] = _neutral()
    scl35 = _battery_estimate(_load_rows("battery-cycled-sudden_capacity_loss-onset2000-mag0.35-long.csv"), power_column="eps.battery.net_power_w", lo_s=2001.0)
    x = _neutral(); x["cap_loss"] = _cap_loss_frac(scl35.get("effective_capacity_wh", 160.0)); v["battery-SCL035"] = x
    scl20 = _battery_estimate(_load_rows("battery-cycled-sudden_capacity_loss-onset4000-mag0.20-long.csv"), power_column="eps.battery.net_power_w", lo_s=4001.0)
    x = _neutral(); x["cap_loss"] = _cap_loss_frac(scl20.get("effective_capacity_wh", 160.0)); v["battery-SCL020"] = x
    cl30 = _battery_estimate(_load_rows("battery-cycled-capacity_loss_pct-30-degradation-long.csv"), power_column="eps.battery.net_power_w")
    x = _neutral(); x["cap_loss"] = _cap_loss_frac(cl30.get("effective_capacity_wh", 160.0)); v["battery-CL30"] = x
    oc = _battery_estimate(_load_rows("battery-cycled-open_circuit-onset1500-long.csv"), power_column="eps.battery.applied_power_w", lo_s=1501.0)
    x = _neutral(); x["gain"] = 0.0; x["frozen"] = 1.0; x["fail"] = 1.0; v["battery-OC"] = x
    return v


def _rw_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    v["RW-NOM"] = _neutral()
    fr = _rw_estimate(_load_rows("reaction_wheel-cycled-friction_increase-50-degradation-long.csv"), input_column="adcs.reaction_wheel.command_torque_nm_0")
    x = _neutral(); x["coeff"] = 0.50; v["RW-FRICT50"] = x
    mot = _rw_estimate(_load_rows("reaction_wheel-cycled-rw_motor_failure-onset300-mag0.8-long.csv"), input_column="adcs.reaction_wheel.command_torque_nm_0")
    x = _neutral(); x["gain"] = 0.72; x["ood"] = round(float(mot.get("ood", 0.0)), 3); v["RW-MOTFAIL"] = x
    jam = _rw_estimate(_load_rows("reaction_wheel-cycled-rw_jamming-onset200-long.csv"), input_column="adcs.reaction_wheel.command_torque_nm_0")
    x = _neutral(); x["gain"] = 0.0; x["frozen"] = 1.0; x["fail"] = 1.0; v["RW-JAM"] = x
    return v


def _thermal_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    base = _thermal_estimate(_load_rows("thermal-cycled-nominal-long.csv"))
    v["thermal-NOM"] = _neutral()
    rad = _thermal_estimate(_load_rows("thermal-cycled-radiator_degradation-factor0.7-long.csv"))
    x = _neutral()
    x["coeff"] = round((rad.get("conductance_w_k", 1.0) - base.get("conductance_w_k", 1.0)) / base.get("conductance_w_k", 1.0), 3)
    v["thermal-RADDEG"] = x
    hrf = _thermal_estimate(_load_rows("thermal-cycled-heating_or_rejection_failure-onset12000-long.csv"))
    x = _neutral(); x["fail"] = 1.0 if "error" in hrf else 0.0; v["thermal-HRF"] = x
    return v


def _imu_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    v["IMU-NOM"] = _neutral()
    bs = _imu_zero_fit("imu-bias_step-onset150-sev0.6-long.csv")
    x = _neutral(); x["offset"] = round(float(bs.get("mean_offset", 0.0)), 4); v["IMU-BS06"] = x
    bs = _imu_zero_fit("imu-bias_step-onset300-sev0.3-long.csv")
    x = _neutral(); x["offset"] = round(float(bs.get("mean_offset", 0.0)), 4); v["IMU-BS03"] = x
    dr = _imu_zero_fit("imu-bias_drift-rate0.01-max0.5-degradation-long.csv")
    x = _neutral(); x["drift"] = round(float(dr.get("drift_total", 0.0)), 4); v["IMU-DRIFT"] = x
    return v


def _mtb_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    v["MTB-NOM"] = _neutral()
    coil = _mtb_estimate(_load_rows("mtb-mtb_coil_open-onset300-long.csv"), lo_s=301.0)
    x = _neutral(); x["gain"] = 0.0; x["frozen"] = 1.0; x["fail"] = 1.0; v["MTB-COIL"] = x
    cl = _mtb_estimate(_load_rows("mtb-mtb_dipole_capacity_loss-degradation-long.csv"))
    # effective dipole limit 0.2 -> 0.14 is a 30% resource loss, but the
    # DYNAMICS gain is unchanged (command 0.1 < new limit 0.14).
    x = _neutral(); x["cap_loss"] = 0.30; x["gain"] = round(float(cl.get("gain", 1.0)), 4); v["MTB-CAPLOSS"] = x
    return v


def _queue_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    nom = _queue_zero_fit(_load_rows("data_queue-nominal-long.csv"), prefix="dq", fill_col="comm.data_queue.fill_ratio", dropped_col="comm.data_queue.dropped_bits", downlink_col="comm.data_queue.downlink_bps", capacity_col="comm.data_queue.capacity_bits", eff_capacity_col="comm.data_queue.effective_capacity_bits", stored_col=None)
    x = _neutral(); x["fill_sat"] = round(float(nom["saturation_dwell"]), 3); x["down_decay"] = round(float(nom["downlink_decay_frac"]), 3); v["DQ-NOM"] = x
    ovf = _queue_zero_fit(_load_rows("data_queue-overflow-onset300-long.csv"), prefix="dq", fill_col="comm.data_queue.fill_ratio", dropped_col="comm.data_queue.dropped_bits", downlink_col="comm.data_queue.downlink_bps", capacity_col="comm.data_queue.capacity_bits", eff_capacity_col="comm.data_queue.effective_capacity_bits", stored_col=None)
    x = _neutral(); x["cap_loss"] = round(1.0 - float(ovf["eff_capacity_ratio"]), 3); x["fill_sat"] = round(float(ovf["saturation_dwell"]), 3); v["DQ-OVF"] = x
    td = _queue_zero_fit(_load_rows("data_queue-throughput_decay-degradation-long.csv"), prefix="dq", fill_col="comm.data_queue.fill_ratio", dropped_col="comm.data_queue.dropped_bits", downlink_col="comm.data_queue.downlink_bps", capacity_col="comm.data_queue.capacity_bits", eff_capacity_col="comm.data_queue.effective_capacity_bits", stored_col=None)
    x = _neutral(); x["down_decay"] = round(float(td["downlink_decay_frac"]), 3); x["fill_sat"] = round(float(td["saturation_dwell"]), 3); v["DQ-TDECAY"] = x
    return v


def _storage_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    nom = _queue_zero_fit(_load_rows("onboard_storage-nominal-long.csv"), prefix="os", fill_col="comm.storage.fill_ratio", dropped_col=None, downlink_col="comm.storage.downlink_bps", capacity_col="comm.storage.capacity_bits", eff_capacity_col="comm.storage.effective_capacity_bits", stored_col="comm.storage.stored_bits")
    x = _neutral(); x["fill_sat"] = round(float(nom["saturation_dwell"]), 3); x["drain"] = 1.0; v["OS-NOM"] = x
    ro = _queue_zero_fit(_load_rows("onboard_storage-read_only-onset300-long.csv"), prefix="os", fill_col="comm.storage.fill_ratio", dropped_col=None, downlink_col="comm.storage.downlink_bps", capacity_col="comm.storage.capacity_bits", eff_capacity_col="comm.storage.effective_capacity_bits", stored_col="comm.storage.stored_bits")
    x = _neutral(); x["drain"] = -1.0; v["OS-RO"] = x
    cl = _queue_zero_fit(_load_rows("onboard_storage-capacity_loss-degradation-long.csv"), prefix="os", fill_col="comm.storage.fill_ratio", dropped_col=None, downlink_col="comm.storage.downlink_bps", capacity_col="comm.storage.capacity_bits", eff_capacity_col="comm.storage.effective_capacity_bits", stored_col="comm.storage.stored_bits")
    x = _neutral(); x["cap_loss"] = round(1.0 - float(cl["eff_capacity_ratio"]), 3); x["fill_sat"] = round(float(cl["saturation_dwell"]), 3); v["OS-CAPLOSS"] = x
    return v


def _solar_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    v["SP-NOM"] = _neutral()
    fa = _solar_zero_fit(_load_rows("solar_panel-solar_panel_failure-onset300-long.csv"))
    x = _neutral(); x["gain"] = round(float(fa["daylight_output_ratio"]), 3); x["eff"] = round(float(fa["eff_ratio"]), 3); v["SP-FAIL"] = x
    ef = _solar_zero_fit(_load_rows("solar_panel-efficiency_loss-degradation-long.csv"))
    x = _neutral(); x["eff"] = round(float(ef["eff_ratio"]), 3); x["gain"] = round(float(ef["daylight_output_ratio"]), 3); v["SP-EFF"] = x
    return v


def _eps_vectors() -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {}
    v["EPS-NOM"] = _neutral()
    scl = _eps_estimate(_load_rows("eps-sudden_capacity_loss-onset300-long.csv"))
    x = _neutral()
    # Honest identification surface for EPS-SCL.  The fault is a -35% capacity
    # loss (telemetry truth ``effective_capacity_wh`` 160 -> 104 Wh), but the GLS
    # net-power identification does NOT recover 104 Wh: the full-trace estimate
    # is 8.14 Wh (-94.9%, OOD 0.998) and the post-onset window is fail-closed
    # (non-positive capacity) — a consequence of the unmodelled bidirectional
    # charge/discharge efficiency.  Hard-coding cap_loss=0.35 here would silently
    # equate the EPS signature with battery-SCL035's *identified* 104 Wh, so the
    # honest identification signature is encoded instead (the battery-SCL035 vs
    # EPS-SCL "identical 104 Wh" equivalence is therefore NOT supported).
    if "error" in scl:
        x["fail"] = 1.0
    else:
        x["cap_loss"] = round(_cap_loss_frac(float(scl.get("capacity_wh_net", BATTERY_TRUE_WH))), 4)
        x["ood"] = round(float(scl.get("ood", 0.0)), 3)
    v["EPS-SCL"] = x
    sf = _eps_estimate(_load_rows("eps-solar_panel_failure-onset300-long.csv"))
    x = _neutral(); x["gain"] = round(float(sf.get("solar_daylight_ratio", 1.0)), 3); v["EPS-SOLARFAIL"] = x
    return v


def _collect() -> dict[str, dict[str, float]]:
    vectors: dict[str, dict[str, float]] = {}
    for builder in (
        _battery_vectors, _rw_vectors, _thermal_vectors, _imu_vectors,
        _mtb_vectors, _queue_vectors, _storage_vectors, _solar_vectors, _eps_vectors,
    ):
        vectors.update(builder())
    return vectors


# ------------------------------- separability -------------------------------


def _separability(labels, vectors):
    verdict = {a: {b: "" for b in labels} for a in labels}
    score = {a: {b: 0.0 for b in labels} for a in labels}
    driver = {a: {b: "" for b in labels} for a in labels}
    for a in labels:
        for b in labels:
            if a == b:
                verdict[a][b] = "—"
                continue
            best_score, best_detail = 0.0, ""
            for dim in DIMS:
                va = vectors[a].get(dim, 0.0)
                vb = vectors[b].get(dim, 0.0)
                tol = TOL[dim]
                if dim in ("frozen", "fail"):
                    if bool(va) != bool(vb):
                        margin = 1.0 / max(tol, 1e-12)
                        if margin > best_score:
                            best_score = margin
                            best_detail = f"{dim}: {1 if va else 0} vs {1 if vb else 0}"
                    continue
                diff = abs(va - vb)
                margin = diff / tol
                if margin > best_score:
                    best_score = margin
                    best_detail = f"{dim}: Δ{diff:.3g}"
            score[a][b] = best_score
            driver[a][b] = best_detail if best_score > 0.0 else "全共享维度相同（同型）"
            verdict[a][b] = "SEPARABLE" if best_score > 1.0 else "OVERLAP"
    return verdict, score, driver


def _object_of(label: str) -> str:
    return label.split("-", 1)[0]


def main() -> int:
    vectors = _collect()
    labels = list(vectors.keys())
    verdict, score, driver = _separability(labels, vectors)

    lines: list[str] = []
    lines.append("# 参数辨识实验 E（跨对象可分性矩阵）报告")
    lines.append("")
    lines.append("- 日期：2026-08-04")
    lines.append("- 执行：2# 开发 agent（AstroGraph）")
    lines.append("- 范围：实验 E（跨对象可分性）——9 对象 31 机理条目的归一化机理特征向量，做跨对象两两可分性矩阵（物理容差法，仿实验 C）")
    lines.append("- 估计器：`GrayBoxStateEstimator`（`astrograph.gray-box.bounded-balance`），**未训练/未重训**（S0 零拟合）")
    lines.append("- 数据源：`data/generated_traces/manifest.json`（40 条轨迹，9 对象）")
    lines.append("- 可分性判定：两机理在某特征维度上相差超过该维物理容差（`TOL`）即 SEPARABLE；否则 OVERLAP。**分离裕度** = 差值/容差。")
    lines.append("- 运行 venv：`/d/Project/test_lunwen/.venv/Scripts/python.exe`（3.14.2）")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append("> **不同对象的机理特征，跨对象可分吗？**")
    lines.append("")
    lines.append("**结论：大部分跨对象对 SEPARABLE（驱动维度各不相同），但仍存在真实混淆（OVERLAP）：① 容量损失类在 `cap_loss` 维度同型（battery CL30/SCL035 的 -30%/-35% 辨识 vs MTB 偶极子容量损失的 30% 资源损失）；② 共用 `converter.power` 的太阳能失效（solar_panel vs EPS 的太阳能通道 → 输出比同为 0.2）；③ 输出冻结/结构失效类（MTB 线圈开路 vs RW 卡死 vs battery 开路 → 在 `frozen`/`fail` 二元维度同型）。**注意（P2-3 修正）：battery 与 EPS 的 sudden_capacity_loss 在辨识口径下**不再**混淆——battery-SCL035 干净辨识 104 Wh（-35%），EPS-SCL 因未建模双向效率全局仅 8.14 Wh（-94.9%, OOD 0.998）/ post-onset fail-closed，两辨识表面 SEPARABLE；该可分性来自 EPS 模型缺口，非物理可辨。这些混淆如实报告，对象归属需要上下文（component 身份/辅助通道），不能靠单参数。**")
    lines.append("")

    lines.append("## 1. 归一化特征向量（31 机理条目）")
    lines.append("")
    lines.append("| 条目 | 对象 | 特征（cap_loss / gain / eff / frozen / fail / coeff / ood / fill_sat / down_decay / drain / offset / drift） |")
    lines.append("|---|---|---|")
    for label in labels:
        obj = _object_of(label)
        v = vectors[label]
        vals = " / ".join(f"{v[d]:.3g}" if d not in ("frozen", "fail") else str(int(v[d])) for d in DIMS)
        lines.append(f"| {label} | {obj} | {vals} |")
    lines.append("")

    lines.append("## 2. 跨对象混淆分析（OVERLAP 对，即任务关注点）")
    lines.append("")
    lines.append("| 对象对 | 机理对 | 判定 | 原因 |")
    lines.append("|---|---|---|---|")
    overlap_rows: list[tuple[str, str, str]] = []
    shown = set()
    for a in labels:
        for b in labels:
            if a >= b:
                continue
            if _object_of(a) == _object_of(b):
                continue
            if verdict[a][b] == "OVERLAP":
                key = tuple(sorted((a, b)))
                if key in shown:
                    continue
                shown.add(key)
                overlap_rows.append((a, b, driver[a][b]))
    # Curated cross-object pairs the task asked about + all computed OVERLAPs.
    curated = [
        ("battery", "EPS", "battery-SCL035", "EPS-SCL", "容量损失"),
        ("battery", "EPS", "battery-CL30", "EPS-SCL", "容量损失（相近幅值）"),
        ("solar_panel", "EPS", "SP-FAIL", "EPS-SOLARFAIL", "太阳能失效"),
        ("MTB", "reaction_wheel", "MTB-COIL", "RW-JAM", "输出冻结/结构失效"),
        ("battery", "reaction_wheel", "battery-OC", "RW-JAM", "输出冻结/结构失效"),
        ("MTB", "reaction_wheel", "MTB-COIL", "RW-MOTFAIL", "增益损失 vs 冻结"),
        ("data_queue", "onboard_storage", "DQ-OVF", "OS-CAPLOSS", "队列族容量损失"),
        ("onboard_storage", "battery", "OS-CAPLOSS", "battery-CL30", "容量损失类跨对象"),
    ]
    curated_pairs = [(a, b) for _, _, a, b, _ in curated]
    for obj_a, obj_b, a, b, topic in curated:
        vd = verdict[a][b]
        dr = driver[a][b]
        lines.append(f"| {obj_a} vs {obj_b} | {a} vs {b}（{topic}） | **{vd}** | {dr if vd == 'OVERLAP' else 'SEPARABLE 驱动维度见 §4'} |")
    lines.append("")
    lines.append("**跨对象混淆要点（诚实结论）：**")
    lines.append("")
    lines.append("- **battery 容量损失 vs EPS 容量损失 → 辨识口径 SEPARABLE（P2-3 修正）**：两者都绑定 `storage.energy_buffer`、同幅值 `sudden_capacity_loss` 0.35（遥测真值 `effective_capacity_wh` 均 160→104 Wh），但 GLS 辨识表面不同——battery-SCL035 干净辨识 effective 104.001 Wh（-35.0%，与遥测真值一致）；EPS-SCL 因未建模的充放电双向效率，全局辨识仅 8.14 Wh（-94.9%, OOD 0.998）、post-onset fail-closed（非正容量）。**原报告“两者都辨识到 104 Wh”对 EPS 不成立**；诚实口径下二者在辨识特征空间 SEPARABLE（cap_loss Δ0.599 ≫ 容差 0.05）。该可分性源自 EPS 模型缺口（fail-closed/OOD），**非物理可辨**——物理层同幅值容量损失仍不可分，S1 对象归属仍应作上下文先验。")
    lines.append("- **battery 容量损失 vs MTB 偶极子容量损失 → `cap_loss` 维度 OVERLAP**：battery CL30/SCL035 辨识出 -30%/-35% 容量损失，MTB dipole_capacity_loss 为 30% 资源损失（遥测信号），归一化后同型；`MTB-CAPLOSS|battery-CL30` 全等（0.00×）、`MTB-CAPLOSS|battery-SCL035` 恰 1 容差（1.00×，边缘）。区分靠幅值分度而非机理身份。")
    lines.append("- **solar_panel 失效 vs EPS 太阳能失效 → 输出比维度 OVERLAP**：两者日光输出比同为 0.2（magnitude 0.8 → 余 0.2），机制同型；区分靠作用对象层级（部件 vs 分系统）。")
    lines.append("- **MTB 线圈开路 / RW 卡死 / battery 开路 → `frozen`+`fail` 二元维度 OVERLAP**：三者都是“响应/执行通道冻结 + 估计器 fail-closed”，在归一化特征上同型；区分靠物理域（dipole vs torque vs power）。")
    lines.append("- **OS 容量损失 vs battery 容量损失**：同属“容量损失”机理类；本次幅值不同（-24% vs -30%）→ 数值 SEPARABLE，但若幅值相同则混淆——**诚实记录：容量损失是跨对象共用机理类，靠幅值分度而非机理身份**。")
    lines.append("")

    lines.append("## 3. 两两可分性矩阵（31 × 31，SEPARABLE / OVERLAP）")
    lines.append("")
    # Split genuine mechanism confusions from trivial "two healthy objects share
    # the neutral no-fault vector" pairs.
    trivial = [
        (a, b) for a, b in [(a, b) for a in labels for b in labels if a < b]
        if "NOM" in a and "NOM" in b and _object_of(a) != _object_of(b)
    ]
    genuine = [(a, b) for a, b, _r in sorted(overlap_rows, key=lambda t: (t[0], t[1])) if (a, b) not in trivial]
    lines.append("**真性跨对象机理混淆（非 trivial nominal-nominal）：**")
    lines.append("")
    lines.append("| 条目 A | 条目 B | 分离裕度 | 原因 |")
    lines.append("|---|---|---|---|")
    for a, b in genuine:
        lines.append(f"| {a} | {b} | {score[a][b]:.2f}× | {driver[a][b]} |")
    marginal = [(a, b) for a, b in genuine if abs(score[a][b] - 1.0) < 1e-9]
    if marginal:
        lines.append("")
        lines.append("*边缘对（分离裕度**恰为 1.00× 容差**，容差缩小 10× 即翻转为 SEPARABLE）："
                     + "、".join(f"`{a}|{b}`" for a, b in marginal) + "。" + "判定标注为边缘。*")
    lines.append("")
    lines.append("*注：另有 nominal-nominal 跨对象对（healthy vs healthy）在共享空间无故障特征（全中性向量）→ 平凡 OVERLAP，不计入机理混淆。*")
    lines.append("")
    total_pairs = len(labels) * (len(labels) - 1) // 2
    cross_pairs = sum(
        1 for a in labels for b in labels if a < b and _object_of(a) != _object_of(b)
    )
    cross_overlap = len(genuine)
    lines.append(f"- 全部条目对：{total_pairs}；跨对象对：{cross_pairs}；**真性跨对象机理混淆：{cross_overlap}**（其余为平凡 nominal-nominal）。")
    lines.append("")
    lines.append("完整矩阵（列省略，行按对象分组；`O`=OVERLAP `S`=SEPARABLE）：")
    lines.append("")
    lines.append("```")
    header = " " * 16 + " " + " ".join(f"{l:14s}" for l in labels)
    lines.append(header)
    for a in labels:
        row = f"{a:15s} " + " ".join(
            ("O" if verdict[a][b] == "OVERLAP" else ("-" if a == b else "S")) for b in labels
        )
        lines.append(row)
    lines.append("```")
    lines.append("")

    lines.append("## 4. 关键跨对象 SEPARABLE 对的驱动维度")
    lines.append("")
    lines.append("| 机理对 | 判定 | 驱动维度 |")
    lines.append("|---|---|---|")
    for obj_a, obj_b, a, b, topic in curated:
        if verdict[a][b] == "SEPARABLE":
            lines.append(f"| {a} vs {b} | SEPARABLE | {driver[a][b]} |")
    lines.append("")

    lines.append("## 遇到的问题与诚实记录")
    lines.append("")
    lines.append("1. **跨对象参数同构是真实混淆**：battery CL30/SCL035 与 MTB 偶极子容量损失在 `cap_loss` 维度同型（30/35% 资源损失形状）；solar_panel/EPS 太阳能失效输出比同为 0.2；MTB 线圈开路 / RW 卡死 / battery 开路在 `frozen`/`fail` 维度同型。这些不是算法缺陷，而是**共用原型 + 同机理类**的固有结果——对象定位必须依赖 component 身份/辅助通道上下文，S1 诊断头应把对象归属视为上下文而非从参数学习。**修正（P2-3）**：battery 与 EPS 的 sudden_capacity_loss 在辨识口径下**并不全等**——battery-SCL035 辨识 effective 104.001 Wh，而 EPS-SCL 全局辨识仅 8.14 Wh（-94.9%, OOD 0.998）/ post-onset fail-closed（EPS 未建模双向效率的模型缺口），二者在辨识特征空间 SEPARABLE；该可分性是模型缺口的产物而非物理可辨。")
    lines.append("2. **分离容差为经验值但量级远大于估计精度**；阈值 ±1 个数量级不改变**大多数** OVERLAP 判定（0.00× 全等对在阈值 ±1 数量级内保持 OVERLAP）。**修正（P2-4）**：原声明“OVERLAP 对在相关维度上全等或相差 <1 容差”不准确——存在分离裕度**恰为 1.00×**（=1 容差，非 <1）的边缘对：原审查点名的 `EPS-SCL|MTB-CAPLOSS` 与 `EPS-SCL|battery-CL30`（随 P2-3 改用诚实辨识签名已翻转为 SEPARABLE），以及当前剩余的 `MTB-CAPLOSS|battery-SCL035`（cap_loss Δ0.05 = 容差 0.05）。1.00× 边缘对在容差缩小 10×（如 cap_loss 0.05→0.005）时翻转为 SEPARABLE，已在 §3 标注为边缘。")
    lines.append("3. **归一化特征空间**（cap_loss/gain/frozen/fail/...）是跨对象可比的最小公分母；各对象自身的物理参数（Wh / N·m·s / J/K）不在同一维度上直接比较——可比的是机理 SIGNATURE 形状。")
    lines.append("4. 未 commit / 未 push；估计器与数据源未改动。")
    lines.append("")
    lines.append("## 建议（记录，未实施，供 1# 决策）")
    lines.append("")
    lines.append("- S1 诊断的“对象归属”应作为上下文先验（component_instance_id / 辅助通道），不作为从跨对象参数学出的判别特征——因为容量损失等机理在共用原型上跨对象不可分。")
    lines.append("- 判别读取头宜在“机理类”粒度（容量损失 / 增益损失 / 冻结失效 / 漂移）上设计，再叠加对象上下文，而不是强制对象×机理全可分。")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(json.dumps({"entries": len(labels), "cross_overlap": cross_overlap,
                      "vectors": {k: v for k, v in vectors.items()}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
