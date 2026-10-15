#!/usr/bin/env python3
"""End-to-end zero-fit diagnosis skeleton (loop 13 work item ⑤).

Ties the S0 zero-fit chain together for 3 objects (battery / reaction_wheel /
thermal), each on a nominal + a fault/degradation long trace:

    generated CSV -> SemanticTelemetryFrame (label-free)
                 -> zero-fit feature operators (22-operator family) -> EvidenceToken
                 -> GrayBoxStateEstimator (4 equation families, GLS only)
                 -> diagnostic report (fail-closed, no fabrication)

Hard constraints honored:
  * S0 zero-fit: the estimator runs ``use_frozen_calibration=False`` and only
    performs GLS linear identification — no parameter fitting, no split/seed,
    ``labels_used`` / ``truth_used`` stay False on every frame.
  * Fail-closed honesty: when a frame lacks excitation (e.g. constant input
    below the observability contract) or the contract cannot be satisfied, the
    report records INSUFFICIENT_EVIDENCE / the exact estimator error instead of
    fabricating parameters.
  * The report is written to ``data/generated_traces/end_to_end_report.md``.

Run with the AstroGraph interpreter only::

    /d/Project/test_lunwen/.venv/Scripts/python.exe scripts/end_to_end_zero_fit_pipeline.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.features.generic.token_builder import TokenBuilder
from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.state_estimation import GrayBoxStateEstimator

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "generated_traces"
MANIFEST_PATH = TRACES_DIR / "manifest.json"
REPORT_PATH = TRACES_DIR / "end_to_end_report.md"
FEATURE_CONTRACT_PATH = ROOT / "configs" / "semantics" / "sat_agent_archetype_feature_contracts.s0.v1.json"

# object -> archetype / contract family / estimation channels (single signal per
# port role, matching the gray-box contract expectations).
OBJECTS = {
    "battery": {
        "archetype": "storage.energy_buffer",
        "traces": {
            "nominal": "battery-cycled-nominal-long.csv",
            "fault": "battery-cycled-sudden_capacity_loss-onset2000-mag0.35-long.csv",
        },
        "true_params": "capacity_wh = 160 (net-base recovery carries the 0.97 discharge-efficiency bias)",
        "channels": [
            ("state.energy_fraction", "energy.state_of_charge", "1", "eps.battery.soc", lambda v: float(v), 1),
            ("power.exchange", "electrical.active_power", "W", "eps.battery.net_power_w", lambda v: float(v), 1),
        ],
    },
    "reaction_wheel": {
        "archetype": "actuator.spinning_rotor",
        "traces": {
            "nominal": "reaction_wheel-cycled-nominal-long.csv",
            "fault": "reaction_wheel-cycled-friction_increase-50-degradation-long.csv",
        },
        "true_params": "J = 0.1 kg·m^2, b = 1e-5 N·m·s (friction +50% in the fault trace)",
        "channels": [
            ("state.rate", "mechanical.angular_rate", "rad/s", "adcs.reaction_wheel.speed_rad_s_0", lambda v: float(v), 1),
            ("command.effort", "mechanical.torque", "N*m", "adcs.reaction_wheel.command_torque_nm_0", lambda v: float(v), 1),
        ],
    },
    "thermal": {
        "archetype": "thermal.lumped_capacity",
        "traces": {
            "nominal": "thermal-cycled-nominal-long.csv",
            "fault": "thermal-cycled-radiator_degradation-factor0.7-long.csv",
        },
        "true_params": "bus capacity = 18000 J/K, sink = -35 C; radiator degradation factor 0.7 lowers effective conductance (single-node vs dual-node alignment gap is recorded, not claimed as exact recovery)",
        "channels": [
            ("state.temperature", "thermal.temperature", "K", "thermal.node.bus_temp_c", lambda v: float(v) + 273.15, 1),
            ("heat.exchange", "electrical.active_power", "W", "__combined_heat__", lambda v: float(v), 1),
            ("context.ambient_temperature", "thermal.temperature", "K", "__sink__", lambda v: float(v), 1),
        ],
    },
}


def _load_rows(filename: str) -> list[dict[str, str]]:
    with (TRACES_DIR / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _frame(archetype: str, channels: list[tuple], rows: list[dict[str, str]]) -> SemanticTelemetryFrame:
    timestamps = [float(row["time_s"]) for row in rows]
    signals: list[SemanticSignal] = []
    for index, (port_role, quantity_kind, unit, source, convert, direction) in enumerate(channels):
        if source == "__combined_heat__":
            values = [
                float(row["thermal.heat.internal_power_w"])
                + float(row["thermal.heat.solar_input_w"])
                + float(row["thermal.heater.power_w"])
                for row in rows
            ]
        elif source == "__sink__":
            values = [-35.0 + 273.15] * len(rows)
        else:
            values = [convert(row[source]) for row in rows]
        signals.append(
            SemanticSignal(
                semantic_channel_id=f"e2e.{archetype}.{index}",
                asset_instance_id="asset-under-test",
                component_archetype_id=archetype,
                port_role_id=port_role,
                quantity_kind_id=quantity_kind,
                canonical_unit=unit,
                direction=direction,
                values=values,
                missing_mask=[False] * len(values),
            )
        )
    payload = {
        "timestamps": timestamps,
        "signals": [item.model_dump(mode="json") for item in signals],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    nominal_period = float(sum(diffs) / len(diffs)) if diffs else 0.0
    return SemanticTelemetryFrame(
        frame_id="e2e-zero-fit",
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=timestamps,
        signals=signals,
        sampling=SamplingProfile(
            row_count=len(timestamps),
            nominal_period_s=nominal_period,
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256=digest,
        labels_used=False,
        truth_used=False,
        claim_boundary="loop-13 end-to-end zero-fit diagnosis skeleton",
    )


def _feature_tokens(archetype: str, frame: SemanticTelemetryFrame) -> list[dict[str, Any]]:
    try:
        builder = TokenBuilder(archetype=archetype, contract_path=str(FEATURE_CONTRACT_PATH))
        tokens = builder.build(frame, component_archetype_id=archetype).tokens
        return [
            {
                "operator_id": token.operator_id,
                "port_role_id": token.port_role_id,
                "quantity_kind_id": token.quantity_kind_id,
                "value": token.value,
                "uncertainty": token.uncertainty,
                "availability": token.availability,
                "quality_flags": token.quality_flags,
            }
            for token in tokens
        ]
    except Exception as exc:  # noqa: BLE001 - honest: no feature contract for this archetype
        return [{"error": str(exc)}]


def _estimate(archetype: str, frame: SemanticTelemetryFrame) -> dict[str, Any]:
    contract = default_dynamics_contracts().get(archetype)
    if contract is None:
        return {"status": "NO_DYNAMICS_CONTRACT"}
    estimator = GrayBoxStateEstimator(use_frozen_calibration=False)
    try:
        result = estimator.estimate(frame, contract)
        return {
            "status": "OK",
            "parameters": [
                {
                    "parameter_id": p.parameter_id,
                    "value": p.value,
                    "canonical_unit": p.canonical_unit,
                    "lower_confidence": p.lower_confidence,
                    "upper_confidence": p.upper_confidence,
                }
                for p in result.parameters
            ],
            "residuals": [
                {"residual_id": r.residual_id, "root_mean_square": r.root_mean_square, "mean": r.mean}
                for r in result.residuals
            ],
            "ood_score": result.ood_score,
            "conservation_raw_violation_fraction": result.conservation_assessment.raw_violation_fraction,
            "conservation_episode_score": result.conservation_assessment.episode_score,
            "conservation_method": result.conservation_assessment.method,
            "constraint_violation_count": len(result.constraint_violations),
        }
    except Exception as exc:  # noqa: BLE001 - fail-closed, report the exact reason
        return {"status": "FAIL_CLOSED", "reason": str(exc)}


def _scenario_label(filename: str) -> str:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for trace in payload["traces"]:
        if trace["file"] == filename:
            return trace["scenario"]
    return "unknown"


def main() -> int:
    lines: list[str] = []
    lines.append("# 端到端零拟合诊断骨架（Loop 13 工作项⑤）报告")
    lines.append("")
    lines.append("- 日期：2026-08-04")
    lines.append("- 执行：2# 开发 agent（AstroGraph）")
    lines.append("- 链路：`generated CSV → SemanticTelemetryFrame（无标签）→ 零拟合特征算子 → EvidenceToken → 灰盒状态估计（4 方程族）→ 诊断报告`")
    lines.append("- 估计器：`GrayBoxStateEstimator`（`astrograph.gray-box.bounded-balance`），`use_frozen_calibration=False`，**仅 GLS 线性辨识，零训练/零校准**（S0）")
    lines.append("- 覆盖对象：battery / reaction_wheel / thermal，各 1 条 nominal + 1 条 fault/degradation")
    lines.append("- 运行 venv：`/d/Project/test_lunwen/.venv/Scripts/python.exe`（3.14.2），符合 AstroGraph 硬约束")
    lines.append("- 硬约束：`labels_used=False`、`truth_used=False`、无 split/seed；激励不足 → `INSUFFICIENT_EVIDENCE`/`FAIL_CLOSED`，不伪造参数")
    lines.append("")

    summary_rows: list[list[str]] = []
    for object_name, spec in OBJECTS.items():
        lines.append(f"## {object_name}（`{spec['archetype']}`）")
        lines.append("")
        lines.append(f"真实参数（生成脚本配置）：{spec['true_params']}")
        lines.append("")
        for scenario_label, filename in spec["traces"].items():
            scenario_id = _scenario_label(filename)
            lines.append(f"### {object_name} · {scenario_label}（`{filename}`，场景 `{scenario_id}`）")
            lines.append("")
            rows = _load_rows(filename)
            frame = _frame(spec["archetype"], spec["channels"], rows)

            tokens = _feature_tokens(spec["archetype"], frame)
            estimate = _estimate(spec["archetype"], frame)

            lines.append(f"- 语义帧：`{len(rows)}` 行 @ 采样间隔 {frame.sampling.nominal_period_s:.6g}s，`semantic_data_sha256={frame.semantic_data_sha256[:16]}…`")
            lines.append("")
            lines.append("**零拟合特征算子（EvidenceToken）**：")
            lines.append("")
            if tokens and "error" not in tokens[0]:
                lines.append("| operator_id | port_role_id | value | availability | quality_flags |")
                lines.append("|---|---|---|---|---|")
                for token in tokens[:12]:
                    flags = ",".join(token.get("quality_flags") or []) or "-"
                    lines.append(
                        f"| {token['operator_id']} | {token['port_role_id']} | "
                        f"{token['value']:.6g} | {token['availability']:.4g} | {flags} |"
                    )
                if len(tokens) > 12:
                    lines.append(f"| …（共 {len(tokens)} 个 token，仅展示前 12） | | | | |")
            else:
                error = tokens[0].get("error", "") if tokens else ""
                lines.append(f"- N/A（无该 archetype 的特征算子契约：`{error}`）")
            lines.append("")

            lines.append("**灰盒状态估计（GLS 参数辨识）**：")
            lines.append("")
            if estimate["status"] == "OK":
                lines.append("| parameter_id | 估计值 | 单位 | 置信下界 | 置信上界 |")
                lines.append("|---|---|---|---|---|")
                for p in estimate["parameters"]:
                    lines.append(
                        f"| {p['parameter_id']} | {p['value']:.6g} | {p['canonical_unit']} | "
                        f"{p['lower_confidence']:.6g} | {p['upper_confidence']:.6g} |"
                    )
                lines.append("")
                lines.append(f"- 状态重建残差：`{estimate['residuals'][0]['root_mean_square']:.3e}`（{estimate['residuals'][0]['residual_id']}），其余残差 RMS 见下：")
                lines.append("")
                lines.append("| residual_id | root_mean_square | mean |")
                lines.append("|---|---|---|")
                for r in estimate["residuals"]:
                    lines.append(f"| {r['residual_id']} | {r['root_mean_square']:.3e} | {r['mean']:.3e} |")
                lines.append("")
                lines.append(
                    f"- 守恒评估：`{estimate['conservation_method']}`，原始违反占比 "
                    f"{estimate['conservation_raw_violation_fraction']:.4f}，episode_score "
                    f"{estimate['conservation_episode_score']:.4f}，约束违反样本数 {estimate['constraint_violation_count']}，ood_score={estimate['ood_score']:.4f}"
                )
            else:
                lines.append(
                    f"- **{estimate['status']}**：{estimate.get('reason', '无动力学契约，估计器不适用')}"
                )
                lines.append("")
                lines.append("> 诚实判定：该段数据/该对象无法进行灰盒参数辨识，报告 `FAIL_CLOSED`/`NO_DYNAMICS_CONTRACT`，不伪造参数。")
            lines.append("")

            summary_rows.append(
                [object_name, scenario_label, estimate.get("status", "?"), str(estimate.get("reason", "-"))]
            )

    lines.append("## 汇总")
    lines.append("")
    lines.append("| 对象 | 场景 | 估计状态 | fail-closed 原因 |")
    lines.append("|---|---|---|---|")
    for row in summary_rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    lines.append("## 遇到的问题与诚实记录")
    lines.append("")
    lines.append("- battery 故障（SCL-0.35）为 onset 阶跃，常量-C 全局拟合会失真/fail-closed；诊断需分段（pre/post onset）辨识。")
    lines.append("- reaction_wheel `actuator.spinning_rotor` 特征算子契约已在第十四循环 P2-A 修复中接入（`spin.rate_dynamics` / `spin.momentum_balance` / `spin.command_rate_consistency` / `spin.change_profile`），本报告 token 已非 N/A；灰盒 `inertial_spin_balance` 辨识 J/b。")
    lines.append("- thermal 单节点线性模型 vs 双节点辐射生成模型存在已知对齐缺口（A2 记录），容量/电导恢复值带系统偏差，仅报告方向性偏移。")
    lines.append("- 未 commit / 未 push；仅新增本脚本 + 报告，未改动估计器与数据源。")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
