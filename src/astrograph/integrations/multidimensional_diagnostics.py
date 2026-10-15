from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.integrations.operator_vocabulary import channel_label_zh


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


DIMENSION_ZH = {
    "LEVEL_SHIFT": "幅值偏移",
    "VARIABILITY": "波动变化",
    "TREND": "趋势漂移",
    "ABRUPTNESS": "突变强度",
    "OSCILLATION": "振荡特征",
    "COUPLING": "通道关联变化",
    "COVERAGE": "数据完整性",
}

CHANNEL_ZH = {
    "adcs.rw.measured_speed_rad_s_2": "反作用轮转速",
    "adcs.rw.motor_current_a_2": "反作用轮电机电流",
    "adcs.pointing_error_deg": "姿态指向误差",
    "adcs.rw.command_execution_residual_nm_2": "反作用轮指令执行残差",
    "adcs.rw.command_execution_residual_norm_nm": "反作用轮执行总残差",
    "adcs.rw.speed_response_residual_nm_2": "反作用轮转速响应残差",
}


class StatisticalDimension(_Strict):
    dimension: str
    label_zh: str
    index: float = Field(ge=0.0, le=100.0)
    level: str
    observation_zh: str
    metric_value: float | None = None


class ChannelStatisticalAssessment(_Strict):
    channel: str
    label_zh: str
    unit: str = ""
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    dimensions: list[StatisticalDimension] = Field(default_factory=list)
    dominant_dimensions: list[str] = Field(default_factory=list)
    operator_observation_zh: str


class CouplingChange(_Strict):
    channel_a: str
    channel_b: str
    label_a_zh: str
    label_b_zh: str
    baseline_correlation: float
    active_correlation: float
    correlation_change: float
    index: float = Field(ge=0.0, le=100.0)
    level: str
    observation_zh: str


class DimensionSummary(_Strict):
    dimension: str
    label_zh: str
    index: float = Field(ge=0.0, le=100.0)
    affected_channels: int = Field(ge=0)
    level: str
    operator_observation_zh: str


class MultidimensionalDiagnosticAssessment(_Strict):
    schema_version: str = "astrograph-multidimensional-diagnostic-assessment.v1"
    status: str
    basis: str = "OBSERVED_TELEMETRY_ONLY"
    focus_start_s: float | None = None
    baseline_end_s: float | None = None
    dimension_summary: list[DimensionSummary] = Field(default_factory=list)
    channels: list[ChannelStatisticalAssessment] = Field(default_factory=list)
    coupling_changes: list[CouplingChange] = Field(default_factory=list)
    maintenance_summary_zh: str = ""
    diagnostic_vote_allowed: bool = False
    probability_claim_allowed: bool = False
    truth_or_labels_used: bool = False
    boundary_zh: str = (
        "多维统计用于维护复核与时序定位，不是故障概率，也不会自动改写已经冻结的核心诊断。"
    )


def _finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _mad(values: list[float]) -> float:
    if len(values) < 2:
        return 1e-12
    center = median(values)
    mad = median([abs(value - center) for value in values])
    if mad > 1e-12:
        return 1.4826 * mad
    mean = fmean(values)
    var = fmean([(value - mean) ** 2 for value in values])
    return max(math.sqrt(var), 1e-12)


def _slope(times: list[float], values: list[float]) -> float:
    if len(times) < 2 or len(values) != len(times):
        return 0.0
    tx = fmean(times)
    vy = fmean(values)
    denom = sum((x - tx) ** 2 for x in times)
    if denom <= 1e-15:
        return 0.0
    return sum((x - tx) * (y - vy) for x, y in zip(times, values)) / denom


def _sign_change_rate(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    scale = _mad(diffs)
    eps = max(scale * 0.1, 1e-12)
    signs = [1 if d > eps else -1 if d < -eps else 0 for d in diffs]
    filtered = [s for s in signs if s]
    if len(filtered) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(filtered, filtered[1:]) if a != b)
    return changes / max(1, len(filtered) - 1)


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 5:
        return None
    ma, mb = fmean(a), fmean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    sa = math.sqrt(sum(x * x for x in da))
    sb = math.sqrt(sum(x * x for x in db))
    if sa <= 1e-12 or sb <= 1e-12:
        return None
    value = sum(x * y for x, y in zip(da, db)) / (sa * sb)
    return max(-1.0, min(1.0, value))


def _index(value: float, scale: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * value / max(scale, 1e-12)))


def _ratio_index(ratio: float) -> float:
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 0.0
    return max(0.0, min(100.0, abs(math.log(ratio, 2.0)) / 2.0 * 100.0))


def _level(score: float) -> str:
    if score >= 70.0:
        return "SIGNIFICANT"
    if score >= 40.0:
        return "WATCH"
    return "STABLE"


def _level_zh(level: str) -> str:
    return {"SIGNIFICANT": "明显", "WATCH": "需关注", "STABLE": "稳定"}.get(level, level)


@dataclass(frozen=True)
class ChannelReference:
    center: float
    scale: float
    variability: float
    diff_scale: float
    oscillation_rate: float


@dataclass
class MultiMetricWindowDetector:
    """Truth-safe multi-metric window detector.

    A channel can become anomalous because of sustained level shift, variance change,
    drift, abrupt derivative energy, or oscillation change. The detector is deliberately
    generic and does not use fault labels or simulator truth. It is suitable for
    first-observable-detection localization, not calibrated fault probability.
    """

    references: dict[str, ChannelReference]
    minimum_channels: int = 2
    baseline_end_s: float | None = None
    semantics: str = "END_STATE"

    @classmethod
    def from_samples(
        cls,
        samples: list[dict[str, Any]],
        channels: list[str],
        *,
        baseline_fraction: float = 0.2,
        minimum_channels: int = 2,
    ) -> "MultiMetricWindowDetector":
        n = max(8, min(len(samples), int(len(samples) * baseline_fraction)))
        references: dict[str, ChannelReference] = {}
        for channel in channels:
            rows = [row for row in samples[:n] if row.get("values", {}).get(channel) is not None]
            values = _finite([row["values"][channel] for row in rows])
            if len(values) < 5:
                continue
            diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
            references[channel] = ChannelReference(
                center=median(values),
                scale=_mad(values),
                variability=_mad(values),
                diff_scale=_mad(diffs) if diffs else max(_mad(values) * 0.1, 1e-12),
                oscillation_rate=_sign_change_rate(values),
            )
        baseline_end_s = None
        if samples[:n]:
            try:
                baseline_end_s = float(samples[n - 1]["t"])
            except (KeyError, TypeError, ValueError):
                baseline_end_s = None
        return cls(references=references, minimum_channels=max(1, minimum_channels), baseline_end_s=baseline_end_s)

    def _window_dimension_scores(self, channel: str, window: list[dict[str, Any]]) -> dict[str, float]:
        ref = self.references[channel]
        rows = [row for row in window if row.get("values", {}).get(channel) is not None]
        if len(rows) < 4:
            return {}
        times = _finite([row.get("t") for row in rows])
        values = _finite([row["values"][channel] for row in rows])
        if len(times) != len(values) or len(values) < 4:
            return {}
        level_z = abs(median(values) - ref.center) / max(ref.scale, 1e-12)
        variability_ratio = _mad(values) / max(ref.variability, 1e-12)
        duration = max(times[-1] - times[0], 1e-9)
        trend_effect = abs(_slope(times, values)) * duration / max(ref.scale, 1e-12)
        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        abrupt_z = (max([abs(x) for x in diffs], default=0.0) / max(ref.diff_scale, 1e-12))
        oscillation_delta = abs(_sign_change_rate(values) - ref.oscillation_rate)
        return {
            "LEVEL_SHIFT": _index(level_z, 6.0),
            "VARIABILITY": _ratio_index(variability_ratio),
            "TREND": _index(trend_effect, 6.0),
            "ABRUPTNESS": _index(abrupt_z, 10.0),
            "OSCILLATION": _index(oscillation_delta, 0.45),
        }

    def detect(self, window: list[dict[str, Any]]) -> tuple[bool, float, list[str]]:
        deviant: list[str] = []
        strengths: list[float] = []
        for channel in self.references:
            scores = sorted(self._window_dimension_scores(channel, window).values(), reverse=True)
            if not scores:
                continue
            first = scores[0]
            second = scores[1] if len(scores) > 1 else 0.0
            # Require either one very strong + corroborating dimension or two strong dimensions.
            anomalous = (first >= 85.0 and second >= 35.0) or (first >= 70.0 and second >= 55.0)
            if anomalous:
                deviant.append(channel)
                strengths.append(min(100.0, first * 0.7 + second * 0.3))
        required = min(self.minimum_channels, max(1, len(self.references)))
        return len(deviant) >= required, (0.0 if not strengths else fmean(strengths)), deviant


def _aligned_rows(visualization: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    channels = list(visualization.get("channels") or [])
    by_time: dict[float, dict[str, Any]] = {}
    for channel in channels:
        name = str(channel.get("name") or "")
        if not name:
            continue
        for point in channel.get("points") or []:
            try:
                t = float(point["t"])
                v = float(point["v"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(t) and math.isfinite(v)):
                continue
            by_time.setdefault(t, {"t": t, "values": {}})["values"][name] = v
    return [by_time[t] for t in sorted(by_time)], channels


def _focus_split(samples: list[dict[str, Any]], temporal: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None, float | None]:
    if not samples:
        return [], [], None, None
    first_detection = None if temporal is None else temporal.get("first_detection_s")
    try:
        focus_start = float(first_detection) if first_detection is not None else None
    except (TypeError, ValueError):
        focus_start = None
    if focus_start is not None and math.isfinite(focus_start):
        before = [row for row in samples if float(row["t"]) < focus_start]
        after = [row for row in samples if float(row["t"]) >= focus_start]
        if len(before) >= 8 and len(after) >= 8:
            baseline_count = max(8, min(len(before), int(len(samples) * 0.2)))
            baseline = before[:baseline_count]
            return baseline, after, focus_start, float(baseline[-1]["t"])
    baseline_count = max(8, min(len(samples), int(len(samples) * 0.2)))
    baseline = samples[:baseline_count]
    active_start = max(baseline_count, int(len(samples) * 0.6))
    active = samples[active_start:]
    focus_start = float(active[0]["t"]) if active else None
    baseline_end = float(baseline[-1]["t"]) if baseline else None
    return baseline, active, focus_start, baseline_end


def _channel_assessment(
    channel_meta: dict[str, Any],
    all_samples: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    active: list[dict[str, Any]],
) -> ChannelStatisticalAssessment | None:
    name = str(channel_meta.get("name") or "")
    if not name:
        return None
    base_rows = [row for row in baseline if row.get("values", {}).get(name) is not None]
    active_rows = [row for row in active if row.get("values", {}).get(name) is not None]
    base_values = _finite([row["values"][name] for row in base_rows])
    active_values = _finite([row["values"][name] for row in active_rows])
    active_times = _finite([row["t"] for row in active_rows])
    all_count = max(1, len(all_samples))
    declared_coverage = channel_meta.get("coverage_ratio")
    try:
        declared_coverage_f = float(declared_coverage)
    except (TypeError, ValueError):
        declared_coverage_f = float("nan")
    if math.isfinite(declared_coverage_f) and 0.0 <= declared_coverage_f <= 1.0:
        coverage = declared_coverage_f
    else:
        coverage = sum(1 for row in all_samples if row.get("values", {}).get(name) is not None) / all_count
    if len(base_values) < 5 or len(active_values) < 5 or len(active_times) != len(active_values):
        return ChannelStatisticalAssessment(
            channel=name,
            label_zh=channel_label_zh(name, CHANNEL_ZH.get(name, str(channel_meta.get("label") or name))),
            unit=str(channel_meta.get("unit") or ""),
            coverage_ratio=coverage,
            dimensions=[StatisticalDimension(
                dimension="COVERAGE", label_zh=DIMENSION_ZH["COVERAGE"], index=max(0.0, min(100.0, (1.0 - coverage) * 100.0)),
                level="WATCH" if coverage < 0.8 else "STABLE", observation_zh="有效数据覆盖不足，统计复核受限。" if coverage < 0.8 else "有效数据覆盖正常。",
                metric_value=coverage,
            )],
            dominant_dimensions=["COVERAGE"] if coverage < 0.8 else [],
            operator_observation_zh="当前通道有效样本不足，建议优先核对数据质量。",
        )
    center = median(base_values)
    scale = _mad(base_values)
    active_scale = _mad(active_values)
    base_diffs = [base_values[i] - base_values[i - 1] for i in range(1, len(base_values))]
    active_diffs = [active_values[i] - active_values[i - 1] for i in range(1, len(active_values))]
    diff_scale = _mad(base_diffs) if base_diffs else max(scale * 0.1, 1e-12)
    duration = max(active_times[-1] - active_times[0], 1e-9)
    level_z = abs(median(active_values) - center) / max(scale, 1e-12)
    variability_ratio = active_scale / max(scale, 1e-12)
    trend_effect = abs(_slope(active_times, active_values)) * duration / max(scale, 1e-12)
    abrupt_z = max([abs(x) for x in active_diffs], default=0.0) / max(diff_scale, 1e-12)
    osc_base = _sign_change_rate(base_values)
    osc_active = _sign_change_rate(active_values)
    osc_delta = abs(osc_active - osc_base)
    raw = {
        "LEVEL_SHIFT": (level_z, _index(level_z, 6.0)),
        "VARIABILITY": (variability_ratio, _ratio_index(variability_ratio)),
        "TREND": (trend_effect, _index(trend_effect, 6.0)),
        "ABRUPTNESS": (abrupt_z, _index(abrupt_z, 10.0)),
        "OSCILLATION": (osc_delta, _index(osc_delta, 0.45)),
        "COVERAGE": (coverage, max(0.0, min(100.0, (1.0 - coverage) * 100.0))),
    }
    observations = {
        "LEVEL_SHIFT": "与前段基线相比，通道中心水平发生偏移。",
        "VARIABILITY": "通道波动幅度相对前段基线发生变化。",
        "TREND": "当前阶段存在持续上升或下降趋势。",
        "ABRUPTNESS": "当前阶段出现相对基线更强的突变/阶跃。",
        "OSCILLATION": "当前阶段的往复振荡特征相对基线发生变化。",
        "COVERAGE": "有效数据覆盖不足，统计结论需谨慎使用。",
    }
    dims: list[StatisticalDimension] = []
    for dimension, (metric, score) in raw.items():
        level = _level(score)
        dims.append(StatisticalDimension(
            dimension=dimension,
            label_zh=DIMENSION_ZH[dimension],
            index=round(score, 3),
            level=level,
            observation_zh=observations[dimension] if level != "STABLE" else f"{DIMENSION_ZH[dimension]}未见明显异常。",
            metric_value=round(float(metric), 6),
        ))
    ranked = sorted(dims, key=lambda item: item.index, reverse=True)
    dominant = [item.dimension for item in ranked if item.level != "STABLE"][:3]
    if dominant:
        phrases = [f"{DIMENSION_ZH[d]}{_level_zh(next(x.level for x in dims if x.dimension == d))}" for d in dominant]
        operator = "、".join(phrases) + "，建议与相关通道及故障机理联动核对。"
    else:
        operator = "多项统计指标未见明显偏离，可作为对照通道。"
    return ChannelStatisticalAssessment(
        channel=name,
        label_zh=channel_label_zh(name, CHANNEL_ZH.get(name, str(channel_meta.get("label") or name))),
        unit=str(channel_meta.get("unit") or ""),
        coverage_ratio=round(coverage, 6),
        dimensions=dims,
        dominant_dimensions=dominant,
        operator_observation_zh=operator,
    )


def _coupling_changes(
    channels: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    active: list[dict[str, Any]],
) -> list[CouplingChange]:
    names = [str(item.get("name") or "") for item in channels if item.get("name")]
    result: list[CouplingChange] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            base_pairs = [(float(r["values"][a]), float(r["values"][b])) for r in baseline if r.get("values", {}).get(a) is not None and r.get("values", {}).get(b) is not None]
            active_pairs = [(float(r["values"][a]), float(r["values"][b])) for r in active if r.get("values", {}).get(a) is not None and r.get("values", {}).get(b) is not None]
            if len(base_pairs) < 6 or len(active_pairs) < 6:
                continue
            base_corr = _pearson([x for x, _ in base_pairs], [y for _, y in base_pairs])
            active_corr = _pearson([x for x, _ in active_pairs], [y for _, y in active_pairs])
            if base_corr is None or active_corr is None:
                continue
            delta = abs(active_corr - base_corr)
            score = max(0.0, min(100.0, delta / 1.0 * 100.0))
            level = _level(score)
            if level == "STABLE":
                continue
            result.append(CouplingChange(
                channel_a=a,
                channel_b=b,
                label_a_zh=CHANNEL_ZH.get(a, a),
                label_b_zh=CHANNEL_ZH.get(b, b),
                baseline_correlation=round(base_corr, 4),
                active_correlation=round(active_corr, 4),
                correlation_change=round(delta, 4),
                index=round(score, 3),
                level=level,
                observation_zh="两个通道之间的同步关系相对前段基线发生明显变化，建议联合查看波形。",
            ))
    result.sort(key=lambda item: item.index, reverse=True)
    return result[:6]


def assess_multidimensional_telemetry(
    visualization: dict[str, Any],
    *,
    temporal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if visualization.get("status") != "AVAILABLE":
        return MultidimensionalDiagnosticAssessment(
            status="UNAVAILABLE",
            maintenance_summary_zh="当前没有可安全使用的遥测曲线，无法进行多维统计复核。",
        ).model_dump(mode="json")
    samples, channel_meta = _aligned_rows(visualization)
    if len(samples) < 12 or not channel_meta:
        return MultidimensionalDiagnosticAssessment(
            status="UNAVAILABLE",
            maintenance_summary_zh="遥测采样点不足，无法稳定计算多维统计指标。",
        ).model_dump(mode="json")
    baseline, active, focus_start, baseline_end = _focus_split(samples, temporal)
    if len(baseline) < 5 or len(active) < 5:
        return MultidimensionalDiagnosticAssessment(
            status="UNAVAILABLE",
            focus_start_s=focus_start,
            baseline_end_s=baseline_end,
            maintenance_summary_zh="基线段或复核段样本不足，无法形成稳定统计对比。",
        ).model_dump(mode="json")
    channel_results: list[ChannelStatisticalAssessment] = []
    for meta in channel_meta[:8]:
        result = _channel_assessment(meta, samples, baseline, active)
        if result is not None:
            channel_results.append(result)
    if not channel_results:
        return MultidimensionalDiagnosticAssessment(
            status="UNAVAILABLE",
            focus_start_s=focus_start,
            baseline_end_s=baseline_end,
            maintenance_summary_zh="关键遥测通道不足，无法形成多维统计复核。",
        ).model_dump(mode="json")
    dimension_summary: list[DimensionSummary] = []
    for dimension in ("LEVEL_SHIFT", "VARIABILITY", "TREND", "ABRUPTNESS", "OSCILLATION"):
        dim_items = [next((d for d in ch.dimensions if d.dimension == dimension), None) for ch in channel_results]
        dim_items = [item for item in dim_items if item is not None]
        if not dim_items:
            continue
        score = max(item.index for item in dim_items)
        affected = sum(1 for item in dim_items if item.level != "STABLE")
        level = _level(score)
        if affected:
            text = f"{affected} 个关键通道在{DIMENSION_ZH[dimension]}维度需要关注。"
        else:
            text = f"关键通道的{DIMENSION_ZH[dimension]}整体稳定。"
        dimension_summary.append(DimensionSummary(
            dimension=dimension,
            label_zh=DIMENSION_ZH[dimension],
            index=round(score, 3),
            affected_channels=affected,
            level=level,
            operator_observation_zh=text,
        ))
    coupling = _coupling_changes(channel_meta[:8], baseline, active)
    if coupling:
        score = max(item.index for item in coupling)
        dimension_summary.append(DimensionSummary(
            dimension="COUPLING",
            label_zh=DIMENSION_ZH["COUPLING"],
            index=round(score, 3),
            affected_channels=len(coupling),
            level=_level(score),
            operator_observation_zh=f"发现 {len(coupling)} 组关键通道关联关系发生变化。",
        ))
    dimension_summary.sort(key=lambda item: item.index, reverse=True)
    strongest = [item.label_zh for item in dimension_summary if item.level != "STABLE"][:3]
    maintenance_summary = (
        "多维统计复核重点关注：" + "、".join(strongest) + "。"
        if strongest else "多维统计复核未发现需要额外关注的显著偏离。"
    )
    return MultidimensionalDiagnosticAssessment(
        status="AVAILABLE",
        focus_start_s=focus_start,
        baseline_end_s=baseline_end,
        dimension_summary=dimension_summary,
        channels=channel_results,
        coupling_changes=coupling,
        maintenance_summary_zh=maintenance_summary,
        diagnostic_vote_allowed=False,
        probability_claim_allowed=False,
        truth_or_labels_used=False,
    ).model_dump(mode="json")


__all__ = [
    "MultidimensionalDiagnosticAssessment",
    "MultiMetricWindowDetector",
    "assess_multidimensional_telemetry",
]
