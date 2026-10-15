from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Protocol

from astrograph.integrations.multidimensional_diagnostics import MultiMetricWindowDetector

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalLocalizationResult(_Strict):
    schema_version: str = "astrograph-temporal-localization.v2"
    status: str
    method: str
    semantics: str
    first_detection_s: float | None = None
    confirmed_detection_s: float | None = None
    estimated_onset_start_s: float | None = None
    estimated_onset_end_s: float | None = None
    window_size_s: float | None = None
    stride_s: float | None = None
    confirmation_windows: int | None = None
    evidence_channels: list[str] = Field(default_factory=list)
    diagnostic_fault_id: str | None = None
    primary_source: str | None = None
    agreement: str = "SINGLE_SOURCE"
    detection_sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence_claim_allowed: bool = False
    truth_or_labels_used: bool = False
    note_zh: str = ""


class WindowDetector(Protocol):
    semantics: str
    def detect(self, window: list[dict[str, Any]]) -> tuple[bool, float, list[str]]: ...


@dataclass
class RobustTelemetryChangeDetector:
    """Truth-safe fallback for first *observable change* localization.

    It deliberately does not claim the exact physical fault onset. A baseline is
    learned from the leading telemetry segment, then windows are flagged when a
    bounded number of selected channels show sustained robust deviation.
    """

    baseline_stats: dict[str, tuple[float, float]]
    z_threshold: float = 5.0
    minimum_channels: int = 2
    semantics: str = "END_STATE"

    @staticmethod
    def _robust_scale(values: list[float]) -> float:
        if len(values) < 2:
            return 1e-12
        center = median(values)
        mad = median([abs(x-center) for x in values])
        if mad > 1e-12:
            return 1.4826 * mad
        mean = fmean(values)
        variance = fmean([(x-mean) ** 2 for x in values])
        return max(math.sqrt(variance), 1e-12)

    @classmethod
    def from_samples(cls, samples: list[dict[str, Any]], channels: list[str], *, baseline_fraction: float = 0.2, z_threshold: float = 5.0, minimum_channels: int = 2) -> "RobustTelemetryChangeDetector":
        n = max(8, min(len(samples), int(len(samples) * baseline_fraction)))
        stats: dict[str, tuple[float, float]] = {}
        for channel in channels:
            values = [float(row["values"][channel]) for row in samples[:n] if row.get("values", {}).get(channel) is not None]
            if len(values) >= 4:
                stats[channel] = (median(values), cls._robust_scale(values))
        return cls(stats, z_threshold=z_threshold, minimum_channels=max(1, minimum_channels))

    def detect(self, window: list[dict[str, Any]]) -> tuple[bool, float, list[str]]:
        deviant: list[str] = []
        strengths: list[float] = []
        for channel, (center, scale) in self.baseline_stats.items():
            values = [float(row["values"][channel]) for row in window if row.get("values", {}).get(channel) is not None]
            if not values:
                continue
            # Median suppresses one-point spikes and better represents a sustained window state.
            z = abs(median(values) - center) / max(scale, 1e-12)
            if z >= self.z_threshold:
                deviant.append(channel)
                strengths.append(z)
        score = 0.0 if not strengths else min(100.0, fmean(strengths))
        return len(deviant) >= min(self.minimum_channels, max(1, len(self.baseline_stats))), score, deviant


def localize_windowed_detection(
    samples: list[dict[str, Any]], detector: WindowDetector, *,
    window_size_s: float = 16.0, stride_s: float = 2.0, confirmation_windows: int = 3,
    diagnostic_fault_id: str | None = None,
) -> TemporalLocalizationResult:
    """Find first and confirmed positive windows without simulator truth.

    For END_STATE semantics, the end of the first positive window is explicitly
    named ``first_detection_s`` rather than physical ``fault_onset_s``.  With a
    preceding negative window, the onset is only bracketed between two detection
    boundaries. This preserves the distinction proposed in the maintenance UI.
    """
    if len(samples) < 4:
        return TemporalLocalizationResult(status="UNAVAILABLE", method="WINDOWED_DETECTION", semantics=getattr(detector, "semantics", "UNKNOWN"), diagnostic_fault_id=diagnostic_fault_id, note_zh="遥测点不足，无法执行窗口定位。")
    times = [float(row["t"]) for row in samples]
    start, end = min(times), max(times)
    if end <= start:
        return TemporalLocalizationResult(status="UNAVAILABLE", method="WINDOWED_DETECTION", semantics=getattr(detector, "semantics", "UNKNOWN"), diagnostic_fault_id=diagnostic_fault_id, note_zh="时间轴无有效跨度。")
    window_size_s = max(float(stride_s), float(window_size_s))
    stride_s = max(1e-9, float(stride_s))
    confirmation_windows = max(1, int(confirmation_windows))
    detections: list[dict[str, Any]] = []
    baseline_end = getattr(detector, "baseline_end_s", None)
    try:
        baseline_end = float(baseline_end) if baseline_end is not None else None
    except (TypeError, ValueError):
        baseline_end = None
    # A detector calibrated on the leading segment must not "detect" inside its own
    # calibration baseline. Begin after that boundary when the detector exposes it.
    cursor = start + window_size_s
    if baseline_end is not None and math.isfinite(baseline_end):
        cursor = max(cursor, baseline_end + stride_s)
    while cursor <= end + 1e-9:
        left = cursor - window_size_s
        window = [row for row in samples if left < float(row["t"]) <= cursor]
        if window:
            detected, score, channels = detector.detect(window)
            detections.append({"end": cursor, "detected": detected, "score": score, "channels": channels})
        cursor += stride_s
    run = 0
    first_index: int | None = None
    confirmed_index: int | None = None
    for i, item in enumerate(detections):
        if item["detected"]:
            if run == 0:
                first_index = i
            run += 1
            if run >= confirmation_windows:
                confirmed_index = i
                break
        else:
            run = 0
            first_index = None
    if first_index is None:
        return TemporalLocalizationResult(
            status="NOT_DETECTED", method="WINDOWED_TELEMETRY_CHANGE", semantics=getattr(detector, "semantics", "END_STATE"),
            window_size_s=window_size_s, stride_s=stride_s, confirmation_windows=confirmation_windows,
            diagnostic_fault_id=diagnostic_fault_id, truth_or_labels_used=False,
            note_zh="当前可观测遥测的窗口检测未形成持续异常；这不等同于证明不存在故障。",
        )
    first = detections[first_index]
    previous_end = detections[first_index-1]["end"] if first_index > 0 else max(start, first["end"] - window_size_s)
    confirmed = detections[confirmed_index]["end"] if confirmed_index is not None else None
    channels: list[str] = []
    for item in detections[first_index:(confirmed_index + 1 if confirmed_index is not None else first_index + 1)]:
        for channel in item["channels"]:
            if channel not in channels:
                channels.append(channel)
    return TemporalLocalizationResult(
        status="DETECTED" if confirmed is not None else "SUSPECTED",
        method="WINDOWED_TELEMETRY_CHANGE",
        semantics=getattr(detector, "semantics", "END_STATE"),
        first_detection_s=float(first["end"]), confirmed_detection_s=(None if confirmed is None else float(confirmed)),
        estimated_onset_start_s=float(previous_end), estimated_onset_end_s=float(first["end"]),
        window_size_s=window_size_s, stride_s=stride_s, confirmation_windows=confirmation_windows,
        evidence_channels=channels, diagnostic_fault_id=diagnostic_fault_id,
        confidence_claim_allowed=False, truth_or_labels_used=False,
        note_zh=("首次阳性窗口末端作为首次检出时间；估计起始区间来自上一阴性窗口末端到首次阳性窗口末端，"
                 "不等同于仿真真值或精确物理故障起始时刻。"),
    )


def localize_from_visualization(visualization: dict[str, Any], *, diagnostic_fault_id: str | None = None) -> dict[str, Any]:
    channels = list(visualization.get("channels") or [])
    if visualization.get("status") != "AVAILABLE" or not channels:
        return TemporalLocalizationResult(status="UNAVAILABLE", method="WINDOWED_TELEMETRY_CHANGE", semantics="END_STATE", diagnostic_fault_id=diagnostic_fault_id, note_zh="没有可安全使用的原始遥测曲线。", truth_or_labels_used=False).model_dump(mode="json")
    # Reconstruct a synchronized sample table from the visualization points. Only
    # values already admitted for maintenance display participate; labels/truth do not.
    by_time: dict[float, dict[str, Any]] = {}
    selected = [str(ch.get("name")) for ch in channels[:6] if ch.get("name")]
    for ch in channels[:6]:
        name = str(ch.get("name") or "")
        for point in ch.get("points") or []:
            t = float(point["t"])
            by_time.setdefault(t, {"t": t, "values": {}})["values"][name] = point.get("v")
    samples = [by_time[t] for t in sorted(by_time)]
    if len(samples) < 12:
        return TemporalLocalizationResult(status="UNAVAILABLE", method="WINDOWED_TELEMETRY_CHANGE", semantics="END_STATE", diagnostic_fault_id=diagnostic_fault_id, note_zh="可视化采样点不足，无法稳定定位。", truth_or_labels_used=False).model_dump(mode="json")
    t0, t1 = float(samples[0]["t"]), float(samples[-1]["t"])
    duration = max(1e-9, t1-t0)
    positive_steps = [
        float(samples[i]["t"]) - float(samples[i - 1]["t"])
        for i in range(1, len(samples))
        if float(samples[i]["t"]) > float(samples[i - 1]["t"])
    ]
    cadence = median(positive_steps) if positive_steps else max(duration / max(1, len(samples) - 1), 0.5)
    stride = max(float(cadence), 0.5)
    window = max(5.0, min(20.0, stride * 5.0))
    # Calibrate from a bounded causal prefix rather than a fraction of the entire
    # file. Appending later telemetry must not retroactively move the baseline.
    baseline_horizon = min(20.0, max(8.0 * stride, 10.0))
    calibration = [row for row in samples if float(row["t"]) <= t0 + baseline_horizon]
    if len(calibration) < 8:
        calibration = samples[: min(len(samples), 20)]
    detector = MultiMetricWindowDetector.from_samples(
        calibration, selected, baseline_fraction=1.0, minimum_channels=max(2, min(3, len(selected)))
    )
    if not detector.references:
        return TemporalLocalizationResult(status="UNAVAILABLE", method="WINDOWED_MULTIMETRIC_TELEMETRY", semantics="END_STATE", diagnostic_fault_id=diagnostic_fault_id, note_zh="关键通道无法形成稳定基线。", truth_or_labels_used=False).model_dump(mode="json")
    result = localize_windowed_detection(
        samples, detector, window_size_s=window, stride_s=stride, confirmation_windows=3,
        diagnostic_fault_id=diagnostic_fault_id,
    ).model_copy(update={
        "method": "WINDOWED_MULTIMETRIC_TELEMETRY",
        "note_zh": (
            "基于幅值偏移、波动变化、趋势漂移、突变强度与振荡特征的多指标窗口检测。"
            "首次阳性窗口末端作为首次检出时间；疑似起始区间不等同于仿真真值或精确物理故障起始时刻。"
        ),
    })
    return result.model_dump(mode="json")



def _canonical_sha256(value: Any) -> str:
    import hashlib
    import json
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def localize_adcs_model_csv(
    telemetry_path: str | Path,
    *,
    diagnostic_fault_id: str | None = None,
    project_root: str | Path | None = None,
    confirmation_windows: int = 3,
) -> dict[str, Any]:
    """Run the frozen ADCS P2 detector over causal 20-sample windows.

    Only the model-declared telemetry channels are read. Dataset labels, simulator
    truth, case names and metadata are never supplied to the model. The endpoint of
    the first positive window is reported as *first detection*, never physical onset.
    """
    import hashlib
    import json
    import pandas as pd
    from astrograph.data_driven import ModelInferenceRequest, ObservationScope
    from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package

    path = Path(telemetry_path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve() if project_root else Path(__file__).resolve().parents[3]
    contract_path = root / "configs" / "agent" / "adcs_native_runtime_contract.v1.json"
    if not path.is_file():
        return {"source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE","reason_zh":"未找到可用于模型窗口检测的遥测 CSV。","truth_or_labels_used":False}
    if not contract_path.is_file():
        return {"source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE","reason_zh":"当前发行包缺少 ADCS 原生模型运行合同。","truth_or_labels_used":False}
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model_path = root / str(contract.get("model_package") or "")
    expected_sha = str(contract.get("model_package_sha256") or "")
    if not model_path.is_file() or len(expected_sha) != 64:
        return {"source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE","reason_zh":"冻结 ADCS 模型包不可用。","truth_or_labels_used":False}
    try:
        provider = load_adcs_three_task_torch_package(model_path, expected_sha256=expected_sha, device="cpu")
    except Exception as exc:
        return {"source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE","reason_zh":f"模型加载失败：{type(exc).__name__}: {exc}","truth_or_labels_used":False}
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE","reason_zh":f"遥测读取失败：{type(exc).__name__}: {exc}","truth_or_labels_used":False}
    time_name = next((name for name in ("time_s","elapsed_s","sim_time_s","time","t") if name in frame.columns), None)
    missing = [name for name in provider.selected_channels if name not in frame.columns]
    if time_name is None or missing:
        reason = "缺少时间列。" if time_name is None else f"缺少模型要求的 {len(missing)} 个遥测通道。"
        return {
            "source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE",
            "reason_zh":reason,"missing_channels":missing[:40],"required_channel_count":len(provider.selected_channels),
            "available_required_channels":len(provider.selected_channels)-len(missing),"truth_or_labels_used":False,
        }
    selected = frame[[time_name, *provider.selected_channels]].copy()
    selected[time_name] = pd.to_numeric(selected[time_name], errors="coerce")
    selected = selected[selected[time_name].notna()].reset_index(drop=True)
    if len(selected) < int(provider.manifest.input_contract["window_length"]):
        return {"source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE","reason_zh":"遥测长度不足一个模型窗口。","truth_or_labels_used":False}
    window_length = int(provider.manifest.input_contract["window_length"])
    requests: list[ModelInferenceRequest] = []
    endpoints: list[float] = []
    for end_index in range(window_length - 1, len(selected)):
        window = selected.iloc[end_index-window_length+1:end_index+1]
        values = window[list(provider.selected_channels)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).tolist()
        payload = {"channel_order": list(provider.selected_channels), "telemetry_window": values}
        digest = _canonical_sha256(payload)
        requests.append(ModelInferenceRequest(
            episode_id=f"temporal:{path.name}:{end_index}",
            model_id=provider.manifest.model_id,
            model_version=provider.manifest.model_version,
            feature_schema_id=provider.manifest.feature_schema_ids[0],
            input_data_sha256=digest,
            registered_fault_ids=list(provider.manifest.supported_fault_ids),
            payload=payload,
            observation_scope=ObservationScope(
                scope_id=f"temporal:{path.name}:{end_index}",
                source_id="uploaded-telemetry.csv",
                source_data_sha256=digest,
            ),
        ))
        endpoints.append(float(window[time_name].iloc[-1]))
    try:
        outputs = provider.predict_raw_batch(requests)
    except Exception as exc:
        return {"source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE","reason_zh":f"模型窗口推理失败：{type(exc).__name__}: {exc}","truth_or_labels_used":False}
    confirmation_windows = max(1, int(confirmation_windows))
    run = 0
    first_i: int | None = None
    confirmed_i: int | None = None
    first_fault: str | None = None
    first_score: float | None = None
    first_matches: bool | None = None
    for idx, output in enumerate(outputs):
        detected = bool((output.metadata or {}).get("detected"))
        attribution = dict(output.fault_attribution or {})
        top_fault = max(attribution, key=attribution.get) if attribution else None
        if detected:
            if run == 0:
                first_i = idx
                first_fault = top_fault
                first_score = None if output.anomaly_score is None else float(output.anomaly_score)
                first_matches = None if diagnostic_fault_id is None else bool(top_fault == diagnostic_fault_id)
            run += 1
            if run >= confirmation_windows:
                confirmed_i = idx
                break
        else:
            run = 0
            first_i = None
            first_fault = None
            first_score = None
            first_matches = None
    if first_i is None:
        return {
            "source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"NOT_DETECTED",
            "method":"ADCS_P2_CAUSAL_WINDOW_DETECTOR","semantics":"END_STATE","window_length_samples":window_length,
            "model_id":provider.manifest.model_id,"model_version":provider.manifest.model_version,
            "model_artifact_sha256":expected_sha,"truth_or_labels_used":False,
            "reason_zh":"冻结模型在当前遥测窗口中未形成持续故障检出；这不等同于证明不存在故障。",
        }
    first = float(endpoints[first_i])
    previous = float(endpoints[first_i-1]) if first_i > 0 else float(selected[time_name].iloc[0])
    confirmed = None if confirmed_i is None else float(endpoints[confirmed_i])
    status = "DETECTED" if confirmed is not None else "SUSPECTED"
    if diagnostic_fault_id and first_fault and first_fault != diagnostic_fault_id:
        status = "DETECTED_OTHER_CANDIDATE"
    dt_values = [b-a for a,b in zip(endpoints,endpoints[1:]) if b>a]
    stride_s = median(dt_values) if dt_values else None
    return {
        "source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":status,
        "method":"ADCS_P2_CAUSAL_WINDOW_DETECTOR","semantics":"END_STATE",
        "first_detection_s":first,"confirmed_detection_s":confirmed,
        "estimated_onset_start_s":previous,"estimated_onset_end_s":first,
        "window_length_samples":window_length,"stride_s":stride_s,"confirmation_windows":confirmation_windows,
        "detected_fault_id":first_fault,"diagnostic_fault_id":diagnostic_fault_id,"candidate_matches":first_matches,
        "detection_score":first_score,"score_semantics":"UNCALIBRATED_MODEL_OUTPUT_SHARED_EVIDENCE",
        "model_id":provider.manifest.model_id,"model_version":provider.manifest.model_version,
        "model_artifact_sha256":expected_sha,"truth_or_labels_used":False,
        "reason_zh":"使用冻结 ADCS P2 模型的故障检测头逐窗口执行；窗口末端是首次检出时间，不是物理故障真实起始时刻。",
    }


def _source_summary_from_statistical(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id":"MULTIMETRIC_STATISTICS","label_zh":"多维统计变化检测","status":result.get("status"),
        "method":result.get("method"),"semantics":result.get("semantics"),
        "first_detection_s":result.get("first_detection_s"),"confirmed_detection_s":result.get("confirmed_detection_s"),
        "estimated_onset_start_s":result.get("estimated_onset_start_s"),"estimated_onset_end_s":result.get("estimated_onset_end_s"),
        "evidence_channels":list(result.get("evidence_channels") or []),"truth_or_labels_used":False,
        "reason_zh":result.get("note_zh") or "",
    }


def localize_from_job_telemetry(
    *,
    expanded: str | Path,
    report: dict[str, Any],
    visualization: dict[str, Any],
    analysis_series: dict[str, Any] | None = None,
    diagnostic_fault_id: str | None = None,
) -> dict[str, Any]:
    """Dual-channel temporal localization: frozen detector + statistical corroboration.

    The fault model is preferred when its exact 33-channel contract is available. The
    statistical detector remains an independent corroborating source. Missing model
    channels fail closed to the statistical route rather than being synthesized.
    """
    statistical = localize_from_visualization(analysis_series or visualization, diagnostic_fault_id=diagnostic_fault_id)
    from astrograph.integrations.platform_visualization import select_truth_safe_telemetry_csv
    csv_path = select_truth_safe_telemetry_csv(Path(expanded), report=report)
    model = localize_adcs_model_csv(csv_path, diagnostic_fault_id=diagnostic_fault_id) if csv_path else {
        "source_id":"ADCS_P2_MODEL","label_zh":"冻结 ADCS 故障检测模型","status":"UNAVAILABLE",
        "reason_zh":"当前任务没有可供模型窗口检测的遥测 CSV。","truth_or_labels_used":False,
    }
    sources = [model, _source_summary_from_statistical(statistical)]
    model_usable = model.get("status") in {"DETECTED","SUSPECTED"} and model.get("candidate_matches") is not False
    stat_usable = statistical.get("status") in {"DETECTED","SUSPECTED"}
    if model_usable:
        primary = dict(model)
        agreement = "SINGLE_SOURCE"
        note = str(model.get("reason_zh") or "")
        if stat_usable:
            m = float(model["first_detection_s"])
            s = float(statistical["first_detection_s"])
            m_stride = float(model.get("stride_s") or 0.0)
            s_stride = float(statistical.get("stride_s") or 0.0)
            tolerance = max(5.0, 2.0 * max(m_stride, s_stride))
            agreement = "AGREE" if abs(m-s) <= tolerance else "DISAGREE"
            relation = "基本一致" if agreement == "AGREE" else "存在明显时间差"
            note += f" 多维统计变化检测于 {s:.2f}s 首次检出，与模型窗口检测{relation}。"
        return TemporalLocalizationResult(
            status=str(primary.get("status")), method="MODEL_PLUS_MULTIMETRIC_TELEMETRY" if stat_usable else str(primary.get("method")),
            semantics="END_STATE", first_detection_s=primary.get("first_detection_s"), confirmed_detection_s=primary.get("confirmed_detection_s"),
            estimated_onset_start_s=primary.get("estimated_onset_start_s"), estimated_onset_end_s=primary.get("estimated_onset_end_s"),
            window_size_s=None, stride_s=primary.get("stride_s"), confirmation_windows=primary.get("confirmation_windows"),
            evidence_channels=list(statistical.get("evidence_channels") or []), diagnostic_fault_id=diagnostic_fault_id,
            primary_source="ADCS_P2_MODEL", agreement=agreement, detection_sources=sources,
            confidence_claim_allowed=False, truth_or_labels_used=False, note_zh=note,
        ).model_dump(mode="json")
    # If the model detects a different candidate, keep that discrepancy visible while
    # allowing the independent statistical route to localize the observable change.
    if stat_usable:
        note = str(statistical.get("note_zh") or "")
        if model.get("status") == "DETECTED_OTHER_CANDIDATE":
            note += f" 冻结模型在首次故障窗口给出的候选为 {model.get('detected_fault_id')}，与当前冻结诊断 {diagnostic_fault_id} 不一致，已保留为复核差异。"
        return TemporalLocalizationResult(
            status=str(statistical.get("status")), method="WINDOWED_MULTIMETRIC_TELEMETRY", semantics="END_STATE",
            first_detection_s=statistical.get("first_detection_s"), confirmed_detection_s=statistical.get("confirmed_detection_s"),
            estimated_onset_start_s=statistical.get("estimated_onset_start_s"), estimated_onset_end_s=statistical.get("estimated_onset_end_s"),
            window_size_s=statistical.get("window_size_s"), stride_s=statistical.get("stride_s"), confirmation_windows=statistical.get("confirmation_windows"),
            evidence_channels=list(statistical.get("evidence_channels") or []), diagnostic_fault_id=diagnostic_fault_id,
            primary_source="MULTIMETRIC_STATISTICS", agreement="DISAGREE" if model.get("status") == "DETECTED_OTHER_CANDIDATE" else "SINGLE_SOURCE",
            detection_sources=sources, confidence_claim_allowed=False, truth_or_labels_used=False, note_zh=note,
        ).model_dump(mode="json")
    statistical = dict(statistical)
    statistical["schema_version"] = "astrograph-temporal-localization.v2"
    statistical["primary_source"] = None
    statistical["agreement"] = "NO_DETECTION"
    statistical["detection_sources"] = sources
    return TemporalLocalizationResult.model_validate(statistical).model_dump(mode="json")


__all__ = ["TemporalLocalizationResult", "WindowDetector", "RobustTelemetryChangeDetector", "localize_windowed_detection", "localize_from_visualization", "localize_adcs_model_csv", "localize_from_job_telemetry"]
