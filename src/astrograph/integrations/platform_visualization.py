"""Truth-safe visualization extraction for the upload platform.

Only observed telemetry already supplied as an executable input (or an explicitly
uploaded non-label CSV) is plotted.  Truth/label files are never used to rank or
select channels.  The module performs display-oriented statistics only and makes no
diagnostic claim.
"""
from __future__ import annotations

import csv
import math
import re
from math import ceil
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from astrograph.integrations.operator_vocabulary import channel_label_zh


VISUALIZATION_SCHEMA_VERSION = "astrograph-platform-visualization.v1"
_MAX_POINTS = 600
_MAX_ROWS_SCANNED = 250_000
_MAX_COLUMNS = 320
_BANNED_DISCOVERY_NAMES = ("label", "ground_truth", "groundtruth", "truth", "annotation", "answer")
_TIME_NAMES = ("time_s", "elapsed_s", "sim_time_s", "time", "timestamp_s", "timestamp", "t")

_FAULT_CHANNEL_HINTS: dict[str, tuple[str, ...]] = {
    "ADCS_RW_JAM": (
        "adcs.rw.command_execution_residual_norm_nm",
        "adcs.rw.speed_response_residual_norm_nm",
        "adcs.pointing_error_deg",
        "adcs.rw.measured_speed_rad_s_0",
        "adcs.rw.measured_speed_rad_s_1",
        "adcs.rw.measured_speed_rad_s_2",
        "adcs.rw.measured_speed_rad_s_3",
        "adcs.rw.motor_current_a_0",
        "adcs.rw.motor_current_a_1",
        "adcs.rw.motor_current_a_2",
        "adcs.rw.motor_current_a_3",
    ),
    "ADCS_RW_FRICTION_INCREASE": (
        "adcs.rw.speed_response_residual_norm_nm",
        "adcs.rw.command_execution_residual_norm_nm",
        "adcs.pointing_error_deg",
        "adcs.rw.measured_speed_rad_s_0",
        "adcs.rw.measured_speed_rad_s_1",
        "adcs.rw.measured_speed_rad_s_2",
        "adcs.rw.measured_speed_rad_s_3",
        "adcs.rw.motor_current_a_0",
        "adcs.rw.motor_current_a_1",
        "adcs.rw.motor_current_a_2",
        "adcs.rw.motor_current_a_3",
    ),
    "ADCS_RW_TORQUE_AUTHORITY_LOSS": (
        "adcs.rw.command_execution_residual_norm_nm",
        "adcs.rw.speed_response_residual_norm_nm",
        "adcs.pointing_error_deg",
        "adcs.rw.command_torque_nm_0",
        "adcs.rw.command_torque_nm_1",
        "adcs.rw.command_torque_nm_2",
        "adcs.rw.command_torque_nm_3",
    ),
}


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit(name: str) -> str:
    low = name.casefold()
    suffixes = [
        ("_deg_c", "°C"), ("_deg", "°"), ("_rad_s", "rad/s"), ("_rad", "rad"),
        ("_nm", "N·m"), ("_rpm", "rpm"), ("_bps", "bps"), ("_wh", "Wh"),
        ("_mah", "mAh"), ("_hz", "Hz"), ("_pa", "Pa"), ("_amp", "A"),
        ("_a", "A"), ("_volt", "V"), ("_v", "V"), ("_w", "W"), ("_k", "K"),
    ]
    for suffix, unit in suffixes:
        if low.endswith(suffix):
            return unit
    return ""


def _time_column(headers: list[str]) -> str | None:
    lower = {name.casefold(): name for name in headers}
    for candidate in _TIME_NAMES:
        if candidate in lower:
            return lower[candidate]
    for name in headers:
        low = name.casefold()
        if low.endswith(".time_s") or low.endswith("_time_s") or "elapsed" in low:
            return name
    return None


def _timestamp_seconds(value: str, base: datetime | None) -> tuple[float | None, datetime | None]:
    number = _float(value)
    if number is not None:
        return number, base
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None, base
    base = base or parsed
    return (parsed - base).total_seconds(), base


def _candidate_hints(report: dict[str, Any]) -> list[str]:
    analysis = report.get("analysis") or {}
    hints: list[str] = []
    pipeline = analysis.get("pipeline_result") or {}
    diagnosis = analysis.get("diagnosis") or {}
    fault_id = str(
        (pipeline.get("fusion") or {}).get("top_fault_id")
        or pipeline.get("fault_id")
        or diagnosis.get("selected_fault_id")
        or diagnosis.get("primary_effect_id")
        or ""
    )
    hints.extend(_FAULT_CHANNEL_HINTS.get(fault_id, ()))
    for item in ((analysis.get("diagnosis") or {}).get("diagnostic_evidence") or []):
        if isinstance(item, dict) and item.get("field"):
            hints.append(str(item["field"]))
    oag = ((analysis.get("oag_result") or {}).get("output") or {}).get("summary") or {}
    for item in oag.get("evidence") or []:
        if isinstance(item, dict):
            for key in ("feature_field", "field", "channel"):
                if item.get(key):
                    hints.append(str(item[key]))
                    break
    pipeline = analysis.get("pipeline_result") or {}

    def walk(value: Any) -> None:
        if len(hints) >= 40:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key.casefold() in {"channel", "feature_field", "telemetry_field"} and isinstance(item, str):
                    hints.append(item)
                elif key.casefold() in {"criteria", "diagnostic_validation", "observations"}:
                    walk(item)
        elif isinstance(value, list):
            for item in value[:80]:
                walk(item)
    walk(pipeline)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in hints:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _select_csv(expanded: Path, *, report: dict[str, Any]) -> Path | None:
    kind = report.get("artifact_kind")
    if kind == "PUBLIC_CASE":
        analysis = report.get("analysis") or {}
        closure = analysis.get("closure") or {}
        case_id = closure.get("case_id")
        manifests = list(expanded.rglob("manifest.json"))
        for manifest_path in manifests:
            try:
                import json
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if case_id and manifest.get("case_id") != case_id:
                continue
            name = str((manifest.get("telemetry") or {}).get("file") or "")
            candidate = (manifest_path.parent / name).resolve()
            if name and candidate.is_file() and expanded in candidate.parents:
                return candidate
    for path in sorted(expanded.rglob("*.csv")):
        low = path.name.casefold()
        if any(token in low for token in _BANNED_DISCOVERY_NAMES):
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                headers = next(reader)
        except Exception:
            continue
        if _time_column([str(h).strip() for h in headers]):
            return path
    return None




def select_truth_safe_telemetry_csv(expanded: Path, *, report: dict[str, Any]) -> Path | None:
    """Return the same truth-safe telemetry CSV used by the visualization layer.

    This public wrapper exists so maintenance analytics can consume the exact observed
    source selected for plotting without rediscovering label/truth files independently.
    """
    return _select_csv(expanded, report=report)

def _match_hints(headers: list[str], hints: list[str]) -> list[str]:
    by_lower = {h.casefold(): h for h in headers}
    selected: list[str] = []
    for hint in hints:
        direct = by_lower.get(hint.casefold())
        if direct and direct not in selected:
            selected.append(direct)
            continue
        tail = hint.casefold().split(".")[-1]
        matches = [h for h in headers if h.casefold().split(".")[-1] == tail]
        if len(matches) == 1 and matches[0] not in selected:
            selected.append(matches[0])
    return selected


def _sample_csv(path: Path, *, hints: list[str]) -> tuple[str | None, list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [str(item).strip() for item in (reader.fieldnames or [])][: _MAX_COLUMNS + 1]
        tcol = _time_column(headers)
        if not headers:
            return None, [], []
        hint_columns = _match_hints(headers, hints)
        # Keep hinted columns plus bounded remaining columns for a display-only variance fallback.
        candidates = [h for h in headers if h != tcol]
        ordered = hint_columns + [h for h in candidates if h not in hint_columns]
        ordered = ordered[:_MAX_COLUMNS]
        sampled: list[dict[str, Any]] = []
        stride = 1
        base_dt: datetime | None = None
        for row_index, row in enumerate(reader):
            if row_index >= _MAX_ROWS_SCANNED:
                break
            if row_index % stride:
                continue
            if tcol:
                t, base_dt = _timestamp_seconds(str(row.get(tcol, "")), base_dt)
            else:
                t = float(row_index)
            if t is None:
                t = float(row_index)
            values = {name: _float(row.get(name)) for name in ordered}
            sampled.append({"t": t, "values": values})
            if len(sampled) > _MAX_POINTS * 2:
                sampled = sampled[::2]
                stride *= 2
        return tcol, sampled, ordered


def _rank_channels(samples: list[dict[str, Any]], columns: list[str], hinted: list[str]) -> list[str]:
    # Diagnostic-context hints are selected first; variance is only a fallback for
    # remaining display slots. This prevents large-magnitude housekeeping counters
    # from displacing fault-relevant telemetry merely because of numeric scale.
    ordered_hints: list[str] = []
    for name in hinted:
        if name in columns and name not in ordered_hints:
            values = [row["values"].get(name) for row in samples]
            finite = [float(v) for v in values if v is not None]
            if len(finite) >= 2:
                ordered_hints.append(name)
    if len(ordered_hints) >= 6:
        return ordered_hints[:6]
    ranked: list[tuple[float, str]] = []
    hint_set = set(ordered_hints)
    for column in columns:
        if column in hint_set:
            continue
        values = [row["values"].get(column) for row in samples]
        finite = [float(v) for v in values if v is not None]
        if len(finite) < 2:
            continue
        mean = fmean(finite)
        variance = fmean([(v - mean) ** 2 for v in finite])
        spread = max(finite) - min(finite)
        score = variance + spread * spread * 0.02
        ranked.append((score, column))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return (ordered_hints + [name for _, name in ranked])[:6]


def _diagnostic_onset(report: dict[str, Any]) -> tuple[float | None, str | None]:
    analysis = report.get("analysis") or {}
    diagnosis = analysis.get("diagnosis") or {}
    onset = _float(diagnosis.get("estimated_onset_time_s"))
    if onset is not None:
        return onset, "DIAGNOSTIC_ESTIMATE"
    closure = analysis.get("closure") or {}
    onset = _float((closure.get("diagnosis") or {}).get("estimated_onset_time_s"))
    if onset is not None:
        return onset, "DIAGNOSTIC_ESTIMATE"
    return None, None




def _scan_selected_csv(
    path: Path,
    *,
    selected: list[str],
    max_points: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int, float | None, float | None]:
    """Read selected channels with bounded, extrema-preserving sampling.

    The first pass computes full admitted-row coverage and per-channel extrema.  The
    second pass keeps a uniform bounded sample plus every selected channel's min/max
    row.  Plot downsampling therefore cannot erase the only peak, and coverage is
    based on source rows rather than only rows that survived display sampling.
    """
    stats = {
        name: {"valid": 0, "minimum": None, "maximum": None, "min_row": None, "max_row": None}
        for name in selected
    }
    row_count = 0
    start_s: float | None = None
    end_s: float | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [str(item).strip() for item in (reader.fieldnames or [])]
        tcol = _time_column(headers)
        base_dt: datetime | None = None
        for row_index, row in enumerate(reader):
            if row_index >= _MAX_ROWS_SCANNED:
                break
            if tcol:
                t, base_dt = _timestamp_seconds(str(row.get(tcol, "")), base_dt)
            else:
                t = float(row_index)
            if t is None:
                t = float(row_index)
            t = float(t)
            start_s = t if start_s is None else min(start_s, t)
            end_s = t if end_s is None else max(end_s, t)
            row_count += 1
            for name in selected:
                value = _float(row.get(name))
                if value is None:
                    continue
                item = stats[name]
                item["valid"] += 1
                if item["minimum"] is None or value < item["minimum"]:
                    item["minimum"] = value
                    item["min_row"] = row_index
                if item["maximum"] is None or value > item["maximum"]:
                    item["maximum"] = value
                    item["max_row"] = row_index

    if row_count == 0:
        return [], stats, 0, start_s, end_s
    forced = {0, row_count - 1}
    for item in stats.values():
        for key in ("min_row", "max_row"):
            if item[key] is not None:
                forced.add(int(item[key]))
    budget = max(1, int(max_points) - len(forced))
    stride = 1 if row_count <= max_points else max(1, ceil(row_count / budget))

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [str(item).strip() for item in (reader.fieldnames or [])]
        tcol = _time_column(headers)
        base_dt = None
        for row_index, row in enumerate(reader):
            if row_index >= row_count:
                break
            keep = row_index in forced or row_index % stride == 0
            if tcol:
                t, base_dt = _timestamp_seconds(str(row.get(tcol, "")), base_dt)
            else:
                t = float(row_index)
            if not keep:
                continue
            if t is None:
                t = float(row_index)
            samples.append({
                "t": float(t),
                "values": {name: _float(row.get(name)) for name in selected},
                "source_row_index": row_index,
            })
    samples.sort(key=lambda row: (float(row["t"]), int(row["source_row_index"])))
    return samples, stats, row_count, start_s, end_s


def _series_payload(
    *,
    expanded: Path,
    report: dict[str, Any],
    max_points: int,
    schema_version: str,
    purpose: str,
) -> dict[str, Any]:
    csv_path = _select_csv(expanded, report=report)
    onset, onset_source = _diagnostic_onset(report)
    base: dict[str, Any] = {
        "schema_version": schema_version,
        "artifact_kind": report.get("artifact_kind"),
        "source": None,
        "status": "NO_SERIES",
        "timeline": {"start_s": None, "end_s": None, "onset_s": onset, "onset_source": onset_source},
        "channels": [],
        "purpose": purpose,
        "truth_or_labels_used": False,
    }
    if csv_path is None:
        base["message"] = "当前任务未找到可安全使用的原始遥测 CSV。"
        return base
    hints = _candidate_hints(report)
    try:
        _, ranking_samples, columns = _sample_csv(csv_path, hints=hints)
    except Exception as exc:
        base["status"] = "FAILED_TO_PARSE_SERIES"
        base["message"] = f"{type(exc).__name__}: {exc}"
        return base
    if not ranking_samples:
        base["message"] = "遥测 CSV 没有可用数据行。"
        return base
    hinted = _match_hints(columns, hints)
    selected = _rank_channels(ranking_samples, columns, hinted)
    if not selected:
        base["message"] = "遥测 CSV 没有足够的数值通道。"
        return base
    samples, stats, row_count, start_s, end_s = _scan_selected_csv(
        csv_path, selected=selected, max_points=max_points
    )
    channels: list[dict[str, Any]] = []
    for name in selected:
        points = [
            {"t": round(float(row["t"]), 9), "v": row["values"].get(name)}
            for row in samples if row["values"].get(name) is not None
        ]
        item = stats[name]
        if int(item["valid"]) < 2:
            continue
        channels.append({
            "name": name,
            "label": channel_label_zh(name),
            "unit": _unit(name),
            "points": points,
            "minimum": item["minimum"],
            "maximum": item["maximum"],
            "source_sample_count": row_count,
            "valid_sample_count": int(item["valid"]),
            "coverage_ratio": (float(item["valid"]) / row_count if row_count else 0.0),
            "importance_reason": "诊断证据字段" if name in hinted else "显示统计排序（方差/幅度）",
            "diagnostic_score": None,
        })
    base.update({
        "status": "AVAILABLE" if channels else "NO_NUMERIC_CHANNELS",
        "source": {"filename": csv_path.name, "relative_path": str(csv_path.relative_to(expanded)).replace("\\", "/")},
        "timeline": {"start_s": start_s, "end_s": end_s, "onset_s": onset, "onset_source": onset_source},
        "channels": channels,
        "source_row_count": row_count,
        "sampled_point_count": len(samples),
        "sampling": {"max_points": max_points, "extrema_preserved": True},
    })
    return base


def build_visualization(*, expanded: Path, report: dict[str, Any]) -> dict[str, Any]:
    payload = _series_payload(
        expanded=expanded, report=report, max_points=_MAX_POINTS,
        schema_version=VISUALIZATION_SCHEMA_VERSION, purpose="DISPLAY",
    )
    payload["display_boundary"] = "曲线与通道排序用于维护可视化，不构成新的诊断概率或投票。"
    return payload


def build_analysis_series(*, expanded: Path, report: dict[str, Any], max_points: int = 20_000) -> dict[str, Any]:
    """Build a truth-safe maintenance-analysis series independent of plot density.

    For ordinary maintenance files up to ``max_points`` admitted rows, no temporal
    downsampling occurs.  Larger files remain bounded but preserve selected-channel
    extrema and source-row coverage metadata.
    """
    payload = _series_payload(
        expanded=expanded, report=report, max_points=max_points,
        schema_version="astrograph-maintenance-analysis-series.v1", purpose="MAINTENANCE_ANALYSIS",
    )
    payload["analysis_boundary"] = (
        "统计/时间定位只使用 truth-safe 原始遥测；该采样合同与绘图密度分离，"
        "不产生新的正式 EvidenceFact。"
    )
    return payload


__all__ = [
    "VISUALIZATION_SCHEMA_VERSION", "select_truth_safe_telemetry_csv",
    "build_visualization", "build_analysis_series",
]
