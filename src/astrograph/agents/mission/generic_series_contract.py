"""Deterministic, truth-free input contract for generic Mission time series.

The contract is deliberately independent from any model family.  It records the
observed channel order/units, data-quality limitations and a canonical content
hash that later model admission can bind to.  It never reads labels or fault
truth and does not infer physical semantics.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .storage import sha


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChannelQuality(_Strict):
    channel: str
    unit: str | None = None
    sample_count: int = Field(ge=0)
    finite_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    positive_infinity_count: int = Field(ge=0)
    negative_infinity_count: int = Field(ge=0)
    constant_finite_signal: bool
    finite_min: float | None = None
    finite_max: float | None = None


class TimestampQuality(_Strict):
    present: bool
    parse_kind: Literal["NONE", "NUMERIC", "DATETIME", "UNPARSEABLE"]
    finite_or_valid_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    monotonic_increasing: bool
    strictly_increasing: bool
    median_step: float | None = None
    relative_step_jitter: float | None = None


class GenericSeriesInputContract(_Strict):
    schema_version: Literal["astrograph-generic-series-input-contract.v1"] = "astrograph-generic-series-input-contract.v1"
    preprocessing_version: Literal["generic-series-contract.post48.dev18.v1"] = "generic-series-contract.post48.dev18.v1"
    source_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    timestamp_column: str | None
    channel_order: tuple[str, ...]
    channel_units: tuple[str | None, ...]
    channel_quality: tuple[ChannelQuality, ...]
    timestamp_quality: TimestampQuality
    quality_status: Literal["VALID", "DEGRADED", "INVALID"]
    reason_codes: tuple[str, ...]
    model_input_eligible: bool
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False


class GenericModelWindow(_Strict):
    schema_version: Literal["astrograph-generic-model-window.v1"] = "astrograph-generic-model-window.v1"
    channel_order: tuple[str, ...]
    channel_units: tuple[str | None, ...]
    window_size: int = Field(ge=1)
    start_row: int = Field(ge=0)
    end_row_exclusive: int = Field(ge=1)
    input_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: tuple[tuple[float, ...], ...]
    labels_used: Literal[False] = False
    truth_used: Literal[False] = False


def _canonical_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp_quality(series: pd.Series | None) -> TimestampQuality:
    if series is None:
        return TimestampQuality(
            present=False, parse_kind="NONE", finite_or_valid_count=0,
            duplicate_count=0, monotonic_increasing=True, strictly_increasing=True,
        )
    numeric = pd.to_numeric(series, errors="coerce")
    finite_numeric = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    if int(finite_numeric.sum()) == len(series):
        arr = numeric.to_numpy(dtype=float)
        diffs = np.diff(arr)
        positive = diffs[diffs > 0]
        median = float(np.median(positive)) if len(positive) else None
        jitter = None
        if median and len(positive):
            jitter = float(np.median(np.abs(positive - median)) / abs(median))
        return TimestampQuality(
            present=True, parse_kind="NUMERIC", finite_or_valid_count=len(series),
            duplicate_count=int(pd.Series(arr).duplicated().sum()),
            monotonic_increasing=bool(np.all(diffs >= 0)), strictly_increasing=bool(np.all(diffs > 0)),
            median_step=median, relative_step_jitter=jitter,
        )
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    valid = parsed.notna()
    if bool(valid.all()):
        arr = parsed.astype("int64").to_numpy(dtype=np.int64) / 1e9
        diffs = np.diff(arr)
        positive = diffs[diffs > 0]
        median = float(np.median(positive)) if len(positive) else None
        jitter = None
        if median and len(positive):
            jitter = float(np.median(np.abs(positive - median)) / abs(median))
        return TimestampQuality(
            present=True, parse_kind="DATETIME", finite_or_valid_count=len(series),
            duplicate_count=int(parsed.duplicated().sum()),
            monotonic_increasing=bool(np.all(diffs >= 0)), strictly_increasing=bool(np.all(diffs > 0)),
            median_step=median, relative_step_jitter=jitter,
        )
    return TimestampQuality(
        present=True, parse_kind="UNPARSEABLE", finite_or_valid_count=int(max(finite_numeric.sum(), valid.sum())),
        duplicate_count=int(series.astype(str).duplicated().sum()), monotonic_increasing=False,
        strictly_increasing=False,
    )


def build_generic_series_input_contract(
    frame: pd.DataFrame, *, timestamp_column: str | None, channel_order: list[str] | tuple[str, ...],
    channel_units: list[str | None] | tuple[str | None, ...], source_data_sha256: str,
) -> GenericSeriesInputContract:
    names = tuple(str(x) for x in channel_order)
    units = tuple(channel_units)
    if len(names) != len(units) or not names or len(names) != len(set(names)):
        raise ValueError("channel order/units must be nonempty, unique and aligned")
    missing = sorted(set(names) - set(map(str, frame.columns)))
    if missing:
        raise ValueError(f"contract frame is missing channels: {missing}")
    qualities: list[ChannelQuality] = []
    reason_codes: list[str] = []
    canonical_rows: list[list[float | None]] = []
    arrays: list[np.ndarray] = []
    for name, unit in zip(names, units, strict=True):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        arrays.append(values)
        finite = np.isfinite(values)
        missing_mask = np.isnan(values)
        posinf = np.isposinf(values)
        neginf = np.isneginf(values)
        finite_values = values[finite]
        qualities.append(ChannelQuality(
            channel=name, unit=unit, sample_count=len(values), finite_count=int(finite.sum()),
            missing_count=int(missing_mask.sum()), positive_infinity_count=int(posinf.sum()),
            negative_infinity_count=int(neginf.sum()),
            constant_finite_signal=bool(len(finite_values) > 0 and np.all(finite_values == finite_values[0])),
            finite_min=float(finite_values.min()) if len(finite_values) else None,
            finite_max=float(finite_values.max()) if len(finite_values) else None,
        ))
        if missing_mask.any(): reason_codes.append(f"MISSING_VALUES:{name}")
        if posinf.any() or neginf.any(): reason_codes.append(f"NONFINITE_INFINITY:{name}")
        if not finite.any(): reason_codes.append(f"NO_FINITE_VALUES:{name}")
    if arrays:
        for row in zip(*arrays, strict=True):
            canonical_rows.append([_canonical_number(v) for v in row])
    timestamp_series = frame[timestamp_column] if timestamp_column else None
    tq = _timestamp_quality(timestamp_series)
    if tq.present and tq.parse_kind == "UNPARSEABLE": reason_codes.append("TIMESTAMP_UNPARSEABLE")
    if tq.present and tq.duplicate_count: reason_codes.append("TIMESTAMP_DUPLICATES")
    if tq.present and not tq.strictly_increasing: reason_codes.append("TIMESTAMP_NOT_STRICTLY_INCREASING")
    if len(frame) == 0: reason_codes.append("EMPTY_SERIES")
    invalid = bool(len(frame) == 0 or any(q.finite_count == 0 for q in qualities) or tq.parse_kind == "UNPARSEABLE")
    degraded = bool(reason_codes) and not invalid
    status = "INVALID" if invalid else "DEGRADED" if degraded else "VALID"
    model_eligible = status == "VALID" and all(q.finite_count == q.sample_count for q in qualities) and (
        not tq.present or tq.strictly_increasing
    )
    timestamp_values = None
    if timestamp_column:
        if tq.parse_kind == "NUMERIC": timestamp_values = [float(x) for x in pd.to_numeric(frame[timestamp_column]).tolist()]
        elif tq.parse_kind == "DATETIME": timestamp_values = [x.isoformat() for x in pd.to_datetime(frame[timestamp_column], utc=True)]
        else: timestamp_values = [str(x) for x in frame[timestamp_column].tolist()]
    payload = {
        "preprocessing_version": "generic-series-contract.post48.dev18.v1",
        "channel_order": list(names), "channel_units": list(units),
        "timestamp_column": timestamp_column, "timestamp_values": timestamp_values,
        "values": canonical_rows,
    }
    return GenericSeriesInputContract(
        source_data_sha256=source_data_sha256, canonical_input_sha256=sha(payload), row_count=len(frame),
        timestamp_column=timestamp_column, channel_order=names, channel_units=units,
        channel_quality=tuple(qualities), timestamp_quality=tq, quality_status=status,
        reason_codes=tuple(dict.fromkeys(reason_codes)), model_input_eligible=model_eligible,
    )


def materialize_model_window(
    frame: pd.DataFrame, *, contract: GenericSeriesInputContract, window_size: int,
    end_row_exclusive: int | None = None,
) -> GenericModelWindow:
    if not contract.model_input_eligible:
        raise ValueError("generic series is not eligible for strict model execution")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    end = len(frame) if end_row_exclusive is None else int(end_row_exclusive)
    start = end - int(window_size)
    if start < 0 or end > len(frame) or end <= start:
        raise ValueError("requested model window is outside the available series")
    selected = frame.loc[:, list(contract.channel_order)].iloc[start:end]
    matrix = selected.to_numpy(dtype=float)
    if matrix.shape != (window_size, len(contract.channel_order)) or not np.isfinite(matrix).all():
        raise ValueError("strict model window must be finite and shape-exact")
    values = tuple(tuple(float(v) for v in row) for row in matrix.tolist())
    payload = {"channel_order": list(contract.channel_order), "telemetry_window": [list(row) for row in values]}
    return GenericModelWindow(
        channel_order=contract.channel_order, channel_units=contract.channel_units, window_size=window_size,
        start_row=start, end_row_exclusive=end, input_data_sha256=sha(payload), values=values,
    )
