from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from astrograph.semantics.leakage import (
    PROHIBITED_CONTAINS,
    PROHIBITED_EXACT,
    PROHIBITED_PREFIXES,
    is_prohibited_field,
)

from .unified_models import DataQualityFinding, TelemetryProfile


# Deliberately small and reversible. Unit-bearing field names are not rewritten.
FIELD_ALIASES: dict[str, str] = {
    "timestamp": "time_s",
    "time": "time_s",
    "t_s": "time_s",
}


@dataclass(frozen=True)
class NormalizedTelemetry:
    frame: pd.DataFrame
    profile: TelemetryProfile


def is_prohibited(column: str) -> bool:
    return is_prohibited_field(column)


def infer_backend(columns: Iterable[str]) -> str:
    names = list(columns)
    dotted = sum("." in name for name in names)
    return "unified" if dotted > max(2, len(names) // 5) else "composite"


def normalize_telemetry_frame(frame: pd.DataFrame, *, requested_backend: str = "auto") -> NormalizedTelemetry:
    findings: list[DataQualityFinding] = []
    rename: dict[str, str] = {}
    normalized_names: list[str] = []
    seen: set[str] = set()
    for original in frame.columns:
        stripped = str(original).strip()
        canonical = FIELD_ALIASES.get(stripped.lower(), stripped)
        if canonical in seen:
            findings.append(DataQualityFinding(
                code="CANONICAL_COLUMN_COLLISION",
                severity="ERROR",
                message=f"multiple source columns normalize to {canonical}",
                fields=[str(original), canonical],
            ))
        seen.add(canonical)
        rename[str(original)] = canonical
        normalized_names.append(canonical)
    normalized = frame.rename(columns=rename).copy()
    prohibited = sorted({column for column in normalized.columns if is_prohibited(column)})
    if prohibited:
        findings.append(DataQualityFinding(
            code="PROHIBITED_TRUTH_COLUMN",
            severity="ERROR",
            message="public diagnosis input contains labels, truth, or direct effect evidence",
            fields=prohibited,
        ))
    if normalized.empty:
        findings.append(DataQualityFinding(
            code="EMPTY_TELEMETRY",
            severity="ERROR",
            message="telemetry contains no rows",
        ))
    numeric_signal_count = 0
    for column in normalized.columns:
        if column == "time_s" or is_prohibited(column):
            continue
        series = pd.to_numeric(normalized[column], errors="coerce")
        if series.notna().any():
            numeric_signal_count += 1
            normalized[column] = series
    if numeric_signal_count == 0:
        findings.append(DataQualityFinding(
            code="NO_NUMERIC_SIGNALS",
            severity="ERROR",
            message="telemetry contains no observable numeric signals",
        ))
    time_column = "time_s" if "time_s" in normalized.columns else None
    time_monotonic: bool | None = None
    if time_column:
        time_values = pd.to_numeric(normalized[time_column], errors="coerce")
        time_monotonic = bool(time_values.notna().all() and time_values.is_monotonic_increasing)
        normalized[time_column] = time_values
        if not time_monotonic:
            findings.append(DataQualityFinding(
                code="NON_MONOTONIC_TIME",
                severity="ERROR",
                message="time_s must be finite and monotonically increasing",
                fields=[time_column],
            ))
    inferred = infer_backend(normalized.columns)
    backend = inferred if requested_backend == "auto" else requested_backend
    if requested_backend != "auto" and requested_backend != inferred:
        findings.append(DataQualityFinding(
            code="BACKEND_PROFILE_MISMATCH",
            severity="WARNING",
            message=f"requested backend {requested_backend} differs from inferred profile {inferred}",
        ))
    status = "FAIL" if any(item.severity == "ERROR" for item in findings) else (
        "WARN" if findings else "PASS"
    )
    profile = TelemetryProfile(
        backend=backend,
        row_count=len(normalized),
        column_count=len(normalized.columns),
        numeric_signal_count=numeric_signal_count,
        time_column=time_column,
        time_monotonic=time_monotonic,
        prohibited_columns=prohibited,
        normalized_columns=list(normalized.columns),
        alias_mapping={source: target for source, target in rename.items() if source != target},
        findings=findings,
        status=status,
    )
    return NormalizedTelemetry(frame=normalized, profile=profile)


__all__ = [
    "NormalizedTelemetry",
    "FIELD_ALIASES",
    "PROHIBITED_PREFIXES",
    "PROHIBITED_EXACT",
    "PROHIBITED_CONTAINS",
    "is_prohibited",
    "infer_backend",
    "normalize_telemetry_frame",
]
