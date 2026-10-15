from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .fault_patterns import FaultPatternSpec, FaultPatternType


class SyntheticDatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-synthetic-dataset-spec.v1"
    dataset_id: str = Field(min_length=1)
    entity_id: str = Field(default="synthetic_case", min_length=1)
    row_count: int = Field(default=1000, ge=10, le=10_000_000)
    sample_interval: float = Field(default=1.0, gt=0.0)
    seed: int = 20260724
    timestamp_column: str = "time_s"
    channels: list[str] = Field(default_factory=list)
    label_column: str = "label.fault_active"
    patterns: list[FaultPatternSpec] = Field(default_factory=list)
    claim_boundary: str = (
        "Synthetic structural test data only; not an ESA-ADB sample and not a "
        "physically calibrated satellite simulation."
    )

    @field_validator("channels")
    @classmethod
    def unique_channels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("channels must be unique")
        return value


def _stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def channels_from_schema_profile(profile: dict[str, Any]) -> tuple[str, list[str], dict[str, str], list[str]]:
    entities = profile.get("entities", [])
    if not entities:
        timestamp = str(profile.get("timestamp_column") or "time_s")
        labels = [str(item) for item in profile.get("label_columns", [])]
        return timestamp, [], {}, labels
    first = entities[0]
    timestamp = str(
        first.get("timestamp_column") or profile.get("timestamp_column") or "time_s"
    )
    labels = [
        str(item)
        for item in (first.get("label_columns") or profile.get("label_columns") or [])
    ]
    columns = first.get("columns", {})
    dtypes = {
        str(name): str(details.get("dtype", "float64"))
        for name, details in columns.items()
        if isinstance(details, dict)
    }
    channels = [
        str(name)
        for name in columns
        if str(name) != timestamp and str(name) not in set(labels)
    ]
    return timestamp, channels, dtypes, labels


def _base_series(name: str, count: int, rng: np.random.Generator, index: int) -> np.ndarray:
    t = np.arange(count, dtype=float)
    phase = (index % 13) * 0.17
    scale = 1.0 + (index % 7) * 0.25
    trend = ((index % 5) - 2) * 0.0005 * t
    noise = rng.normal(0.0, 0.02 * scale, size=count)
    return scale * np.sin(t / (20.0 + index % 11) + phase) + trend + noise


def _coerce_generated_dtype(values: np.ndarray, dtype: str) -> Any:
    lowered = dtype.lower()
    if "bool" in lowered:
        return values > np.median(values)
    if any(token in lowered for token in ("int", "uint")):
        return np.round(values * 10).astype("int64")
    if any(token in lowered for token in ("float", "double", "number")):
        return values.astype(float)
    # Object/string columns are deterministic categorical placeholders, never
    # copied from the source profile.
    return np.where(values >= 0, "STATE_A", "STATE_B")


def _window_indices(pattern: FaultPatternSpec, count: int) -> tuple[int, int]:
    start = min(count - 1, max(0, int(round(pattern.start_fraction * (count - 1)))))
    end = min(count, max(start + 1, int(round(pattern.end_fraction * count))))
    return start, end


def apply_fault_patterns(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    label_column: str,
    patterns: list[FaultPatternSpec],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy(deep=True)
    if label_column not in result:
        result[label_column] = False
    applied: list[dict[str, Any]] = []
    for pattern in patterns:
        start, end = _window_indices(pattern, len(result))
        channel = pattern.channel
        if channel and channel not in result.columns:
            raise KeyError(f"fault pattern references missing channel: {channel}")
        if pattern.pattern == FaultPatternType.SPIKE:
            location = start
            result.loc[location, channel] = pd.to_numeric(
                pd.Series([result.loc[location, channel]]), errors="coerce"
            ).iloc[0] + pattern.magnitude
            result.loc[location, label_column] = True
        elif pattern.pattern == FaultPatternType.SHIFT:
            values = pd.to_numeric(result.loc[start:end - 1, channel], errors="coerce")
            result.loc[start:end - 1, channel] = values + pattern.magnitude
            result.loc[start:end - 1, label_column] = True
        elif pattern.pattern == FaultPatternType.DRIFT:
            values = pd.to_numeric(result.loc[start:end - 1, channel], errors="coerce")
            drift = np.linspace(0.0, pattern.magnitude, len(values))
            result.loc[start:end - 1, channel] = values.to_numpy() + drift
            result.loc[start:end - 1, label_column] = True
        elif pattern.pattern == FaultPatternType.PERIODIC:
            indices = np.arange(start, end)
            active = indices[(indices - start) % pattern.period == 0]
            values = pd.to_numeric(result.loc[active, channel], errors="coerce")
            result.loc[active, channel] = values + pattern.magnitude
            result.loc[active, label_column] = True
        elif pattern.pattern == FaultPatternType.DROPOUT:
            result.loc[start:end - 1, channel] = np.nan
            result.loc[start:end - 1, label_column] = True
        elif pattern.pattern == FaultPatternType.NOISE:
            values = pd.to_numeric(result.loc[start:end - 1, channel], errors="coerce")
            result.loc[start:end - 1, channel] = values.to_numpy() + rng.normal(
                0.0, abs(pattern.magnitude), size=len(values)
            )
            result.loc[start:end - 1, label_column] = True
        elif pattern.pattern == FaultPatternType.MULTI_RATE:
            indices = np.arange(start, end)
            dropped = indices[(indices - start) % pattern.keep_every != 0]
            result.loc[dropped, channel] = np.nan
            result.loc[start:end - 1, label_column] = True
        elif pattern.pattern == FaultPatternType.DUPLICATE_TIMESTAMP:
            if len(result) >= 2:
                duplicate_at = min(max(1, start), len(result) - 1)
                result.loc[duplicate_at, timestamp_column] = result.loc[
                    duplicate_at - 1, timestamp_column
                ]
                result.loc[duplicate_at, label_column] = True
        elif pattern.pattern == FaultPatternType.REVERSE_TIMESTAMP:
            segment = result.loc[start:end - 1, timestamp_column].to_numpy()[::-1]
            result.loc[start:end - 1, timestamp_column] = segment
            result.loc[start:end - 1, label_column] = True
        elif pattern.pattern == FaultPatternType.MISSING_CHANNEL:
            result = result.drop(columns=[channel])
            result.loc[start:end - 1, label_column] = True
        else:  # pragma: no cover
            raise ValueError(f"unsupported pattern: {pattern.pattern}")
        applied.append(
            {
                "pattern": pattern.pattern.value,
                "channel": channel,
                "start_index": start,
                "end_index_exclusive": end,
                "magnitude": pattern.magnitude,
                "period": pattern.period,
                "keep_every": pattern.keep_every,
            }
        )
    return result, applied


def generate_synthetic_dataset(
    spec: SyntheticDatasetSpec,
    output_root: str | Path,
    *,
    schema_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    entity_dir = output / spec.entity_id
    entity_dir.mkdir(parents=True, exist_ok=True)

    timestamp = spec.timestamp_column
    channels = list(spec.channels)
    dtypes: dict[str, str] = {}
    profile_labels: list[str] = []
    if schema_profile:
        profile_timestamp, profile_channels, dtypes, profile_labels = channels_from_schema_profile(
            schema_profile
        )
        if spec.timestamp_column == "time_s" and profile_timestamp:
            timestamp = profile_timestamp
        if not channels:
            channels = profile_channels
    if not channels:
        channels = [
            "sensor.eps.battery_voltage_v",
            "sensor.eps.battery_current_a",
            "sensor.thermal.payload_temp_c",
        ]
    # Never use truth/label columns as generated model channels.
    excluded_tokens = ("label.", "truth.")
    channels = [
        name for name in channels if name != timestamp and not name.lower().startswith(excluded_tokens)
    ]

    rng = np.random.default_rng(spec.seed)
    frame = pd.DataFrame(
        {timestamp: np.arange(spec.row_count, dtype=float) * spec.sample_interval}
    )
    for index, channel in enumerate(channels):
        values = _base_series(channel, spec.row_count, rng, index)
        frame[channel] = _coerce_generated_dtype(values, dtypes.get(channel, "float64"))

    label_column = spec.label_column
    if profile_labels and spec.label_column == "label.fault_active":
        # Use the first compatible profile label name but generate fresh values.
        label_column = profile_labels[0]
    frame[label_column] = False
    frame, applied = apply_fault_patterns(
        frame,
        timestamp_column=timestamp,
        label_column=label_column,
        patterns=spec.patterns,
        rng=rng,
    )

    telemetry_path = entity_dir / "telemetry.csv"
    frame.to_csv(telemetry_path, index=False, encoding="utf-8")
    data_sha256 = hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "astrograph-synthetic-dataset-manifest.v1",
        "dataset_id": spec.dataset_id,
        "entity_id": spec.entity_id,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "timestamp_column": timestamp,
        "label_column": label_column,
        "channels": [c for c in frame.columns if c not in {timestamp, label_column}],
        "seed": spec.seed,
        "sample_interval": spec.sample_interval,
        "patterns": applied,
        "telemetry_sha256": data_sha256,
        "source_profile_sha256": None
        if schema_profile is None
        else _stable_digest(schema_profile),
        "claim_boundary": spec.claim_boundary,
    }
    manifest["manifest_sha256"] = _stable_digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    (entity_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
