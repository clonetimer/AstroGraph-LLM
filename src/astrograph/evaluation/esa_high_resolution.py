"""High-resolution transient anomaly screen for ESA Mission1-style telemetry.

The R20/R21 ESA path operates on non-overlapping five-minute medians.  This
module provides a separate high-resolution branch that works on the admitted
raw sampling grid.  It intentionally implements a small, auditable detector:
per-channel lag-one deltas are robustly normalized from development reference
windows and coincident large deltas across channels produce transient alarms.

This is not a probability model and it does not use anomaly truth at scoring
time. Dataset-specific code is responsible for selecting development reference
windows and for respecting the data-exposure registry.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ESADeltaChannelReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    channel_id: str = Field(min_length=1)
    delta_center: float
    delta_scale: float = Field(gt=0.0)
    reference_delta_count: int = Field(ge=1)


class ESAHighResolutionReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-esa-high-resolution-reference.v1"
    channels: tuple[ESADeltaChannelReference, ...]
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_semantics: str = "ROBUST_LAG_ONE_DELTA_Z_NOT_PROBABILITY"
    claim_boundary: str = (
        "Development-fitted robust first-difference reference. It preserves the raw sampling grid but is not a "
        "probability, event classifier, causal diagnosis, or external-validation result."
    )

    @model_validator(mode="after")
    def integrity(self) -> "ESAHighResolutionReference":
        ids = tuple(item.channel_id for item in self.channels)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("ESA high-resolution reference channels must be unique and sorted")
        payload = {"channels": [item.model_dump(mode="json") for item in self.channels]}
        if _digest(payload) != self.reference_sha256:
            raise ValueError("ESA high-resolution reference_sha256 mismatch")
        return self


class ESAHighResolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    robust_z_threshold: float = Field(gt=0.0)
    minimum_coincident_channels: int = Field(ge=1)
    alarm_group_gap_s: float = Field(gt=0.0, default=40.0)


class ESAHighResolutionScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-esa-high-resolution-score.v1"
    channel_ids: tuple[str, ...]
    time_ns: tuple[int, ...]
    coincident_channel_count: tuple[int, ...]
    max_robust_delta_z: tuple[float, ...]
    alarm: tuple[bool, ...]
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_semantics: str = "ROBUST_LAG_ONE_DELTA_Z_NOT_PROBABILITY"

    @model_validator(mode="after")
    def lengths_and_hash(self) -> "ESAHighResolutionScore":
        n = len(self.time_ns)
        if not (len(self.coincident_channel_count) == len(self.max_robust_delta_z) == len(self.alarm) == n):
            raise ValueError("ESA high-resolution score arrays must have equal length")
        payload = {
            "channel_ids": self.channel_ids,
            "time_ns": self.time_ns,
            "coincident_channel_count": self.coincident_channel_count,
            "max_robust_delta_z": self.max_robust_delta_z,
            "alarm": self.alarm,
        }
        if _digest(payload) != self.score_sha256:
            raise ValueError("ESA high-resolution score_sha256 mismatch")
        return self


def fit_delta_reference(training_windows: Mapping[str, Sequence[Sequence[float]]]) -> ESAHighResolutionReference:
    """Fit robust lag-one-delta references from already-admitted development windows."""
    from .data_exposure import guard_development
    guard_development("FIT")
    references: list[ESADeltaChannelReference] = []
    if not training_windows:
        raise ValueError("ESA_HIGH_RESOLUTION_REFERENCE_EMPTY")
    for channel_id in sorted(training_windows):
        chunks: list[np.ndarray] = []
        for raw in training_windows[channel_id]:
            values = np.asarray(raw, dtype=float)
            if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
                raise ValueError(f"ESA_HIGH_RESOLUTION_REFERENCE_INVALID:{channel_id}")
            chunks.append(np.diff(values))
        if not chunks:
            raise ValueError(f"ESA_HIGH_RESOLUTION_REFERENCE_EMPTY:{channel_id}")
        deltas = np.concatenate(chunks)
        center = float(np.median(deltas))
        mad = float(np.median(np.abs(deltas - center)))
        scale = max(1.4826 * mad, 1e-6)
        references.append(
            ESADeltaChannelReference(
                channel_id=channel_id,
                delta_center=center,
                delta_scale=scale,
                reference_delta_count=int(deltas.size),
            )
        )
    payload = {"channels": [item.model_dump(mode="json") for item in references]}
    return ESAHighResolutionReference(channels=tuple(references), reference_sha256=_digest(payload))


def score_high_resolution_transients(
    *,
    reference: ESAHighResolutionReference,
    time_ns: Sequence[int],
    channel_values: Mapping[str, Sequence[float]],
    config: ESAHighResolutionConfig,
) -> ESAHighResolutionScore:
    """Score one equal-timeline multichannel segment without changing sampling resolution."""
    from .data_exposure import guard_prediction_generation
    guard_prediction_generation()
    times = np.asarray(time_ns, dtype=np.int64)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) < 0):
        raise ValueError("ESA_HIGH_RESOLUTION_TIME_INVALID")
    refs = {item.channel_id: item for item in reference.channels}
    if tuple(sorted(channel_values)) != tuple(sorted(refs)):
        raise ValueError("ESA_HIGH_RESOLUTION_CHANNEL_SET_MISMATCH")
    if config.minimum_coincident_channels > len(refs):
        raise ValueError("ESA_HIGH_RESOLUTION_MIN_CHANNELS_EXCEEDS_AVAILABLE")

    z_columns: list[np.ndarray] = []
    for channel_id in sorted(refs):
        values = np.asarray(channel_values[channel_id], dtype=float)
        if values.ndim != 1 or values.size != times.size or not np.isfinite(values).all():
            raise ValueError(f"ESA_HIGH_RESOLUTION_VALUES_INVALID:{channel_id}")
        delta = np.r_[0.0, np.diff(values)]
        ref = refs[channel_id]
        z_columns.append(np.abs(delta - ref.delta_center) / ref.delta_scale)
    z = np.vstack(z_columns).T
    count = np.sum(z >= config.robust_z_threshold, axis=1).astype(int)
    max_z = np.max(z, axis=1)
    alarms = count >= config.minimum_coincident_channels
    payload = {
        "channel_ids": tuple(sorted(refs)),
        "time_ns": tuple(int(x) for x in times),
        "coincident_channel_count": tuple(int(x) for x in count),
        "max_robust_delta_z": tuple(float(x) for x in max_z),
        "alarm": tuple(bool(x) for x in alarms),
    }
    return ESAHighResolutionScore(**payload, score_sha256=_digest(payload))


def group_alarm_intervals(
    score: ESAHighResolutionScore,
    *,
    gap_s: float,
) -> tuple[tuple[int, int], ...]:
    """Group alarm samples into deterministic intervals on the raw timestamp grid."""
    if not math.isfinite(gap_s) or gap_s <= 0:
        raise ValueError("ESA_HIGH_RESOLUTION_GROUP_GAP_INVALID")
    gap_ns = int(gap_s * 1_000_000_000)
    alarm_times = [t for t, alarm in zip(score.time_ns, score.alarm) if alarm]
    groups: list[list[int]] = []
    for value in alarm_times:
        if not groups or value - groups[-1][-1] > gap_ns:
            groups.append([value])
        else:
            groups[-1].append(value)
    return tuple((items[0], items[-1]) for items in groups)


__all__ = [
    "ESADeltaChannelReference",
    "ESAHighResolutionConfig",
    "ESAHighResolutionReference",
    "ESAHighResolutionScore",
    "fit_delta_reference",
    "group_alarm_intervals",
    "score_high_resolution_transients",
]
