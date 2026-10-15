"""Truth-free, shadow-only OPS-SAT probe measurements.

The functions in this module inspect an already admitted segment.  They do not
read labels, select a threshold, or create formal evidence.  Every result is
bound to the caller-supplied segment digest so an observation cannot be reused
against a different segment.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from astrograph.evaluation.ops_sat_adapter import FEATURE_FIELDS, validate_online_fields


OPS_SAT_PROBE_LIBRARY_VERSION = "ops-sat-shadow-probes.v2"
OPS_SAT_PROBE_IDS = (
    "early_late_shift",
    "log_volatility_ratio",
    "peak_concentration_delta",
    "normalized_linear_trend",
    "lag1_autocorrelation_delta",
    "high_frequency_energy_delta",
)
OPS_SAT_OFFICIAL_FEATURE_BUNDLE_VERSION = "ops-sat-official-derived-features.v1"
OPS_SAT_LENGTH_FREE_FEATURE_IDS = tuple(
    field for field in FEATURE_FIELDS
    if field not in {"duration", "len", "len_weighted", "var_div_duration", "var_div_len", "gaps_squared"}
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _correlation_lag1(values: np.ndarray) -> float:
    left, right = values[:-1], values[1:]
    if left.size < 2 or float(np.std(left)) <= 1e-15 or float(np.std(right)) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _high_frequency_energy(values: np.ndarray) -> float:
    variance = float(np.var(values))
    if variance <= 1e-24:
        return 0.0
    return float(np.mean(np.diff(values) ** 2) / variance)


def measure_ops_sat_probe(points: list[dict], probe_id: str, *, source_sha256: str) -> dict:
    """Measure one registered segment statistic without access to offline truth."""
    if probe_id not in OPS_SAT_PROBE_IDS:
        raise ValueError("UNREGISTERED_OPS_SAT_PROBE")
    if not (
        isinstance(source_sha256, str)
        and len(source_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("OPS_SAT_PROBE_SOURCE_SHA256_REQUIRED")
    if not points:
        raise ValueError("OPS_SAT_PROBE_REQUIRES_SEGMENT")
    identities = {
        (str(point.get("segment")), str(point.get("channel")), str(point.get("sampling")))
        for point in points
    }
    if len(identities) != 1:
        raise ValueError("OPS_SAT_PROBE_SEGMENT_IDENTITY_DRIFT")
    values = np.asarray([float(point["value"]) for point in points], dtype=float)
    if values.size < 8 or not np.isfinite(values).all():
        raise ValueError("OPS_SAT_PROBE_REQUIRES_AT_LEAST_EIGHT_FINITE_SAMPLES")
    split = values.size // 2
    early, late = values[:split], values[split:]
    scale = max(float(np.std(values, ddof=1)), abs(float(np.median(values))) * 1e-9, 1e-12)

    if probe_id == "early_late_shift":
        value = (float(np.mean(late)) - float(np.mean(early))) / scale
        semantics = "late-minus-early mean divided by full-segment sample standard deviation"
    elif probe_id == "log_volatility_ratio":
        early_std = max(float(np.std(early, ddof=1)), scale * 1e-9, 1e-12)
        late_std = max(float(np.std(late, ddof=1)), scale * 1e-9, 1e-12)
        value = math.log(late_std / early_std)
        semantics = "natural log of late-half to early-half sample-standard-deviation ratio"
    elif probe_id == "peak_concentration_delta":
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        robust_scale = max(1.4826 * mad, scale * 1e-6, 1e-12)
        early_rate = float(np.mean(np.abs(early - center) / robust_scale >= 3.0))
        late_rate = float(np.mean(np.abs(late - center) / robust_scale >= 3.0))
        value = late_rate - early_rate
        semantics = "late-minus-early fraction beyond three full-segment robust standard deviations"
    elif probe_id == "normalized_linear_trend":
        x = np.arange(values.size, dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])
        value = slope * float(values.size - 1) / scale
        semantics = "least-squares end-to-end linear change divided by sample standard deviation"
    elif probe_id == "lag1_autocorrelation_delta":
        value = _correlation_lag1(late) - _correlation_lag1(early)
        semantics = "late-minus-early lag-one Pearson autocorrelation"
    else:
        value = _high_frequency_energy(late) - _high_frequency_energy(early)
        semantics = "late-minus-early first-difference energy normalized by half-segment variance"
    if not math.isfinite(value):
        raise ValueError("OPS_SAT_PROBE_PRODUCED_NONFINITE_VALUE")

    segment, channel, sampling = next(iter(identities))
    binding = {
        "probe_library_version": OPS_SAT_PROBE_LIBRARY_VERSION,
        "probe_id": probe_id,
        "segment": segment,
        "channel": channel,
        "sampling": sampling,
        "source_sha256": source_sha256,
        "sample_count": int(values.size),
        "value": float(value),
        "semantics": semantics,
    }
    return {
        "observation_id": "OPSOBS-" + _digest(binding)[:16].upper(),
        **binding,
        "creates_new_raw_observation": True,
        "formal_evidence_generated": False,
        "formal_evidence_eligible": False,
        "claim_boundary": "Shadow observation only; not a diagnosis, probability, or formal EvidenceFact.",
    }


def measure_ops_sat_probe_bundle(points: list[dict], *, source_sha256: str) -> dict:
    """Return all registered measurements as one bounded shadow observation."""
    components = [
        measure_ops_sat_probe(points, probe_id, source_sha256=source_sha256)
        for probe_id in OPS_SAT_PROBE_IDS
    ]
    binding = {
        "probe_library_version": OPS_SAT_PROBE_LIBRARY_VERSION,
        "source_sha256": source_sha256,
        "component_observation_ids": [component["observation_id"] for component in components],
    }
    return {
        "observation_id": "OPSBUNDLE-" + _digest(binding)[:16].upper(),
        **binding,
        "components": [
            {
                "probe_id": component["probe_id"],
                "value": component["value"],
                "semantics": component["semantics"],
            }
            for component in components
        ],
        "component_count": len(components),
        "creates_new_raw_observation": True,
        "formal_evidence_generated": False,
        "formal_evidence_eligible": False,
        "claim_boundary": (
            "One bounded shadow action containing multiple deterministic segment statistics; "
            "not a diagnosis, probability, or formal EvidenceFact."
        ),
    }


def measure_ops_sat_official_feature_bundle(feature_row: dict, *, source_sha256: str) -> dict:
    """Expose one admitted official feature row as a non-voting derived observation."""
    if not (
        isinstance(source_sha256, str)
        and len(source_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("OPS_SAT_FEATURE_SOURCE_SHA256_REQUIRED")
    validate_online_fields(feature_row, track="feature")
    values = {field: float(feature_row[field]) for field in FEATURE_FIELDS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("OPS_SAT_FEATURE_BUNDLE_REQUIRES_FINITE_VALUES")
    row_binding = {
        "bundle_version": OPS_SAT_OFFICIAL_FEATURE_BUNDLE_VERSION,
        "segment": str(feature_row["segment"]),
        "channel": str(feature_row["channel"]),
        "sampling": str(feature_row["sampling"]),
        "values": values,
        "source_sha256": source_sha256,
    }
    row_sha256 = _digest(row_binding)
    return {
        "observation_id": "OPSFEATURES-" + row_sha256[:16].upper(),
        **row_binding,
        "row_sha256": row_sha256,
        "feature_count": len(FEATURE_FIELDS),
        "length_free_feature_ids": list(OPS_SAT_LENGTH_FREE_FEATURE_IDS),
        "newly_inspected_input": True,
        "creates_new_raw_observation": False,
        "creates_new_derived_observation": True,
        "formal_evidence_generated": False,
        "formal_evidence_eligible": False,
        "claim_boundary": (
            "Newly inspected official full-segment derived features, not a newly acquired sensor sample. "
            "Shadow observation only; not a diagnosis, probability, or formal EvidenceFact."
        ),
    }


def measure_ops_sat_length_free_feature_bundle(feature_row: dict, *, source_sha256: str) -> dict:
    """Expose only the predeclared feature subset that excludes segment-length proxies."""
    full = measure_ops_sat_official_feature_bundle(feature_row, source_sha256=source_sha256)
    values = {field: full["values"][field] for field in OPS_SAT_LENGTH_FREE_FEATURE_IDS}
    binding = {
        "bundle_version": "ops-sat-length-free-derived-features.v1",
        "segment": full["segment"],
        "channel": full["channel"],
        "sampling": full["sampling"],
        "feature_ids": list(OPS_SAT_LENGTH_FREE_FEATURE_IDS),
        "values": values,
        "source_sha256": source_sha256,
        "upstream_full_row_sha256": full["row_sha256"],
    }
    digest = _digest(binding)
    return {
        "observation_id": "OPSLENGTHFREE-" + digest[:16].upper(),
        **binding,
        "observation_sha256": digest,
        "feature_count": len(values),
        "newly_inspected_input": True,
        "creates_new_raw_observation": False,
        "creates_new_derived_observation": True,
        "formal_evidence_generated": False,
        "formal_evidence_eligible": False,
        "claim_boundary": (
            "Official retrospective full-segment shape/dynamics features with duration and length proxies removed. "
            "Shadow observation only; not formal evidence or an online early-warning claim."
        ),
    }
__all__ = [
    "OPS_SAT_PROBE_IDS",
    "OPS_SAT_PROBE_LIBRARY_VERSION",
    "OPS_SAT_OFFICIAL_FEATURE_BUNDLE_VERSION",
    "OPS_SAT_LENGTH_FREE_FEATURE_IDS",
    "measure_ops_sat_probe",
    "measure_ops_sat_probe_bundle",
    "measure_ops_sat_official_feature_bundle",
    "measure_ops_sat_length_free_feature_bundle",
]
