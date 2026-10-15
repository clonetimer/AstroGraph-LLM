from __future__ import annotations

from typing import Any

from astrograph.anomaly.paired_residual import DetectionRecord


def build_observability_gap(run_id: str, case_id: str, records: list[DetectionRecord], *, top_n: int = 5) -> dict[str, Any]:
    detected = [item for item in records if item.detected]
    ranked = sorted(
        records,
        key=lambda item: (-(item.peak_residual_abs / max(item.threshold_abs, 1e-15)), item.channel_id),
    )
    top_channels = [
        {
            "channel_id": item.channel_id,
            "anomaly_id": item.anomaly_id,
            "peak_residual_abs": item.peak_residual_abs,
            "threshold_abs": item.threshold_abs,
            "threshold_ratio": item.peak_residual_abs / max(item.threshold_abs, 1e-15),
            "expected_direction": item.expected_direction,
            "direction_matched": item.direction_matched,
            "semantic_role": item.semantic_role,
        }
        for item in ranked[:top_n]
    ]
    if detected:
        status = "OBSERVABLE"
        recommendation = "No observability gap: at least one configured anomaly rule was satisfied."
    elif records:
        status = "BELOW_CONFIGURED_THRESHOLDS"
        recommendation = (
            "Keep diagnosis UNKNOWN. Review direct root-cause channels, sensor resolution, paired baseline, "
            "and training-only threshold calibration; do not lower thresholds ad hoc."
        )
    else:
        status = "NO_CONFIGURED_CHANNEL"
        recommendation = "Add a reviewed channel-to-anomaly rule for this fault and operating condition."
    return {
        "schema_version": "astrograph-observability-gap.v1",
        "run_id": run_id,
        "case_id": case_id,
        "status": status,
        "configured_channel_count": len(records),
        "detected_channel_count": len(detected),
        "top_channels": top_channels,
        "recommendation": recommendation,
        "claim_boundary": "Synthetic engineering observability review; not a sensor qualification result.",
    }
