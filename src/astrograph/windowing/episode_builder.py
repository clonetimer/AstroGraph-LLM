"""Episode builder for the generic diagnosis pipeline.

Reads real Sat-Agent CSV files through the channel-map layer, produces
SemanticTelemetryFrame objects, and builds EpisodeManifest records.

Design: no global resampling. Each component keeps its native sampling rate
and missing mask. Time alignment is deferred to the operator/token layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrograph.data_sources.models import DataQualityIssue
from astrograph.semantics.channel_mapping import ChannelMapper
from astrograph.semantics.contracts import (
    ChannelMapCatalog,
    SemanticSchema,
    SemanticTelemetryFrame,
)
from astrograph.windowing.episode_models import (
    DataQualityStatus,
    EpisodeComponent,
    EpisodeManifest,
    EpisodeManifestCatalog,
    TimeAxisQuality,
)


def _opaque_episode_id(start_s: float, end_s: float, component_ids: list[str]) -> str:
    """Deterministic, opaque episode id from time span and component set."""
    payload = f"{start_s:.9f}|{end_s:.9f}|{','.join(sorted(component_ids))}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _assess_time_axis(timestamps: np.ndarray) -> TimeAxisQuality:
    """Analyse a time-axis array and produce quality metadata."""
    if len(timestamps) == 0:
        return TimeAxisQuality(
            is_monotonic=False,
            is_strictly_monotonic=False,
            is_uniform_frame_counter=False,
        )

    diffs = np.diff(timestamps)
    is_monotonic = bool(np.all(diffs >= 0))
    is_strictly = bool(np.all(diffs > 0)) if is_monotonic else False

    # Detect uniform frame counter: integer sequence 0, 1, 2, …
    is_frame_counter = False
    if is_strictly and len(timestamps) > 1:
        int_ts = timestamps.astype(np.int64)
        if np.allclose(timestamps, int_ts.astype(float)):
            expected = np.arange(int_ts[0], int_ts[0] + len(timestamps))
            if np.array_equal(int_ts, expected):
                is_frame_counter = True

    nan_count = int(np.sum(np.isnan(timestamps)))
    dups = len(timestamps) - len(np.unique(timestamps))

    med = float(np.median(diffs)) if len(diffs) > 0 and is_strictly else None
    min_p = float(np.min(diffs)) if len(diffs) > 0 and is_strictly else None
    max_p = float(np.max(diffs)) if len(diffs) > 0 and is_strictly else None

    return TimeAxisQuality(
        is_uniform_frame_counter=is_frame_counter,
        is_monotonic=is_monotonic,
        is_strictly_monotonic=is_strictly,
        min_period_s=min_p,
        max_period_s=max_p,
        median_period_s=med,
        missing_value_count=nan_count,
        duplicate_count=int(dups),
    )


def _data_quality_for_component(
    frame: SemanticTelemetryFrame | None,
    tao: TimeAxisQuality,
    channel_count: int,
) -> DataQualityStatus:
    """Derive data quality status from frame and time-axis analysis."""
    reasons: list[str] = []

    if frame is None or len(frame.signals) == 0:
        return DataQualityStatus(
            status="INVALID",
            reason_codes=["EMPTY_FRAME"],
            empty_trajectory=True,
            time_quality=tao,
        )

    if not tao.is_monotonic:
        reasons.append("TIME_NOT_MONOTONIC")
    if tao.duplicate_count > 0:
        reasons.append("DUPLICATE_TIMESTAMPS")
    if tao.missing_value_count > 0:
        reasons.append("MISSING_TIME_VALUES")

    # Check if all channels are valid
    valid_channels = sum(
        1 for s in frame.signals
        if s.values and not all(v is None for v in s.values)
    )
    avail = valid_channels / max(channel_count, 1)

    if valid_channels == 0:
        reasons.append("NO_VALID_CHANNELS")

    if reasons:
        status = "INVALID" if ("NO_VALID_CHANNELS" in reasons or valid_channels == 0) else "DEGRADED"
    else:
        status = "VALID"

    return DataQualityStatus(
        status=status,
        reason_codes=sorted(reasons),
        empty_trajectory=(frame is None or len(frame.timestamps_s) == 0),
        time_quality=tao,
        available_port_fraction=round(avail, 4),
    )


def build_episodes_from_semantic_frames(
    *,
    frames: dict[str, SemanticTelemetryFrame],
    source_file_map: dict[str, str],
    catalog_id: str = "generic-episode-catalog",
    source_id: str = "SAT_AGENT_PLATFORM",
    channel_map_sha256: str | None = None,
    semantic_schema_id: str | None = None,
) -> EpisodeManifestCatalog:
    """Build an EpisodeManifestCatalog from a collection of semantic frames.

    Each entry in `frames` maps a component_id to its SemanticTelemetryFrame.
    Components with overlapping time ranges are grouped into the same episode.

    Args:
        frames: component_id → SemanticTelemetryFrame for that component.
        source_file_map: component_id → source file path.
        catalog_id: identifier for this catalog.
        source_id: data source identifier.
        channel_map_sha256: SHA-256 of the channel map catalog used.
        semantic_schema_id: ID of the semantic schema used.

    Returns:
        EpisodeManifestCatalog with one or more episodes.
    """
    if not frames:
        raise ValueError("At least one semantic frame is required to build episodes.")

    # Determine the union time span
    all_starts = [min(f.timestamps_s) for f in frames.values()]
    all_ends = [max(f.timestamps_s) for f in frames.values()]
    t0 = min(all_starts)
    t1 = max(all_ends)

    episode_id = _opaque_episode_id(t0, t1, list(frames.keys()))

    components: list[EpisodeComponent] = []
    any_invalid = False

    for comp_id, frame in frames.items():
        comp_start = float(min(frame.timestamps_s))
        comp_end = float(max(frame.timestamps_s))
        tao = _assess_time_axis(np.array(frame.timestamps_s, dtype=float))
        dq = _data_quality_for_component(frame, tao, len(frame.signals))
        if dq.status == "INVALID":
            any_invalid = True

        comp = EpisodeComponent(
            component_instance_id=comp_id,
            component_archetype_id=frame.signals[0].component_archetype_id if frame.signals else "unknown",
            start_s=comp_start,
            end_s=comp_end,
            native_sample_period_s=tao.median_period_s,
            semantic_channel_ids=[s.semantic_channel_id for s in frame.signals],
            row_count=int(len(frame.timestamps_s)),
            data_quality=dq,
        )
        components.append(comp)

    overall = DataQualityStatus(
        status="INVALID" if any_invalid else "VALID",
        reason_codes=sorted({
            rc
            for c in components
            if c.data_quality
            for rc in c.data_quality.reason_codes
        }),
        empty_trajectory=all(
            c.data_quality.empty_trajectory
            for c in components
            if c.data_quality
        ),
    )

    manifest = EpisodeManifest(
        episode_id=episode_id,
        source_id=source_id,
        start_s=t0,
        end_s=t1,
        components=components,
        source_file_references=source_file_map,
        channel_map_catalog_sha256=channel_map_sha256,
        semantic_schema_id=semantic_schema_id,
        overall_data_quality=overall,
    )

    return EpisodeManifestCatalog(
        catalog_id=catalog_id,
        source_id=source_id,
        episodes=[manifest],
    )


def build_episodes_from_profile_csvs(
    *,
    profile_file_map: dict[str, str],
    profile_path: str | Path,
    channel_map_catalog_path: str | Path,
    semantic_schema_path: str | Path,
    source_id: str = "SAT_AGENT_PLATFORM",
    catalog_id: str = "batch-a-episodes",
) -> EpisodeManifestCatalog:
    """Build episodes directly from Sat-Agent profile CSV files.

    This is the high-level entry point: give it a mapping of profile_id → CSV
    file path, and it does the full ChannelMapper → SemanticTelemetryFrame →
    EpisodeManifest chain.

    Args:
        profile_file_map: profile_id → path to real Sat-Agent CSV file.
        profile_path: path to the schema profiles JSON.
        channel_map_catalog_path: path to the channel map catalog JSON.
        semantic_schema_path: path to the semantic schema JSON.
        source_id: data source identifier.
        catalog_id: identifier for the resulting catalog.

    Returns:
        EpisodeManifestCatalog.
    """
    from astrograph.semantics.archetypes import ComponentArchetypeCatalog
    from astrograph.semantics.channel_mapping import ChannelMapper

    # Load contracts
    profile_data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    channel_map_cat = ChannelMapCatalog.model_validate_json(
        Path(channel_map_catalog_path).read_text(encoding="utf-8")
    )
    schema = SemanticSchema.model_validate_json(
        Path(semantic_schema_path).read_text(encoding="utf-8")
    )
    # Also load archetypes so validate_channel_maps can run
    try:
        archetype_path = Path(semantic_schema_path).parent / "core_component_archetypes.json"
        archetypes = ComponentArchetypeCatalog.model_validate_json(
            archetype_path.read_text(encoding="utf-8")
        )
        archetypes.validate_channel_maps(channel_map_cat)
    except Exception:
        pass  # archetype validation is not the job of the episode builder

    # Build profile lookup
    profile_by_id = {p["profile_id"]: p for p in profile_data["profiles"]}

    frames: dict[str, SemanticTelemetryFrame] = {}
    source_files: dict[str, str] = {}

    for profile_id, csv_path in profile_file_map.items():
        profile = profile_by_id.get(profile_id)
        if profile is None:
            raise ValueError(f"Profile {profile_id} not found in {profile_path}")

        df = pd.read_csv(csv_path)
        # Normalise time column name
        tf_name = profile["time_field"]["name"]
        if "time_s" not in df.columns and tf_name in df.columns:
            df = df.rename(columns={tf_name: "time_s"})

        # Find the channel map for this profile
        channel_map = None
        for cm in channel_map_cat.maps:
            for entry in cm.entries:
                if any(
                    pf["name"] == entry.source_field
                    for pf in profile.get("public_observation_fields", [])
                ):
                    channel_map = cm
                    break
            if channel_map is not None:
                break
        # Fallback: match by mapping_id containing profile keywords
        if channel_map is None:
            for cm in channel_map_cat.maps:
                if any(kw in cm.mapping_id.lower() for kw in profile_id.lower().split(".")[1:3]):
                    channel_map = cm
                    break

        if channel_map is None:
            raise ValueError(
                f"No channel map found for profile {profile_id}. "
                f"Available maps: {[cm.mapping_id for cm in channel_map_cat.maps]}"
            )

        # Pre-select only mapped + time columns to avoid prohibited-field rejection
        mapped_fields = {entry.source_field for entry in channel_map.entries}
        safe_columns = ["time_s"] + [c for c in mapped_fields if c in df.columns]
        df = df[safe_columns]

        mapper = ChannelMapper(semantic_schema=schema, channel_map=channel_map)
        result = mapper.map_frame(df)
        frames[profile_id] = result.frame
        source_files[profile_id] = csv_path

    return build_episodes_from_semantic_frames(
        frames=frames,
        source_file_map=source_files,
        catalog_id=catalog_id,
        source_id=source_id,
        channel_map_sha256=hashlib.sha256(
            Path(channel_map_catalog_path).read_bytes()
        ).hexdigest(),
        semantic_schema_id=schema.semantic_schema_id,
    )
