"""Independent label adapter for labeled episodes (G2 data-loop work item).

Labels — health state, fault ids, degradation flags, scenario family — are the
supervision signal that data-driven diagnosis consumes *after* the model-input
path.  Per ``DEV_GOAL.md`` and ``EPISODE_MANIFEST`` design:

* Labels are injected through this adapter ONLY.  They never enter an
  ``EpisodeManifest`` record, whose ``labels_used`` / ``truth_used`` fields are
  ``False`` and frozen.
* An ``EpisodeManifest`` is built from the *public* view of a Sat-Agent trace
  (only ``time_s`` + channel-map bound fields, via ``episode_builder``).  The
  label adapter reads the *same* source trace and produces a separate
  ``LabeledEpisode`` record linked by ``episode_id``.
* No global resampling: each component keeps its native sampling rate; the
  manifest and the label record only reference descriptive summary fields
  (time span, row count, native period) — never signal values.

S0 constraint: this module performs no fitting, no calibration and creates no
formal train/validation split or seed.  It only materialises (manifest,
labeled) pairs so that grouped holdout can be defined later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from astrograph.assurance import ArtifactRoot
from astrograph.windowing.episode_builder import build_episodes_from_profile_csvs
from astrograph.windowing.episode_models import EpisodeManifest

__all__ = [
    "EpisodeLabels",
    "EpisodeGrouping",
    "LabeledEpisode",
    "LabeledEpisodeCatalog",
    "extract_labels_from_trace",
    "build_labeled_episode",
    "materialize_labeled_episodes",
    "load_labeled_episode_catalog",
    "load_episode_manifest",
]

# Severity ordering used to aggregate a run-wide health state from the
# per-row ``label.health_state`` column.  Ordered least -> most severe.
_HEALTH_SEVERITY: dict[str, int] = {
    "nominal": 0,
    "healthy": 0,
    "degraded_unverified": 1,
    "degraded": 2,
    "fault": 3,
    "faulted": 3,
    "critical": 4,
}

# Deterministic iteration order for scenario families.
SCENARIO_FAMILIES: tuple[str, ...] = ("nominal", "fault", "degradation")

_DEFAULT_SCENARIO_HEALTH: dict[str, str] = {
    "nominal": "nominal",
    "fault": "fault",
    "degradation": "degraded",
}


class EpisodeLabels(BaseModel):
    """Supervision labels for one episode.

    This record is the ONLY place health state / fault ids / degradation may
    appear.  It is physically separated from ``EpisodeManifest`` and is never a
    model input.  ``model_config = extra=forbid`` keeps the record closed.
    """

    model_config = ConfigDict(extra="forbid")

    label_source: str = Field(
        description='"trace_columns" when label.* columns were read from the '
        "source trace; 'scenario_declared' when the trace had no label columns "
        "and only the file-declared scenario family was available.",
    )
    scenario_family: str = Field(description="nominal | fault | degradation")
    health_state: str = Field(
        description="Aggregate health state (max severity seen across the run).",
    )
    health_states_seen: list[str] = Field(
        default_factory=list,
        description="Distinct label.health_state values present in the run.",
    )
    fault_active: bool = False
    degradation_active: bool = False
    modifier_active: bool = False
    constraint_active: bool | None = Field(
        default=None,
        description="True when label.constraint_active fired; None if the column is absent.",
    )
    fault_ids: list[str] = Field(
        default_factory=list,
        description="Fault/degradation instance ids from modifier.active_ids (never model input).",
    )
    label_columns_present: list[str] = Field(default_factory=list)
    signal_row_count: int = Field(default=0, ge=0)


class EpisodeGrouping(BaseModel):
    """Grouping keys for future grouped holdout.

    Every field is provenance / label metadata, never a model input.  An
    episode carries these so the dataset can be grouped by instance, by
    configuration family, or by scenario family before any split is built
    (the split itself is an S1-authorized step).
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = "SAT_AGENT_PLATFORM"
    asset_instance_id: str = "asset-under-test"
    component_instance_id: str
    component_archetype_id: str
    config_id: str = Field(
        description="Simulator capability/config family, e.g. component.imu.v1.",
    )
    scenario_family: str = Field(description="nominal | fault | degradation")
    scenario_id: str = Field(
        description="Sat-Agent scenario/task id, e.g. component_imu_v1_bias_degradation.",
    )
    seed: int | None = Field(
        default=None,
        description="Source simulation seed (provenance only; never a model input).",
    )
    operating_condition: str | None = Field(
        default=None,
        description="Unavailable (None) in this dataset: Sat-Agent synthetic traces "
        "carry no dedicated operating-condition column. Not fabricated.",
    )


class LabeledEpisode(BaseModel):
    """A labeled episode: ``episode_id`` + public summary + labels.

    The public summary (time span, row count, native period, channel count) is
    descriptive only and is NOT a model input — no signal values are stored.
    Labels live in ``labels`` and reference the manifest only through
    ``episode_id``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-labeled-episode.v1"
    episode_id: str
    source_id: str = "SAT_AGENT_PLATFORM"

    # Public summary (non-inference): descriptive statistics only.
    start_s: float
    end_s: float
    row_count: int = Field(default=0, ge=0)
    native_sample_period_s: float | None = None
    public_signal_channel_count: int = Field(default=0, ge=0)

    grouping: EpisodeGrouping
    labels: EpisodeLabels
    label_source_ref: str = Field(
        description="Provenance: source trace path the labels were extracted from.",
    )


class LabeledEpisodeCatalog(BaseModel):
    """A collection of labeled episodes — the materialized dataset index."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-labeled-episode-catalog.v1"
    catalog_id: str
    source_id: str = "SAT_AGENT_PLATFORM"
    source_count: int = 1
    generated_at: str | None = Field(
        default=None,
        description="Left unset for byte-deterministic replay; set only by callers that accept non-determinism.",
    )
    grouping_dimensions: list[str] = Field(
        default_factory=lambda: [
            "source_id",
            "asset_instance_id",
            "component_instance_id",
            "component_archetype_id",
            "config_id",
            "scenario_family",
            "scenario_id",
        ],
    )
    claim_boundary: str = (
        "Labels are supervision signal held in LabeledEpisode records only. "
        "EpisodeManifest records in this dataset never contain labels "
        "(labels_used/truth_used stay False and are frozen). LabeledEpisode is "
        "NOT a model input; the split/seed that turns this dataset into a "
        "train/eval setup is an S1-authorized step."
    )
    episodes: list[LabeledEpisode] = Field(default_factory=list)


def _scenario_episode_id(profile_id: str, scenario_family: str, start_s: float, end_s: float) -> str:
    """Deterministic, opaque episode id that distinguishes scenario families.

    The scenario family is mixed into the hashed payload so that the
    nominal / fault / degradation variants of the same object+time-span get
    distinct episode ids, while the emitted id stays opaque (a SHA-256 prefix;
    no label string is embedded).
    """
    payload = f"{profile_id}|{scenario_family}|{start_s:.9f}|{end_s:.9f}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _any_true(trace: pd.DataFrame, column: str) -> bool | None:
    """Return True if a boolean-ish label column ever fires; None if absent."""
    if column not in trace.columns:
        return None
    series = trace[column].dropna()
    if len(series) == 0:
        return None
    lowered = series.astype(str).str.strip().str.lower()
    return bool(lowered.isin({"true", "1", "yes"}).any())


def _aggregate_health_state(trace: pd.DataFrame, scenario_family: str) -> tuple[str, list[str]]:
    """Return (aggregate health state, distinct health states seen)."""
    if "label.health_state" in trace.columns:
        seen = sorted(
            {
                str(v).strip()
                for v in trace["label.health_state"].dropna().tolist()
                if str(v).strip()
            }
        )
        if seen:
            agg = max(seen, key=lambda h: _HEALTH_SEVERITY.get(h, -1))
            return agg, seen
    # No usable label column: fall back to the declared scenario family.  This
    # is the file-declared run mode, not a fabricated label.
    return _DEFAULT_SCENARIO_HEALTH.get(scenario_family, scenario_family), []


def extract_labels_from_trace(
    trace: pd.DataFrame,
    *,
    scenario_family: str,
    scenario_id: str,
    label_source_ref: str,
) -> EpisodeLabels:
    """Extract supervision labels from a Sat-Agent source trace.

    Reads only the ``label.*`` / ``modifier.*`` columns.  Signal columns are
    ignored here; they are owned by the public episode path.  Missing boolean
    label columns are not fabricated — when a column is absent the value falls
    back to the declared scenario family (``scenario_family``), which is real
    file-declared provenance.
    """
    label_columns = sorted(
        c for c in trace.columns if c.startswith("label.") or c.startswith("modifier.")
    )

    fault_active = _any_true(trace, "label.fault_active")
    if fault_active is None:
        fault_active = scenario_family == "fault"
    degradation_active = _any_true(trace, "label.degradation_active")
    if degradation_active is None:
        degradation_active = scenario_family == "degradation"
    modifier_active = _any_true(trace, "label.modifier_active")
    if modifier_active is None:
        modifier_active = bool(fault_active or degradation_active)
    constraint_active = _any_true(trace, "label.constraint_active")

    fault_ids: list[str] = []
    if "modifier.active_ids" in trace.columns:
        seen: set[str] = set()
        for raw in trace["modifier.active_ids"].dropna().astype(str).tolist():
            for token in str(raw).split(","):
                token = token.strip()
                if token and token.lower() != "nan":
                    seen.add(token)
        fault_ids = sorted(seen)

    health_state, health_states_seen = _aggregate_health_state(trace, scenario_family)
    label_source = (
        "trace_columns"
        if any(c.startswith("label.") for c in label_columns)
        else "scenario_declared"
    )

    return EpisodeLabels(
        label_source=label_source,
        scenario_family=scenario_family,
        health_state=health_state,
        health_states_seen=health_states_seen,
        fault_active=bool(fault_active),
        degradation_active=bool(degradation_active),
        modifier_active=bool(modifier_active),
        constraint_active=constraint_active,
        fault_ids=fault_ids,
        label_columns_present=label_columns,
        signal_row_count=int(len(trace)),
    )


def grouping_from_manifest(
    manifest: EpisodeManifest,
    *,
    scenario_family: str,
    scenario_id: str,
    config_id: str,
    seed: int | None,
) -> EpisodeGrouping:
    """Derive grouping keys from the public manifest + declared scenario."""
    comp = manifest.components[0] if manifest.components else None
    return EpisodeGrouping(
        source_id=manifest.source_id,
        asset_instance_id=manifest.asset_instance_id,
        component_instance_id=comp.component_instance_id if comp else "unknown",
        component_archetype_id=comp.component_archetype_id if comp else "unknown",
        config_id=config_id,
        scenario_family=scenario_family,
        scenario_id=scenario_id,
        seed=seed,
        operating_condition=None,  # no dedicated column; never fabricated (see model doc)
    )


def build_labeled_episode(
    *,
    manifest: EpisodeManifest,
    trace: pd.DataFrame,
    scenario_family: str,
    scenario_id: str,
    label_source_ref: str,
    config_id: str | None = None,
    seed: int | None = None,
) -> LabeledEpisode:
    """Pair an EpisodeManifest with labels extracted from the source trace.

    The returned ``LabeledEpisode`` shares ``episode_id`` with ``manifest`` and
    carries grouping + labels.  Nothing in this function writes a label into
    the manifest.
    """
    labels = extract_labels_from_trace(
        trace,
        scenario_family=scenario_family,
        scenario_id=scenario_id,
        label_source_ref=label_source_ref,
    )
    grouping = grouping_from_manifest(
        manifest,
        scenario_family=scenario_family,
        scenario_id=scenario_id,
        config_id=config_id or manifest.episode_id,
        seed=seed,
    )

    comp = manifest.components[0] if manifest.components else None
    row_count = int(sum(c.row_count for c in manifest.components))
    native_period: float | None = None
    if comp is not None and comp.data_quality is not None and comp.data_quality.time_quality is not None:
        native_period = comp.data_quality.time_quality.median_period_s
    channel_count = len(comp.semantic_channel_ids) if comp is not None else 0

    return LabeledEpisode(
        episode_id=manifest.episode_id,
        source_id=manifest.source_id,
        start_s=float(manifest.start_s),
        end_s=float(manifest.end_s),
        row_count=row_count,
        native_sample_period_s=native_period,
        public_signal_channel_count=int(channel_count),
        grouping=grouping,
        labels=labels,
        label_source_ref=label_source_ref,
    )


# --- Provenance readers (read-only; never modify the source tree) ---


def _read_scenario_id(trace: pd.DataFrame, source_path: Path, profile_id: str, scenario_family: str) -> str:
    if "task_id" in trace.columns:
        first = trace["task_id"].dropna().astype(str)
        if len(first) and first.iloc[0].strip():
            return first.iloc[0].strip()
    return f"{profile_id}:{scenario_family}"


def _read_seed(source_path: Path) -> int | None:
    manifest_path = source_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = payload.get("task", {}).get("seed")
        return int(seed) if isinstance(seed, int) else None
    except (OSError, ValueError, TypeError):
        return None


def _read_config_id(profile_id: str, profile_path: Path) -> str:
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        for profile in payload.get("profiles", []):
            if profile.get("profile_id") == profile_id:
                return profile.get("capability_id") or profile_id
    except (OSError, ValueError):
        pass
    return profile_id


# --- Materialization ---


def materialize_labeled_episodes(
    *,
    profile_sources: dict[str, dict[str, str]],
    profile_path: str | Path,
    channel_map_catalog_path: str | Path,
    semantic_schema_path: str | Path,
    output_dir: str | Path,
    sat_root: str | Path | None = None,
    source_id: str = "SAT_AGENT_PLATFORM",
    catalog_id: str = "sat-agent-labeled-episodes",
) -> LabeledEpisodeCatalog:
    """Materialize (EpisodeManifest, LabeledEpisode) pairs.

    ``profile_sources`` maps ``profile_id -> {scenario_family -> source path}``.
    Paths are resolved against ``sat_root`` when given (the Sat-Agent platform,
    opened read-only).  The public manifest is built by ``episode_builder``,
    which selects ONLY ``time_s`` + channel-map bound fields, so the manifest
    stays label-free even though the same source trace carries label columns.
    Labels are extracted through ``extract_labels_from_trace`` into a separate
    ``LabeledEpisode`` record.

    Output layout under ``output_dir``::

        catalog.json             # LabeledEpisodeCatalog (the dataset index)
        manifests/<episode_id>.json   # EpisodeManifest (label-free)
        labels/<episode_id>.json      # LabeledEpisode (labels + grouping)

    The result is deterministic: iteration is sorted, ids are hashes, and JSON
    is written with stable field order and no timestamps.
    """
    root = Path(sat_root).expanduser().resolve() if sat_root is not None else None
    out = Path(output_dir)
    project_root = Path(__file__).resolve().parents[3]
    artifacts = ArtifactRoot(out, source_root=project_root)
    artifacts.ensure()

    episodes: list[LabeledEpisode] = []

    for profile_id in sorted(profile_sources):
        scenario_map = profile_sources[profile_id]
        config_id = _read_config_id(profile_id, Path(profile_path))
        for scenario_family in sorted(scenario_map):
            rel = scenario_map[scenario_family]
            source_path = Path(rel)
            if root is not None:
                source_path = root / rel
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Labeled-episode source missing: {source_path}"
                )

            trace = pd.read_csv(source_path, encoding="utf-8-sig")
            catalog = build_episodes_from_profile_csvs(
                profile_file_map={profile_id: str(source_path)},
                profile_path=str(profile_path),
                channel_map_catalog_path=str(channel_map_catalog_path),
                semantic_schema_path=str(semantic_schema_path),
                source_id=source_id,
                catalog_id=f"{catalog_id}:{profile_id}:{scenario_family}",
            )
            manifest = catalog.episodes[0]

            episode_id = _scenario_episode_id(
                profile_id, scenario_family, float(manifest.start_s), float(manifest.end_s)
            )
            # Portability: keep the scenario-bearing path only for the label
            # record (which is NOT a model input).  The manifest's
            # source_file_references must stay NEUTRAL: the raw source path
            # contains scenario tokens (e.g. .../component_mtb__fault__alias/
            # runs/trace.csv) which the deny rules forbid in any model-input
            # surface.  Map it to a scenario-free `source:<profile_id>` id.
            ref = str(rel) if root is not None else str(source_path)
            # Re-emit the manifest under the scenario-distinguishing opaque id
            # (a hash — no scenario token leaks) and a neutral source ref.
            manifest = manifest.model_copy(
                update={
                    "episode_id": episode_id,
                    "source_file_references": {profile_id: f"source:{profile_id}"},
                }
            )

            scenario_id = _read_scenario_id(trace, source_path, profile_id, scenario_family)
            seed = _read_seed(source_path)

            labeled = build_labeled_episode(
                manifest=manifest,
                trace=trace,
                scenario_family=scenario_family,
                scenario_id=scenario_id,
                label_source_ref=ref,
                config_id=config_id,
                seed=seed,
            )
            episodes.append(labeled)

            artifacts.write_text(
                f"manifests/{episode_id}.json",
                manifest.model_dump_json(indent=2) + "\n",
            )
            artifacts.write_text(
                f"labels/{episode_id}.json",
                labeled.model_dump_json(indent=2) + "\n",
            )

    catalog = LabeledEpisodeCatalog(
        catalog_id=catalog_id,
        source_id=source_id,
        episodes=episodes,
    )
    artifacts.write_text(
        "catalog.json",
        catalog.model_dump_json(indent=2) + "\n",
    )
    return catalog


def load_labeled_episode_catalog(dataset_dir: str | Path) -> LabeledEpisodeCatalog:
    """Load a materialized dataset from its ``catalog.json``."""
    path = Path(dataset_dir) / "catalog.json"
    return LabeledEpisodeCatalog.model_validate_json(path.read_text(encoding="utf-8"))


def load_episode_manifest(dataset_dir: str | Path, episode_id: str) -> EpisodeManifest:
    """Load a label-free EpisodeManifest from a materialized dataset."""
    path = Path(dataset_dir) / "manifests" / f"{episode_id}.json"
    return EpisodeManifest.model_validate_json(path.read_text(encoding="utf-8"))
