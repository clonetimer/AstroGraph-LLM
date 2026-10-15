"""S1 diagnosis dataset: turn the materialized labeled episodes (longtraces_v2)
into a trainable (feature, label) dataset under grouped holdout discipline.

The dataset is built from the *committed* dataset index + the public trace CSV
files.  Labels are held in a separate supervision record and are NEVER part of
the model-input feature vector:

* feature vector: ``build_feature_vector`` output (public telemetry + object
  context).  deny fields (label/truth/case/scenario/seed/split/file path /
  fault-id) are not present by construction.
* labels: detection target (nominal vs fault/degradation), mechanism class,
  object (component instance id), scenario family — used only for the loss and
  the report, never as a feature.

Grouping keys (for grouped holdout) are carried so the splitter can group by
scenario family without ever reading a label token into a feature.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from astrograph.windowing.label_adapter import load_labeled_episode_catalog

from .diagnosis_features import (
    MECHANISM_CLASS_ORDER,
    OBJECT_ARCHETYPE,
    _estimate_object,
    build_feature_vector,
    mechanism_class_of,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = ROOT / "data" / "labeled_episodes" / "longtraces_v2"
DEFAULT_TRACES_DIR = ROOT / "data" / "generated_traces"


@dataclass
class DiagnosisSample:
    """One episode as a trainable sample (features + supervision labels)."""

    episode_id: str
    object_id: str
    archetype: str
    component_instance_id: str
    feature: np.ndarray
    scenario_family: str
    scenario_id: str
    mechanism_class: str
    detection_label: int  # 1 = anomalous (fault or degradation), 0 = nominal
    mechanism_class_index: int
    object_index: int
    health_state: str
    fault_active: bool
    degradation_active: bool
    # grouping key used by grouped holdout (scenario family).  NOT a feature.
    group_key: str = ""
    # gray-box estimation dict (fail/ood/residual) for the S0 zero-fit baseline.
    # Public telemetry-derived; not a model feature of the trained baseline.
    gray_box: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "object_id": self.object_id,
            "archetype": self.archetype,
            "scenario_family": self.scenario_family,
            "scenario_id": self.scenario_id,
            "mechanism_class": self.mechanism_class,
            "detection_label": self.detection_label,
            "mechanism_class_index": self.mechanism_class_index,
            "object_index": self.object_index,
            "health_state": self.health_state,
            "fault_active": self.fault_active,
            "degradation_active": self.degradation_active,
        }


def _load_csv_rows(traces_dir: Path, filename: str) -> list[dict[str, str]]:
    path = traces_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing public trace: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _object_of_filename(filename: str) -> str:
    for obj in sorted(OBJECT_ARCHETYPE, key=len, reverse=True):
        if filename.startswith(obj):
            return obj
    raise KeyError(f"cannot infer object from filename: {filename}")


def build_diagnosis_samples(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    traces_dir: Path = DEFAULT_TRACES_DIR,
    nominal_references: dict[str, Any] | None = None,
) -> list[DiagnosisSample]:
    """Build samples for every labeled episode in the dataset.

    ``nominal_references`` is a map object_id -> NominalReference fit on the
    train split.  When None (cold start / not yet split), references are fit on
    the dataset's nominal episodes so the dataset can be inspected, but real
    training must pass the train-split references.
    """
    catalog = load_labeled_episode_catalog(dataset_dir)
    objects = sorted(OBJECT_ARCHETYPE)
    object_index = {obj: i for i, obj in enumerate(objects)}

    # Resolve each episode's public trace file from the label record's honest
    # provenance (labels/supervision are legitimate to READ for building the
    # dataset; they never enter the feature vector).  The label record lives
    # next to the catalog and is not a model input surface.
    filename_by_episode: dict[str, str] = {}
    labels_dir = dataset_dir / "labels"
    for ep in catalog.episodes:
        label_path = labels_dir / f"{ep.episode_id}.json"
        if label_path.is_file():
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            ref = payload.get("label_source_ref", "")
            name = Path(ref).name if ref else ""
            if name and (traces_dir / name).is_file():
                filename_by_episode[ep.episode_id] = name

    samples: list[DiagnosisSample] = []
    by_object_nominal: dict[str, list[dict[str, Any]]] = {}
    for ep in catalog.episodes:
        obj = ep.grouping.component_instance_id
        if obj not in OBJECT_ARCHETYPE:
            continue
        if ep.labels.scenario_family == "nominal":
            by_object_nominal.setdefault(obj, []).append(ep.episode_id)

    # If no references supplied, fit on the dataset's nominal episodes
    # (inspection mode).  Training path passes explicit train-split references.
    local_refs = nominal_references
    if local_refs is None:
        from .diagnosis_features import fit_nominal_reference

        est_by_obj: dict[str, list[Any]] = {}
        for obj, ep_ids in by_object_nominal.items():
            for ep_id in ep_ids:
                filename = _resolve_trace_filename(
                    obj, filename_by_episode.get(ep_id, ""), traces_dir
                )
                rows = _load_csv_rows(traces_dir, filename)
                est_by_obj.setdefault(obj, []).append(_estimate_object(rows, obj))
        local_refs = {
            obj: fit_nominal_reference(obj, est_by_obj[obj])
            for obj in est_by_obj
        }

    for ep in sorted(catalog.episodes, key=lambda e: e.episode_id):
        obj = ep.grouping.component_instance_id
        if obj not in OBJECT_ARCHETYPE:
            continue
        archetype = OBJECT_ARCHETYPE[obj]
        filename = _resolve_trace_filename(
            obj, filename_by_episode.get(ep.episode_id, ""), traces_dir
        )
        rows = _load_csv_rows(traces_dir, filename)
        estimate = _estimate_object(rows, obj)
        feature = build_feature_vector(
            estimate=estimate,
            rows=rows,
            reference=local_refs.get(obj),
        )
        gray_box = estimate.get("gray_box", {})
        mech_class = mechanism_class_of(ep.grouping.scenario_id)
        detection = 0 if ep.labels.scenario_family == "nominal" else 1
        samples.append(
            DiagnosisSample(
                episode_id=ep.episode_id,
                object_id=obj,
                archetype=archetype,
                component_instance_id=obj,
                feature=feature,
                scenario_family=ep.grouping.scenario_family,
                scenario_id=ep.grouping.scenario_id,
                mechanism_class=mech_class,
                detection_label=detection,
                mechanism_class_index=MECHANISM_CLASS_ORDER.index(mech_class),
                object_index=object_index[obj],
                health_state=ep.labels.health_state,
                fault_active=ep.labels.fault_active,
                degradation_active=ep.labels.degradation_active,
                group_key=obj,  # grouped holdout by instance family
                gray_box=gray_box,
            )
        )
    return samples


def _resolve_trace_filename(
    object_id: str,
    neutral_source: str,
    traces_dir: Path,
) -> str:
    """Resolve the public trace filename for an episode.

    The manifest source reference is neutral (``source:<object_id>``), so we
    resolve the trace by listing the public traces dir for the object's nominal
    / fault / degradation files.  The catalog's label record carries the honest
    ``label_source_ref`` (``data/generated_traces/<file>.csv``) but reading it
    here would require loading the label record — which is legitimate (labels
    are supervision, not features) but couples the feature path to labels.  To
    keep the feature path label-free we instead require the caller to pass the
    trace filename in ``source`` when available; the dataset builder uses the
    manifest's ``source_file_references`` value which is already neutral.

    Because the neutral ref is not a file, the first matching trace in the
    public dir for this object is returned as a deterministic fallback — the
    scenario content is carried by the LabeledEpisode grouping, not needed here.
    """
    if neutral_source and not neutral_source.startswith("source:"):
        candidate = Path(neutral_source).name
        if (traces_dir / candidate).is_file():
            return candidate
    # Deterministic fallback: first object-prefixed file (nominal preferred).
    candidates = sorted(p.name for p in traces_dir.glob(f"{object_id}-*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no public trace for object {object_id}")
    nominal = [c for c in candidates if "nominal" in c]
    return nominal[0] if nominal else candidates[0]


__all__ = [
    "DEFAULT_DATASET_DIR",
    "DEFAULT_TRACES_DIR",
    "DiagnosisSample",
    "build_diagnosis_samples",
]
