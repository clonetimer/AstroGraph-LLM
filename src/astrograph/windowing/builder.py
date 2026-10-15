from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrograph.conditions import ConditionCatalog, OperatingConditionResolver
from astrograph.assurance.artifact_paths import ArtifactRoot

from astrograph.data.raw_adapter import D0DatasetAdapter, STREAM_FILES
from astrograph.windowing.models import OperatingCondition, StreamSliceReference, TelemetryWindow, WindowLabels


FAULT_LABEL_TO_ID = {
    "label.eps_capacity_loss_active": "FAULT.EPS.BATTERY.CAPACITY_LOSS",
    "label.solar_efficiency_loss_active": "FAULT.EPS.SOLAR_ARRAY.EFFICIENCY_LOSS",
    "label.thermal_radiator_loss_active": "FAULT.THERMAL.RADIATOR.REJECTION_LOSS",
    "label.adcs_fault_affects_pointing": "FAULT.ADCS.RW.JAM",
    "label.comm_link_loss_active": "FAULT.COMM.DOWNLINK.LOSS",
}


def _opaque_window_id(dataset_id: str, run_id: str, start_s: float, end_s: float, version: str) -> str:
    payload = f"{dataset_id}|{run_id}|{start_s:.9f}|{end_s:.9f}|{version}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def default_condition_resolver() -> OperatingConditionResolver:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "operating_conditions.json"
    return OperatingConditionResolver(ConditionCatalog.from_json(config_path))


def infer_operating_condition(
    legacy: pd.DataFrame,
    adcs: pd.DataFrame,
    *,
    resolver: OperatingConditionResolver | None = None,
) -> OperatingCondition:
    return (resolver or default_condition_resolver()).resolve(legacy, adcs)


def _window_labels(joined_legacy: pd.DataFrame) -> WindowLabels:
    label_columns = [column for column in FAULT_LABEL_TO_ID if column in joined_legacy.columns]
    fault_ids = [
        fault_id
        for column, fault_id in FAULT_LABEL_TO_ID.items()
        if column in joined_legacy.columns
        and joined_legacy[column].astype(str).str.lower().isin({"true", "1"}).any()
    ]
    fault_active = bool(joined_legacy["label.fault_active"].astype(str).str.lower().isin({"true", "1"}).any())
    degradation_active = bool(joined_legacy["label.degradation_active"].astype(str).str.lower().isin({"true", "1"}).any())
    health_values = joined_legacy["label.health_state"].astype(str).value_counts()
    health_state = str(health_values.index[0]) if not health_values.empty else "unknown"
    if fault_active and not fault_ids and label_columns:
        fault_ids = ["FAULT.UNSPECIFIED"]
    return WindowLabels(
        health_state=health_state,
        fault_active=fault_active,
        degradation_active=degradation_active,
        fault_ids=sorted(fault_ids),
    )


def build_window_index(
    adapter: D0DatasetAdapter,
    output_path: str | Path,
    *,
    window_s: float = 60.0,
    stride_s: float = 30.0,
    include_splits: set[str] | None = None,
    condition_resolver: OperatingConditionResolver | None = None,
) -> dict:
    if window_s <= 0 or stride_s <= 0:
        raise ValueError("window_s and stride_s must be positive")
    output = Path(output_path)
    project_root = Path(__file__).resolve().parents[3]
    artifacts = ArtifactRoot(output.parent, source_root=project_root)
    resolver = condition_resolver or default_condition_resolver()
    count = 0
    counts_by_split: dict[str, int] = {}
    counts_by_case: dict[str, int] = {}
    counts_by_condition: dict[str, int] = {}

    def _write_index(temporary: Path) -> None:
        nonlocal count
        with temporary.open("w", encoding="utf-8") as handle:
            for run in adapter.iter_runs():
                if include_splits and run.split not in include_splits:
                    continue
                joined_legacy = adapter.load_joined_stream(run.run_id, "legacy")
                stream_frames = {stream: adapter.load_stream(run.run_id, stream) for stream in STREAM_FILES}
                stream_times = {
                    stream: frame["time_s"].to_numpy(dtype=float)
                    for stream, frame in stream_frames.items()
                }
                stream_periods = {
                    stream: (float(pd.Series(times).diff().dropna().median()) if len(times) > 1 else 0.0)
                    for stream, times in stream_times.items()
                }
                stream_channels = {
                    stream: [column for column in frame.columns if column not in {"record_id", "run_id", "time_s"}]
                    for stream, frame in stream_frames.items()
                }
                legacy_join_times = joined_legacy["time_s"].to_numpy(dtype=float)
                duration_s = float(
                    adapter.run_manifest(run.run_id).get(
                        "duration_s", adapter.manifest.get("duration_s", 0.0)
                    )
                )
                start = 0.0
                while start + window_s <= duration_s + 1e-9:
                    end = start + window_s
                    slices: dict[str, pd.DataFrame] = {}
                    for stream, frame in stream_frames.items():
                        times = stream_times[stream]
                        left = int(np.searchsorted(times, start, side="left"))
                        right = int(np.searchsorted(times, end, side="left"))
                        slices[stream] = frame.iloc[left:right]
                    legacy_left = int(np.searchsorted(legacy_join_times, start, side="left"))
                    legacy_right = int(np.searchsorted(legacy_join_times, end, side="left"))
                    legacy_join = joined_legacy.iloc[legacy_left:legacy_right]
                    if legacy_join.empty:
                        start += stride_s
                        continue
                    refs: list[StreamSliceReference] = []
                    for stream, frame in slices.items():
                        refs.append(
                            StreamSliceReference(
                                stream=stream,
                                source_file=f"aggregate/{STREAM_FILES[stream]}",
                                start_s=start,
                                end_s=end,
                                sample_period_s=stream_periods[stream],
                                row_count=int(len(frame)),
                                channel_ids=stream_channels[stream],
                            )
                        )
                    condition = infer_operating_condition(
                        slices["legacy"], slices["adcs"], resolver=resolver
                    )
                    window = TelemetryWindow(
                        window_id=_opaque_window_id(adapter.dataset_id, run.run_id, start, end, "v1"),
                        dataset_id=adapter.dataset_id,
                        run_id=run.run_id,
                        paired_nominal_run_id=run.paired_nominal_run_id,
                        split=run.split,
                        case_type=run.case_type,
                        case_id=run.case_id,
                        start_s=start,
                        end_s=end,
                        operating_condition=condition,
                        streams=refs,
                        labels=_window_labels(legacy_join),
                        metadata={
                            "window_s": window_s,
                            "stride_s": stride_s,
                            "labels_not_features": True,
                            "condition_inference_status": "E0_SYNTHETIC_HEURISTIC",
                            "condition_catalog_version": resolver.catalog.catalog_version,
                        },
                    )
                    handle.write(window.model_dump_json() + "\n")
                    count += 1
                    counts_by_split[run.split] = counts_by_split.get(run.split, 0) + 1
                    counts_by_case[run.case_id] = counts_by_case.get(run.case_id, 0) + 1
                    for condition_id in condition.condition_ids:
                        counts_by_condition[condition_id] = counts_by_condition.get(condition_id, 0) + 1
                    start += stride_s

    artifacts.write_generated(output.name, _write_index)
    summary = {
        "schema_version": "astrograph-window-index-summary.v1",
        "dataset_id": adapter.dataset_id,
        "window_schema_version": "astrograph-telemetry-window.v1",
        "window_s": window_s,
        "stride_s": stride_s,
        "window_count": count,
        "counts_by_split": counts_by_split,
        "counts_by_case": counts_by_case,
        "counts_by_condition": counts_by_condition,
        "condition_catalog_version": resolver.catalog.catalog_version,
        "condition_resolver_id": resolver.resolver_id,
        "claim_boundary": "Window labels are evaluation references and must not be used as model features.",
        "output": str(output),
    }
    artifacts.write_json(output.with_suffix(".summary.json").name, summary)
    return summary
