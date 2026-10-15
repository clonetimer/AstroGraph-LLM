#!/usr/bin/env python3
"""R25 development-only ESA high-resolution transient remediation experiment.

Never regenerates predictions for R21 seen confirmatory/external partitions.
The development metric is an engineering screen: alarm-group precision plus
anomaly-ID recall inside event-centered development windows. It is not the
ESA-ADB corrected event-wise benchmark metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrograph.evaluation.data_exposure import dataset_use, guard_development, guard_prediction_generation, registry
from astrograph.evaluation.esa_high_resolution import (
    ESAHighResolutionConfig,
    fit_delta_reference,
    group_alarm_intervals,
    score_high_resolution_transients,
)
from astrograph.evaluation.external_dataset_contract import seal

CHANNELS = tuple(f"channel_{i}" for i in range(41, 47))
THRESHOLDS = (6.0, 8.0, 10.0, 12.0)
MIN_CHANNELS = (2, 3, 4)
PRE_REFERENCE = pd.Timedelta(hours=2)
REFERENCE_GAP = pd.Timedelta(minutes=10)
EVAL_MARGIN = pd.Timedelta(hours=2)
GROUP_GAP_S = 40.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _f05(p: float, r: float) -> float:
    b2 = 0.25
    return 0.0 if p + r == 0 else (1 + b2) * p * r / (b2 * p + r)


def _intervals(labels: pd.DataFrame, ids: list[str]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    subset = labels[labels["ID"].isin(ids) & labels["Channel"].isin(CHANNELS)].copy()
    if set(ids) != set(subset["ID"]):
        raise ValueError("R25_ESA_LIGHTWEIGHT_LABEL_COVERAGE_INCOMPLETE")
    subset["S"] = pd.to_datetime(subset["StartTime"], utc=True).dt.tz_convert(None)
    subset["E"] = pd.to_datetime(subset["EndTime"], utc=True).dt.tz_convert(None)
    return {item: (part["S"].min(), part["E"].max()) for item, part in subset.groupby("ID")}


def _overlap(group: tuple[int, int], interval: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    start = int(interval[0].value); end = int(interval[1].value)
    return group[0] <= end and group[1] >= start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esa-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--unpacked-channel-root", default=None, help="Optional performance-only mirror of channel pickle bytes; source identity remains bound to ESA zip files.")
    args = ap.parse_args()
    root = Path(args.esa_root).resolve()
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)

    reg = registry()["datasets"]["ESA_MISSION1"]
    train_ids = list(reg["group_ids_by_partition"]["MODEL_TRAIN"])
    dev_ids = list(reg["group_ids_by_partition"]["DEVELOPMENT_SELECTION"])
    labels = pd.read_csv(root / "labels.csv")
    types = pd.read_csv(root / "anomaly_types.csv").set_index("ID")
    train_intervals = _intervals(labels, train_ids)
    dev_intervals = _intervals(labels, dev_ids)
    anomaly_ids = [x for x in dev_ids if str(types.loc[x, "Category"]) == "Anomaly"]
    rare_ids = [x for x in dev_ids if str(types.loc[x, "Category"]) != "Anomaly"]

    training_windows: dict[str, list[np.ndarray]] = {ch: [] for ch in CHANNELS}
    eval_windows: dict[str, dict[str, np.ndarray]] = {item: {} for item in dev_ids}
    eval_times: dict[str, np.ndarray] = {}
    source_hashes = {}

    for channel in CHANNELS:
        path = root / "channels" / f"{channel}.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        source_hashes[channel] = _sha(path)
        read_path = (Path(args.unpacked_channel_root) / channel) if args.unpacked_channel_root else path
        frame = pd.read_pickle(read_path)
        if list(frame.columns) != [channel] or not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"R25_ESA_CHANNEL_SCHEMA_INVALID:{channel}")
        series = frame[channel]
        with dataset_use("ESA_MISSION1", "MODEL_TRAIN", "FIT", group_ids=train_ids):
            guard_development("FIT", group_ids=train_ids)
            for event_id in train_ids:
                event_start = train_intervals[event_id][0]
                values = series.loc[event_start - PRE_REFERENCE : event_start - REFERENCE_GAP].to_numpy(dtype=float)
                if values.size < 3:
                    raise ValueError(f"R25_ESA_REFERENCE_WINDOW_TOO_SHORT:{channel}:{event_id}")
                training_windows[channel].append(values)
        with dataset_use("ESA_MISSION1", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=dev_ids):
            guard_prediction_generation()
            for event_id in dev_ids:
                s, e = dev_intervals[event_id]
                part = series.loc[s - EVAL_MARGIN : e + EVAL_MARGIN]
                times = part.index.view("int64")
                if event_id not in eval_times:
                    eval_times[event_id] = times.copy()
                elif not np.array_equal(eval_times[event_id], times):
                    raise ValueError(f"R25_ESA_LIGHTWEIGHT_TIMELINE_DRIFT:{event_id}:{channel}")
                eval_windows[event_id][channel] = part.to_numpy(dtype=float)
        del frame, series

    with dataset_use("ESA_MISSION1", "MODEL_TRAIN", "FIT", group_ids=train_ids):
        guard_development("FIT", group_ids=train_ids)
        reference = fit_delta_reference(training_windows)

    candidates = []
    with dataset_use("ESA_MISSION1", "DEVELOPMENT_SELECTION", "SELECT", group_ids=dev_ids):
        guard_development("SELECT", group_ids=dev_ids)
        for threshold in THRESHOLDS:
            for minimum in MIN_CHANNELS:
                config = ESAHighResolutionConfig(
                    robust_z_threshold=threshold,
                    minimum_coincident_channels=minimum,
                    alarm_group_gap_s=GROUP_GAP_S,
                )
                groups_by_id = {}
                all_groups = []
                for event_id in dev_ids:
                    score = score_high_resolution_transients(
                        reference=reference,
                        time_ns=eval_times[event_id],
                        channel_values=eval_windows[event_id],
                        config=config,
                    )
                    groups = group_alarm_intervals(score, gap_s=GROUP_GAP_S)
                    groups_by_id[event_id] = groups
                    all_groups.extend((event_id, group) for group in groups)
                positive_groups = sum(
                    1 for event_id, group in all_groups
                    if event_id in anomaly_ids and _overlap(group, dev_intervals[event_id])
                )
                hit_events = sum(
                    1 for event_id in anomaly_ids
                    if any(_overlap(group, dev_intervals[event_id]) for group in groups_by_id[event_id])
                )
                group_precision = positive_groups / len(all_groups) if all_groups else 0.0
                event_recall = hit_events / len(anomaly_ids) if anomaly_ids else 0.0
                false_groups = len(all_groups) - positive_groups
                candidates.append({
                    "robust_z_threshold": threshold,
                    "minimum_coincident_channels": minimum,
                    "alarm_group_gap_s": GROUP_GAP_S,
                    "alarm_group_count": len(all_groups),
                    "positive_alarm_group_count": positive_groups,
                    "false_alarm_group_count": false_groups,
                    "anomaly_event_count": len(anomaly_ids),
                    "hit_anomaly_event_count": hit_events,
                    "alarm_group_precision": group_precision,
                    "anomaly_event_recall": event_recall,
                    "engineering_f0_5": _f05(group_precision, event_recall),
                    "alarm_groups_by_development_id": {k: len(v) for k, v in groups_by_id.items()},
                })
    selected = sorted(
        candidates,
        key=lambda row: (-row["engineering_f0_5"], row["false_alarm_group_count"], row["robust_z_threshold"], row["minimum_coincident_channels"]),
    )[0]
    payload = {
        "schema_version": "astrograph-r25-esa-high-resolution-development-report.v1",
        "dataset_id": "ESA_MISSION1",
        "partitions_consumed": ["MODEL_TRAIN", "DEVELOPMENT_SELECTION"],
        "seen_partitions_regenerated": False,
        "confirmatory_external_predictions_generated": False,
        "channels": list(CHANNELS),
        "source_channel_zip_sha256": source_hashes,
        "reference": reference.model_dump(mode="json"),
        "reference_policy": {"pre_event_hours": 2, "gap_before_event_minutes": 10},
        "development_window_policy": {"margin_hours_each_side": 2},
        "development_anomaly_ids": anomaly_ids,
        "development_rare_event_ids": rare_ids,
        "candidate_grid": candidates,
        "selected_candidate": selected,
        "metric_contract": {
            "precision": "alarm groups overlapping the same development anomaly ID / all alarm groups in event-centered windows",
            "recall": "development anomaly IDs with >=1 overlapping alarm group / development anomaly IDs",
            "warning": "engineering development screen; not ESA-ADB corrected event-wise metric and not external validation",
        },
        "status": "EXPLORATORY_DEVELOPMENT_SELECTED",
        "qualification": "NOT_EXTERNAL_VALIDATION_NOT_FLIGHT_QUALIFIED",
    }
    sealed = seal(payload)
    output.write_text(json.dumps(sealed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": sealed["status"], "selected": selected, "receipt_sha256": sealed["receipt_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
