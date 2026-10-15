#!/usr/bin/env python3
"""Leakage-controlled ADCS v5.2.2 end-to-end regression.

This is an engineering regression harness, not a performance benchmark.  It
uses the frozen trajectory-group split, fits preprocessing on train only, and
compares a statistical baseline, sklearn logistic regression, and a small
past-only-window 1-D CNN across detection, component localization, and
component-conditioned fault classification.  Each family is adapted to
AstroGraph's common DataDrivenPrediction contract before entering the main
workflow.  The LLM is an evidence-constrained reasoner, never a source of
telemetry truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.evaluation_protocol import three_task_evaluation_audit
from astrograph.data_driven.alarm_policy import alarm_event_metrics
from astrograph.data_driven.ordered_features import read_csv_in_declared_order
from astrograph.data_driven.preprocessing import ChannelPreprocessor, fit_channel_preprocessor

DEFAULT_DATASET = Path(
    os.environ.get(
        "ASTROGRAPH_ADCS_V522_DATA_ROOT",
        str(ROOT / "data" / "external" / "whole_spacecraft_adcs_paired_dataset_v5_2_2"),
    )
)
DEFAULT_OUTPUT = ROOT / "scripts" / "integration" / "out" / "adcs_v522_regression"
PROTOCOL_ROOT = ROOT / "protocols" / "phm_project_a"
EVIDENCE_PACKAGE = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "runtime_evidence.json"
DATASET_PROFILE_REGISTRY = ROOT / "configs" / "dataset_profiles" / "registry.json"
RUNTIME_POLICY_REGISTRY = ROOT / "configs" / "runtime_policies" / "registry.json"
FAULT_CLASS_MAP = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "fault_class_map.json"
DIAGNOSIS_HIERARCHY = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "diagnosis_hierarchy.json"
DOR_SEMANTIC_PACKAGE = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "dor_semantic_package.json"
CANDIDATE_UNIVERSE = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "candidate_universe.json"
GENERIC_FAULT_ONTOLOGY = ROOT / "configs" / "knowledge" / "generic_fault_ontology.json"
FAULT_ONTOLOGY_BINDINGS = ROOT / "configs" / "datasets" / "adcs_v5_2_2" / "fault_ontology_bindings.json"
COMPONENT_ARCHETYPES = ROOT / "configs" / "semantics" / "core_component_archetypes.json"
SEMANTIC_SCHEMA = ROOT / "configs" / "semantics" / "core_semantic_schema.json"
DEFAULT_DOR_EXECUTABLE = Path(
    os.environ.get("ASTROGRAPH_DOR_EXECUTABLE", shutil.which("dor") or "__optional_dor_not_found__")
)
WINDOW = 20
TRAIN_STRIDE = 5
EVAL_STRIDE = 1
SEED = 20260812

SPLIT_INDICES = {
    "train": frozenset({0, 1, 2, 5, 6, 7}),
    "validation": frozenset({3, 8}),
    "test": frozenset({4, 9}),
}

# Only exact semantic correspondences are allowed into the current production
# ontology.  Other v5.2.2 classes remain visible in offline metrics but are not
# silently renamed into a registered AstroGraph fault.
CLASS_TO_REGISTERED = json.loads(FAULT_CLASS_MAP.read_text(encoding="utf-8"))["mappings"]
_HIERARCHY = json.loads(DIAGNOSIS_HIERARCHY.read_text(encoding="utf-8"))
FAULT_TO_COMPONENT = _HIERARCHY["fault_to_component"]
COMPONENT_NAMES = _HIERARCHY["components"]

FEATURE_COLUMNS = (
    "adcs.pointing_error_deg",
    "adcs.sensor.gyro_measured_rad_s_x",
    "adcs.sensor.gyro_measured_rad_s_y",
    "adcs.sensor.gyro_measured_rad_s_z",
    "adcs.sensor.imu.valid",
    "adcs.sensor.gyro_axis_valid_x",
    "adcs.sensor.gyro_axis_valid_y",
    "adcs.sensor.gyro_axis_valid_z",
    "adcs.star_tracker.valid",
    "adcs.star_tracker.solution_fresh",
    "adcs.sun_sensor.valid",
    "adcs.sun_sensor.saturated",
    "adcs.sun_sensor.cell_healthy",
    "adcs.rw.command_execution_residual_norm_nm",
    "adcs.rw.speed_response_residual_norm_nm",
    "adcs.rw.speed_sensor_invalid_count",
    "adcs.sensor.gyro_invalid_axis_count",
    *tuple(f"adcs.rw.measured_speed_rad_s_{i}" for i in range(4)),
    *tuple(f"adcs.rw.command_torque_nm_{i}" for i in range(4)),
    *tuple(f"adcs.rw.motor_current_a_{i}" for i in range(4)),
)


@dataclass(frozen=True)
class Trajectory:
    sample_id: str
    trajectory_pair_id: str
    sample_index: int
    scenario_id: str
    diagnostic_class: str
    telemetry_path: Path
    labels_path: Path
    split: str


@dataclass
class WindowSet:
    x: np.ndarray
    detection: np.ndarray
    isolation: np.ndarray
    trajectory_ids: list[str]
    end_times_s: np.ndarray
    onset_times_s: np.ndarray
    component: np.ndarray | None = None


def split_for_index(sample_index: int) -> str:
    matches = [name for name, values in SPLIT_INDICES.items() if sample_index in values]
    if len(matches) != 1:
        raise ValueError(f"sample_index {sample_index} does not belong to exactly one split")
    return matches[0]


def load_trajectories(dataset_root: Path) -> list[Trajectory]:
    rows = list(csv.DictReader((dataset_root / "sample_index.csv").open(encoding="utf-8")))
    result = []
    for row in rows:
        result.append(
            Trajectory(
                sample_id=row["sample_id"],
                trajectory_pair_id=row["trajectory_pair_id"],
                sample_index=int(row["sample_index"]),
                scenario_id=row["scenario_id"],
                diagnostic_class=row["diagnostic_class"],
                telemetry_path=dataset_root / row["telemetry_file"],
                labels_path=dataset_root / row["labels_file"],
                split=split_for_index(int(row["sample_index"])),
            )
        )
    return result


def audit_split(rows: list[Trajectory]) -> dict[str, Any]:
    by_split = {name: [row for row in rows if row.split == name] for name in SPLIT_INDICES}
    groups = {name: {row.trajectory_pair_id for row in values} for name, values in by_split.items()}
    overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    report: dict[str, Any] = {
        "schema_version": "astrograph-adcs-v522-split-audit.v1",
        "primary_group_key": "trajectory_pair_id",
        "assignment": "sample_index modulo 5; train={0,1,2}, validation={3}, test={4}",
        "group_overlap": overlaps,
        "group_overlap_count": sum(map(len, overlaps.values())),
        "splits": {},
    }
    for name, values in by_split.items():
        report["splits"][name] = {
            "trajectory_count": len(values),
            "fault_trajectory_count": sum(row.diagnostic_class != "nominal" for row in values),
            "nominal_trajectory_count": sum(row.diagnostic_class == "nominal" for row in values),
            "trajectory_group_count": len(groups[name]),
            "scenario_counts": dict(sorted(Counter(row.scenario_id for row in values).items())),
            "class_counts": dict(sorted(Counter(row.diagnostic_class for row in values).items())),
        }
    expected = {
        "train": (546, 528, 18, 18),
        "validation": (182, 176, 6, 6),
        "test": (182, 176, 6, 6),
    }
    report["frozen_counts_match"] = all(
        (
            item["trajectory_count"], item["fault_trajectory_count"],
            item["nominal_trajectory_count"], item["trajectory_group_count"],
        ) == expected[name]
        for name, item in report["splits"].items()
    )
    report["status"] = (
        "PASS" if report["group_overlap_count"] == 0 and report["frozen_counts_match"] else "FAIL"
    )
    return report


def _read_truth(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    labels = pd.read_csv(path, usecols=["time_s", "label.fault_active"])
    active = labels["label.fault_active"].astype(str).str.lower().isin({"1", "true"}).to_numpy()
    times = labels["time_s"].to_numpy(dtype=float)
    onset = float(times[np.flatnonzero(active)[0]]) if active.any() else math.nan
    return times, active, onset


def build_windows(
    rows: list[Trajectory], class_to_index: dict[str, int], component_to_index: dict[str, int],
    *, feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> dict[str, WindowSet]:
    buffers: dict[str, dict[str, Any]] = {
        name: {"x": [], "d": [], "i": [], "c": [], "ids": [], "times": [], "onsets": []}
        for name in SPLIT_INDICES
    }
    for number, row in enumerate(rows, start=1):
        # Pandas does not preserve the ``usecols`` argument order.  Restore the
        # declared tensor contract before conversion so training and runtime
        # cannot silently disagree about channel positions.
        frame = read_csv_in_declared_order(row.telemetry_path, feature_columns)
        values = frame.to_numpy(dtype=np.float32)
        times, active, onset = _read_truth(row.labels_path)
        if len(values) != len(active):
            raise ValueError(f"telemetry/label length mismatch: {row.sample_id}")
        stride = TRAIN_STRIDE if row.split in {"train", "validation"} else EVAL_STRIDE
        target = buffers[row.split]
        for end in range(WINDOW - 1, len(values), stride):
            target["x"].append(values[end - WINDOW + 1 : end + 1])
            target["d"].append(int(active[end]))
            target["i"].append(class_to_index[row.diagnostic_class] if active[end] else -1)
            target["c"].append(
                component_to_index[FAULT_TO_COMPONENT[row.diagnostic_class]] if active[end] else -1
            )
            target["ids"].append(row.sample_id)
            target["times"].append(float(times[end]))
            target["onsets"].append(onset)
        if number % 100 == 0:
            print(f"[windows] {number}/{len(rows)} trajectories", flush=True)
    return {
        name: WindowSet(
            x=np.asarray(item["x"], dtype=np.float32),
            detection=np.asarray(item["d"], dtype=np.int64),
            isolation=np.asarray(item["i"], dtype=np.int64),
            trajectory_ids=item["ids"],
            end_times_s=np.asarray(item["times"], dtype=np.float64),
            onset_times_s=np.asarray(item["onsets"], dtype=np.float64),
            component=np.asarray(item["c"], dtype=np.int64),
        )
        for name, item in buffers.items()
    }


def fit_train_scaler(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the historical standard scaler."""
    state = fit_channel_preprocessor(train_x, method="standard")
    return state.impute, state.center, state.scale


def fit_train_preprocessor(
    train_x: np.ndarray, *, method: str = "standard"
) -> ChannelPreprocessor:
    return fit_channel_preprocessor(train_x, method=method)


def scale_windows(
    item: WindowSet, impute: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> WindowSet:
    filled = np.where(np.isfinite(item.x), item.x, impute)
    return WindowSet(
        x=((filled - mean) / scale).astype(np.float32), detection=item.detection,
        isolation=item.isolation, trajectory_ids=item.trajectory_ids,
        end_times_s=item.end_times_s, onset_times_s=item.onset_times_s,
        component=item.component,
    )


def preprocess_windows(item: WindowSet, state: ChannelPreprocessor) -> WindowSet:
    return WindowSet(
        x=state.transform(item.x), detection=item.detection,
        isolation=item.isolation, trajectory_ids=item.trajectory_ids,
        end_times_s=item.end_times_s, onset_times_s=item.onset_times_s,
        component=item.component,
    )


def summary_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x.mean(1), x.std(1), x[:, -1], x[:, -1] - x[:, 0]), axis=1)


def _metrics(truth: np.ndarray, predicted: np.ndarray, labels: list[int] | None = None) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_precision": float(precision_score(
            truth, predicted, labels=labels, average="macro", zero_division=0,
        )),
        "macro_recall": float(recall_score(
            truth, predicted, labels=labels, average="macro", zero_division=0,
        )),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
    }


def component_class_indices(
    class_names: list[str], component_names: list[str]
) -> dict[int, np.ndarray]:
    component_to_index = {name: index for index, name in enumerate(component_names)}
    result = {
        index: np.asarray(
            [
                class_index
                for class_index, fault_name in enumerate(class_names)
                if component_to_index[FAULT_TO_COMPONENT[fault_name]] == index
            ],
            dtype=np.int64,
        )
        for index in range(len(component_names))
    }
    if any(len(values) == 0 for values in result.values()):
        raise ValueError("every diagnosis component must own at least one fault class")
    return result


def _conditioned_class_probabilities(
    all_class_prob: np.ndarray,
    component_prob: np.ndarray,
    class_by_component: dict[int, np.ndarray],
    *,
    mode: str = "hard",
) -> tuple[np.ndarray, np.ndarray]:
    predicted_component = component_prob.argmax(1)
    if mode == "soft":
        weighted = np.zeros_like(all_class_prob)
        for component_index, class_indices in class_by_component.items():
            weighted[:, class_indices] = (
                all_class_prob[:, class_indices] * component_prob[:, component_index, None]
            )
        total = weighted.sum(1, keepdims=True)
        conditioned = weighted / np.maximum(total, 1e-12)
        return conditioned, predicted_component
    if mode != "hard":
        raise ValueError(f"unknown class conditioning mode: {mode}")
    conditioned = np.zeros_like(all_class_prob)
    for row_index, component_index in enumerate(predicted_component):
        allowed = class_by_component[int(component_index)]
        values = all_class_prob[row_index, allowed]
        total = float(values.sum())
        conditioned[row_index, allowed] = values / total if total > 0 else 1.0 / len(allowed)
    return conditioned, predicted_component


def _oracle_class_prediction(
    all_class_prob: np.ndarray,
    true_component: np.ndarray,
    class_by_component: dict[int, np.ndarray],
) -> np.ndarray:
    predicted = np.zeros(len(all_class_prob), dtype=np.int64)
    for row_index, component_index in enumerate(true_component):
        allowed = class_by_component[int(component_index)]
        predicted[row_index] = int(allowed[np.argmax(all_class_prob[row_index, allowed])])
    return predicted


def three_stage_report(
    *,
    family: str,
    test: WindowSet,
    det_score: np.ndarray,
    component_prob: np.ndarray,
    all_class_prob: np.ndarray,
    class_names: list[str],
    component_names: list[str],
    mechanism: str,
    extra: dict[str, Any] | None = None,
    detection_threshold: float = 0.5,
    detection_predicted: np.ndarray | None = None,
    conditioning_mode: str = "hard",
) -> tuple[dict[str, Any], np.ndarray]:
    if test.component is None:
        raise ValueError("three-stage evaluation requires evaluator-side component truth")
    active = test.isolation >= 0
    class_by_component = component_class_indices(class_names, component_names)
    conditioned_prob, predicted_component = _conditioned_class_probabilities(
        all_class_prob, component_prob, class_by_component, mode=conditioning_mode
    )
    oracle_class = _oracle_class_prediction(
        all_class_prob[active], test.component[active], class_by_component
    )
    predicted_class = conditioned_prob[active].argmax(1)
    if detection_predicted is None:
        det_pred = (det_score >= detection_threshold).astype(np.int64)
    else:
        det_pred = np.asarray(detection_predicted, dtype=np.int64)
        if det_pred.shape != test.detection.shape:
            raise ValueError("detection_predicted must match evaluator truth shape")
        if not np.isin(det_pred, [0, 1]).all():
            raise ValueError("detection_predicted must be binary")
    cascade_truth = np.where(test.detection == 0, 0, test.isolation + 1)
    cascade_pred = np.where(det_pred == 0, 0, conditioned_prob.argmax(1) + 1)
    detected_active = active & (det_pred == 1)
    localized_active = detected_active & (predicted_component == test.component)
    failure_attribution = {
        "fault_windows": int(active.sum()),
        "first_wrong_stage_detection": int((active & (det_pred == 0)).sum()),
        "first_wrong_stage_localization": int(
            (detected_active & (predicted_component != test.component)).sum()
        ),
        "first_wrong_stage_classification": int(
            (
                localized_active
                & (conditioned_prob.argmax(1) != test.isolation)
            ).sum()
        ),
        "fully_correct_fault_windows": int(
            (localized_active & (conditioned_prob.argmax(1) == test.isolation)).sum()
        ),
    }
    report: dict[str, Any] = {
        "family": family,
        "detection": _metrics(test.detection, det_pred, [0, 1]),
        "localization": _metrics(
            test.component[active], predicted_component[active], list(range(len(component_names)))
        ),
        "classification_oracle_component": _metrics(
            test.isolation[active], oracle_class, list(range(len(class_names)))
        ),
        "classification_predicted_component": _metrics(
            test.isolation[active], predicted_class, list(range(len(class_names)))
        ),
        # Compatibility alias: isolation now means predicted-component classification.
        "isolation": _metrics(
            test.isolation[active], predicted_class, list(range(len(class_names)))
        ),
        "end_to_end_15_class": _metrics(
            cascade_truth, cascade_pred, list(range(len(class_names) + 1))
        ),
        "failure_attribution": failure_attribution,
        "delay": detection_delays(
            test, det_score, threshold=detection_threshold, predicted=det_pred
        ),
        "alarm_event_metrics": alarm_event_metrics(
            test.detection.tolist(), det_pred.tolist(), test.trajectory_ids,
            test.end_times_s.tolist(), test.onset_times_s.tolist(),
        ),
        "detection_threshold": detection_threshold,
        "conditioning_mode": conditioning_mode,
        "mechanism": mechanism,
        "evaluation_audit": three_task_evaluation_audit(
            group_ids=test.trajectory_ids,
            detection_truth=test.detection.tolist(),
            detection_predicted=det_pred.tolist(),
            detection_positive_scores=det_score.tolist(),
            component_truth=test.component.tolist(),
            component_predicted=predicted_component.tolist(),
            class_truth=test.isolation.tolist(),
            class_predicted=conditioned_prob.argmax(1).tolist(),
            component_labels=component_names,
            class_labels=class_names,
            group_key="sample_id",
        ),
    }
    if extra:
        report.update(extra)
    return report, conditioned_prob


def detection_delays(
    item: WindowSet,
    scores: np.ndarray,
    threshold: float = 0.5,
    predicted: np.ndarray | None = None,
) -> dict[str, Any]:
    alarms = (scores >= threshold) if predicted is None else np.asarray(predicted, dtype=bool)
    if alarms.shape != scores.shape:
        raise ValueError("predicted alarms must match scores shape")
    first: dict[str, float] = {}
    onset: dict[str, float] = {}
    for trajectory_id, time_s, onset_s, alarm in zip(
        item.trajectory_ids, item.end_times_s, item.onset_times_s, alarms, strict=True
    ):
        if math.isnan(onset_s):
            continue
        onset[trajectory_id] = onset_s
        if time_s >= onset_s and alarm and trajectory_id not in first:
            first[trajectory_id] = time_s
    delays = [first[key] - value for key, value in onset.items() if key in first]
    return {
        "eligible_fault_trajectories": len(onset), "detected_trajectories": len(delays),
        "detection_coverage": len(delays) / len(onset) if onset else 0.0,
        "mean_delay_s": float(np.mean(delays)) if delays else None,
        "median_delay_s": float(np.median(delays)) if delays else None,
    }


def _centroid_probabilities(
    train_f: np.ndarray, train_labels: np.ndarray, test_f: np.ndarray, n_labels: int
) -> np.ndarray:
    centroids = np.stack([
        train_f[train_labels == index].mean(0) for index in range(n_labels)
    ])
    distances = ((test_f[:, None, :] - centroids[None, :, :]) ** 2).mean(2)
    probabilities = np.exp(-(distances - distances.min(1, keepdims=True)))
    return probabilities / probabilities.sum(1, keepdims=True)


def run_statistical(
    train: WindowSet, test: WindowSet, class_names: list[str], component_names: list[str]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    train_f = summary_features(train.x)
    test_f = summary_features(test.x)
    normal = train_f[train.detection == 0]
    centre = normal.mean(0)
    distance_train = np.linalg.norm(train_f - centre, axis=1)
    threshold = float(np.quantile(distance_train[train.detection == 0], 0.99))
    distance_test = np.linalg.norm(test_f - centre, axis=1)
    scale = max(float(np.std(distance_train[train.detection == 0])), 1e-6)
    det_score = 1.0 / (1.0 + np.exp(-(distance_test - threshold) / scale))
    active_train = train.isolation >= 0
    if train.component is None:
        raise ValueError("component labels are required for localization training")
    component_prob = _centroid_probabilities(
        train_f[active_train], train.component[active_train], test_f, len(component_names)
    )
    all_class_prob = _centroid_probabilities(
        train_f[active_train], train.isolation[active_train], test_f, len(class_names)
    )
    report, conditioned = three_stage_report(
        family="statistical", test=test, det_score=det_score,
        component_prob=component_prob, all_class_prob=all_class_prob,
        class_names=class_names, component_names=component_names,
        mechanism=("train-normal distance detection + train-fault component centroid localization "
                   "+ component-conditioned fault centroid classification"),
    )
    return report, det_score, component_prob, conditioned


def run_ml(
    train: WindowSet, test: WindowSet, class_names: list[str], component_names: list[str]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    train_f = summary_features(train.x)
    test_f = summary_features(test.x)
    detector = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    detector.fit(train_f, train.detection)
    det_prob = detector.predict_proba(test_f)[:, list(detector.classes_).index(1)]
    active_train = train.isolation >= 0
    if train.component is None:
        raise ValueError("component labels are required for localization training")
    localizer = LogisticRegression(max_iter=700, class_weight="balanced", random_state=SEED)
    localizer.fit(train_f[active_train], train.component[active_train])
    component_prob = np.zeros((len(test_f), len(component_names)), dtype=float)
    component_prob[:, localizer.classes_.astype(int)] = localizer.predict_proba(test_f)

    class_by_component = component_class_indices(class_names, component_names)
    all_class_prob = np.zeros((len(test_f), len(class_names)), dtype=float)
    for component_index, allowed in class_by_component.items():
        if len(allowed) == 1:
            all_class_prob[:, allowed[0]] = 1.0
            continue
        routed_train = active_train & (train.component == component_index)
        classifier = LogisticRegression(
            max_iter=700, class_weight="balanced", random_state=SEED
        )
        classifier.fit(train_f[routed_train], train.isolation[routed_train])
        all_class_prob[:, classifier.classes_.astype(int)] = classifier.predict_proba(test_f)
    report, conditioned = three_stage_report(
        family="machine_learning", test=test, det_score=det_prob,
        component_prob=component_prob, all_class_prob=all_class_prob,
        class_names=class_names, component_names=component_names,
        mechanism=("balanced logistic detection + balanced multinomial component localization "
                   "+ per-component balanced logistic fault classifiers"),
    )
    return report, det_prob, component_prob, conditioned


def run_deep(
    train: WindowSet, test: WindowSet, class_names: list[str],
    component_names: list[str], epochs: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    class ThreeStageWindowCNN(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(channels, 32, 5, padding=2), nn.ReLU(),
                nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            )
            self.detector = nn.Linear(32, 2)
            self.localizer = nn.Linear(32, len(component_names))
            class_by_component = component_class_indices(class_names, component_names)
            self.classifiers = nn.ModuleList([
                nn.Linear(32, len(class_by_component[index]))
                for index in range(len(component_names))
            ])

        def forward(self, x: Any) -> tuple[Any, Any, list[Any]]:
            h = self.encoder(x.transpose(1, 2)).squeeze(-1)
            return self.detector(h), self.localizer(h), [head(h) for head in self.classifiers]

    if train.component is None:
        raise ValueError("component labels are required for localization training")
    class_by_component = component_class_indices(class_names, component_names)
    global_to_local = {
        component_index: {int(value): local for local, value in enumerate(allowed)}
        for component_index, allowed in class_by_component.items()
    }
    model = ThreeStageWindowCNN(train.x.shape[-1])
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.x), torch.from_numpy(train.detection),
            torch.from_numpy(train.component), torch.from_numpy(train.isolation),
        ),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    det_loss = nn.CrossEntropyLoss()
    loc_loss = nn.CrossEntropyLoss()
    cls_loss = nn.CrossEntropyLoss()
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for x, detection, component, isolation in loader:
            optimizer.zero_grad(set_to_none=True)
            dlogits, llogits, class_logits = model(x)
            active = isolation >= 0
            loss = det_loss(dlogits, detection)
            if active.any():
                loss = loss + loc_loss(llogits[active], component[active])
                for component_index, logits in enumerate(class_logits):
                    routed = active & (component == component_index)
                    if routed.any():
                        local_target = torch.tensor(
                            [global_to_local[component_index][int(value)] for value in isolation[routed]],
                            dtype=torch.long,
                        )
                        loss = loss + cls_loss(logits[routed], local_target)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(x)
        print(f"[deep] epoch={epoch + 1}/{epochs} loss={running / len(train.x):.6f}", flush=True)
    model.eval()
    det_parts, component_parts, class_parts = [], [], []
    with torch.no_grad():
        for start in range(0, len(test.x), 1024):
            dlogits, llogits, class_logits = model(
                torch.from_numpy(test.x[start:start + 1024])
            )
            det_parts.append(torch.softmax(dlogits, 1)[:, 1].numpy())
            component_parts.append(torch.softmax(llogits, 1).numpy())
            batch_class = np.zeros((len(dlogits), len(class_names)), dtype=float)
            for component_index, logits in enumerate(class_logits):
                batch_class[:, class_by_component[component_index]] = torch.softmax(logits, 1).numpy()
            class_parts.append(batch_class)
    det_prob = np.concatenate(det_parts)
    component_prob = np.concatenate(component_parts)
    all_class_prob = np.concatenate(class_parts)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report, conditioned = three_stage_report(
        family="deep_learning", test=test, det_score=det_prob,
        component_prob=component_prob, all_class_prob=all_class_prob,
        class_names=class_names, component_names=component_names,
        mechanism=("past-only 20 s window CNN encoder + binary detector + five-component "
                   "localizer + component-specific fault heads"),
        extra={"epochs": epochs, "parameter_count": parameter_count,
               "training_seconds": time.perf_counter() - started},
    )
    return report, det_prob, component_prob, conditioned


def make_prediction(
    *, family: str, sample: Trajectory, window: np.ndarray, class_names: list[str],
    component_names: list[str], det_score: float, component_prob: np.ndarray,
    iso_prob: np.ndarray, episode_id: str,
):
    from astrograph.data_driven import (
        DataDrivenModelManifest, DataDrivenTaskType, ModelCapability, ModelInferenceRequest,
        ModelOutputSemantics, ModelRuntime, RawModelOutput, adapt_raw_model_output,
    )
    # Map only audited exact correspondences; all unsupported mass is retained in
    # metadata and cannot influence BiRRF under a false registered identity.
    from astrograph.data_driven.candidate_mapping import map_runtime_fault_attribution

    universe = json.loads(CANDIDATE_UNIVERSE.read_text(encoding="utf-8"))
    mapping = map_runtime_fault_attribution(
        dict(zip(class_names, map(float, iso_prob), strict=True)),
        CLASS_TO_REGISTERED,
        set(universe["runtime_fault_ids"]),
    )
    mapped = mapping.mapped_attribution
    unsupported = mapping.unsupported_probabilities
    runtime = {
        "statistical": ModelRuntime.CUSTOM_CALLABLE,
        "machine_learning": ModelRuntime.SCIKIT_LEARN,
        "deep_learning": ModelRuntime.PYTORCH,
    }[family]
    model_id = f"adcs-v522-{family}-regression"
    version = "2026.08.12"
    manifest = DataDrivenModelManifest(
        model_id=model_id, model_version=version, display_name=model_id,
        task_type=DataDrivenTaskType.ANOMALY_ATTRIBUTION, runtime=runtime,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS, ModelCapability.TEMPORAL_CONTEXT],
        supported_fault_ids=sorted(mapped), normal_class_id="NORMAL",
        feature_schema_ids=["adcs-v522-causal-20s-v1"],
        training_dataset_ids=["whole_spacecraft_adcs_paired_dataset_v5_2_2"],
        claim_boundary=("Regression-only synthetic ADCS signal; unsupported v5.2.2 classes are not renamed; "
                        "not calibrated, not production-bound, not a flight claim."),
    )
    digest = hashlib.sha256(window.tobytes()).hexdigest()
    request = ModelInferenceRequest(
        episode_id=episode_id, model_id=model_id, model_version=version,
        feature_schema_id="adcs-v522-causal-20s-v1", input_data_sha256=digest,
        registered_fault_ids=sorted(mapped),
    )
    from astrograph.knowledge import load_dataset_fault_bindings, load_generic_fault_ontology
    ontology = load_generic_fault_ontology(GENERIC_FAULT_ONTOLOGY)
    bindings = load_dataset_fault_bindings(FAULT_ONTOLOGY_BINDINGS, ontology)
    binding_by_class = {item.fault_class_id: item for item in bindings.bindings}
    predicted_dataset_class = class_names[int(np.argmax(iso_prob))]
    abstract_binding = binding_by_class[predicted_dataset_class]
    raw = RawModelOutput(
        episode_id=episode_id, model_id=model_id, model_version=version,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        anomaly_score=float(np.clip(det_score, 0.0, 1.0)), fault_attribution=mapped,
        out_of_distribution_score=mapping.unsupported_mass,
        metadata={
            "offline_component_probabilities": dict(
                zip(component_names, map(float, component_prob), strict=True)
            ),
            "predicted_component": component_names[int(np.argmax(component_prob))],
            "component_routing": "predicted_component",
            "offline_class_probabilities": dict(
                zip(class_names, map(float, iso_prob), strict=True)
            ),
            "predicted_dataset_class": predicted_dataset_class,
            "abstract_failure_mechanism_id": abstract_binding.mechanism_id,
            "abstract_expected_effect_pattern_ids": abstract_binding.expected_effect_pattern_ids,
            "abstract_ontology_version": ontology.version,
            "unsupported_class_mass": unsupported,
            "sample_id": sample.sample_id,
        },
    )
    return adapt_raw_model_output(manifest, request, raw)


def run_chain(
    *, dataset_root: Path, sample: Trajectory, prediction: Any, output_dir: Path, reasoner: str,
    vllm_url: str, vllm_model: str, dor_executable: Path,
    experience_ledger_path: Path | None = None,
    memory_store_path: Path | None = None,
) -> dict[str, Any]:
    from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
    from astrograph.orchestration.models import ExecutionPolicy, PermissionScope, TaskEnvelope, WorkflowContext
    from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    from astrograph.dataset_profiles import DatasetProfileRegistry
    from astrograph.policy_profiles import RuntimePolicyRegistry
    from astrograph.simulation import SimulationRunCompleted

    source = pd.read_csv(sample.telemetry_path)
    nominal_row = next(
        row for row in load_trajectories(dataset_root)
        if row.diagnostic_class == "nominal" and row.trajectory_pair_id == sample.trajectory_pair_id
    )
    run_dir = output_dir / reasoner / "run"
    nominal_dir = output_dir / reasoner / "nominal"
    artifact_dir = output_dir / reasoner / "artifacts"
    for path in (run_dir, nominal_dir, artifact_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    shutil.copy2(sample.telemetry_path, run_dir / "telemetry.csv")
    shutil.copy2(nominal_row.telemetry_path, nominal_dir / "telemetry.csv")
    manifest = {
        "schema_version": "astrograph-adcs-v522-runtime.v1", "scenario": "", "events": [],
        "event_count": 0, "row_count": len(source), "seed": SEED,
        "context_of_use": "synthetic ADCS v5.2.2 offline causal-window regression",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    event = SimulationRunCompleted(
        event_id=f"ADCS-V522-{reasoner}", run_id=f"adcs-v522-{reasoner}",
        run_dir=str(run_dir), nominal_run_dir=str(nominal_dir), emitted_at="2026-08-12T00:00:00Z",
        metadata={"source": "whole_spacecraft_adcs_paired_dataset_v5_2_2", "split": "test"},
    )
    bundle, _ = ConfiguredRuntimeEvidenceBuilder(EVIDENCE_PACKAGE).build(event)
    from astrograph.data_driven.adapters import rebind_prediction_episode
    prediction = rebind_prediction_episode(prediction, bundle.episode.episode_id)
    permissions = [
        PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE, PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT, PermissionScope.WRITE_ARTIFACTS,
    ]
    experience_inputs = {}
    if experience_ledger_path is not None:
        experience_inputs["pre_reasoning_experience_ledger_path"] = str(
            experience_ledger_path.resolve()
        )
    if memory_store_path is not None:
        experience_inputs["pre_reasoning_memory_store_path"] = str(memory_store_path.resolve())
    envelope = TaskEnvelope(
        task_id=f"ADCS-V522-{reasoner}", task_type="simulation.online_diagnosis",
        input_payload={"simulation_event": event.model_dump(mode="json"),
                       "dataset_profile_id": "adcs.v5_2_2",
                       "artifact_output_dir": str(artifact_dir),
                       "dor_semantic_audit": {
                           "semantic_package_path": str(DOR_SEMANTIC_PACKAGE),
                           "executable_path": str(dor_executable),
                           "release_gate": True,
                           "min_support": 1.0,
                       },
                       "shadow_mode": True, "reasoner_mode": reasoner,
                       "open_set_policy": {
                           "minimum_top_probability": 0.0,
                           "minimum_top_margin": 0.0,
                           "maximum_ood_score": json.loads(
                               CANDIDATE_UNIVERSE.read_text(encoding="utf-8")
                           )["maximum_unsupported_mass_for_fusion"],
                           "suppress_data_branch_on_reject": True,
                       },
                       "data_driven_prediction": prediction.model_dump(mode="json"),
                       **experience_inputs},
        execution_policy=ExecutionPolicy(deterministic_only=reasoner == "rule", allow_llm=reasoner == "llm"),
        permission_scope=permissions,
    )
    context = WorkflowContext(
        workflow_id=WholeSpacecraftOnlineDiagnosisWorkflow.workflow_id,
        run_id=f"adcs-v522-{reasoner}", permissions=permissions,
    )
    backend = None
    if reasoner == "llm":
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=vllm_url, provider="vllm", model=vllm_model, max_tokens=800,
            timeout_s=180, prompt_profile="evidence_protocol_v059_compact", candidate_top_k=2,
            tokenizer_mode="disabled", context_length=16384,
        ))
    result = WholeSpacecraftOnlineDiagnosisWorkflow(
        reasoning_backend=backend,
        dataset_profile_registry=DatasetProfileRegistry(index_path=DATASET_PROFILE_REGISTRY),
        runtime_policy_registry=RuntimePolicyRegistry(RUNTIME_POLICY_REGISTRY),
    ).invoke(envelope, context)
    return result.model_dump(mode="json")


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for item in report["models"]:
        rows.append(
            f"| {item['family']} | {item['detection']['macro_f1']:.4f} | "
            f"{item['localization']['macro_f1']:.4f} | "
            f"{item['classification_oracle_component']['macro_f1']:.4f} | "
            f"{item['classification_predicted_component']['macro_f1']:.4f} | "
            f"{item['end_to_end_15_class']['macro_f1']:.4f} | "
            f"{item['delay']['detection_coverage']:.4f} |"
        )
    chains = []
    for key, value in report["chains"].items():
        payload = value.get("output_payload", {})
        decision = payload.get("decision_record", {})
        dor_audit = payload.get("dor_semantic_audit") or {}
        chains.append(
            f"| {key} | {value.get('status')} | {value.get('code')} | "
            f"{decision.get('disposition')} | {decision.get('accepted_fault_id')} | "
            f"{dor_audit.get('status', 'NOT_CONFIGURED')} |"
        )
    return f"""# ADCS v5.2.2 全链路回归报告

生成时间：{report['generated_at']}
性质：工程回归，不是性能选型或飞行能力声明。

## 1. 数据与泄漏控制

- 数据集：`{report['dataset_root']}`
- 划分：`trajectory_pair_id` 组级 60/20/20；先分轨迹、后切窗口。
- 轨迹数：Train 546 / Validation 182 / Test 182。
- 组数：Train 18 / Validation 6 / Test 6；交集 {report['split_audit']['group_overlap_count']}。
- 因果窗：20 s；Train/Validation stride=5 s，Test stride=1 s。
- 缺失值中位数和 scaler：都只在 Train 窗口上拟合；Val/Test 只 transform。
- 划分审计：**{report['split_audit']['status']}**。
- 运行时证据配置：`{report['runtime_evidence_package']}`。
- 配置 SHA-256：`{report['runtime_evidence_package_sha256']}`。
- 原始遥测 schema SHA-256：`{report['runtime_source_schema_sha256']}`；未生成伪造的协议通道。

## 2. 模型回归结果

| 方法族 | Detection Macro-F1 | Component Macro-F1 | Oracle Class Macro-F1 | Cascade Class Macro-F1 | E2E 15-class Macro-F1 | 检测覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`Oracle Class` 使用真实部件只做离线误差上界；`Cascade Class` 才是实际推理口径，使用模型预测部件路由。真实部件标签不进入任何推理输入。指标只证明本次固定环境下脚本可运行并产生有限实验结果；没有多种子、容量匹配和调参预算控制，不能据此宣称某模型机制更优。

## 3. 全链路结果

| 链路 | 状态 | 代码 | 决策 | 接受故障 | DOR语义审计 |
|---|---|---|---|---|---|
{chr(10).join(chains)}

链路：原始遥测 → 因果窗口 → 数据驱动预测 → 公共预测协议；并行的确定性特征/证据构造 → DOR语义资格审计 → KG/证据覆盖 → BiRRF → Decision Governor → 发布门 → 结构化输出。DOR不是BiRRF第四排序分支，LLM只读取已构造证据，二者都不读取labels。

DOR配置：`{report['dor_semantic_audit']['package']}`；SHA-256：`{report['dor_semantic_audit']['package_sha256']}`；root-cause调用：`{report['dor_semantic_audit']['root_cause_enabled']}`。

LLM 执行状态：`{report['llm_execution']['status']}`；原因：`{report['llm_execution'].get('reason') or '无'}`。

## 4. 组件卡、比较矩阵与架构方案

- 统计组件：Train 正常分布距离阈值用于检测，Train 故障类质心用于隔离；无梯度、低容量，边界是非线性和多模态类分布。
- ML 组件：平衡 Logistic Regression；输入为同一因果窗的 mean/std/last/delta，线性决策边界，优化稳定但表达有限。
- 深度组件：共享 1-D CNN 编码器，输入 `[B,20,{len(FEATURE_COLUMNS)}]`，在时间轴局部卷积后池化，接检测头、五部件定位头和五个部件内分类头；参数 {report['models'][2].get('parameter_count')}。池化丢失精确时间位置，且本轮训练预算不足以支持优越性结论。
- 三者使用同一数据、划分、特征字段和 Train-only scaler；统计/ML 使用确定性摘要表示，CNN 使用完整标准化窗口，表示容量并不完全匹配，因此性能只作条件性参考。
- 架构是成熟组件组合和数据/表示方案，不声称新基础算子或新架构创新。

## 5. 实验计划与不可宣称边界

- 本轮覆盖 B0/S0/A0 的可运行性，不覆盖参数匹配 C0、计算匹配 C1、机制破坏 M0 和多随机种子 V0。
- 后续正式实验需执行五次旋转 Group CV、独立检测/隔离调参、多种子、延迟/显存测量及稳定故障切片。
- 未映射的 v5.2.2 类不会被伪装为现有 AstroGraph 注册 fault ID；映射表见 JSON 报告。
- 不得声称真实航天器泛化、因果根因证明、安全认证或生产可用。

## 6. 技术报告扫描与组件登记

本任务未提出“最新/首次”组件，也未引入新算子；扫描结论为“无新组件创新声明”。登记项仅为本项目回归组合：统计质心、Logistic Regression、共享 CNN 三阶段多头，证据等级为本地代码与单次实验事实。

## 7. 技能资源使用追踪

| 资源 | 对本次工作的约束 |
|---|---|
| README / core-principles | 从泄漏和全链路问题出发，不从模型名称出发 |
| component-ontology | 将窗口、CNN、双头分别归为表示、局部算子和架构组合 |
| research-protocol | 未做新颖性声明，记录“无新组件创新声明” |
| design-validation-protocol | 保留统计低复杂度对照，并限制单次指标结论 |
| learning-protocol / learning log | 使用 L2 应用级审计，不把个人学习状态当实验事实 |
| full-design-principles | 明确信息边界、Train-only scaler、池化损失和部署边界 |
| component-card template | 合并到第4节的输入、机制、作用轴、代价和失败边界 |
| comparison-matrix template | 合并到第4节的统一条件和不可公平比较项 |
| architecture-proposal template | 合并到第3、4节的数据流、接口和组件必要性 |
| experiment-plan template | 合并到第5节的已覆盖/未覆盖实验矩阵 |
| technical-report-scan template | 合并到第6节，无新组件声明 |
| component-registry template | 合并到第6节，仅登记项目级组合，不污染通用注册表 |

## 8. 运行环境

```json
{json.dumps(report['environment'], ensure_ascii=False, indent=2)}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--vllm-model", default="Qwen3.5-9B")
    parser.add_argument("--dor-executable", type=Path, default=DEFAULT_DOR_EXECUTABLE)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-llm-reason")
    args = parser.parse_args()
    if args.skip_llm_reason and not args.skip_llm:
        parser.error("--skip-llm-reason requires --skip-llm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(args.dataset_root)
    from astrograph.knowledge import GraphShapeValidator, project_generic_fault_ontology
    abstract_graph_report = GraphShapeValidator().validate(project_generic_fault_ontology(
        GENERIC_FAULT_ONTOLOGY, COMPONENT_ARCHETYPES, SEMANTIC_SCHEMA
    ))
    if abstract_graph_report.status != "PASS":
        raise RuntimeError(abstract_graph_report.model_dump(mode="json"))
    split_audit = audit_split(trajectories)
    if split_audit["status"] != "PASS":
        raise RuntimeError(f"split audit failed: {split_audit}")
    class_names = sorted({row.diagnostic_class for row in trajectories if row.diagnostic_class != "nominal"})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if set(class_names) != set(FAULT_TO_COMPONENT):
        raise RuntimeError(
            "diagnosis hierarchy and dataset classes differ: "
            f"dataset_only={sorted(set(class_names) - set(FAULT_TO_COMPONENT))}, "
            f"hierarchy_only={sorted(set(FAULT_TO_COMPONENT) - set(class_names))}"
        )
    component_names = list(COMPONENT_NAMES)
    component_to_index = {name: index for index, name in enumerate(component_names)}
    windows = build_windows(trajectories, class_to_index, component_to_index)
    impute, mean, scale = fit_train_scaler(windows["train"].x)
    scaled = {name: scale_windows(item, impute, mean, scale) for name, item in windows.items()}
    model_outputs = []
    family_predictions = {}
    for runner in (run_statistical, run_ml):
        item, det, component, iso = runner(
            scaled["train"], scaled["test"], class_names, component_names
        )
        model_outputs.append(item)
        family_predictions[item["family"]] = (det, component, iso)
    item, det, component, iso = run_deep(
        scaled["train"], scaled["test"], class_names, component_names, args.epochs
    )
    model_outputs.append(item)
    family_predictions[item["family"]] = (det, component, iso)

    selected = next(
        row for row in trajectories
        if row.split == "test" and row.diagnostic_class == "adcs_rw_jamming"
    )
    indices = [index for index, value in enumerate(scaled["test"].trajectory_ids) if value == selected.sample_id]
    # Select a fixed replay time without consulting fault_active.  Labels remain
    # evaluator-only and are used for training/metrics, never inference routing.
    selected_index = min(indices, key=lambda index: abs(scaled["test"].end_times_s[index] - 200.0))
    chains = {}
    predictions = {}
    for family, (det_values, component_values, iso_values) in family_predictions.items():
        prediction = make_prediction(
            family=family, sample=selected, window=scaled["test"].x[selected_index],
            class_names=class_names, component_names=component_names,
            det_score=float(det_values[selected_index]),
            component_prob=component_values[selected_index],
            iso_prob=iso_values[selected_index], episode_id=f"preflight-{family}",
        )
        predictions[family] = prediction.model_dump(mode="json")
        chains[f"{family}_rule"] = run_chain(
            dataset_root=args.dataset_root, sample=selected, prediction=prediction,
            output_dir=args.output_dir / "chains" / family,
            reasoner="rule", vllm_url=args.vllm_url, vllm_model=args.vllm_model,
            dor_executable=args.dor_executable,
        )
    if not args.skip_llm:
        deep_prediction = make_prediction(
            family="deep_learning", sample=selected, window=scaled["test"].x[selected_index],
            class_names=class_names, component_names=component_names,
            det_score=float(family_predictions["deep_learning"][0][selected_index]),
            component_prob=family_predictions["deep_learning"][1][selected_index],
            iso_prob=family_predictions["deep_learning"][2][selected_index], episode_id="preflight-deep-llm",
        )
        chains["deep_learning_llm"] = run_chain(
            dataset_root=args.dataset_root, sample=selected, prediction=deep_prediction,
            output_dir=args.output_dir / "chains" / "deep_learning",
            reasoner="llm", vllm_url=args.vllm_url, vllm_model=args.vllm_model,
            dor_executable=args.dor_executable,
        )

    import platform
    import sklearn
    import torch
    report = {
        "schema_version": "astrograph-adcs-v522-regression-report.v1",
        "generated_at": "2026-08-12T00:00:00+08:00",
        "dataset_root": str(args.dataset_root.resolve()),
        "split_audit": split_audit,
        "window_protocol": {"length_s": WINDOW, "sampling_hz": 1, "train_stride_s": TRAIN_STRIDE,
                            "validation_stride_s": TRAIN_STRIDE, "test_stride_s": EVAL_STRIDE,
                            "label": "window_endpoint", "imputer_fit": "train_only_median",
                            "scaler_fit": "train_only"},
        "window_counts": {name: len(item.x) for name, item in scaled.items()},
        "feature_columns": list(FEATURE_COLUMNS), "class_names": class_names,
        "component_names": component_names,
        "fault_to_component": FAULT_TO_COMPONENT,
        "diagnosis_hierarchy": str(DIAGNOSIS_HIERARCHY.resolve()),
        "diagnosis_hierarchy_sha256": hashlib.sha256(DIAGNOSIS_HIERARCHY.read_bytes()).hexdigest(),
        "inference_contract": {
            "stages": ["fault_detection", "component_localization", "component_conditioned_classification"],
            "routing": "predicted_component",
            "oracle_component_usage": "offline_metrics_only",
            "component_truth_in_model_input": False,
        },
        "class_to_registered_fault": CLASS_TO_REGISTERED,
        "runtime_evidence_package": str(EVIDENCE_PACKAGE.resolve()),
        "runtime_evidence_package_sha256": hashlib.sha256(EVIDENCE_PACKAGE.read_bytes()).hexdigest(),
        "dor_semantic_audit": {
            "package": str(DOR_SEMANTIC_PACKAGE.resolve()),
            "package_sha256": hashlib.sha256(DOR_SEMANTIC_PACKAGE.read_bytes()).hexdigest(),
            "executable": str(args.dor_executable),
            "role": "independent semantic qualification and conservative release gate; not a BiRRF ranker",
            "root_cause_enabled": False,
        },
        "runtime_source_schema_sha256": json.loads(
            EVIDENCE_PACKAGE.read_text(encoding="utf-8")
        )["source_schema_sha256"],
        "fault_class_map": str(FAULT_CLASS_MAP.resolve()),
        "fault_class_map_sha256": hashlib.sha256(FAULT_CLASS_MAP.read_bytes()).hexdigest(),
        "llm_execution": {
            "requested": not args.skip_llm,
            "status": "NOT_EXECUTED" if args.skip_llm else "EXECUTED",
            "reason": args.skip_llm_reason if args.skip_llm else None,
        },
        "models": model_outputs, "selected_chain_sample": selected.sample_id,
        "abstract_fault_ontology": {
            "path": str(GENERIC_FAULT_ONTOLOGY),
            "binding_path": str(FAULT_ONTOLOGY_BINDINGS),
            "shape_report": abstract_graph_report.model_dump(mode="json"),
        },
        "selected_chain_window_end_s": float(scaled["test"].end_times_s[selected_index]),
        "predictions": predictions, "chains": chains,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "torch": torch.__version__, "torch_cuda_available": torch.cuda.is_available(),
                        "vllm_url": args.vllm_url, "vllm_model": args.vllm_model},
        "claim_boundary": "Single-seed engineering regression on synthetic data; not model selection or flight evidence.",
    }
    json_path = args.output_dir / "adcs_v522_regression_report.json"
    md_path = args.output_dir / "adcs_v522_regression_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[report] {json_path}")
    print(f"[report] {md_path}")
    successful_statuses = {"SUCCEEDED", "completed"}
    return 0 if all(value.get("status") in successful_statuses for value in chains.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
