#!/usr/bin/env python3
"""Unknown-onset causal-window three-stage screening for PHM v6.0.

Feature selection, imputation, scaling, checkpoint selection, detector-source
selection and hierarchy blending are learned from train/validation only.  The
test split is evaluated once after the complete selection state is frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.phm_v60 import audit_split, is_allowed_channel, load_records, metric_bundle
from astrograph.data_driven.evaluation_protocol import three_task_evaluation_audit
from astrograph.data_driven.preprocessing import fit_channel_preprocessor
from astrograph.data_driven.alarm_policy import (
    AlarmPolicy, alarm_event_metrics, apply_alarm_policy, candidate_alarm_policies,
)
from astrograph.data_driven.three_task_tcn import (
    hierarchy_js_loss,
    make_causal_multiview_detector_tcn,
    make_decoupled_tcn,
    make_hierarchy_event_tcn,
    make_hierarchical_mechanism_gat_tcn,
    make_mechanism_gated_gat_tcn,
    prepare_torch_device,
    task_gradient_statistics,
    trajectory_uniform_weights,
    mean_plus_cvar,
)
from astrograph.data_driven.mechanism_graph import (
    load_hierarchical_mechanism_graph,
    load_mechanism_graph,
)
from astrograph.data_driven.sota_backbones import (
    SOTA_MODEL_FAMILIES, make_sota_multitask_model,
)

DEFAULT_DATASET = Path(os.environ.get(
    "ASTROGRAPH_PHM_V60_DATA_ROOT",
    str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0"),
))
HIERARCHY = ROOT / "configs/datasets/phm_v6_0/diagnosis_hierarchy.json"
MECHANISM_GRAPH_CONFIG = ROOT / "configs/data_driven/mechanism_graphs.json"
HIERARCHICAL_GRAPH_CONFIG = ROOT / "configs/data_driven/hierarchical_mechanism_graphs.json"
WINDOW = 60
TRAIN_STRIDE = 5
EVAL_STRIDE = 1


@dataclass
class WindowSet:
    x: np.ndarray
    detection: np.ndarray
    component: np.ndarray
    isolation: np.ndarray
    sample_ids: list[str]
    end_times_s: np.ndarray
    onset_times_s: np.ndarray
    injection_detection: np.ndarray
    injection_onset_times_s: np.ndarray


def supervision_active(
    record: Any,
    times: np.ndarray,
    event_active: np.ndarray,
    label_contract: str,
) -> np.ndarray:
    """Return supervision state without exposing evaluator timestamps as inputs."""
    if label_contract == "event_active":
        return event_active
    if label_contract != "manifestation_aware":
        raise ValueError(f"unknown PHM label contract: {label_contract}")
    if record.condition_id == "nominal":
        return np.zeros(len(times), dtype=bool)
    if not math.isfinite(record.fault_manifestation_time_s):
        raise ValueError(f"fault record lacks manifestation time: {record.sample_id}")
    return times >= record.fault_manifestation_time_s


def select_train_channels(
    records: list[Any], class_names: list[str], limit: int,
    *, required_channels: list[str] | tuple[str, ...] = (),
    label_contract: str = "event_active",
) -> tuple[list[str], np.ndarray]:
    """Select raw telemetry channels from train-only, sparsely sampled rows."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer

    first = pd.read_csv(records[0].telemetry_path, nrows=1)
    candidates = [str(name) for name in first.columns if is_allowed_channel(str(name))]
    missing_required = sorted(set(required_channels).difference(candidates))
    if missing_required:
        raise ValueError(f"required mechanism channels are missing: {missing_required}")
    if len(required_channels) > limit:
        raise ValueError("required mechanism channels exceed the channel limit")
    rows, labels = [], []
    class_index = {name: index + 1 for index, name in enumerate(class_names)}
    for number, record in enumerate((row for row in records if row.split == "train"), start=1):
        full_frame = pd.read_csv(
            record.telemetry_path, usecols=["time_s", *candidates], low_memory=False
        )
        frame = full_frame[candidates].iloc[::TRAIN_STRIDE]
        truth = pd.read_csv(record.telemetry_path.parent / "labels.csv", usecols=["event_active"])
        event_active = truth["event_active"].astype(bool).to_numpy()
        times = pd.to_numeric(full_frame["time_s"], errors="coerce").to_numpy(dtype=float)
        active = supervision_active(
            record, times, event_active, label_contract
        )[::TRAIN_STRIDE]
        rows.append(frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
        active_class = class_index.get(record.condition_id, 0)
        labels.append(np.where(active, active_class, 0))
        if number % 100 == 0:
            print(f"[channel-selection] {number} train traces", flush=True)
    values = np.concatenate(rows)
    targets = np.concatenate(labels)
    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(values)
    usable = np.isfinite(filled).all(0) & (np.nanstd(filled, axis=0) > 1e-10)
    usable_names = [name for name, keep in zip(candidates, usable, strict=True) if keep]
    filled = filled[:, usable]
    unusable_required = sorted(set(required_channels).difference(usable_names))
    if unusable_required:
        raise ValueError(f"required mechanism channels are unusable on Train: {unusable_required}")
    remaining_names = [name for name in usable_names if name not in required_channels]
    remaining_positions = [usable_names.index(name) for name in remaining_names]
    selected = list(required_channels)
    remaining_limit = min(limit - len(selected), len(remaining_names))
    if remaining_limit:
        selector = SelectKBest(f_classif, k=remaining_limit)
        selector.fit(filled[:, remaining_positions], targets)
        selected.extend(
            name for name, keep in zip(remaining_names, selector.get_support(), strict=True) if keep
        )
    # Store per-channel train medians; the window scaler is fitted separately below.
    positions = [candidates.index(name) for name in selected]
    medians = np.asarray(imputer.statistics_)[positions].astype(np.float32)
    return selected, medians


def build_windows(
    records: list[Any],
    selected: list[str],
    class_names: list[str],
    component_names: list[str],
    fault_to_component: dict[str, str],
    *,
    label_contract: str = "event_active",
) -> dict[str, WindowSet]:
    class_index = {name: index for index, name in enumerate(class_names)}
    component_index = {name: index for index, name in enumerate(component_names)}
    buffers = {
        name: {
            "x": [], "d": [], "c": [], "i": [], "ids": [], "times": [],
            "onsets": [], "injection_d": [], "injection_onsets": [],
        }
        for name in ("train", "validation", "test")
    }
    for number, record in enumerate(records, start=1):
        frame = pd.read_csv(record.telemetry_path, usecols=["time_s", *selected], low_memory=False)
        values = frame[selected].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
        labels = pd.read_csv(record.telemetry_path.parent / "labels.csv", usecols=["event_active"])
        event_active = labels["event_active"].astype(bool).to_numpy()
        if len(event_active) != len(values):
            raise ValueError(f"telemetry/label length mismatch: {record.sample_id}")
        active = supervision_active(record, times, event_active, label_contract)
        active_indices = np.flatnonzero(active)
        onset = float(times[active_indices[0]]) if len(active_indices) else math.nan
        injection_indices = np.flatnonzero(event_active)
        injection_onset = (
            float(times[injection_indices[0]]) if len(injection_indices) else math.nan
        )
        stride = TRAIN_STRIDE if record.split in {"train", "validation"} else EVAL_STRIDE
        target = buffers[record.split]
        for end in range(WINDOW - 1, len(values), stride):
            is_active = bool(active[end])
            target["x"].append(values[end - WINDOW + 1 : end + 1])
            target["d"].append(int(is_active))
            target["c"].append(
                component_index[fault_to_component[record.condition_id]] if is_active else -1
            )
            target["i"].append(class_index[record.condition_id] if is_active else -1)
            target["ids"].append(record.sample_id)
            target["times"].append(float(times[end]))
            target["onsets"].append(onset)
            target["injection_d"].append(int(event_active[end]))
            target["injection_onsets"].append(injection_onset)
        if number % 100 == 0:
            print(f"[windows] {number}/{len(records)} traces", flush=True)
    return {
        name: WindowSet(
            x=np.asarray(item["x"], dtype=np.float32),
            detection=np.asarray(item["d"], dtype=np.int64),
            component=np.asarray(item["c"], dtype=np.int64),
            isolation=np.asarray(item["i"], dtype=np.int64),
            sample_ids=item["ids"],
            end_times_s=np.asarray(item["times"], dtype=float),
            onset_times_s=np.asarray(item["onsets"], dtype=float),
            injection_detection=np.asarray(item["injection_d"], dtype=np.int64),
            injection_onset_times_s=np.asarray(item["injection_onsets"], dtype=float),
        )
        for name, item in buffers.items()
    }


def scale_windows(
    windows: dict[str, WindowSet], fallback_median: np.ndarray, *, method: str = "standard",
    return_state: bool = False,
) -> Any:
    state = fit_channel_preprocessor(
        windows["train"].x, method=method, fallback_impute=fallback_median
    )
    result = {}
    for name, item in windows.items():
        result[name] = WindowSet(
            x=state.transform(item.x), detection=item.detection,
            component=item.component, isolation=item.isolation, sample_ids=item.sample_ids,
            end_times_s=item.end_times_s, onset_times_s=item.onset_times_s,
            injection_detection=item.injection_detection,
            injection_onset_times_s=item.injection_onset_times_s,
        )
    return (result, state) if return_state else result


def summary_features(x: np.ndarray, *, profile: str = "legacy_summary") -> np.ndarray:
    legacy = (x.mean(1), x.std(1), x[:, -1], x[:, -1] - x[:, 0])
    if profile == "legacy_summary":
        return np.concatenate(legacy, axis=1)
    if profile != "change_summary":
        raise ValueError(f"unknown detector feature profile: {profile}")
    delta = np.diff(x, axis=1, prepend=x[:, :1])
    anchor_length = max(4, x.shape[1] // 4)
    anchor = x[:, :anchor_length]
    anchor_center = anchor.mean(1)
    anchor_scale = np.maximum(anchor.std(1), 0.25)
    relative_last = (x[:, -1] - anchor_center) / anchor_scale
    return np.concatenate((*legacy, delta.mean(1), delta.std(1),
                           np.abs(delta).mean(1), relative_last), axis=1)


def logistic_detector(
    train: WindowSet, data: WindowSet, seed: int, *, feature_profile: str = "legacy_summary"
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(summary_features(train.x, profile=feature_profile), train.detection)
    probability = model.predict_proba(summary_features(data.x, profile=feature_profile))
    return probability[:, list(model.classes_).index(1)]


def normalize_detection_scores(
    scores: np.ndarray,
    sample_ids: list[str],
    end_times_s: np.ndarray,
    *,
    profile: str = "none",
    baseline_duration_s: float = 20.0,
) -> np.ndarray:
    """Apply a causal episode baseline without reading event labels or onset."""
    values = np.asarray(scores, dtype=float)
    if profile == "none":
        return values
    if profile != "initial_logit_delta":
        raise ValueError(f"unknown detection score normalization: {profile}")
    if not (len(values) == len(sample_ids) == len(end_times_s)):
        raise ValueError("score normalization inputs must have equal length")
    if baseline_duration_s < 0.0:
        raise ValueError("baseline duration must be nonnegative")
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped))
    normalized = np.empty_like(values)
    for sample_id in sorted(set(sample_ids)):
        positions = np.asarray(
            [index for index, value in enumerate(sample_ids) if value == sample_id], dtype=int
        )
        order = positions[np.argsort(end_times_s[positions], kind="stable")]
        first_time = float(end_times_s[order[0]])
        baseline = order[end_times_s[order] <= first_time + baseline_duration_s]
        reference = float(np.median(logits[baseline]))
        residual = np.clip(logits[order] - reference, -20.0, 20.0)
        normalized[order] = 1.0 / (1.0 + np.exp(-residual))
    return normalized


def weights(values: np.ndarray, count: int):
    import torch

    frequency = np.bincount(values[values >= 0], minlength=count).astype(float)
    result = np.sqrt(frequency.sum() / np.maximum(frequency * count, 1.0))
    return torch.tensor(result / result.mean(), dtype=torch.float32)


def phm_subsystem_groups(
    selected_channels: list[str], component_names: list[str], class_names: list[str],
    fault_to_component: dict[str, str],
) -> tuple[list[list[int]], list[int], list[int], list[str]]:
    """Bind telemetry families to deployable spacecraft subsystem roles."""
    group_names = ["adcs", "eps", "payload", "comm_data", "thermal", "propulsion"]

    def channel_group(name: str) -> str:
        prefix = name.split(".", 1)[0]
        return "comm_data" if prefix in {"comm", "data"} else prefix

    component_group_name = {
        "battery": "eps", "bus_load": "eps", "pdu": "eps", "solar_array": "eps",
        "downlink": "comm_data", "onboard_storage": "comm_data", "transmitter": "comm_data",
        "payload_instrument": "payload", "fuel_tank": "propulsion", "thruster": "propulsion",
        "heater": "thermal", "thermal_node": "thermal", "radiator": "thermal",
    }
    groups = [
        [index for index, name in enumerate(selected_channels) if channel_group(name) == group]
        for group in group_names
    ]
    if any(not values for values in groups):
        raise ValueError(f"every PHM subsystem requires selected observable channels: {dict(zip(group_names, groups, strict=True))}")
    group_index = {name: index for index, name in enumerate(group_names)}
    component_group = [group_index[component_group_name[name]] for name in component_names]
    component_index = {name: index for index, name in enumerate(component_names)}
    class_component = [component_index[fault_to_component[name]] for name in class_names]
    return groups, component_group, class_component, group_names


def make_model(
    kind: str, channels: int, components: int, classes: int, *,
    subsystem_groups: list[list[int]] | None = None,
    component_group: list[int] | None = None,
    class_component: list[int] | None = None,
    graph_edges: list[tuple[int, int]] | None = None,
    hierarchical_contract: dict[str, Any] | None = None,
):
    import torch
    from torch import nn

    if kind in SOTA_MODEL_FAMILIES:
        return make_sota_multitask_model(
            kind, channels, components, classes, window_size=WINDOW
        )

    class Block(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, width), nn.GELU(),
            )

        def forward(self, x):
            return x + self.net(x)

    class TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Conv1d(channels, 64, 1)
            self.blocks = nn.Sequential(*(Block(64, value) for value in (1, 2, 4)))

        def forward(self, x):
            h = self.blocks(self.input(x.transpose(1, 2)))
            return 0.5 * (h.mean(-1) + h[:, :, -1])

    class GRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(channels, 64, num_layers=2, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.norm(0.5 * (h.mean(1) + h[:, -1]))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = {"residual_tcn": TCN, "gru": GRU}[kind]()
            self.detector = nn.Linear(64, 2)
            self.localizer = nn.Linear(64, components)
            self.classifier = nn.Linear(64, classes)

        def forward(self, x):
            h = self.encoder(x)
            return self.detector(h), self.localizer(h), self.classifier(h)

    class SubsystemAwareModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if subsystem_groups is None or component_group is None or class_component is None:
                raise ValueError("subsystem_tcn requires hierarchy bindings")
            self.global_input = nn.Conv1d(channels, 32, 1)
            self.global_blocks = nn.Sequential(*(Block(32, value) for value in (1, 2, 4)))
            self.local_inputs = nn.ModuleList(nn.Conv1d(len(indices), 16, 1) for indices in subsystem_groups)
            self.local_blocks = nn.ModuleList(
                nn.Sequential(*(Block(16, value) for value in (1, 2))) for _ in subsystem_groups
            )
            for index, values in enumerate(subsystem_groups):
                self.register_buffer(f"subsystem_index_{index}", torch.tensor(values, dtype=torch.long))
            self.detector = nn.Linear(32 + 16 * len(subsystem_groups), 2)
            self.localizer = nn.ModuleList(nn.Linear(48, 1) for _ in component_group)
            self.classifier = nn.ModuleList(nn.Linear(48, 1) for _ in class_component)
            self.component_group = tuple(component_group)
            self.class_component = tuple(class_component)

        @staticmethod
        def pool(value):
            return 0.5 * (value.mean(-1) + value[:, :, -1])

        def forward(self, x):
            temporal = x.transpose(1, 2)
            global_state = self.pool(self.global_blocks(self.global_input(temporal)))
            local_states = []
            for index, (projection, blocks) in enumerate(zip(self.local_inputs, self.local_blocks, strict=True)):
                channel_index = getattr(self, f"subsystem_index_{index}")
                local_states.append(self.pool(blocks(projection(temporal.index_select(1, channel_index)))))
            detection = self.detector(torch.cat([global_state, *local_states], dim=1))
            localization = torch.cat([
                head(torch.cat([global_state, local_states[self.component_group[index]]], dim=1))
                for index, head in enumerate(self.localizer)
            ], dim=1)
            classification = torch.cat([
                head(torch.cat([
                    global_state,
                    local_states[self.component_group[self.class_component[index]]],
                ], dim=1))
                for index, head in enumerate(self.classifier)
            ], dim=1)
            return detection, localization, classification

    if kind == "subsystem_tcn":
        return SubsystemAwareModel()
    if kind == "hierarchy_event_tcn":
        return make_hierarchy_event_tcn(
            channels, components, classes, window_size=WINDOW
        )
    if kind == "causal_multiview_detector_tcn":
        return make_causal_multiview_detector_tcn(
            channels, components, classes, window_size=WINDOW
        )
    if kind == "mechanism_gated_gat_tcn":
        if subsystem_groups is None or component_group is None or class_component is None:
            raise ValueError("mechanism_gated_gat_tcn requires hierarchy bindings")
        return make_mechanism_gated_gat_tcn(
            channels, components, classes,
            subsystem_groups=subsystem_groups,
            component_group=component_group,
            class_component=class_component,
            graph_edges=graph_edges or [],
            temporal_norm_groups=8,
        )
    if kind == "hierarchical_mechanism_gat_tcn":
        if hierarchical_contract is None or class_component is None:
            raise ValueError("hierarchical_mechanism_gat_tcn requires hierarchy bindings")
        return make_hierarchical_mechanism_gat_tcn(
            channels, components, classes,
            component_groups=hierarchical_contract["component_groups"],
            subsystem_groups=hierarchical_contract["subsystem_groups"],
            component_subsystem=hierarchical_contract["component_subsystem"],
            class_component=class_component,
            subsystem_edges=hierarchical_contract["subsystem_edges"],
        )
    if kind in {"pooled_task_adapter_tcn", "late_fork_tcn", "tied_late_fork_tcn", "capacity_matched_shared_tcn"}:
        return make_decoupled_tcn(kind, channels, components, classes)
    return Model()


def infer(model: Any, data: WindowSet, batch_size: int = 1024, *, device: str = "cpu"):
    import torch

    outputs = [[], [], []]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.x), batch_size):
            logits = model(torch.from_numpy(data.x[start : start + batch_size]).to(device))
            for target, value in zip(outputs, logits, strict=True):
                target.append(torch.softmax(value, 1).cpu().numpy())
    return tuple(np.concatenate(value) for value in outputs)


def component_from_classes(
    probabilities: np.ndarray, class_names: list[str], component_names: list[str],
    fault_to_component: dict[str, str],
) -> np.ndarray:
    component_index = {name: index for index, name in enumerate(component_names)}
    result = np.zeros((len(probabilities), len(component_names)), dtype=float)
    for index, name in enumerate(class_names):
        result[:, component_index[fault_to_component[name]]] += probabilities[:, index]
    return result


def condition_classes(
    probabilities: np.ndarray, component_probability: np.ndarray, class_names: list[str],
    component_names: list[str], fault_to_component: dict[str, str], *, mode: str = "hard",
) -> np.ndarray:
    if mode == "soft":
        component_index = {name: index for index, name in enumerate(component_names)}
        weighted = probabilities.copy()
        for index, name in enumerate(class_names):
            weighted[:, index] *= component_probability[:, component_index[fault_to_component[name]]]
        return weighted / np.maximum(weighted.sum(1, keepdims=True), 1e-12)
    if mode != "hard":
        raise ValueError(f"unknown class conditioning mode: {mode}")
    result = np.zeros_like(probabilities)
    predicted_component = np.asarray(component_names)[component_probability.argmax(1)]
    for row, component in enumerate(predicted_component):
        allowed = [index for index, name in enumerate(class_names) if fault_to_component[name] == component]
        values = probabilities[row, allowed]
        result[row, allowed] = values / values.sum() if values.sum() else 1.0 / len(allowed)
    return result


def macro_f1(truth: np.ndarray, predicted: np.ndarray, labels: list[int]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0))


def select_validation_state(
    deep: tuple[np.ndarray, np.ndarray, np.ndarray], logistic: np.ndarray, data: WindowSet,
    class_names: list[str], component_names: list[str], fault_to_component: dict[str, str],
    *, alarm_policy_mode: str = "point",
    score_normalization_profiles: tuple[str, ...] = ("none",),
    baseline_duration_s: float = 20.0,
    fixed_detection_threshold: float | None = None,
) -> dict[str, Any]:
    deep_detection, direct_component, classes = deep
    derived = component_from_classes(classes, class_names, component_names, fault_to_component)
    active = data.isolation >= 0
    best_detection: dict[str, Any] = {"detection_macro_f1": -1.0}
    for detector_name, raw_detection_score in (("deep", deep_detection[:, 1]), ("logistic", logistic)):
        for normalization in score_normalization_profiles:
            detection_score = normalize_detection_scores(
                raw_detection_score, data.sample_ids, data.end_times_s,
                profile=normalization, baseline_duration_s=baseline_duration_s,
            )
            # A point alarm still requires train/Validation-only probability
            # calibration; fixing it at 0.5 silently assumes calibrated scores.
            thresholds = (
                (float(fixed_detection_threshold),)
                if fixed_detection_threshold is not None else
                (0.3, 0.4, 0.5, 0.6, 0.7)
            )
            for policy in candidate_alarm_policies(alarm_policy_mode, thresholds):
                alarm = np.asarray(apply_alarm_policy(
                    detection_score.tolist(), data.sample_ids, data.end_times_s.tolist(), policy,
                ))
                detection_f1 = macro_f1(data.detection, alarm, [0, 1])
                if detection_f1 > best_detection["detection_macro_f1"]:
                    best_detection = {
                        "detector": detector_name,
                        "detection_score_normalization": normalization,
                        "detection_baseline_duration_s": baseline_duration_s,
                        "detection_threshold": policy.on_threshold,
                        "alarm_on_threshold": policy.on_threshold,
                        "alarm_off_threshold": policy.off_threshold,
                        "alarm_minimum_consecutive": policy.minimum_consecutive,
                        "detection_macro_f1": detection_f1,
                    }
    best_diagnosis: dict[str, Any] = {"diagnosis_score": -1.0}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        component = alpha * direct_component + (1.0 - alpha) * derived
        conditioned = condition_classes(
            classes, component, class_names, component_names, fault_to_component, mode="hard",
        )
        localization_f1 = macro_f1(
            data.component[active], component[active].argmax(1), list(range(len(component_names)))
        )
        classification_f1 = macro_f1(
            data.isolation[active], conditioned[active].argmax(1), list(range(len(class_names)))
        )
        diagnosis_score = 0.3 * localization_f1 + 0.5 * classification_f1
        if diagnosis_score > best_diagnosis["diagnosis_score"]:
            best_diagnosis = {
                "conditioning_mode": "hard",
                "hierarchy_blend_alpha": alpha,
                "localization_macro_f1": localization_f1,
                "classification_macro_f1": classification_f1,
                "diagnosis_score": diagnosis_score,
            }
    result = {**best_detection, **best_diagnosis}
    result.pop("diagnosis_score")
    result["selection_score"] = (
        0.2 * result["detection_macro_f1"]
        + 0.3 * result["localization_macro_f1"]
        + 0.5 * result["classification_macro_f1"]
    )
    return result


def delays(
    data: WindowSet,
    score: np.ndarray,
    threshold: float = 0.5,
    predicted: np.ndarray | None = None,
) -> dict[str, Any]:
    alarms = (score >= threshold) if predicted is None else np.asarray(predicted, dtype=bool)
    if alarms.shape != score.shape:
        raise ValueError("predicted alarms must match scores shape")
    onset, first = {}, {}
    for sample, end, start, alarm in zip(
        data.sample_ids, data.end_times_s, data.onset_times_s, alarms, strict=True
    ):
        if math.isnan(start):
            continue
        onset[sample] = start
        if end >= start and alarm and sample not in first:
            first[sample] = end
    values = [first[name] - start for name, start in onset.items() if name in first]
    return {
        "eligible_fault_traces": len(onset), "detected_fault_traces": len(values),
        "coverage": len(values) / len(onset) if onset else 0.0,
        "mean_delay_s": float(np.mean(values)) if values else None,
        "median_delay_s": float(np.median(values)) if values else None,
        "p95_delay_s": float(np.percentile(values, 95)) if values else None,
    }


def evaluate_test(
    model: Any, data: WindowSet, logistic: np.ndarray, selection: dict[str, Any],
    class_names: list[str], component_names: list[str], fault_to_component: dict[str, str],
    *, device: str = "cpu",
    detection_model: Any | None = None,
    detection_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnosis_detection, direct, classes = infer(model, data, device=device)
    deep_detection = (
        infer(detection_model, data, device=device)[0]
        if detection_model is not None else diagnosis_detection
    )
    if detection_selection is not None:
        selection = {
            **selection,
            **{
                key: detection_selection[key]
                for key in (
                    "detector", "detection_threshold", "alarm_on_threshold",
                    "alarm_off_threshold", "alarm_minimum_consecutive",
                    "detection_score_normalization", "detection_baseline_duration_s",
                )
            },
        }
    derived = component_from_classes(classes, class_names, component_names, fault_to_component)
    alpha = float(selection["hierarchy_blend_alpha"])
    component = alpha * direct + (1.0 - alpha) * derived
    conditioned = condition_classes(
        classes, component, class_names, component_names, fault_to_component,
        mode=str(selection["conditioning_mode"]),
    )
    raw_detection_score = logistic if selection["detector"] == "logistic" else deep_detection[:, 1]
    detection_score = normalize_detection_scores(
        raw_detection_score, data.sample_ids, data.end_times_s,
        profile=str(selection.get("detection_score_normalization", "none")),
        baseline_duration_s=float(selection.get("detection_baseline_duration_s", 20.0)),
    )
    threshold = float(selection["detection_threshold"])
    alarm_policy = AlarmPolicy(
        on_threshold=float(selection["alarm_on_threshold"]),
        off_threshold=float(selection["alarm_off_threshold"]),
        minimum_consecutive=int(selection["alarm_minimum_consecutive"]),
    )
    detection_pred = np.asarray(apply_alarm_policy(
        detection_score.tolist(), data.sample_ids, data.end_times_s.tolist(), alarm_policy,
    ))
    active = data.isolation >= 0
    component_pred = component.argmax(1)
    class_pred = conditioned.argmax(1)
    cascade_truth = np.where(data.detection == 0, 0, data.isolation + 1)
    cascade_pred = np.where(detection_pred == 0, 0, class_pred + 1)
    injection_detection_f1 = macro_f1(
        data.injection_detection, detection_pred, [0, 1]
    )
    injection_delay_data = copy.copy(data)
    injection_delay_data.onset_times_s = data.injection_onset_times_s
    return {
        "detection": {"macro_f1": macro_f1(data.detection, detection_pred, [0, 1])},
        "injection_awareness_detection": {"macro_f1": injection_detection_f1},
        "localization": {"macro_f1": macro_f1(
            data.component[active], component_pred[active], list(range(len(component_names)))
        )},
        "classification_predicted_component": {"macro_f1": macro_f1(
            data.isolation[active], class_pred[active], list(range(len(class_names)))
        )},
        "classification_unconditioned": {"macro_f1": macro_f1(
            data.isolation[active], classes[active].argmax(1), list(range(len(class_names)))
        )},
        "end_to_end_18_class": {"macro_f1": macro_f1(
            cascade_truth, cascade_pred, list(range(len(class_names) + 1)
        ))},
        "delay": delays(data, detection_score, threshold, predicted=detection_pred),
        "injection_referenced_delay": delays(
            injection_delay_data, detection_score, threshold, predicted=detection_pred
        ),
        "alarm_event_metrics": alarm_event_metrics(
            data.detection.tolist(), detection_pred.tolist(), data.sample_ids,
            data.end_times_s.tolist(), data.onset_times_s.tolist(),
        ),
        "detection_threshold": threshold,
        "alarm_policy": alarm_policy.to_dict(),
        "detector": selection["detector"],
        "detection_score_normalization": selection.get("detection_score_normalization", "none"),
        "conditioning_mode": selection["conditioning_mode"],
        "failure_attribution": {
            "fault_windows": int(active.sum()),
            "first_wrong_stage_detection": int(np.sum(active & (detection_pred == 0))),
            "first_wrong_stage_localization": int(np.sum(
                active & (detection_pred == 1) & (component_pred != data.component)
            )),
            "fully_correct_fault_windows": int(np.sum(
                active & (detection_pred == 1) & (class_pred == data.isolation)
            )),
        },
        "evaluation_audit": three_task_evaluation_audit(
            group_ids=data.sample_ids,
            detection_truth=data.detection.tolist(),
            detection_predicted=detection_pred.tolist(),
            detection_positive_scores=detection_score.tolist(),
            component_truth=data.component.tolist(),
            component_predicted=component_pred.tolist(),
            class_truth=data.isolation.tolist(),
            class_predicted=class_pred.tolist(),
            component_labels=component_names,
            class_labels=class_names,
            group_key="sample_id",
        ),
    }


def train_one(
    kind: str, train: WindowSet, validation: WindowSet, test: WindowSet,
    class_names: list[str], component_names: list[str], fault_to_component: dict[str, str],
    seed: int, epochs: int, selected_channels: list[str] | None = None,
    device: str = "cpu",
    batch_size: int = 384,
    minimum_epochs: int = 0,
    early_stop_patience: int | None = None,
    evaluate_test_split: bool = True,
    gradient_audit: bool = False,
    alarm_policy_mode: str = "point",
    label_contract: str = "event_active",
    training_sampling: str = "window_shuffle",
    task_scope: str = "joint",
    detector_feature_profile: str = "legacy_summary",
    detection_score_normalization_profiles: tuple[str, ...] = ("none",),
    detection_baseline_duration_s: float = 20.0,
    diagnosis_auxiliary_weight: float = 0.2,
    fixed_detection_threshold: float | None = None,
    nominal_trajectory_weight: float = 1.0,
    detection_cvar_weight: float = 0.0,
    detection_cvar_fraction: float = 0.2,
    return_runtime: bool = False,
) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch_device = prepare_torch_device(device, seed)
    if task_scope not in {"joint", "detection", "diagnosis", "detection_aux"}:
        raise ValueError(f"unknown task scope: {task_scope}")
    if diagnosis_auxiliary_weight < 0.0:
        raise ValueError("diagnosis auxiliary weight must be nonnegative")
    if nominal_trajectory_weight <= 0.0:
        raise ValueError("nominal trajectory weight must be positive")
    if detection_cvar_weight < 0.0 or not 0.0 < detection_cvar_fraction <= 1.0:
        raise ValueError("invalid detection CVaR contract")
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    validation_logistic = logistic_detector(
        train, validation, seed, feature_profile=detector_feature_profile
    )
    test_logistic = (
        logistic_detector(train, test, seed, feature_profile=detector_feature_profile)
        if evaluate_test_split else None
    )
    subsystem_groups = component_group = None
    component_index = {name: index for index, name in enumerate(component_names)}
    class_component = [component_index[fault_to_component[name]] for name in class_names]
    group_names: list[str] | None = None
    graph_edges: list[tuple[int, int]] = []
    graph_receipt = None
    hierarchical_contract = None
    if kind in {"subsystem_tcn", "mechanism_gated_gat_tcn"}:
        if selected_channels is None:
            raise ValueError(f"{kind} requires selected channel names")
        subsystem_groups, component_group, class_component, group_names = phm_subsystem_groups(
            selected_channels, component_names, class_names, fault_to_component
        )
    if kind == "mechanism_gated_gat_tcn":
        graph_edges, graph_receipt = load_mechanism_graph(
            MECHANISM_GRAPH_CONFIG, "phm_v6_0", group_names or []
        )
    if kind == "hierarchical_mechanism_gat_tcn":
        if selected_channels is None:
            raise ValueError(f"{kind} requires selected channel names")
        hierarchical_contract = load_hierarchical_mechanism_graph(
            HIERARCHICAL_GRAPH_CONFIG, "phm_v6_0", selected_channels, component_names
        )
        graph_receipt = {
            key: value for key, value in hierarchical_contract.items()
            if key not in {"component_groups", "subsystem_groups", "component_subsystem", "subsystem_edges"}
        }
    model = make_model(
        kind, train.x.shape[-1], len(component_names), len(class_names),
        subsystem_groups=subsystem_groups, component_group=component_group,
        class_component=class_component, graph_edges=graph_edges,
        hierarchical_contract=hierarchical_contract,
    ).to(device)
    # Equalize optimization-time stochasticity across differently sized models.
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    dataset = TensorDataset(
        torch.from_numpy(train.x), torch.from_numpy(train.detection),
        torch.from_numpy(train.component), torch.from_numpy(train.isolation),
    )
    generator = torch.Generator().manual_seed(seed)
    if training_sampling == "trajectory_uniform":
        sample_weights = torch.as_tensor(
            trajectory_uniform_weights(
                train.sample_ids, nominal_multiplier=nominal_trajectory_weight
            ),
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train.x), replacement=True, generator=generator
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            pin_memory=device.startswith("cuda"),
        )
    elif training_sampling == "window_shuffle":
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator,
            pin_memory=device.startswith("cuda"),
        )
    else:
        raise ValueError(f"unknown training sampling mode: {training_sampling}")
    detection_class_weights = weights(train.detection, 2).to(device)
    criteria = (
        nn.CrossEntropyLoss(weight=weights(train.component, len(component_names)).to(device)),
        nn.CrossEntropyLoss(weight=weights(train.isolation, len(class_names)).to(device)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state, best_selection, best_epoch = copy.deepcopy(model.state_dict()), {"selection_score": -1.0}, 0
    history, gradient_history, started = [], [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        epoch_gradient = None
        for x, detection, component, isolation in loader:
            x = x.to(device, non_blocking=True)
            detection = detection.to(device, non_blocking=True)
            component = component.to(device, non_blocking=True)
            isolation = isolation.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            dlogits, llogits, clogits = model(x)
            active = isolation >= 0
            task_losses = []
            if task_scope in {"joint", "detection", "detection_aux"}:
                per_window_detection = nn.functional.cross_entropy(
                    dlogits, detection, weight=detection_class_weights, reduction="none"
                )
                task_losses.append(mean_plus_cvar(
                    per_window_detection,
                    cvar_weight=detection_cvar_weight,
                    cvar_fraction=detection_cvar_fraction,
                ))
            if task_scope in {"joint", "diagnosis", "detection_aux"} and active.any():
                diagnosis_scale = diagnosis_auxiliary_weight if task_scope == "detection_aux" else 1.0
                task_losses.extend([
                    diagnosis_scale * criteria[0](llogits[active], component[active]),
                    diagnosis_scale * criteria[1](clogits[active], isolation[active]),
                ])
                consistency_weight = float(getattr(model, "consistency_weight", 0.0))
                if consistency_weight:
                    task_losses.append(diagnosis_scale * consistency_weight * hierarchy_js_loss(
                        llogits, clogits, class_component, active
                    ))
                if getattr(model, "last_subsystem_logits", None) is not None:
                    component_owner = torch.as_tensor(
                        model.component_subsystem, dtype=torch.long, device=device
                    )
                    subsystem_target = component_owner[component[active]]
                    task_losses.append(
                        diagnosis_scale * 0.2 * nn.functional.cross_entropy(
                            model.last_subsystem_logits[active], subsystem_target
                        )
                    )
            regularization = (
                model.graph_regularization_loss()
                if task_scope in {"joint", "diagnosis", "detection_aux"}
                and hasattr(model, "graph_regularization_loss") else 0.0
            )
            if not task_losses:
                task_losses.append(dlogits.sum() * 0.0)
            loss = sum(task_losses) + regularization
            if gradient_audit and epoch_gradient is None:
                epoch_gradient = task_gradient_statistics(task_losses, model.parameters())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach()) * len(x)
        scheduler.step()
        if epoch_gradient is not None:
            gradient_history.append({"epoch": epoch, **epoch_gradient})
        selection = select_validation_state(
            infer(model, validation, device=device), validation_logistic, validation,
            class_names, component_names, fault_to_component,
            alarm_policy_mode=alarm_policy_mode,
            score_normalization_profiles=detection_score_normalization_profiles,
            baseline_duration_s=detection_baseline_duration_s,
            fixed_detection_threshold=fixed_detection_threshold,
        )
        selection.update({"epoch": epoch, "train_loss": running / len(train.x)})
        selection["task_checkpoint_score"] = (
            selection["detection_macro_f1"]
            if task_scope in {"detection", "detection_aux"} else
            0.375 * selection["localization_macro_f1"]
            + 0.625 * selection["classification_macro_f1"]
            if task_scope == "diagnosis" else
            selection["selection_score"]
        )
        history.append(selection)
        print(
            f"[{kind}] epoch={epoch} loss={selection['train_loss']:.4f} "
            f"val={selection['selection_score']:.4f} det={selection['detector']} "
            f"alpha={selection['hierarchy_blend_alpha']}", flush=True,
        )
        best_task_score = best_selection.get("task_checkpoint_score", -1.0)
        if selection["task_checkpoint_score"] > best_task_score:
            best_state, best_selection, best_epoch = (
                copy.deepcopy(model.state_dict()), selection, epoch
            )
        if (
            early_stop_patience is not None
            and epoch >= max(1, minimum_epochs)
            and epoch - best_epoch >= early_stop_patience
        ):
            print(f"[{kind}] early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    run_metadata = {
        "family": kind,
        "seed": seed,
        "best_epoch": best_epoch,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "device": device,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch_device) if device.startswith("cuda") else None
        ),
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "minimum_epochs": minimum_epochs,
        "early_stop_patience": early_stop_patience,
        "epochs_completed": len(history),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch_device))
            if device.startswith("cuda") else None
        ),
        "receptive_field": getattr(model, "receptive_field", 29),
        "hierarchy_consistency_weight": float(getattr(model, "consistency_weight", 0.0)),
        "gradient_strategy": "sum",
        "gradient_audit": gradient_history,
        "label_contract": label_contract,
        "training_sampling": training_sampling,
        "task_scope": task_scope,
        "detector_feature_profile": detector_feature_profile,
        "detection_score_normalization_profiles": list(detection_score_normalization_profiles),
        "detection_baseline_duration_s": detection_baseline_duration_s,
        "diagnosis_auxiliary_weight": diagnosis_auxiliary_weight,
        "fixed_detection_threshold": fixed_detection_threshold,
        "nominal_trajectory_weight": nominal_trajectory_weight,
        "detection_cvar_weight": detection_cvar_weight,
        "detection_cvar_fraction": detection_cvar_fraction,
        "best_validation": best_selection,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "architecture": {
            "type": "global_plus_subsystem_causal_tcn" if kind == "subsystem_tcn" else kind,
            "subsystem_groups": group_names,
            "component_group_binding": component_group,
            "branching_mode": getattr(model, "branching_mode", "fully_shared"),
            "mechanism_graph": graph_receipt,
            "graph_diagnostics": (
                model.graph_diagnostics() if hasattr(model, "graph_diagnostics") else None
            ),
        },
    }
    if not evaluate_test_split:
        report = {
            **run_metadata,
            "evaluation_scope": "validation_only; test scores were not computed",
        }
        return (report, {"model": model, "validation_logistic": validation_logistic}) if return_runtime else report
    result = evaluate_test(
        model, test, test_logistic, best_selection,
        class_names, component_names, fault_to_component, device=device,
    )
    report = {
        **run_metadata,
        **result,
    }
    runtime = {"model": model, "test_logistic": test_logistic}
    return (report, runtime) if return_runtime else report


def render(report: dict[str, Any]) -> str:
    rows = []
    for item in report["models"]:
        rows.append(
            f"| {item['family']} / {item['seed']} | {item['best_validation']['detector']} | "
            f"{item['best_validation']['hierarchy_blend_alpha']} | {item['detection']['macro_f1']:.4f} | "
            f"{item['localization']['macro_f1']:.4f} | "
            f"{item['classification_predicted_component']['macro_f1']:.4f} | "
            f"{item['end_to_end_18_class']['macro_f1']:.4f} |"
        )
    return """# PHM v6.0 未知起点因果滑窗筛选

不读取 `activation_time_s` 构造输入。Train-only 通道选择、填补与标准化；Validation 选择 checkpoint、检测器来源和层级融合系数；Test 冻结后一次评价。

| 模型/种子 | Detector | hierarchy alpha | Detection F1 | Localization F1 | Classification F1 | E2E F1 |
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "scripts/integration/out/phm_v60_causal_window_comparison",
    )
    parser.add_argument("--models", nargs="+", choices=["residual_tcn", "gru", "subsystem_tcn", "hierarchy_event_tcn", "causal_multiview_detector_tcn", "pooled_task_adapter_tcn", "late_fork_tcn", "tied_late_fork_tcn", "capacity_matched_shared_tcn", "mechanism_gated_gat_tcn", "hierarchical_mechanism_gat_tcn", *SOTA_MODEL_FAMILIES], default=["inception_time", "timesnet", "modern_tcn", "residual_tcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260814])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channel-limit", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--minimum-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--gradient-audit", action="store_true")
    parser.add_argument("--preprocessing", choices=["standard", "robust_clip"], default="standard")
    parser.add_argument("--alarm-policy", choices=["point", "hysteresis"], default="point")
    parser.add_argument(
        "--label-contract", choices=["event_active", "manifestation_aware"],
        default="event_active",
    )
    parser.add_argument(
        "--training-sampling", choices=["window_shuffle", "trajectory_uniform"],
        default="window_shuffle",
    )
    parser.add_argument(
        "--detector-feature-profile", choices=["legacy_summary", "change_summary"],
        default="legacy_summary",
    )
    parser.add_argument(
        "--feature-profile", choices=["selected_only", "mechanism_required"],
        default="selected_only",
    )
    args = parser.parse_args()
    hierarchy = json.loads(HIERARCHY.read_text())
    records = load_records(args.dataset_root)
    split_audit = audit_split(records)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    class_names = sorted(hierarchy["fault_to_component"])
    component_names = sorted(set(hierarchy["fault_to_component"].values()))
    from astrograph.data_driven.mechanism_features import load_feature_bank, profile_columns

    bank = load_feature_bank(ROOT / "configs/data_driven/mechanism_feature_banks.json")
    required_channels = profile_columns(bank, "phm_v6_0", args.feature_profile)
    selected, median = select_train_channels(
        records, class_names, args.channel_limit, required_channels=required_channels,
        label_contract=args.label_contract,
    )
    windows = scale_windows(
        build_windows(
            records, selected, class_names, component_names, hierarchy["fault_to_component"],
            label_contract=args.label_contract,
        ),
        median, method=args.preprocessing,
    )
    models = [
        train_one(
            kind, windows["train"], windows["validation"], windows["test"],
            class_names, component_names, hierarchy["fault_to_component"], seed, args.epochs,
            selected_channels=selected, device=args.device, batch_size=args.batch_size,
            minimum_epochs=args.minimum_epochs, early_stop_patience=args.early_stop_patience,
            evaluate_test_split=not args.validation_only,
            gradient_audit=args.gradient_audit,
            alarm_policy_mode=args.alarm_policy,
            label_contract=args.label_contract,
            training_sampling=args.training_sampling,
            detector_feature_profile=args.detector_feature_profile,
        )
        for kind in args.models for seed in args.seeds
    ]
    report = {
        "schema_version": "astrograph-phm-v60-causal-window-comparison.v1",
        "split_audit": split_audit,
        "window": {"length_s": WINDOW, "train_validation_stride_s": TRAIN_STRIDE, "test_stride_s": EVAL_STRIDE},
        "selected_channels": selected,
        "selection_scope": "train rows only; labels used only for supervised training/selection",
        "preprocessing": {
            "schema_version": "astrograph-channel-preprocessor.v1",
            "method": args.preprocessing,
            "fit_scope": "train_only",
        },
        "feature_profile": args.feature_profile,
        "required_mechanism_channels": required_channels,
        "alarm_policy_mode": args.alarm_policy,
        "label_contract": args.label_contract,
        "training_sampling": args.training_sampling,
        "detector_feature_profile": args.detector_feature_profile,
        "models": models,
        "claim_boundary": (
            "E0 synthetic unknown-onset causal-window screening. Evaluator-only onset/manifestation "
            "timestamps define labels and delay references but are never model inputs or window boundaries."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not args.validation_only:
        (args.output_dir / "report.md").write_text(render(report))
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
