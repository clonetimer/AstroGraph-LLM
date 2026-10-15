#!/usr/bin/env python3
"""Compare three causal-window encoders on the frozen ADCS v5.2.2 split.

This is an architecture-screening experiment.  Validation selects checkpoints;
the test split is evaluated once after each model finishes training.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.integration import run_adcs_v522_regression as adcs
from astrograph.data_driven.three_task_tcn import (
    hierarchy_js_loss,
    make_decoupled_tcn,
    make_hierarchy_event_tcn,
    make_mechanism_gated_gat_tcn,
    make_shared_residual_tcn,
    prepare_torch_device,
    task_gradient_statistics,
)
from astrograph.data_driven.mechanism_graph import (
    load_mechanism_graph,
    load_progressive_hierarchy,
)
from astrograph.data_driven.progressive_hierarchy import (
    make_progressive_hierarchical_gat_tcn,
)
from astrograph.data_driven.mechanism_features import (
    compose_columns, load_feature_bank, profile_columns, require_available,
)
from astrograph.data_driven.alarm_policy import (
    AlarmPolicy, alarm_event_metrics, apply_alarm_policy, candidate_alarm_policies,
)
from astrograph.evaluation.data_exposure import dataset_use
from astrograph.data_driven.sota_backbones import (
    SOTA_MODEL_FAMILIES, make_sota_multitask_model,
)

MECHANISM_FEATURE_BANK = ROOT / "configs/data_driven/mechanism_feature_banks.json"
MECHANISM_GRAPH_CONFIG = ROOT / "configs/data_driven/mechanism_graphs.json"
PROGRESSIVE_HIERARCHY_CONFIG = ROOT / "configs/data_driven/progressive_hierarchies.json"


def class_weights(values: np.ndarray, count: int):
    import torch

    frequencies = np.bincount(values[values >= 0], minlength=count).astype(np.float64)
    weights = frequencies.sum() / np.maximum(frequencies * count, 1.0)
    weights = np.sqrt(weights)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def metrics(truth: np.ndarray, predicted: np.ndarray, labels: list[int]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0))


def trajectory_relative_features(data: adcs.WindowSet) -> np.ndarray:
    """Causal change features relative to each trajectory's first window."""
    absolute = adcs.summary_features(data.x)
    relative = np.empty_like(absolute)
    ids = np.asarray(data.trajectory_ids)
    for sample_id in sorted(set(data.trajectory_ids)):
        positions = np.flatnonzero(ids == sample_id)
        first = positions[np.argmin(data.end_times_s[positions])]
        relative[positions] = absolute[positions] - absolute[first]
    return relative


def fit_logistic_detector(train: adcs.WindowSet, seed: int):
    """Fit the frozen trajectory-relative detector using Train windows only."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed
    )
    model.fit(trajectory_relative_features(train), train.detection)
    return model


def logistic_detection_scores(
    train: adcs.WindowSet, data: adcs.WindowSet, seed: int, *, model=None,
) -> np.ndarray:
    """Score a split with the frozen Train-only low-capacity detector."""
    model = model if model is not None else fit_logistic_detector(train, seed)
    probabilities = model.predict_proba(trajectory_relative_features(data))
    return probabilities[:, list(model.classes_).index(1)]


def component_probabilities_from_classes(
    class_probabilities: np.ndarray,
    class_names: list[str],
    component_names: list[str],
) -> np.ndarray:
    component_index = {name: index for index, name in enumerate(component_names)}
    result = np.zeros((len(class_probabilities), len(component_names)), dtype=float)
    for class_index, fault_name in enumerate(class_names):
        result[:, component_index[adcs.FAULT_TO_COMPONENT[fault_name]]] += class_probabilities[:, class_index]
    return result


def hierarchy_blend(
    direct: np.ndarray,
    classes: np.ndarray,
    alpha: float,
    class_names: list[str],
    component_names: list[str],
) -> np.ndarray:
    derived = component_probabilities_from_classes(classes, class_names, component_names)
    blended = alpha * direct + (1.0 - alpha) * derived
    return blended / np.maximum(blended.sum(1, keepdims=True), 1e-12)


def adcs_subsystem_groups(
    feature_names: tuple[str, ...] | list[str], component_names: list[str], class_names: list[str]
) -> tuple[list[list[int]], list[int]]:
    """Bind channels and fault classes to stable ADCS component roles.

    The binding uses semantic channel families, not dataset row values.  The
    global branch still sees every channel so cross-subsystem effects are not
    hidden from the detector.
    """
    def owner(name: str) -> str:
        if ".rw." in name:
            return "reaction_wheel"
        if ".star_tracker." in name:
            return "star_tracker"
        if ".sun_sensor." in name:
            return "sun_sensor"
        if ".sensor." in name:
            return "imu"
        return "control_loop"

    groups = [
        [index for index, name in enumerate(feature_names) if owner(name) == component]
        for component in component_names
    ]
    if any(not group for group in groups):
        raise ValueError(f"every ADCS component requires at least one observable channel: {groups}")
    component_index = {name: index for index, name in enumerate(component_names)}
    class_component = [component_index[adcs.FAULT_TO_COMPONENT[name]] for name in class_names]
    return groups, class_component


def make_model(
    name: str,
    channels: int,
    components: int,
    classes: int,
    *,
    subsystem_groups: list[list[int]] | None = None,
    component_group: list[int] | None = None,
    class_component: list[int] | None = None,
    graph_edges: list[tuple[int, int]] | None = None,
    progressive_contract: dict[str, Any] | None = None,
    specialist_seed: int | None = None,
):
    import torch
    from torch import nn

    if name in SOTA_MODEL_FAMILIES:
        return make_sota_multitask_model(
            name, channels, components, classes, window_size=adcs.WINDOW
        )

    if name == "residual_tcn":
        return make_shared_residual_tcn(channels, components, classes)

    class ResidualTCNBlock(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(4, width), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(4, width), nn.GELU(),
            )

        def forward(self, x):
            return x + self.net(x)

    class EncoderHeads(nn.Module):
        def __init__(self, encoder: nn.Module, width: int) -> None:
            super().__init__()
            self.encoder = encoder
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, components)
            self.classifier = nn.Linear(width, classes)

        def forward(self, x):
            h = self.encoder(x)
            return self.detector(h), self.localizer(h), self.classifier(h)

    class TCNEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Conv1d(channels, 64, 1)
            self.blocks = nn.Sequential(*(ResidualTCNBlock(64, d) for d in (1, 2, 4)))

        def forward(self, x):
            h = self.blocks(self.input(x.transpose(1, 2)))
            return 0.5 * (h.mean(-1) + h[:, :, -1])

    class GRUEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(channels, 64, num_layers=2, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            sequence, _ = self.gru(x)
            return self.norm(0.5 * (sequence.mean(1) + sequence[:, -1]))

    class TransformerEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(channels, 64)
            self.position = nn.Parameter(torch.zeros(1, adcs.WINDOW, 64))
            layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            h = self.transformer(self.input(x) + self.position[:, : x.shape[1]])
            return self.norm(0.5 * (h.mean(1) + h[:, -1]))

    class SubsystemAwareTCN(nn.Module):
        """Global temporal context plus component-bound local encoders."""

        def __init__(self) -> None:
            super().__init__()
            if subsystem_groups is None or class_component is None:
                raise ValueError("subsystem_tcn requires channel and class hierarchy bindings")
            if len(subsystem_groups) != components or len(class_component) != classes:
                raise ValueError("hierarchy binding dimensions do not match model outputs")
            self.global_input = nn.Conv1d(channels, 32, 1)
            self.global_blocks = nn.Sequential(*(ResidualTCNBlock(32, d) for d in (1, 2, 4)))
            self.local_inputs = nn.ModuleList(nn.Conv1d(len(indices), 16, 1) for indices in subsystem_groups)
            self.local_blocks = nn.ModuleList(
                nn.Sequential(*(ResidualTCNBlock(16, d) for d in (1, 2)))
                for _ in subsystem_groups
            )
            for index, values in enumerate(subsystem_groups):
                self.register_buffer(f"subsystem_index_{index}", torch.tensor(values, dtype=torch.long))
            fused_width = 32 + 16 * len(subsystem_groups)
            self.detector = nn.Linear(fused_width, 2)
            self.localizer = nn.ModuleList(nn.Linear(48, 1) for _ in subsystem_groups)
            self.classifier = nn.ModuleList(nn.Linear(48, 1) for _ in class_component)
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
                head(torch.cat([global_state, local_states[index]], dim=1))
                for index, head in enumerate(self.localizer)
            ], dim=1)
            classification = torch.cat([
                head(torch.cat([global_state, local_states[self.class_component[index]]], dim=1))
                for index, head in enumerate(self.classifier)
            ], dim=1)
            return detection, localization, classification

    if name == "subsystem_tcn":
        return SubsystemAwareTCN()
    if name == "hierarchy_event_tcn":
        return make_hierarchy_event_tcn(
            channels, components, classes, window_size=adcs.WINDOW
        )
    if name == "mechanism_gated_gat_tcn":
        if subsystem_groups is None or component_group is None or class_component is None:
            raise ValueError("mechanism_gated_gat_tcn requires hierarchy bindings")
        return make_mechanism_gated_gat_tcn(
            channels, components, classes,
            subsystem_groups=subsystem_groups,
            component_group=component_group,
            class_component=class_component,
            graph_edges=graph_edges or [],
        )
    if name == "progressive_hierarchical_gat_tcn":
        if progressive_contract is None or class_component is None:
            raise ValueError("progressive_hierarchical_gat_tcn requires its reviewed contract")
        return make_progressive_hierarchical_gat_tcn(
            channels, components, classes,
            parameter_groups=progressive_contract["parameter_groups"],
            parameter_part=progressive_contract["parameter_part"],
            part_component=progressive_contract["part_component"],
            component_module=progressive_contract["component_module"],
            module_subsystem=progressive_contract["module_subsystem"],
            class_component=class_component,
        )
    if name == "gru_inception_specialist":
        if specialist_seed is None:
            raise ValueError("gru_inception_specialist requires specialist_seed")
        # Match each standalone reference model's initialization. Constructing
        # one branch must not change the other branch through RNG consumption.
        torch.manual_seed(specialist_seed)
        detection_model = EncoderHeads(GRUEncoder(), 64)
        torch.manual_seed(specialist_seed)
        diagnosis_model = make_sota_multitask_model(
            "inception_time", channels, components, classes, window_size=adcs.WINDOW,
        )

        class GRUInceptionSpecialist(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.detection_expert = detection_model
                self.diagnosis_expert = diagnosis_model
                self.branching_mode = "gru_detection_plus_inception_diagnosis_specialists"
                self.receptive_field = adcs.WINDOW

            def forward(self, value):
                detection = self.detection_expert(value)
                diagnosis = self.diagnosis_expert(value)
                return detection[0], diagnosis[1], diagnosis[2]

        return GRUInceptionSpecialist()
    if name in {"pooled_task_adapter_tcn", "late_fork_tcn", "tied_late_fork_tcn", "capacity_matched_shared_tcn"}:
        return make_decoupled_tcn(name, channels, components, classes)
    encoders = {"residual_tcn": TCNEncoder, "gru": GRUEncoder, "transformer": TransformerEncoder}
    return EncoderHeads(encoders[name](), 64)


def predict(model: Any, data: adcs.WindowSet, batch_size: int = 1024, *, device: str = "cpu"):
    import torch

    model.eval()
    outputs = [[], [], []]
    with torch.no_grad():
        for start in range(0, len(data.x), batch_size):
            logits = model(torch.from_numpy(data.x[start : start + batch_size]).to(device))
            for target, value in zip(outputs, logits, strict=True):
                target.append(torch.softmax(value, 1).cpu().numpy())
    return tuple(np.concatenate(item) for item in outputs)


def concatenate_windows(parts: list[adcs.WindowSet]) -> adcs.WindowSet:
    """Combine frozen partitions for a separately generated nominal cohort."""
    return adcs.WindowSet(
        x=np.concatenate([part.x for part in parts]),
        detection=np.concatenate([part.detection for part in parts]),
        isolation=np.concatenate([part.isolation for part in parts]),
        trajectory_ids=[item for part in parts for item in part.trajectory_ids],
        end_times_s=np.concatenate([part.end_times_s for part in parts]),
        onset_times_s=np.concatenate([part.onset_times_s for part in parts]),
        component=np.concatenate([part.component for part in parts]),
    )


def namespace_windows(data: adcs.WindowSet, namespace: str) -> adcs.WindowSet:
    """Prevent independently generated cohorts with reused sample IDs merging."""
    return adcs.WindowSet(
        x=data.x,
        detection=data.detection,
        isolation=data.isolation,
        trajectory_ids=[f"{namespace}:{item}" for item in data.trajectory_ids],
        end_times_s=data.end_times_s,
        onset_times_s=data.onset_times_s,
        component=data.component,
    )


def nominal_envelope_report(
    model: Any,
    logistic_detector: Any,
    train: adcs.WindowSet,
    cohort: adcs.WindowSet,
    selection: dict[str, Any],
    *,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Measure false alarms on an independently seeded all-nominal cohort."""
    if np.any(cohort.detection != 0):
        raise ValueError("nominal envelope cohort contains fault windows")
    deep, _, _ = predict(model, cohort, device=device)
    low_capacity = logistic_detection_scores(
        train, cohort, seed, model=logistic_detector
    )
    alpha = float(selection["detector_deep_blend_alpha"])
    scores = alpha * deep[:, 1] + (1.0 - alpha) * low_capacity
    policy = AlarmPolicy(
        on_threshold=float(selection["alarm_on_threshold"]),
        off_threshold=float(selection["alarm_off_threshold"]),
        minimum_consecutive=int(selection["alarm_minimum_consecutive"]),
    )
    alarms = np.asarray(apply_alarm_policy(
        scores.tolist(), cohort.trajectory_ids, cohort.end_times_s.tolist(), policy,
    ), dtype=bool)
    trajectories = sorted(set(cohort.trajectory_ids))
    alarmed = sorted({sample for sample, alarm in zip(cohort.trajectory_ids, alarms, strict=True) if alarm})
    exposure_s = sum(
        float(cohort.end_times_s[[sample == item for item in cohort.trajectory_ids]].max())
        for sample in trajectories
    )
    return {
        "trajectory_count": len(trajectories),
        "window_count": len(alarms),
        "alarmed_trajectory_count": len(alarmed),
        "trajectory_false_alarm_rate": len(alarmed) / len(trajectories),
        "false_alarm_window_count": int(alarms.sum()),
        "window_false_positive_rate": float(alarms.mean()),
        "false_alarm_windows_per_hour": float(alarms.sum()) / (exposure_s / 3600.0),
        "alarmed_trajectory_ids": alarmed,
        "selection_frozen_before_cohort_scoring": True,
    }


def validation_score(
    model: Any,
    data: adcs.WindowSet,
    class_names: list[str],
    component_names: list[str],
    detection_score: np.ndarray,
    *,
    device: str = "cpu",
    alarm_policy_mode: str = "point",
) -> dict[str, Any]:
    detection, component, classes = predict(model, data, device=device)
    active = data.isolation >= 0
    class_by_component = adcs.component_class_indices(class_names, component_names)
    best_detection: dict[str, Any] = {
        "detection_selection_score": -1.0,
        "_policy_rank": (-1, float("-inf"), float("-inf")),
    }
    thresholds = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995)
    policies = candidate_alarm_policies(alarm_policy_mode, thresholds)
    if alarm_policy_mode == "hysteresis":
        policies.extend(
            AlarmPolicy(value, max(0.0, value - 0.1), consecutive)
            for value in thresholds for consecutive in (5, 10)
        )
        policies = list(dict.fromkeys(policies))
    for detector_blend in (0.0, 0.25, 0.5, 0.75, 1.0):
        calibrated_detection = detector_blend * detection[:, 1] + (1.0 - detector_blend) * detection_score
        for policy in policies:
            alarm = np.asarray(apply_alarm_policy(
                calibrated_detection.tolist(), data.trajectory_ids,
                data.end_times_s.tolist(), policy,
            ))
            detection_f1 = metrics(data.detection, alarm, [0, 1])
            event = alarm_event_metrics(
                data.detection.tolist(), alarm.tolist(), data.trajectory_ids,
                data.end_times_s.tolist(), data.onset_times_s.tolist(),
            )
            nominal_groups = {
                group for group in set(data.trajectory_ids)
                if not np.any(data.detection[np.asarray(data.trajectory_ids) == group])
            }
            alarmed_nominal = {
                group for group, predicted in zip(data.trajectory_ids, alarm, strict=True)
                if group in nominal_groups and predicted
            }
            nominal_trajectory_far = (
                len(alarmed_nominal) / len(nominal_groups) if nominal_groups else 0.0
            )
            selection_score = (
                0.70 * float(event["fault_event_recall"]) + 0.30 * detection_f1
            )
            meets_false_alarm_constraint = nominal_trajectory_far <= 0.05
            policy_rank = (
                int(meets_false_alarm_constraint),
                selection_score if meets_false_alarm_constraint else -nominal_trajectory_far,
                selection_score,
            )
            if policy_rank > best_detection["_policy_rank"]:
                best_detection = {
                    "detection_macro_f1": detection_f1,
                    "detection_selection_score": selection_score,
                    "fault_event_recall": event["fault_event_recall"],
                    "nominal_trajectory_false_alarm_rate": nominal_trajectory_far,
                    "false_alarm_episodes_per_hour": event["false_alarm_episodes_per_hour"],
                    "nominal_trajectory_false_alarm_constraint": 0.05,
                    "meets_nominal_false_alarm_constraint": meets_false_alarm_constraint,
                    "detector_deep_blend_alpha": detector_blend,
                    "detection_threshold": policy.on_threshold,
                    "alarm_on_threshold": policy.on_threshold,
                    "alarm_off_threshold": policy.off_threshold,
                    "alarm_minimum_consecutive": policy.minimum_consecutive,
                    "_policy_rank": policy_rank,
                }
    best_diagnosis: dict[str, Any] = {"diagnosis_score": -1.0}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = hierarchy_blend(component, classes, alpha, class_names, component_names)
        conditioned, _ = adcs._conditioned_class_probabilities(
            classes, blended, class_by_component, mode="hard"
        )
        localization_f1 = metrics(
            data.component[active], blended[active].argmax(1), list(range(len(component_names)))
        )
        classification_f1 = metrics(
            data.isolation[active], conditioned[active].argmax(1), list(range(len(class_names)))
        )
        diagnosis_score = 0.3 * localization_f1 + 0.5 * classification_f1
        if diagnosis_score > best_diagnosis["diagnosis_score"]:
            best_diagnosis = {
                "localization_macro_f1": localization_f1,
                "classification_macro_f1": classification_f1,
                "hierarchy_blend_alpha": alpha,
                "conditioning_mode": "hard",
                "diagnosis_score": diagnosis_score,
            }
    result = {**best_detection, **best_diagnosis}
    result.pop("diagnosis_score")
    result.pop("_policy_rank")
    result["selection_score"] = (
        0.2 * result["detection_selection_score"]
        + 0.3 * result["localization_macro_f1"]
        + 0.5 * result["classification_macro_f1"]
    )
    return result


def checkpoint_validation_score(
    model: Any,
    data: adcs.WindowSet,
    class_names: list[str],
    component_names: list[str],
    detection_score: np.ndarray,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Cheap epoch score; full trajectory policy search runs after training."""
    detection, component, classes = predict(model, data, device=device)
    active = data.isolation >= 0
    class_by_component = adcs.component_class_indices(class_names, component_names)
    detection_f1 = max(
        metrics(
            data.detection,
            (alpha * detection[:, 1] + (1.0 - alpha) * detection_score >= 0.5).astype(int),
            [0, 1],
        )
        for alpha in (0.0, 0.5, 1.0)
    )
    best_diagnosis = 0.0
    localization_f1 = 0.0
    classification_f1 = 0.0
    for alpha in (0.0, 0.5, 1.0):
        blended = hierarchy_blend(component, classes, alpha, class_names, component_names)
        conditioned, _ = adcs._conditioned_class_probabilities(
            classes, blended, class_by_component, mode="hard"
        )
        local = metrics(
            data.component[active], blended[active].argmax(1), list(range(len(component_names)))
        )
        classification = metrics(
            data.isolation[active], conditioned[active].argmax(1), list(range(len(class_names)))
        )
        combined = 0.3 * local + 0.5 * classification
        if combined > best_diagnosis:
            best_diagnosis = combined
            localization_f1 = local
            classification_f1 = classification
    return {
        "detection_macro_f1": detection_f1,
        "localization_macro_f1": localization_f1,
        "classification_macro_f1": classification_f1,
        "selection_score": 0.2 * detection_f1 + best_diagnosis,
    }


def train_model(
    name: str,
    train: adcs.WindowSet,
    validation: adcs.WindowSet,
    test: adcs.WindowSet,
    class_names: list[str],
    component_names: list[str],
    *,
    epochs: int,
    seed: int,
    device: str = "cpu",
    batch_size: int = 384,
    minimum_epochs: int = 0,
    early_stop_patience: int | None = None,
    evaluate_test_split: bool = True,
    gradient_audit: bool = False,
    selected_channels: list[str] | tuple[str, ...] | None = None,
    alarm_policy_mode: str = "point",
    artifact_callback=None,
    external_nominal: adcs.WindowSet | None = None,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch_device = prepare_torch_device(device, seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    validation_detection_score = logistic_detection_scores(train, validation, seed)
    test_detection_score = (
        logistic_detection_scores(train, test, seed) if evaluate_test_split else None
    )
    subsystem_groups, class_component = adcs_subsystem_groups(
        selected_channels or adcs.FEATURE_COLUMNS, component_names, class_names
    )
    component_group = list(range(len(component_names)))
    graph_edges: list[tuple[int, int]] = []
    graph_receipt = None
    if name == "mechanism_gated_gat_tcn":
        graph_edges, graph_receipt = load_mechanism_graph(
            MECHANISM_GRAPH_CONFIG, "adcs_v5_2_2", component_names
        )
    progressive_contract = None
    if name == "progressive_hierarchical_gat_tcn":
        progressive_contract = load_progressive_hierarchy(
            PROGRESSIVE_HIERARCHY_CONFIG, "adcs_v5_2_2",
            list(selected_channels or adcs.FEATURE_COLUMNS), component_names,
        )
        graph_receipt = progressive_contract
    model = make_model(
        name, train.x.shape[-1], len(component_names), len(class_names),
        subsystem_groups=subsystem_groups, component_group=component_group,
        class_component=class_component, graph_edges=graph_edges,
        progressive_contract=progressive_contract, specialist_seed=seed,
    ).to(device)
    # Model families consume different amounts of RNG during construction.
    # Reset before optimization so shared paths see the same stochastic stream.
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.x), torch.from_numpy(train.detection),
            torch.from_numpy(train.component), torch.from_numpy(train.isolation),
        ),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
        pin_memory=device.startswith("cuda"),
    )
    losses = (
        nn.CrossEntropyLoss(weight=class_weights(train.detection, 2).to(device)),
        nn.CrossEntropyLoss(weight=class_weights(train.component, len(component_names)).to(device)),
        nn.CrossEntropyLoss(weight=class_weights(train.isolation, len(class_names)).to(device)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_validation: dict[str, float] = {"selection_score": -1.0}
    best_epoch = 0
    specialist = name == "gru_inception_specialist"
    task_best_epochs = {"detection": 0, "diagnosis": 0}
    task_best_scores = {"detection": -1.0, "diagnosis": -1.0}
    task_best_states: dict[str, dict[str, Any]] = {}
    history = []
    gradient_history = []
    started = time.perf_counter()
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
            if specialist:
                detection_outputs = model.detection_expert(x)
                diagnosis_outputs = model.diagnosis_expert(x)
                dlogits = detection_outputs[0]
                llogits, clogits = diagnosis_outputs[1:]
            else:
                dlogits, llogits, clogits = model(x)
            active = isolation >= 0
            task_losses = [losses[0](dlogits, detection)]
            if specialist:
                task_losses.append(losses[0](diagnosis_outputs[0], detection))
            if active.any():
                task_losses.extend([
                    losses[1](llogits[active], component[active]),
                    losses[2](clogits[active], isolation[active]),
                ])
                consistency_weight = float(getattr(model, "consistency_weight", 0.0))
                if consistency_weight:
                    task_losses.append(consistency_weight * hierarchy_js_loss(
                        llogits, clogits, class_component, active
                    ))
                if specialist:
                    # Keep both experts' original three-task auxiliary
                    # supervision, while inference selects the strongest head
                    # for each task.
                    task_losses.extend([
                        losses[1](detection_outputs[1][active], component[active]),
                        losses[2](detection_outputs[2][active], isolation[active]),
                    ])
            regularization = (
                model.graph_regularization_loss()
                if hasattr(model, "graph_regularization_loss") else 0.0
            )
            loss = sum(task_losses) + regularization
            if gradient_audit and epoch_gradient is None:
                epoch_gradient = task_gradient_statistics(task_losses, model.parameters())
            loss.backward()
            if specialist:
                # Preserve independent optimization: a large gradient in one
                # expert must not rescale the other expert's update.
                nn.utils.clip_grad_norm_(
                    model.detection_expert.parameters(), 2.0,
                )
                nn.utils.clip_grad_norm_(
                    model.diagnosis_expert.parameters(), 2.0,
                )
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach()) * len(x)
        scheduler.step()
        if epoch_gradient is not None:
            gradient_history.append({"epoch": epoch, **epoch_gradient})
        val = checkpoint_validation_score(
            model, validation, class_names, component_names, validation_detection_score,
            device=device,
        )
        val["epoch"] = epoch
        val["train_loss"] = running / len(train.x)
        history.append(val)
        print(f"[{name}] epoch={epoch} loss={val['train_loss']:.4f} val={val['selection_score']:.4f}", flush=True)
        if val["selection_score"] > best_validation["selection_score"]:
            best_validation = val
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if specialist:
            diagnosis_score = (
                0.375 * val["localization_macro_f1"]
                + 0.625 * val["classification_macro_f1"]
            )
            if val["detection_macro_f1"] > task_best_scores["detection"]:
                task_best_scores["detection"] = val["detection_macro_f1"]
                task_best_epochs["detection"] = epoch
                task_best_states["detection_expert"] = copy.deepcopy(
                    model.detection_expert.state_dict()
                )
            if diagnosis_score > task_best_scores["diagnosis"]:
                task_best_scores["diagnosis"] = diagnosis_score
                task_best_epochs["diagnosis"] = epoch
                task_best_states["diagnosis_expert"] = copy.deepcopy(
                    model.diagnosis_expert.state_dict()
                )
            best_epoch = max(task_best_epochs.values())
        if (
            early_stop_patience is not None
            and epoch >= max(1, minimum_epochs)
            and epoch - best_epoch >= early_stop_patience
        ):
            print(f"[{name}] early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    if specialist:
        model.detection_expert.load_state_dict(task_best_states["detection_expert"])
        model.diagnosis_expert.load_state_dict(task_best_states["diagnosis_expert"])
    else:
        model.load_state_dict(best_state)
    final_validation = validation_score(
        model, validation, class_names, component_names, validation_detection_score,
        device=device, alarm_policy_mode=alarm_policy_mode,
    )
    final_validation["epoch"] = best_epoch
    final_validation["checkpoint_selection_score"] = best_validation["selection_score"]
    best_validation = final_validation
    logistic_detector = fit_logistic_detector(train, seed)
    if artifact_callback is not None:
        artifact_callback(model, logistic_detector, best_validation)
    run_metadata = {
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "task_best_epochs": task_best_epochs if specialist else None,
        "task_best_scores": task_best_scores if specialist else None,
        "best_validation": best_validation,
        "training_seconds": time.perf_counter() - started,
        "seed": seed,
        "device": device,
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
        "cuda_device_name": (
            torch.cuda.get_device_name(torch_device) if device.startswith("cuda") else None
        ),
        "branching_mode": getattr(model, "branching_mode", "fully_shared"),
        "mechanism_graph": graph_receipt,
        "graph_diagnostics": (
            model.graph_diagnostics() if hasattr(model, "graph_diagnostics") else None
        ),
    }
    if external_nominal is not None:
        run_metadata["independent_nominal_envelope"] = nominal_envelope_report(
            model, logistic_detector, train, external_nominal, best_validation,
            seed=seed, device=device,
        )
    if not evaluate_test_split:
        return {
            "family": name,
            **run_metadata,
            "validation_history": history,
            "evaluation_scope": "validation_only; test scores were not computed",
        }
    detection, component, classes = predict(model, test, device=device)
    detection_score = (
        float(best_validation["detector_deep_blend_alpha"]) * detection[:, 1]
        + (1.0 - float(best_validation["detector_deep_blend_alpha"])) * test_detection_score
    )
    component = hierarchy_blend(
        component,
        classes,
        float(best_validation["hierarchy_blend_alpha"]),
        class_names,
        component_names,
    )
    alarm_policy = AlarmPolicy(
        on_threshold=float(best_validation["alarm_on_threshold"]),
        off_threshold=float(best_validation["alarm_off_threshold"]),
        minimum_consecutive=int(best_validation["alarm_minimum_consecutive"]),
    )
    detection_predicted = np.asarray(apply_alarm_policy(
        detection_score.tolist(), test.trajectory_ids, test.end_times_s.tolist(), alarm_policy,
    ))
    report, _ = adcs.three_stage_report(
        family=f"logistic_detector_{name}_hierarchy_blend",
        test=test,
        det_score=detection_score,
        component_prob=component,
        all_class_prob=classes,
        class_names=class_names,
        component_names=component_names,
        mechanism={
            "residual_tcn": "three residual dilated temporal-convolution blocks; mean+last pooling; global three-task heads",
            "gru": "two-layer GRU finite-state encoder; mean+last pooling; global three-task heads",
            "transformer": "two-layer four-head self-attention over the past-only window; mean+last pooling; global three-task heads",
            "subsystem_tcn": "global residual TCN plus component-bound causal TCN branches; localization and class logits read only global plus their bound component state",
            "hierarchy_event_tcn": "window-covering residual dilated TCN; mean+max+last event pooling; train-only class-to-component consistency regularization",
            "pooled_task_adapter_tcn": "fully shared residual TCN with separate zero-initialized bottleneck residual adapters for detection and diagnosis states",
            "late_fork_tcn": "two shared residual temporal blocks followed by independent detection and diagnosis dilation-4 blocks",
            "tied_late_fork_tcn": "late-fork computation with the final detection and diagnosis block parameters tied as a mechanism-destruction control",
            "capacity_matched_shared_tcn": "widened fully shared residual TCN used as a parameter-capacity control",
            "mechanism_gated_gat_tcn": "shared global causal TCN plus subsystem-local causal encoders and low-initialized masked GAT for localization/classification; detector logits remain graph-free and topology comes from reviewed configuration",
            "progressive_hierarchical_gat_tcn": "graph-free global detector plus reviewed parameter-to-part-to-component-to-functional-module-to-subsystem-to-spacecraft gated attention for localization/classification",
            "gru_inception_specialist": "independently optimized full multitask GRU and InceptionTime experts; inference uses the GRU detection head and InceptionTime localization/classification heads",
            "inception_time": "six-module InceptionTime adaptation with multi-scale temporal kernels and residual shortcuts; common three-task heads",
            "timesnet": "two TimesBlocks with FFT-selected periods and temporal 2D inception modeling; common three-task heads",
            "modern_tcn": "four large-kernel depthwise temporal blocks with pointwise channel mixing; common three-task heads",
        }[name],
        extra={
            **run_metadata,
            "detector": "class-balanced logistic regression on causal-window summary",
            "hierarchy_blend_alpha": best_validation["hierarchy_blend_alpha"],
            "detector_deep_blend_alpha": best_validation["detector_deep_blend_alpha"],
            "alarm_policy": alarm_policy.to_dict(),
            "alarm_policy_mode": alarm_policy_mode,
        },
        detection_threshold=float(best_validation["detection_threshold"]),
        detection_predicted=detection_predicted,
        conditioning_mode=str(best_validation["conditioning_mode"]),
    )
    report["validation_history"] = history
    return report


def render(report: dict[str, Any]) -> str:
    rows = []
    for item in report["models"]:
        rows.append(
            f"| {item['family']} | {item['parameter_count']} | {item['best_epoch']} | "
            f"{item['detection']['macro_f1']:.4f} | {item['localization']['macro_f1']:.4f} | "
            f"{item['classification_predicted_component']['macro_f1']:.4f} | "
            f"{item['end_to_end_15_class']['macro_f1']:.4f} | {item['training_seconds']:.1f}s |"
        )
    return """# ADCS v5.2.2 深度编码器筛选

同一 frozen group split、20 s 因果窗口、Train-only scaler。Validation 选择 checkpoint，Test 每模型只评估一次。

| 模型 | 参数 | best epoch | Detection F1 | Localization F1 | Cascade class F1 | E2E F1 | 训练时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n\n单种子架构筛选，不支持机制优越性或正式模型晋升声明。\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=adcs.DEFAULT_DATASET)
    parser.add_argument("--development-nominal-root", type=Path)
    parser.add_argument("--external-nominal-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/integration/out/adcs_v522_deep_comparison")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=["residual_tcn", "gru", "transformer", "subsystem_tcn", "hierarchy_event_tcn", "pooled_task_adapter_tcn", "late_fork_tcn", "tied_late_fork_tcn", "capacity_matched_shared_tcn", "mechanism_gated_gat_tcn", "progressive_hierarchical_gat_tcn", "gru_inception_specialist", *SOTA_MODEL_FAMILIES], default=["inception_time", "timesnet", "modern_tcn", "residual_tcn"])
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--minimum-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--gradient-audit", action="store_true")
    parser.add_argument("--preprocessing", choices=["standard", "robust_clip"], default="standard")
    parser.add_argument("--alarm-policy", choices=["point", "hysteresis"], default="point")
    parser.add_argument(
        "--feature-profile",
        choices=["base", "innovation", "rw_execution", "rw_current_pair", "full"],
        default="base",
    )
    args = parser.parse_args()

    trajectories = adcs.load_trajectories(args.dataset_root)
    split_audit = adcs.audit_split(trajectories)
    if split_audit["status"] != "PASS":
        raise RuntimeError(split_audit)
    class_names = sorted({row.diagnostic_class for row in trajectories if row.diagnostic_class != "nominal"})
    component_names = list(adcs.COMPONENT_NAMES)
    bank = load_feature_bank(MECHANISM_FEATURE_BANK)
    feature_columns = compose_columns(
        adcs.FEATURE_COLUMNS, profile_columns(bank, "adcs_v5_2_2", args.feature_profile)
    )
    available = list(pd.read_csv(trajectories[0].telemetry_path, nrows=0).columns)
    require_available(feature_columns, available)
    windows = adcs.build_windows(
        trajectories,
        {name: index for index, name in enumerate(class_names)},
        {name: index for index, name in enumerate(component_names)},
        feature_columns=feature_columns,
    )
    if args.development_nominal_root is not None:
        development_trajectories = adcs.load_trajectories(args.development_nominal_root)
        if any(item.diagnostic_class != "nominal" for item in development_trajectories):
            raise ValueError("--development-nominal-root must contain only nominal trajectories")
        development_parts = {
            name: namespace_windows(part, "development-normal")
            for name, part in adcs.build_windows(
                development_trajectories,
                {name: index for index, name in enumerate(class_names)},
                {name: index for index, name in enumerate(component_names)},
                feature_columns=feature_columns,
            ).items()
        }
        windows["train"] = concatenate_windows([windows["train"], development_parts["train"]])
        windows["validation"] = concatenate_windows([
            windows["validation"], development_parts["validation"],
            development_parts["test"],
        ])
    train_groups = sorted(set(windows["train"].trajectory_ids))
    with dataset_use(
        "SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT", group_ids=train_groups,
    ):
        preprocessor = adcs.fit_train_preprocessor(
            windows["train"].x, method=args.preprocessing
        )
    scaled = {
        name: adcs.preprocess_windows(value, preprocessor) for name, value in windows.items()
    }
    external_nominal = None
    if args.external_nominal_root is not None:
        external_trajectories = adcs.load_trajectories(args.external_nominal_root)
        if any(item.diagnostic_class != "nominal" for item in external_trajectories):
            raise ValueError("--external-nominal-root must contain only nominal trajectories")
        external_parts = adcs.build_windows(
            external_trajectories,
            {name: index for index, name in enumerate(class_names)},
            {name: index for index, name in enumerate(component_names)},
            feature_columns=feature_columns,
        )
        external_nominal = adcs.preprocess_windows(
            namespace_windows(
                concatenate_windows(list(external_parts.values())), "confirmation-normal",
            ),
            preprocessor,
        )
    models = [
        train_model(
            name, scaled["train"], scaled["validation"], scaled["test"],
            class_names, component_names, epochs=args.epochs, seed=args.seed, device=args.device,
            batch_size=args.batch_size, minimum_epochs=args.minimum_epochs,
            early_stop_patience=args.early_stop_patience,
            evaluate_test_split=not args.validation_only,
            gradient_audit=args.gradient_audit,
            selected_channels=feature_columns,
            alarm_policy_mode=args.alarm_policy,
            external_nominal=external_nominal,
        )
        for name in args.models
    ]
    report = {
        "schema_version": "astrograph-adcs-v522-deep-comparison.v1",
        "split_audit": split_audit,
        "window_counts": {name: len(value.x) for name, value in scaled.items()},
        "selection": "validation weighted macro-F1: detection .2 + localization .3 + classification .5",
        "preprocessing": preprocessor.receipt(),
        "feature_profile": args.feature_profile,
        "feature_columns": feature_columns,
        "alarm_policy_mode": args.alarm_policy,
        "external_nominal_root": (
            str(args.external_nominal_root.resolve()) if args.external_nominal_root else None
        ),
        "development_nominal_root": (
            str(args.development_nominal_root.resolve())
            if args.development_nominal_root else None
        ),
        "development_nominal_partition_policy": (
            "indices 0,1,2,5,6,7 augment train; 3,4,8,9 augment validation"
            if args.development_nominal_root else None
        ),
        "models": models,
        "claim_boundary": "Single-seed architecture screening on E0 synthetic data; no production promotion.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not args.validation_only:
        (args.output_dir / "report.md").write_text(render(report))
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
