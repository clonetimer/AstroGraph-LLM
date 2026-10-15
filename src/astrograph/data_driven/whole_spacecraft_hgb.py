"""Whole-spacecraft hierarchical gradient-boosting candidate.

STATUS (recorded 2026-08-04, ADR-004): DEVELOPMENT / NOT WIRED.  This module is
an offline engineering candidate and is NOT part of the S1 data-driven mainline
(diagnosis_baseline) nor reachable from unified/CLI.  It is kept for research
but must not be cited as a current capability.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astrograph.data_driven.contracts import (
    DataDrivenModelManifest,
    ModelInferenceRequest,
    RawModelOutput,
    DataDrivenTaskType,
    ModelCapability,
    ModelOutputSemantics,
    ModelRuntime,
    ObservationScope,
)
from astrograph.data_driven.credibility import (
    TemperatureScaler,
    TemperatureScalingRecord,
    calibration_metrics,
)
from astrograph.data_driven.models import DataDrivenPrediction
from astrograph.data_driven.registry import DataDrivenModelRegistry
from astrograph.evidence.executable import stable_id
from astrograph.assurance.feature_admission import canonical_sha256


CASE_TO_FAULT_ID = {
    "battery_capacity_loss": "EPS_BATTERY_CAPACITY_LOSS",
    "battery_internal_resistance_increase": "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE",
    "downlink_service_loss": "COMM_DOWNLINK_SERVICE_LOSS",
    "gyro_bias_drift": "ADCS_GYRO_BIAS_DRIFT",
    "heater_stuck_on": "THERMAL_HEATER_STUCK_ON",
    "radiator_rejection_loss": "THERMAL_RADIATOR_REJECTION_LOSS",
    "reaction_wheel_friction_increase": "ADCS_RW_FRICTION_INCREASE",
    "reaction_wheel_jam": "ADCS_RW_JAM",
    "reaction_wheel_torque_authority_loss": "ADCS_RW_TORQUE_AUTHORITY_LOSS",
    "solar_array_string_failure": "EPS_SOLAR_ARRAY_STRING_FAILURE",
    "solar_efficiency_loss": "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS",
    "star_tracker_dropout": "ADCS_STAR_TRACKER_DROPOUT",
    "storage_write_failure": "DATA_STORAGE_WRITE_FAILURE",
    "temperature_sensor_bias": "THERMAL_TEMPERATURE_SENSOR_BIAS",
    "temperature_sensor_drift": "THERMAL_TEMPERATURE_SENSOR_DRIFT",
    "transmitter_output_power_loss": "COMM_TRANSMITTER_OUTPUT_POWER_LOSS",
    "nominal": "NORMAL",
}

COMPOSITE_MEMBERS = {
    "eps_capacity_plus_solar": ["EPS_BATTERY_CAPACITY_LOSS", "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS"],
    "radiator_plus_heater": ["THERMAL_RADIATOR_REJECTION_LOSS", "THERMAL_HEATER_STUCK_ON"],
    "rw_friction_plus_star_tracker": ["ADCS_RW_FRICTION_INCREASE", "ADCS_STAR_TRACKER_DROPOUT"],
    "transmitter_plus_storage_write": ["COMM_TRANSMITTER_OUTPUT_POWER_LOSS", "DATA_STORAGE_WRITE_FAILURE"],
}

_ALLOWED_PREFIXES = (
    "sensor.",
    "estimate.",
    "status.",
    "counter.",
    "adcs.pointing_error_deg",
    "adcs.body_rate_rad_s_",
    "eps.net_power_w",
    "data.storage_bits",
    "data.net_baud_bps",
    "thermal.payload_temp_c",
    "comm.downlink_bps",
)


@dataclass
class WholeSpacecraftHGBModel:
    classifier: Any
    feature_columns: list[str]
    telemetry_fields: list[str]
    model_id: str
    model_version: str
    feature_schema_id: str
    training_metadata: dict[str, Any]
    manifest: DataDrivenModelManifest
    calibrator: TemperatureScaler
    calibration_record: TemperatureScalingRecord


@dataclass
class WholeSpacecraftHGBProvider:
    """Registered provider wrapper for the bundled HGB engineering baseline."""

    model: WholeSpacecraftHGBModel

    @property
    def manifest(self) -> DataDrivenModelManifest:
        return self.model.manifest

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput:
        features = request.payload.get("features")
        if not isinstance(features, dict):
            raise ValueError("HGB provider expects payload.features mapping")
        expected = list(self.model.feature_columns)
        missing = [name for name in expected if name not in features]
        unknown = sorted(set(features) - set(expected))
        if missing or unknown:
            raise ValueError(
                f"HGB feature contract mismatch; missing={missing}, unknown={unknown}"
            )
        values = [float(features[name]) for name in expected]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("HGB provider features must be finite")
        frame = pd.DataFrame([values], columns=expected)
        raw_probabilities = self.model.classifier.predict_proba(frame)
        probabilities = self.model.calibrator.transform(raw_probabilities)[0]
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=self.model.model_id,
            model_version=self.model.model_version,
            output_semantics=ModelOutputSemantics.PROBABILITIES,
            class_values={
                str(class_id): float(probability)
                for class_id, probability in zip(self.model.classifier.classes_, probabilities, strict=True)
            },
            metadata={
                "provider": "WholeSpacecraftHGBProvider",
                "calibration_id": self.model.calibration_record.calibration_id,
            },
        )


def _numeric_observable_fields(frame: pd.DataFrame) -> list[str]:
    fields = []
    for column in frame.columns:
        if column == "time_s" or column.startswith(("label.", "truth.", "event.")):
            continue
        if not any(column.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            fields.append(column)
    return sorted(fields)


def _window_feature_dict(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    fields: list[str],
) -> dict[str, float]:
    if len(current) != len(baseline):
        raise ValueError("current and baseline windows must have equal length")
    if len(current) < 2:
        raise ValueError("at least two samples are required")
    time_s = pd.to_numeric(current["time_s"], errors="coerce").to_numpy(dtype=float)
    centered = time_s - float(np.mean(time_s))
    denominator = float(np.sum(centered * centered))
    split = max(1, len(current) // 2)
    result: dict[str, float] = {}
    for field in fields:
        current_values = pd.to_numeric(current[field], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        baseline_values = pd.to_numeric(baseline[field], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        residual = current_values - baseline_values
        result[f"{field}__mean"] = float(np.mean(current_values))
        result[f"{field}__std"] = float(np.std(current_values))
        result[f"{field}__slope"] = (
            float(np.sum(centered * (current_values - float(np.mean(current_values)))) / denominator)
            if denominator > 0.0
            else 0.0
        )
        result[f"{field}__half_window_difference"] = float(
            np.mean(current_values[split:]) - np.mean(current_values[:split])
        )
        result[f"{field}__residual_mean_abs"] = float(np.mean(np.abs(residual)))
        result[f"{field}__residual_max_abs"] = float(np.max(np.abs(residual)))
    return result


def _scene_interval(scene_dir: Path, *, composite: bool = False) -> tuple[float, float]:
    if scene_dir.name == "nominal":
        return 0.0, 120.0
    events = json.loads((scene_dir / "run_manifest.json").read_text(encoding="utf-8"))["events"]
    if composite:
        return (
            min(float(item["start_s"]) for item in events),
            max(float(item["start_s"]) + float(item["duration_s"]) for item in events),
        )
    event = events[0]
    start = float(event["start_s"])
    return start, start + float(event["duration_s"])


def _aligned_interval(
    scene_dir: Path,
    nominal: pd.DataFrame,
    *,
    composite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = pd.read_csv(scene_dir / "telemetry.csv")
    start, end = _scene_interval(scene_dir, composite=composite)
    selected = current[(current["time_s"] >= start) & (current["time_s"] <= end)].reset_index(drop=True)
    baseline = nominal[nominal["time_s"].isin(selected["time_s"])].reset_index(drop=True)
    if len(selected) != len(baseline):
        raise ValueError(f"nominal alignment failed for {scene_dir.name}")
    return selected, baseline


def train_whole_spacecraft_hgb(
    scene_root: str | Path,
    *,
    window_samples: int = 21,
    step_samples: int = 4,
    seed: int = 20260722,
) -> WholeSpacecraftHGBModel:
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT')
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install AstroGraph with the 'ml' optional dependency") from exc

    root = Path(scene_root)
    nominal = pd.read_csv(root / "nominal" / "telemetry.csv")
    fields = _numeric_observable_fields(nominal)
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    split_names: list[str] = []
    for case_id in sorted(CASE_TO_FAULT_ID):
        current, baseline = _aligned_interval(root / case_id, nominal)
        if len(current) < window_samples:
            raise ValueError(f"scene {case_id} is shorter than the configured window")
        max_start = len(current) - window_samples
        for start in range(0, max_start + 1, step_samples):
            stop = start + window_samples
            rows.append(_window_feature_dict(current.iloc[start:stop], baseline.iloc[start:stop], fields))
            labels.append(CASE_TO_FAULT_ID[case_id])
            progress = start / max(1, max_start)
            split_names.append("train" if progress < 0.6 else "validation" if progress < 0.8 else "test")

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    label_series = pd.Series(labels)
    split_series = pd.Series(split_names)
    classifier = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        l2_regularization=0.03,
        max_leaf_nodes=15,
        random_state=seed,
    )
    classifier.fit(frame.loc[split_series == "train"], label_series.loc[split_series == "train"])
    validation_mask = split_series == "validation"
    validation_truth = label_series.loc[validation_mask]
    validation_probabilities = classifier.predict_proba(frame.loc[validation_mask])
    class_to_index = {str(class_id): index for index, class_id in enumerate(classifier.classes_)}
    validation_truth_indices = np.asarray([class_to_index[str(value)] for value in validation_truth], dtype=int)
    calibration_id = stable_id(
        "CAL",
        {
            "model": "whole-spacecraft-hgb-window-baseline",
            "seed": seed,
            "window_samples": window_samples,
            "step_samples": step_samples,
        },
    )
    calibrator, calibration_record = TemperatureScaler.fit(
        validation_probabilities,
        validation_truth_indices,
        calibration_id=calibration_id,
    )

    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("validation", "test"):
        mask = split_series == split_name
        truth = label_series.loc[mask]
        predicted = classifier.predict(frame.loc[mask])
        probabilities = classifier.predict_proba(frame.loc[mask])
        calibrated = calibrator.transform(probabilities)
        truth_indices = np.asarray([class_to_index[str(value)] for value in truth], dtype=int)
        top3_hits = []
        for row, label in zip(calibrated, truth, strict=True):
            indices = np.argsort(row)[::-1][:3]
            top3_hits.append(str(label) in set(classifier.classes_[indices]))
        split_metrics[split_name] = {
            "window_count": int(mask.sum()),
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "top3_recall": float(np.mean(top3_hits)) if top3_hits else 0.0,
            "calibration_before": calibration_metrics(probabilities, truth_indices).model_dump(mode="json"),
            "calibration_after": calibration_metrics(calibrated, truth_indices).model_dump(mode="json"),
        }
    feature_schema_id = stable_id(
        "FEATURE-SCHEMA",
        {"fields": fields, "aggregations": ["mean", "std", "slope", "half", "residual_mean_abs", "residual_max_abs"]},
    )
    model_id = "whole-spacecraft-hgb-window-baseline"
    model_version = "0.5.15-e0"
    manifest = DataDrivenModelManifest(
        model_id=model_id,
        model_version=model_version,
        display_name="Whole-spacecraft HGB engineering baseline",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=ModelRuntime.SCIKIT_LEARN,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[
            ModelCapability.FAULT_RANKING,
            ModelCapability.NORMAL_CLASS,
            ModelCapability.CALIBRATED_PROBABILITY,
            ModelCapability.TEMPORAL_CONTEXT,
        ],
        supported_fault_ids=sorted(set(CASE_TO_FAULT_ID.values()) - {"NORMAL"}),
        normal_class_id="NORMAL",
        feature_schema_ids=[feature_schema_id],
        calibration_id=calibration_record.calibration_id,
        training_dataset_ids=["whole-spacecraft-e0-21-scenario-source"],
        input_contract={
            "payload.features": "mapping[str,float]",
            "window_samples": window_samples,
            "paired_nominal_required": True,
        },
        metadata={
            "baseline_only": True,
            "estimator": "HistGradientBoostingClassifier",
            "not_the_only_supported_model_family": True,
        },
    )
    training_metadata = {
        "window_samples": window_samples,
        "step_samples": step_samples,
        "seed": seed,
        "training_window_count": int((split_series == "train").sum()),
        "validation_window_count": int((split_series == "validation").sum()),
        "test_window_count": int((split_series == "test").sum()),
        "split_metrics": split_metrics,
        "temperature_scaling": calibration_record.model_dump(mode="json"),
        "split_boundary": (
            "Within-run temporal split on one E0 trajectory per class. It is leakage-controlled by time order "
            "but is not independent-run, cross-satellite, or flight generalization evidence."
        ),
    }
    return WholeSpacecraftHGBModel(
        classifier=classifier,
        feature_columns=list(frame.columns),
        telemetry_fields=fields,
        model_id=model_id,
        model_version=model_version,
        feature_schema_id=feature_schema_id,
        training_metadata=training_metadata,
        manifest=manifest,
        calibrator=calibrator,
        calibration_record=calibration_record,
    )


def predict_scene(
    model: WholeSpacecraftHGBModel,
    scene_root: str | Path,
    case_id: str,
    *,
    episode_id: str,
) -> DataDrivenPrediction:
    root = Path(scene_root)
    nominal = pd.read_csv(root / "nominal" / "telemetry.csv")
    current, baseline = _aligned_interval(root / case_id, nominal, composite=case_id in COMPOSITE_MEMBERS)
    features = _window_feature_dict(current, baseline, model.telemetry_fields)
    payload = {"features": features}
    start_s = None
    end_s = None
    if "time_s" in current and not current.empty:
        numeric_time = pd.to_numeric(current["time_s"], errors="coerce").dropna()
        if not numeric_time.empty:
            start_s = float(numeric_time.iloc[0])
            end_s = float(numeric_time.iloc[-1])
    request = ModelInferenceRequest(
        episode_id=episode_id,
        model_id=model.model_id,
        model_version=model.model_version,
        feature_schema_id=model.feature_schema_id,
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=model.manifest.supported_fault_ids,
        payload=payload,
        observation_scope=ObservationScope(
            scope_id=f"whole-spacecraft-hgb:{episode_id}",
            source_id="whole-spacecraft-e0-offline-replay",
            window_start_s=start_s,
            window_end_s=end_s,
            trajectory_group_id=episode_id,
        ),
    )
    registry = DataDrivenModelRegistry.for_research()
    registry.register(WholeSpacecraftHGBProvider(model))
    prediction = registry.predict(request)
    metadata = dict(prediction.metadata)
    metadata.update(
        {
            "case_id_used_for_offline_replay_only": case_id,
            "training_metadata": model.training_metadata,
            "model_manifest": model.manifest.model_dump(mode="json"),
            "temperature_scaling": model.calibration_record.model_dump(mode="json"),
            "feature_count": len(model.feature_columns),
            "telemetry_field_count": len(model.telemetry_fields),
            "inference_path": "DataDrivenModelRegistry -> WholeSpacecraftHGBProvider -> common output adapter",
        }
    )
    return prediction.model_copy(update={"metadata": metadata})

