from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from astrograph.data_driven.credibility import TemperatureScaler
from astrograph.data_driven.whole_spacecraft_hgb import CASE_TO_FAULT_ID, COMPOSITE_MEMBERS


ACTIVITY_AXIS_SOURCES: dict[str, str] = {
    "fault_timing": "v0522_fault_timing_runtime_effect",
    "severity": "v0528_all_fault_severity_mapping",
    "noise": "v0523_sensor_noise_runtime_effect",
    "dropout": "v0524_sensor_dropout_runtime_effect",
    "operating_condition": "v0525_operating_condition_runtime_effect",
    "initial_state": "v0526_initial_state_runtime_effect",
    "satellite_instance": "v0527_satellite_instance_runtime_effect",
}

AGGREGATIONS = (
    "mean",
    "std",
    "min",
    "max",
    "slope",
    "residual_mean_abs",
    "residual_max_abs",
)


class FrozenModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    estimator_class: str = Field(min_length=1)
    artifact_file: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_id: str = Field(min_length=1)
    calibration_temperature: float = Field(gt=0.0)
    classes: list[str]


class FrozenModelBundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-frozen-model-bundle.v1"
    bundle_id: str = Field(min_length=1)
    feature_schema_id: str = Field(min_length=1)
    feature_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_count: int = Field(ge=1)
    train_run_ids: list[str]
    validation_run_ids: list[str]
    test_run_ids: list[str]
    random_seed: int
    models: list[FrozenModelArtifact]
    claim_boundary: str = (
        "Model artifacts are deterministically fitted from the frozen V0.5.17 feature matrix and split manifest. "
        "They remain E0 simulator artifacts and do not establish cross-condition or flight generalization."
    )


class ActivityAxisSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    run_dir: str = Field(min_length=1)
    telemetry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    truth_class_ids: list[str]
    label_mode: str
    controlled_axis_record: dict[str, Any] = Field(default_factory=dict)


class ActivityAxisPredictionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    axis: str
    variant_id: str
    scenario_id: str
    label_mode: str
    truth_class_ids: list[str]
    model_id: str
    model_version: str
    input_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_class_id: str
    top_probability: float = Field(ge=0.0, le=1.0)
    top3_class_ids: list[str]
    top1_correct: bool | None = None
    member_hit_at_3: bool | None = None
    latency_ms: float = Field(ge=0.0)


@dataclass
class _LoadedFrozenModel:
    model_id: str
    model_version: str
    estimator: Any
    scaler: TemperatureScaler
    classes: list[str]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20].upper()}"


def _normalise_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0)
    return values / values.sum(axis=1, keepdims=True)


def _model_specs(seed: int) -> list[tuple[str, str, Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install AstroGraph with the 'ml' optional dependency") from exc

    return [
        (
            "lab-hist-gradient-boosting",
            "0.5.17-independent-run-frozen-r4f2",
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.05,
                l2_regularization=0.03,
                max_leaf_nodes=15,
                random_state=seed,
            ),
        ),
        (
            "lab-random-forest",
            "0.5.17-independent-run-frozen-r4f2",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        (
            "lab-logistic-regression",
            "0.5.17-independent-run-frozen-r4f2",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
            ),
        ),
    ]


def freeze_seed_models(
    *,
    feature_matrix_path: str | Path,
    feature_manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 20260722,
) -> FrozenModelBundleManifest:
    """Fit and persist deterministic model artifacts from the frozen V0.5.17 matrix.

    This function intentionally consumes only the pre-existing feature matrix and its frozen split manifest.
    Activity-axis telemetry is never used for fitting or calibration.
    """

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - supplied by scikit-learn
        raise RuntimeError("joblib is required to persist frozen model artifacts") from exc

    feature_matrix_path = Path(feature_matrix_path)
    feature_manifest_path = Path(feature_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(feature_matrix_path).set_index("run_id")
    if "truth_class_id" not in matrix.columns:
        raise ValueError("frozen feature matrix must contain truth_class_id")
    manifest_payload = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    split_ids = manifest_payload["split_run_ids"]
    train_ids = list(split_ids["TRAIN"])
    validation_ids = list(split_ids["VALIDATION"])
    test_ids = list(split_ids["TEST"])
    all_ids = train_ids + validation_ids + test_ids
    missing = sorted(set(all_ids) - set(matrix.index))
    if missing:
        raise ValueError(f"frozen feature matrix is missing run IDs: {missing}")

    labels = matrix.pop("truth_class_id").astype(str)
    features = matrix.astype(float)
    artifacts: list[FrozenModelArtifact] = []
    for model_id, model_version, estimator in _model_specs(seed):
        estimator.fit(features.loc[train_ids], labels.loc[train_ids])
        validation_raw = _normalise_probabilities(estimator.predict_proba(features.loc[validation_ids]))
        classes = [str(value) for value in estimator.classes_]
        class_to_index = {class_id: index for index, class_id in enumerate(classes)}
        validation_truth_indices = np.asarray(
            [class_to_index[str(value)] for value in labels.loc[validation_ids]],
            dtype=int,
        )
        calibration_id = _stable_id(
            "CAL",
            {
                "model_id": model_id,
                "model_version": model_version,
                "validation_run_ids": validation_ids,
                "feature_matrix_sha256": sha256_file(feature_matrix_path),
            },
        )
        scaler, calibration_record = TemperatureScaler.fit(
            validation_raw,
            validation_truth_indices,
            calibration_id=calibration_id,
        )
        artifact_name = f"{model_id}.joblib"
        artifact_path = output_dir / artifact_name
        joblib.dump(
            {
                "schema_version": "astrograph-frozen-model-artifact.v1",
                "model_id": model_id,
                "model_version": model_version,
                "estimator": estimator,
                "classes": classes,
                "calibration": calibration_record.model_dump(mode="json"),
                "feature_columns": list(features.columns),
                "feature_schema_id": str(manifest_payload["feature_schema_id"]),
                "feature_matrix_sha256": sha256_file(feature_matrix_path),
                "train_run_ids": train_ids,
                "validation_run_ids": validation_ids,
                "test_run_ids": test_ids,
            },
            artifact_path,
            compress=3,
        )
        artifacts.append(
            FrozenModelArtifact(
                model_id=model_id,
                model_version=model_version,
                estimator_class=estimator.__class__.__name__,
                artifact_file=artifact_name,
                artifact_sha256=sha256_file(artifact_path),
                calibration_id=calibration_record.calibration_id,
                calibration_temperature=calibration_record.temperature,
                classes=classes,
            )
        )

    bundle = FrozenModelBundleManifest(
        bundle_id=_stable_id(
            "FROZEN-MODEL-BUNDLE",
            {
                "feature_matrix_sha256": sha256_file(feature_matrix_path),
                "feature_schema_id": manifest_payload["feature_schema_id"],
                "seed": seed,
                "models": [item.model_id for item in artifacts],
            },
        ),
        feature_schema_id=str(manifest_payload["feature_schema_id"]),
        feature_matrix_sha256=sha256_file(feature_matrix_path),
        feature_count=int(manifest_payload["feature_count"]),
        train_run_ids=train_ids,
        validation_run_ids=validation_ids,
        test_run_ids=test_ids,
        random_seed=seed,
        models=artifacts,
    )
    (output_dir / "frozen_model_bundle_manifest.json").write_text(
        bundle.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def load_frozen_models(bundle_dir: str | Path) -> tuple[FrozenModelBundleManifest, list[_LoadedFrozenModel], list[str]]:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("joblib is required to load frozen model artifacts") from exc

    bundle_dir = Path(bundle_dir)
    manifest = FrozenModelBundleManifest.model_validate_json(
        (bundle_dir / "frozen_model_bundle_manifest.json").read_text(encoding="utf-8")
    )
    loaded: list[_LoadedFrozenModel] = []
    feature_columns: list[str] | None = None
    for item in manifest.models:
        path = bundle_dir / item.artifact_file
        if sha256_file(path) != item.artifact_sha256:
            raise ValueError(f"frozen model artifact hash mismatch: {item.artifact_file}")
        payload = joblib.load(path)
        if payload["feature_matrix_sha256"] != manifest.feature_matrix_sha256:
            raise ValueError("frozen model artifact references a different feature matrix")
        current_columns = list(payload["feature_columns"])
        if feature_columns is None:
            feature_columns = current_columns
        elif feature_columns != current_columns:
            raise ValueError("frozen model artifacts use inconsistent feature schemas")
        loaded.append(
            _LoadedFrozenModel(
                model_id=item.model_id,
                model_version=item.model_version,
                estimator=payload["estimator"],
                scaler=TemperatureScaler(item.calibration_temperature),
                classes=list(item.classes),
            )
        )
    if feature_columns is None:
        raise ValueError("frozen model bundle contains no models")
    return manifest, loaded, feature_columns


def _event_interval(run_dir: Path, frame: pd.DataFrame) -> tuple[float, float]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8-sig"))
    events = manifest.get("events", [])
    if not events:
        return float(frame["time_s"].min()), float(frame["time_s"].max())
    start = min(float(item["start_s"]) for item in events)
    end = max(float(item["start_s"]) + float(item["duration_s"]) for item in events)
    return start, end


def _slope(values: np.ndarray, time_s: np.ndarray) -> float:
    centered = time_s - float(np.mean(time_s))
    denominator = float(np.sum(centered * centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(centered * (values - float(np.mean(values)))) / denominator)


def feature_row_for_run(
    *,
    run_dir: str | Path,
    nominal_dir: str | Path,
    feature_columns: Iterable[str],
) -> pd.Series:
    """Build the exact V0.5.17 aggregate feature row for an arbitrary runtime output."""

    run_dir = Path(run_dir)
    nominal_dir = Path(nominal_dir)
    current = pd.read_csv(run_dir / "telemetry.csv")
    nominal = pd.read_csv(nominal_dir / "telemetry.csv")
    start, end = _event_interval(run_dir, current)
    current_window = current[(current["time_s"] >= start) & (current["time_s"] <= end)].reset_index(drop=True)
    nominal_window = nominal[(nominal["time_s"] >= start) & (nominal["time_s"] <= end)].reset_index(drop=True)
    common_times = set(pd.to_numeric(current_window["time_s"], errors="coerce").dropna())
    nominal_window = nominal_window[nominal_window["time_s"].isin(common_times)].reset_index(drop=True)
    current_window = current_window[current_window["time_s"].isin(set(nominal_window["time_s"]))].reset_index(drop=True)
    if len(current_window) != len(nominal_window) or len(current_window) < 2:
        raise ValueError(f"cannot align activity-axis run with nominal baseline: {run_dir}")

    required_telemetry_columns = sorted({column.rsplit("__", 1)[0] for column in feature_columns})
    missing_current = sorted(set(required_telemetry_columns) - set(current_window.columns))
    missing_nominal = sorted(set(required_telemetry_columns) - set(nominal_window.columns))
    if missing_current or missing_nominal:
        raise ValueError(
            f"telemetry schema mismatch: missing_current={missing_current}, missing_nominal={missing_nominal}"
        )

    time_s = pd.to_numeric(current_window["time_s"], errors="coerce").to_numpy(dtype=float)
    features: dict[str, float] = {}
    for telemetry_column in required_telemetry_columns:
        current_values = pd.to_numeric(current_window[telemetry_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        nominal_values = pd.to_numeric(nominal_window[telemetry_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        residual = current_values - nominal_values
        values = {
            "mean": float(np.mean(current_values)),
            "std": float(np.std(current_values)),
            "min": float(np.min(current_values)),
            "max": float(np.max(current_values)),
            "slope": _slope(current_values, time_s),
            "residual_mean_abs": float(np.mean(np.abs(residual))),
            "residual_max_abs": float(np.max(np.abs(residual))),
        }
        for aggregation, value in values.items():
            features[f"{telemetry_column}__{aggregation}"] = value

    row = pd.Series(features, dtype=float).reindex(list(feature_columns))
    if row.isna().any():
        missing_features = list(row.index[row.isna()])
        raise ValueError(f"feature extraction did not produce required columns: {missing_features}")
    return row


def _axis_sample_id(axis: str, variant: str, scenario: str, telemetry_sha256: str) -> str:
    return _stable_id(
        "AXIS-SAMPLE",
        {"axis": axis, "variant": variant, "scenario": scenario, "telemetry_sha256": telemetry_sha256},
    )


def discover_activity_axis_samples(outputs_root: str | Path) -> list[ActivityAxisSample]:
    outputs_root = Path(outputs_root)
    samples: list[ActivityAxisSample] = []
    for axis, source_dir in ACTIVITY_AXIS_SOURCES.items():
        root = outputs_root / source_dir
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.rglob("run_manifest.json")):
            run_dir = manifest_path.parent
            telemetry_path = run_dir / "telemetry.csv"
            if not telemetry_path.is_file():
                continue
            relative_parts = manifest_path.relative_to(root).parts
            variant = relative_parts[0] if len(relative_parts) > 2 else "default"
            payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            scenario = str(payload.get("scenario") or run_dir.name)
            controlled_record = payload.get("activity_axes", {}).get(axis, {})
            if not isinstance(controlled_record, dict) or controlled_record.get("controlled") is not True:
                continue
            if scenario == "nominal":
                truth_class_ids = ["NORMAL"]
                label_mode = "SINGLE_LABEL"
            elif scenario in CASE_TO_FAULT_ID:
                truth_class_ids = [CASE_TO_FAULT_ID[scenario]]
                label_mode = "SINGLE_LABEL"
            elif scenario in COMPOSITE_MEMBERS:
                truth_class_ids = list(COMPOSITE_MEMBERS[scenario])
                label_mode = "COMPOSITE_RESEARCH"
            else:
                truth_class_ids = []
                label_mode = "UNREGISTERED_SCENARIO"
            telemetry_hash = sha256_file(telemetry_path)
            samples.append(
                ActivityAxisSample(
                    sample_id=_axis_sample_id(axis, variant, scenario, telemetry_hash),
                    axis=axis,
                    variant_id=variant,
                    scenario_id=scenario,
                    run_dir=str(run_dir),
                    telemetry_sha256=telemetry_hash,
                    manifest_sha256=sha256_file(manifest_path),
                    truth_class_ids=truth_class_ids,
                    label_mode=label_mode,
                    controlled_axis_record=controlled_record,
                )
            )
    return samples


def _canonical_nominal_dir(inventory_path: Path, *, project_root: Path, preferred_seed: int = 20260722) -> Path:
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    candidates = [
        item
        for item in records
        if item.get("class_ids") == ["NORMAL"] and int(item.get("random_seed") or -1) == preferred_seed
    ]
    if not candidates:
        candidates = [item for item in records if item.get("class_ids") == ["NORMAL"]]
    if not candidates:
        raise ValueError("independent run inventory contains no nominal run")
    record = sorted(candidates, key=lambda item: str(item.get("split")))[0]
    run_id = str(record["run_id"])
    scenario_id = str(record.get("metadata", {}).get("scenario_id") or "nominal")
    local = project_root / "outputs" / "v0516_campaign" / "runtime_outputs" / "runs" / run_id / scenario_id
    if local.is_dir():
        return local
    configured = Path(str(record.get("metadata", {}).get("scenario_dir", "")))
    if configured.is_dir():
        return configured
    raise FileNotFoundError(f"cannot resolve canonical nominal run for {run_id}")


def run_activity_axis_inference(
    *,
    project_root: str | Path,
    frozen_bundle_dir: str | Path,
    inventory_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    manifest, models, feature_columns = load_frozen_models(frozen_bundle_dir)
    samples = discover_activity_axis_samples(project_root / "outputs")
    nominal_dir = _canonical_nominal_dir(Path(inventory_path), project_root=project_root)

    rows: list[ActivityAxisPredictionRow] = []
    errors: list[dict[str, str]] = []
    for sample in samples:
        if sample.label_mode == "UNREGISTERED_SCENARIO":
            errors.append({"sample_id": sample.sample_id, "error": "UNREGISTERED_SCENARIO"})
            continue
        try:
            feature_row = feature_row_for_run(
                run_dir=sample.run_dir,
                nominal_dir=nominal_dir,
                feature_columns=feature_columns,
            )
        except Exception as exc:
            errors.append({"sample_id": sample.sample_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        feature_row_hash = hashlib.sha256(
            json.dumps(
                [(name, float(feature_row[name])) for name in feature_columns],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        frame = pd.DataFrame([feature_row.to_dict()]).reindex(columns=feature_columns)
        for model in models:
            started = time.perf_counter()
            raw = _normalise_probabilities(model.estimator.predict_proba(frame))
            probabilities = model.scaler.transform(raw)[0]
            latency_ms = (time.perf_counter() - started) * 1000.0
            ranked = sorted(
                zip(model.classes, probabilities, strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )
            top3 = [class_id for class_id, _ in ranked[:3]]
            top_class_id, top_probability = ranked[0]
            top1_correct = None
            member_hit_at_3 = None
            if sample.label_mode == "SINGLE_LABEL":
                top1_correct = top_class_id == sample.truth_class_ids[0]
            elif sample.label_mode == "COMPOSITE_RESEARCH":
                member_hit_at_3 = any(class_id in top3 for class_id in sample.truth_class_ids)
            rows.append(
                ActivityAxisPredictionRow(
                    sample_id=sample.sample_id,
                    axis=sample.axis,
                    variant_id=sample.variant_id,
                    scenario_id=sample.scenario_id,
                    label_mode=sample.label_mode,
                    truth_class_ids=sample.truth_class_ids,
                    model_id=model.model_id,
                    model_version=model.model_version,
                    input_data_sha256=sample.telemetry_sha256,
                    feature_row_sha256=feature_row_hash,
                    top_class_id=top_class_id,
                    top_probability=float(top_probability),
                    top3_class_ids=top3,
                    top1_correct=top1_correct,
                    member_hit_at_3=member_hit_at_3,
                    latency_ms=latency_ms,
                )
            )

    metrics: dict[str, Any] = {}
    for model in models:
        model_rows = [row for row in rows if row.model_id == model.model_id]
        single = [row for row in model_rows if row.label_mode == "SINGLE_LABEL"]
        composites = [row for row in model_rows if row.label_mode == "COMPOSITE_RESEARCH"]
        by_axis: dict[str, Any] = {}
        for axis in ACTIVITY_AXIS_SOURCES:
            axis_rows = [row for row in single if row.axis == axis]
            if not axis_rows:
                continue
            by_axis[axis] = {
                "sample_count": len(axis_rows),
                "top1_accuracy": sum(bool(row.top1_correct) for row in axis_rows) / len(axis_rows),
                "normal_sample_count": sum(row.truth_class_ids == ["NORMAL"] for row in axis_rows),
                "normal_false_positive_rate": (
                    sum(row.truth_class_ids == ["NORMAL"] and row.top_class_id != "NORMAL" for row in axis_rows)
                    / max(1, sum(row.truth_class_ids == ["NORMAL"] for row in axis_rows))
                ),
            }
        metrics[model.model_id] = {
            "single_label_sample_count": len(single),
            "single_label_top1_accuracy": (
                sum(bool(row.top1_correct) for row in single) / len(single) if single else None
            ),
            "normal_sample_count": sum(row.truth_class_ids == ["NORMAL"] for row in single),
            "normal_false_positive_rate": (
                sum(row.truth_class_ids == ["NORMAL"] and row.top_class_id != "NORMAL" for row in single)
                / max(1, sum(row.truth_class_ids == ["NORMAL"] for row in single))
            ),
            "composite_research_sample_count": len(composites),
            "composite_member_hit_at_3": (
                sum(bool(row.member_hit_at_3) for row in composites) / len(composites) if composites else None
            ),
            "by_axis": by_axis,
        }

    expected_prediction_count = len(samples) * len(models)
    report = {
        "schema_version": "astrograph-activity-axis-frozen-model-inference.v1",
        "frozen_model_bundle_id": manifest.bundle_id,
        "feature_schema_id": manifest.feature_schema_id,
        "canonical_nominal_dir": str(nominal_dir),
        "activity_axis_sample_count": len(samples),
        "model_count": len(models),
        "expected_prediction_count": expected_prediction_count,
        "prediction_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "samples": [sample.model_dump(mode="json") for sample in samples],
        "predictions": [row.model_dump(mode="json") for row in rows],
        "metrics": metrics,
        "status": "PASS" if len(rows) == expected_prediction_count and not errors else "FAIL",
        "claim_boundary": (
            "This report proves that frozen model artifacts consumed activity-axis telemetry without retraining. "
            "Metrics are E0 cross-condition observations only; no cross-condition, cross-satellite, OOD, or flight "
            "release claim is implied. Composite runs are research-only and scored by member hit-at-3."
        ),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
